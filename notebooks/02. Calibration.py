#!/usr/bin/env python3
# ============================================================
# SINGLE-SCENARIO pyWBM calibration
#
# Purpose:
#   Run exactly ONE calibration scenario + ONE metric per process.
#   Task ID encodes both: task_id = scenario_idx * n_metrics + metric_idx
#   This avoids the JAX/XLA memory buildup that crashed the
#   all-scenarios batch script.
#
# Recommended usage:
#   python single_scenario_calibration.py 0
#   or with SLURM_ARRAY_TASK_ID
#
# SLURM array size = n_scenarios * len(METRICS_TO_RUN)
#   e.g. if n_scenarios=570 and 4 metrics: --array=0-2279
#
# Output:
#   - one scenario NetCDF per task
# ============================================================

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
from datetime import datetime

import numpy as np
import xarray as xr
import jax
import jax.numpy as jnp
import optax

# ============================================================
# PATHS
# ============================================================
main_dir = "/data/keeling/a/tahsina2/Alam_et_al_2026" #Update with yours
DATA_DIR = f"{main_dir}/data"
SRC_DIR  = f"{main_dir}/src"

INPUTS_NC   = f"{DATA_DIR}/inputs_by_station_noleap.nc"
HIST_RAW_NC = f"{DATA_DIR}/Historical_Inputs.nc"
HIST_BC_NC  = f"{DATA_DIR}/Historical_Inputs_biascorrected.nc"
LAI_NC      = f"{DATA_DIR}/LAI_GLDAS_clima_NLDASgrid.nc"

sys.path.append(SRC_DIR)

# ============================================================
# IMPORT MODEL / PARAMS
# ============================================================
from water_balance_jax import wbm_jax, construct_Kpet_crop, construct_Kpet_gen
from initial_params import constants as CONSTANTS_VEC, initial_params as INITIAL_THETA
from param_bounds import params_lower as PARAMS_LOWER, params_upper as PARAMS_UPPER
from param_names import param_names as PARAM_NAMES

# ============================================================
# USER SETTINGS
# ============================================================
INSITU_FORCING_SOURCE = "gridmet"

METRICS_TO_RUN = ["kgeprime", "rmse", "outer50rmse", "kge"]

LEARNING_RATE = 1e-3
N_EPOCHS_MIN = 10
N_EPOCHS_MAX = 30
PATIENCE = 5

INSITU20_ENDDATE = "2022-12-31"
INSITU7_STARTDATE = "2016-01-01"
INSITU7_ENDDATE = "2022-12-31"
SAT_STARTDATE = "2016-01-01"
SAT_ENDDATE = "2022-12-31"
# ============================================================
# RUN TAG
# ============================================================
if "SLURM_ARRAY_JOB_ID" in os.environ:
    RUN_TAG = f"job{os.environ['SLURM_ARRAY_JOB_ID']}"
elif "SLURM_JOB_ID" in os.environ:
    RUN_TAG = f"job{os.environ['SLURM_JOB_ID']}"
else:
    RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")

# ============================================================
# SCENARIOS
# ============================================================

def build_scenarios(ds_inputs, ds_hist_raw):
    input_stations = [str(s).strip() for s in ds_inputs["station"].values]
    hist_stations  = [str(s).strip() for s in ds_hist_raw["station"].values]

    stations = sorted(set(input_stations).intersection(set(hist_stations)))

    missing_from_hist = sorted(set(input_stations) - set(hist_stations))
    if missing_from_hist:
        print("Skipping stations missing from Historical_Inputs:")
        print(missing_from_hist)

    print(f"Using {len(stations)} stations:")
    print(stations)

    scenarios = []

    for st in stations:

        scenarios.append(
            dict(
                station=st,
                data_source="Insitu",
                use_bias_corrected=False,
                startdate=INSITU7_STARTDATE,
                enddate=INSITU7_ENDDATE,
                scenario_label="Insitu7"
            )
        )

    for st in stations:
        for src in ["SMAP", "VIC", "MOSAIC", "NOAH"]:
            for bc in [False]:
                scenarios.append(
                    dict(
                        station=st,
                        data_source=src,
                        use_bias_corrected=bc,
                        startdate=SAT_STARTDATE,
                        enddate=SAT_ENDDATE,
                        scenario_label=f"{src}_{'bc' if bc else 'raw'}"
                    )
                )

    print(f"Total scenarios: {len(scenarios)}")
    return scenarios

# ============================================================
# REG CONST
# ============================================================
def choose_reg_const(metric_name: str) -> float:
    metric_name = metric_name.lower()
    if metric_name in ["kge", "kgeprime", "nse"]:
        return 0.001
    if metric_name == "mse":
        return 0.1
    # rmse, ubrmse, mae, ubmae, ubmse, outer50rmse, outer20rmse
    return 0.01

# ============================================================
# METRICS
# ============================================================
def _rmse(pred, y):
    return jnp.sqrt(jnp.nanmean((pred - y) ** 2))

def _ubrmse(pred, y):
    pred_c = pred - jnp.nanmean(pred)
    y_c    = y    - jnp.nanmean(y)
    return jnp.sqrt(jnp.nanmean((pred_c - y_c) ** 2))

def _mse(pred, y):
    return jnp.nanmean((pred - y) ** 2)

def _ubmse(pred, y):
    pred_c = pred - jnp.nanmean(pred)
    y_c    = y    - jnp.nanmean(y)
    return jnp.nanmean((pred_c - y_c) ** 2)

def _mae(pred, y):
    return jnp.nanmean(jnp.abs(pred - y))

def _ubmae(pred, y):
    pred_c = pred - jnp.nanmean(pred)
    y_c    = y    - jnp.nanmean(y)
    return jnp.nanmean(jnp.abs(pred_c - y_c))

def _kge(pred, y, eps=1e-6):
    pred_m = jnp.nanmean(pred)
    y_m    = jnp.nanmean(y)
    pred_s = jnp.nanstd(pred)
    y_s    = jnp.nanstd(y)

    denom = jnp.maximum(pred_s * y_s, eps)
    corr = jnp.nanmean((pred - pred_m) * (y - y_m)) / denom

    mean_ratio = pred_m / jnp.maximum(y_m, eps)
    std_ratio  = pred_s / jnp.maximum(y_s, eps)

    kge = 1 - jnp.sqrt((corr - 1) ** 2 + (mean_ratio - 1) ** 2 + (std_ratio - 1) ** 2)
    return -kge

def _kgeprime(pred, y, eps=1e-6):
    """Modified KGE (KGE') as per Kling et al. (2012).
    Uses CV ratio (gamma = CV_sim / CV_obs) instead of std ratio.
    KGE' = 1 - sqrt((r-1)^2 + (gamma-1)^2 + (beta-1)^2)
    where:
        r     = Pearson correlation
        gamma = (sigma_s / mu_s) / (sigma_e / mu_e)  [CV ratio]
        beta  = mu_s / mu_e                           [mean ratio]
    """
    pred_m = jnp.nanmean(pred)
    y_m    = jnp.nanmean(y)
    pred_s = jnp.nanstd(pred)
    y_s    = jnp.nanstd(y)

    # Pearson correlation (r)
    denom_corr = jnp.maximum(pred_s * y_s, eps)
    corr = jnp.nanmean((pred - pred_m) * (y - y_m)) / denom_corr

    # Beta: mean ratio (mu_s / mu_e)
    beta = pred_m / jnp.maximum(y_m, eps)

    # Gamma: CV ratio (CV_sim / CV_obs)
    cv_sim = pred_s / jnp.maximum(pred_m, eps)
    cv_obs = y_s    / jnp.maximum(y_m,    eps)
    gamma  = cv_sim / jnp.maximum(cv_obs, eps)

    kgeprime = 1 - jnp.sqrt((corr - 1) ** 2 + (gamma - 1) ** 2 + (beta - 1) ** 2)
    return -kgeprime

def _outer50rmse(pred, y):
    """RMSE on outer 50% of obs: values below 25th or above 75th percentile of y."""
    q25 = jnp.nanquantile(y, 0.25)
    q75 = jnp.nanquantile(y, 0.75)
    mask = (y < q25) | (y > q75)
    n = jnp.maximum(jnp.sum(mask), 1)
    return jnp.sqrt(jnp.sum(jnp.where(mask, (pred - y) ** 2, 0.0)) / n)

def _outer20rmse(pred, y):
    """RMSE on outer 20% of obs: values below 10th or above 90th percentile of y."""
    q10 = jnp.nanquantile(y, 0.10)
    q90 = jnp.nanquantile(y, 0.90)
    mask = (y < q10) | (y > q90)
    n = jnp.maximum(jnp.sum(mask), 1)
    return jnp.sqrt(jnp.sum(jnp.where(mask, (pred - y) ** 2, 0.0)) / n)

def _nse(pred, y, eps=1e-6):
    denom = jnp.nansum((y - jnp.nanmean(y)) ** 2)
    denom = jnp.maximum(denom, eps)
    nse = 1 - jnp.nansum((y - pred) ** 2) / denom
    return -nse

ERROR_FNS = {
    "rmse":        _rmse,
    "ubrmse":      _ubrmse,
    "mse":         _mse,
    "ubmse":       _ubmse,
    "mae":         _mae,
    "ubmae":       _ubmae,
    "kge":         _kge,
    "kgeprime":    _kgeprime,
    "nse":         _nse,
    "outer50rmse": _outer50rmse,
    "outer20rmse": _outer20rmse,
}

# ============================================================
# HELPERS
# ============================================================
def _target_var_name(data_source: str, use_bc: bool) -> str:
    ds = data_source.strip().lower()
    if ds == "insitu":
        return "Insitu_sm"

    name_map = {
        "smap":   "SMAP_sm",
        "vic":    "VIC_sm",
        "mosaic": "MOSAIC_sm",
        "noah":   "NOAH_sm",
    }
    if ds not in name_map:
        raise ValueError(f"Unknown data_source='{data_source}'")

    base = name_map[ds]
    return f"{base}_bc" if use_bc else base

# ============================================================
# FORCING VARIABLE NAMES
# ============================================================
def _forcing_var_names(data_source: str):
    return "Gridmet_tas", "Gridmet_pr"

def _safe_get_da(ds: xr.Dataset, name: str) -> xr.DataArray:
    if name not in ds:
        raise KeyError(f"Variable '{name}' not found. Available: {list(ds.data_vars)}")
    return ds[name]

def _station_scalar(ds: xr.Dataset, var: str, station: str) -> float:
    da = _safe_get_da(ds, var).sel(station=station)
    if "time" in da.dims:
        return float(da.isel(time=0).values)
    if "doy" in da.dims:
        return float(da.isel(doy=0).values)
    return float(da.values)

def _load_lai_365_nearest(st_lat: float, st_lon: float) -> np.ndarray:
    with xr.open_dataset(LAI_NC) as ds_lai:
        lai_pt = ds_lai["LAI"].sel(lat=st_lat, lon=st_lon, method="nearest")
        lai_365 = np.asarray(lai_pt.values, dtype=np.float64)
    if lai_365.shape[0] != 365:
        raise ValueError(f"LAI time length is {lai_365.shape[0]} not 365.")
    return lai_365

def _tile_to_nt(x_365: np.ndarray, nt: int) -> np.ndarray:
    if nt % 365 != 0:
        raise ValueError(f"nt={nt} is not a multiple of 365.")
    return np.tile(x_365, nt // 365).astype(np.float64)

def _build_x_maps(ds: xr.Dataset, station: str) -> np.ndarray:
    rootDepth = _station_scalar(ds, "rootDepth", station)
    clayfrac  = _station_scalar(ds, "clayfrac", station)
    sandfrac  = _station_scalar(ds, "sandfrac", station)
    siltfrac  = _station_scalar(ds, "siltfrac", station)
    elev_std  = _station_scalar(ds, "elev_std", station)
    lats      = float(ds["latitude"].sel(station=station).values)

    corn              = _station_scalar(ds, "corn", station)
    cotton            = _station_scalar(ds, "cotton", station)
    rice              = _station_scalar(ds, "rice", station)
    sorghum           = _station_scalar(ds, "sorghum", station)
    soybeans          = _station_scalar(ds, "soybeans", station)
    durum_wheat       = _station_scalar(ds, "durum_wheat", station)
    spring_wheat      = _station_scalar(ds, "spring_wheat", station)
    wheat = float(durum_wheat + spring_wheat)

    cropland_other       = _station_scalar(ds, "cropland_other", station)
    evergreen_needleleaf = _station_scalar(ds, "evergreen_needleleaf", station)
    evergreen_broadleaf  = _station_scalar(ds, "evergreen_broadleaf", station)
    deciduous_needleleaf = _station_scalar(ds, "deciduous_needleleaf", station)
    deciduous_broadleaf  = _station_scalar(ds, "deciduous_broadleaf", station)
    mixed_forest         = _station_scalar(ds, "mixed_forest", station)
    woodland             = _station_scalar(ds, "woodland", station)
    wooded_grassland     = _station_scalar(ds, "wooded_grassland", station)
    closed_shurbland     = _station_scalar(ds, "closed_shurbland", station)
    open_shrubland       = _station_scalar(ds, "open_shrubland", station)
    grassland            = _station_scalar(ds, "grassland", station)
    barren               = _station_scalar(ds, "barren", station)
    urban                = _station_scalar(ds, "urban", station)

    return np.array(
        [
            rootDepth, clayfrac, sandfrac, siltfrac, elev_std, lats,
            corn, cotton, rice, sorghum, soybeans, wheat,
            cropland_other,
            evergreen_needleleaf, evergreen_broadleaf,
            deciduous_needleleaf, deciduous_broadleaf,
            mixed_forest, woodland, wooded_grassland,
            closed_shurbland, open_shrubland, grassland, barren, urban,
        ],
        dtype=np.float64,
    )

# ============================================================
# MODEL
# ============================================================
WS_INIT_GLOBAL = None

def make_prediction_and_kpet(theta_log, constants, x_forcing_nt, lai_365, x_maps, phi, awCap_base, wiltingp_base):
    rootDepth, clayfrac, sandfrac, siltfrac, elev_std, lats = x_maps[:6]
    w = x_maps[6:]

    th = jnp.exp(theta_log)
    theta_dict = {name: th[i] for i, name in enumerate(PARAM_NAMES)}

    Ts, Tm, Wi_init, Sp_init = constants

    awCap_scalar    = theta_dict["awCap_scalar"]
    wiltingp_scalar = theta_dict["wiltingp_scalar"]

    alpha = 1.0 + (
        theta_dict["alpha_claycoef"] * clayfrac
        + theta_dict["alpha_sandcoef"] * sandfrac
        + theta_dict["alpha_siltcoef"] * siltfrac
    )

    betaHBV = 1.0 + (
        theta_dict["betaHBV_claycoef"] * clayfrac
        + theta_dict["betaHBV_sandcoef"] * sandfrac
        + theta_dict["betaHBV_siltcoef"] * siltfrac
        + theta_dict["betaHBV_elevcoef"] * elev_std
    )

    awCap_scaled    = awCap_scalar * awCap_base
    wiltingp_scaled = wiltingp_scalar * wiltingp_base
    Ws_init = WS_INIT_GLOBAL
    lai = jnp.asarray(lai_365)

    def crop_kpet(crop):
        GS_start = theta_dict[f"GS_start_{crop}"]
        GS_end   = theta_dict[f"GS_end_{crop}"]

        L_ini  = theta_dict[f"L_ini_{crop}"]
        L_dev  = theta_dict[f"L_dev_{crop}"]
        L_mid  = theta_dict[f"L_mid_{crop}"]
        L_late = 1.0 - (L_ini + L_dev + L_mid)
        L_late = jnp.maximum(L_late, 1e-6)

        Kc_ini = theta_dict[f"Kc_ini_{crop}"]
        Kc_mid = theta_dict[f"Kc_mid_{crop}"]
        Kc_end = theta_dict[f"Kc_end_{crop}"]

        Kmin  = theta_dict[f"Kmin_{crop}"]
        Kmax  = theta_dict[f"Kmax_{crop}"]
        c_lai = theta_dict[f"c_lai_{crop}"]

        return construct_Kpet_crop(
            GS_start, GS_end,
            L_ini, L_dev, L_mid, L_late,
            Kc_ini, Kc_mid, Kc_end,
            Kmin, Kmax,
            c_lai,
            lai,
        )

    def gen_kpet(name):
        return construct_Kpet_gen(
            theta_dict[f"Kmin_{name}"],
            theta_dict[f"Kmax_{name}"],
            theta_dict[f"c_lai_{name}"],
            lai,
        )

    Kpet_corn     = crop_kpet("corn")
    Kpet_cotton   = crop_kpet("cotton")
    Kpet_rice     = crop_kpet("rice")
    Kpet_sorghum  = crop_kpet("sorghum")
    Kpet_soybeans = crop_kpet("soybeans")
    Kpet_wheat    = crop_kpet("wheat")

    Kpet_cropland_other       = gen_kpet("cropland_other")
    Kpet_evergreen_needleleaf = gen_kpet("evergreen_needleleaf")
    Kpet_evergreen_broadleaf  = gen_kpet("evergreen_broadleaf")
    Kpet_deciduous_needleleaf = gen_kpet("deciduous_needleleaf")
    Kpet_deciduous_broadleaf  = gen_kpet("deciduous_broadleaf")
    Kpet_mixed_forest         = gen_kpet("mixed_forest")
    Kpet_woodland             = gen_kpet("woodland")
    Kpet_wooded_grassland     = gen_kpet("wooded_grassland")
    Kpet_closed_shurbland     = gen_kpet("closed_shurbland")
    Kpet_open_shrubland       = gen_kpet("open_shrubland")
    Kpet_grassland            = gen_kpet("grassland")
    Kpet_barren               = gen_kpet("barren")
    Kpet_urban                = gen_kpet("urban")

    Kpets = jnp.array(
        [
            Kpet_corn, Kpet_cotton, Kpet_rice, Kpet_sorghum, Kpet_soybeans, Kpet_wheat,
            Kpet_cropland_other,
            Kpet_evergreen_needleleaf, Kpet_evergreen_broadleaf,
            Kpet_deciduous_needleleaf, Kpet_deciduous_broadleaf,
            Kpet_mixed_forest, Kpet_woodland, Kpet_wooded_grassland,
            Kpet_closed_shurbland, Kpet_open_shrubland, Kpet_grassland, Kpet_barren, Kpet_urban,
        ]
    )
    weights = jnp.asarray(w)
    kpet_365 = jnp.average(Kpets, weights=weights, axis=0)

    tas = x_forcing_nt[0]
    pr  = x_forcing_nt[1]

    params_wbm = (
        Ts,
        Tm,
        awCap_scaled,
        wiltingp_scaled,
        rootDepth,
        alpha,
        betaHBV,
    )

    pred = wbm_jax(
        tas, pr, kpet_365,
        Ws_init, Wi_init, Sp_init,
        lai, phi, params_wbm,
    )

    return pred, kpet_365, lai

# ============================================================
# LOSS / TRAIN
# ============================================================
def reg_loss(theta, theta0, lower, upper, eps=1e-6, cap=1e6):
    denom1 = jnp.maximum(theta - lower, eps)
    denom2 = jnp.maximum(upper - theta, eps)
    r = jnp.nansum((theta - theta0) ** 2 / (denom1 * denom2))
    return jnp.minimum(r, cap)

BIG = 1e8

def total_loss(theta, theta0, lower, upper, reg_const, constants, x_forcing_nt, lai_365, x_maps, y, metric_fn, phi, awCap_base, wiltingp_base):
    pred, _, _ = make_prediction_and_kpet(
        theta, constants, x_forcing_nt, lai_365, x_maps, phi,
        awCap_base, wiltingp_base
    )
    pred_finite = jnp.all(jnp.isfinite(pred))
    y_finite    = jnp.all(jnp.isfinite(y))

    def _ok():
        p_loss = metric_fn(pred, y)
        r_loss = reg_loss(theta, theta0, lower, upper)
        L = p_loss + reg_const * r_loss
        L      = jnp.where(jnp.isfinite(L),      L,      BIG)
        p_loss = jnp.where(jnp.isfinite(p_loss), p_loss, BIG)
        r_loss = jnp.where(jnp.isfinite(r_loss), r_loss, BIG)
        return L, p_loss, r_loss

    def _bad():
        return (jnp.array(BIG), jnp.array(BIG), jnp.array(BIG))

    return jax.lax.cond(pred_finite & y_finite, _ok, _bad)

def train_one(theta_init, theta0, lower, upper, reg_const,
              constants, x_forcing_nt, lai_365, x_maps, y,
              metric_fn, phi, awCap_base, wiltingp_base):

    opt = optax.adam(learning_rate=LEARNING_RATE)
    opt_state = opt.init(theta_init)
    theta = theta_init
    valid = True

    def _loss_only(th):
        L, _, _ = total_loss(
            th, theta0, lower, upper, reg_const,
            constants, x_forcing_nt, lai_365, x_maps, y,
            metric_fn, phi, awCap_base, wiltingp_base
        )
        return L

    loss_and_grad = jax.value_and_grad(_loss_only)

    train_loss = np.full(N_EPOCHS_MAX + 1, np.nan, dtype=np.float64)
    pred_loss  = np.full(N_EPOCHS_MAX + 1, np.nan, dtype=np.float64)
    reg_loss_a = np.full(N_EPOCHS_MAX + 1, np.nan, dtype=np.float64)

    L0, P0, R0 = total_loss(
        theta, theta0, lower, upper, reg_const,
        constants, x_forcing_nt, lai_365, x_maps, y,
        metric_fn, phi, awCap_base, wiltingp_base
    )
    if not bool(jnp.isfinite(L0)):
        return theta, train_loss, pred_loss, reg_loss_a, False

    train_loss[0] = float(L0)
    pred_loss[0]  = float(P0)
    reg_loss_a[0] = float(R0)

    stagnation = 0
    buffer = 1e-4

    for epoch in range(1, N_EPOCHS_MAX + 1):
        Lval, g = loss_and_grad(theta)

        if (not bool(jnp.isfinite(Lval))) or (not bool(jnp.all(jnp.isfinite(g)))):
            valid = False
            break

        updates, opt_state = opt.update(g, opt_state)
        theta = optax.apply_updates(theta, updates)
        theta = jnp.clip(theta, lower + buffer, upper - buffer)

        if not bool(jnp.all(jnp.isfinite(theta))):
            valid = False
            break

        L, P, R = total_loss(
            theta, theta0, lower, upper, reg_const,
            constants, x_forcing_nt, lai_365, x_maps, y,
            metric_fn, phi, awCap_base, wiltingp_base
        )

        if not bool(jnp.isfinite(L)):
            valid = False
            break

        train_loss[epoch] = float(L)
        pred_loss[epoch]  = float(P)
        reg_loss_a[epoch] = float(R)

        if epoch > 1 and np.isfinite(pred_loss[epoch - 1]) and pred_loss[epoch - 1] != 0:
            pct_diff = abs(pred_loss[epoch] - pred_loss[epoch - 1]) / abs(pred_loss[epoch - 1])
            if (pred_loss[epoch] < pred_loss[epoch - 1]) and (pct_diff < 0.01):
                stagnation += 1
            else:
                stagnation = 0

        if (stagnation > PATIENCE) and (epoch > N_EPOCHS_MIN):
            break

    return theta, train_loss, pred_loss, reg_loss_a, valid

def pick_best_run(pred_losses: np.ndarray) -> int:
    best = None
    best_val = np.inf
    for r in range(pred_losses.shape[0]):
        row = pred_losses[r]
        finite = np.where(np.isfinite(row))[0]
        if finite.size == 0:
            continue
        val = row[finite[-1]]
        if val < best_val:
            best_val = val
            best = r
    return 0 if best is None else int(best)

# ============================================================
# SAVE
# ============================================================
def save_outputs(out: dict, scenario_id: int, out_dir: str, metric: str):
    best_run = int(out["best_run"])

    tag = (
        f"{out['station']}_{out['data_source']}"
        f"_{'bc' if out['use_bias_corrected'] else 'raw'}"
        f"_{out['startdate'].replace('-', '')}"
    )
    nc_path = os.path.join(out_dir, f"scenario_{scenario_id:04d}_{tag}.nc")

    pred_loss_best  = out["pred_loss"][best_run]
    reg_loss_best   = out["reg_loss"][best_run]
    train_loss_best = out["train_loss"][best_run]

    final_pred_loss  = pred_loss_best[np.isfinite(pred_loss_best)][-1]  if np.isfinite(pred_loss_best).any()  else np.nan
    final_reg_loss   = reg_loss_best[np.isfinite(reg_loss_best)][-1]    if np.isfinite(reg_loss_best).any()   else np.nan
    final_train_loss = train_loss_best[np.isfinite(train_loss_best)][-1] if np.isfinite(train_loss_best).any() else np.nan

    ds = xr.Dataset(
        coords={
            "param": np.array(PARAM_NAMES, dtype=str),
            "const": np.array(["Ts", "Tm", "Wi_init", "Sp_init"], dtype=str),
            "epoch": np.arange(N_EPOCHS_MAX + 1, dtype=np.int64),
            "doy":   np.arange(1, 366, dtype=np.int64),
            "time":  out["time_coord"],
            "run":   np.arange(out["pred_loss"].shape[0], dtype=np.int64),
        }
    )

    ds["run_valid"]           = (("run",),   out["run_valid"])
    ds["best_run"]            = best_run

    ds["initial_theta_best"]  = (("param",), out["initial_theta"][best_run])
    ds["final_theta_best"]    = (("param",), out["final_theta"][best_run])
    ds["constants"]           = (("const",), out["constants"])

    ds["pred_loss_best"]      = (("epoch",), pred_loss_best)
    ds["reg_loss_best"]       = (("epoch",), reg_loss_best)
    ds["train_loss_best"]     = (("epoch",), train_loss_best)

    ds["pred_sm_best"]        = (("time",),  out["pred_sm"][best_run])
    ds["target_sm"]           = (("time",),  out["target_sm"])
    ds["lai_used"]            = (("time",),  out["lai_used"])
    ds["kpet_init_best"]      = (("doy",),   out["kpet_used_init"][best_run])
    ds["kpet_final_best"]     = (("doy",),   out["kpet_used_final"][best_run])

    ds.attrs["station"]                    = out["station"]
    ds.attrs["latitude"]                   = float(out["latitude"])
    ds.attrs["longitude"]                  = float(out["longitude"])
    ds.attrs["data_source"]                = out["data_source"]
    ds.attrs["use_bias_corrected"]         = int(out["use_bias_corrected"])
    ds.attrs["startdate"]                  = out["startdate"]
    ds.attrs["enddate"]                    = out["enddate"]
    ds.attrs["target_var"]                 = out["target_var"]
    ds.attrs["forcing_tas_var"]            = out["forcing_tas_var"]
    ds.attrs["forcing_pr_var"]             = out["forcing_pr_var"]
    ds.attrs["metric"]                     = metric
    ds.attrs["learning_rate"]              = float(LEARNING_RATE)
    ds.attrs["reg_const"]                  = float(out["reg_const"])
    ds.attrs["best_run"]                   = int(best_run)
    ds.attrs["final_pred_loss_best"]       = float(final_pred_loss)
    ds.attrs["final_reg_loss_best"]        = float(final_reg_loss)
    ds.attrs["final_train_loss_best"]      = float(final_train_loss)
    ds.attrs["created_local"]              = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ds.attrs["awCap_base_from_training"]   = float(out["awCap_base"])
    ds.attrs["wiltingp_base_from_training"]= float(out["wiltingp_base"])

    ds.to_netcdf(nc_path)
    return nc_path

# ============================================================
# ONE SCENARIO
# ============================================================
def run_one_scenario(ds_inputs, ds_hist_raw, ds_hist_bc, scenario,
                     metric, metric_fn, reg_const):
    global WS_INIT_GLOBAL

    station    = scenario["station"]
    data_source= scenario["data_source"]
    use_bc     = scenario["use_bias_corrected"]
    startdate  = scenario["startdate"]
    enddate    = scenario["enddate"]

    st_lat = float(ds_inputs["latitude"].sel(station=station).values)
    st_lon = float(ds_inputs["longitude"].sel(station=station).values)
    phi = st_lat

    ds_hist = ds_hist_bc if use_bc else ds_hist_raw

    target_var = _target_var_name(data_source, use_bc)
    if target_var not in ds_hist:
        raise KeyError(f"Target variable '{target_var}' not found. Available: {list(ds_hist.data_vars)}")

    tas_var, pr_var = _forcing_var_names(data_source)
    if tas_var not in ds_hist or pr_var not in ds_hist:
        raise KeyError(f"Forcing vars '{tas_var}'/'{pr_var}' not found. Available: {list(ds_hist.data_vars)}")

    y = np.asarray(
        ds_hist[target_var].sel(station=station).sel(time=slice(startdate, enddate)).values,
        dtype=np.float64,
    )

    y_finite = y[np.isfinite(y)]
    if y_finite.size == 0:
        raise ValueError(f"No finite target data for scenario: {scenario}")

    awCap_base    = float(np.nanmax(y_finite) - np.nanmin(y_finite))
    wiltingp_base = float(np.nanmin(y_finite))

    if awCap_base <= 0:
        raise ValueError(f"Computed awCap_base <= 0 for scenario: {scenario}")

    tas = np.asarray(
        ds_hist[tas_var].sel(station=station).sel(time=slice(startdate, enddate)).values,
        dtype=np.float64,
    )
    pr = np.asarray(
        ds_hist[pr_var].sel(station=station).sel(time=slice(startdate, enddate)).values,
        dtype=np.float64,
    )
    time_coord = ds_hist.sel(time=slice(startdate, enddate)).time.values

    if len(y) == 0:
        raise ValueError(f"No target data for scenario: {scenario}")

    if len(y) != len(tas) or len(y) != len(pr):
        raise ValueError(f"Length mismatch for scenario {scenario}: y={len(y)}, tas={len(tas)}, pr={len(pr)}")

    nt = len(y)
    if nt % 365 != 0:
        raise ValueError(f"Scenario {scenario} gives nt={nt}, not a multiple of 365.")

    lai_365      = _load_lai_365_nearest(st_lat, st_lon)
    lai_used_time = _tile_to_nt(lai_365, nt)

    WS_INIT_GLOBAL = jnp.asarray(_station_scalar(ds_inputs, "soilMoist_init", station))
    x_maps_np = _build_x_maps(ds_inputs, station)

    y_j    = jnp.asarray(y,   dtype=jnp.float32)
    x_nt   = jnp.stack([jnp.asarray(tas, dtype=jnp.float32), jnp.asarray(pr, dtype=jnp.float32)], axis=0)
    x_maps = jnp.asarray(x_maps_np, dtype=jnp.float32)
    lai_365_j = jnp.asarray(lai_365, dtype=jnp.float32)

    theta0 = jnp.asarray(INITIAL_THETA)
    lower  = jnp.asarray(PARAMS_LOWER)
    upper  = jnp.asarray(PARAMS_UPPER)
    consts = jnp.asarray(CONSTANTS_VEC)

    rng = np.random.default_rng(42)
    theta_inits = [np.asarray(INITIAL_THETA, dtype=np.float64)]
    for _ in range(4):
        eps = 1e-3
        theta_inits.append(
            rng.uniform(
                low=np.asarray(PARAMS_LOWER) + eps,
                high=np.asarray(PARAMS_UPPER) - eps,
            ).astype(np.float64)
        )

    n_runs  = 5
    n_param = len(PARAM_NAMES)

    final_thetas = np.full((n_runs, n_param), np.nan, dtype=np.float64)
    init_thetas  = np.full((n_runs, n_param), np.nan, dtype=np.float64)
    train_losses = np.full((n_runs, N_EPOCHS_MAX + 1), np.nan, dtype=np.float64)
    pred_losses  = np.full((n_runs, N_EPOCHS_MAX + 1), np.nan, dtype=np.float64)
    reg_losses   = np.full((n_runs, N_EPOCHS_MAX + 1), np.nan, dtype=np.float64)

    pred_sm_runs    = np.full((n_runs, nt),  np.nan, dtype=np.float64)
    kpet_init_runs  = np.full((n_runs, 365), np.nan, dtype=np.float64)
    kpet_final_runs = np.full((n_runs, 365), np.nan, dtype=np.float64)
    run_valid       = np.zeros(n_runs, dtype=np.int8)

    for r in range(n_runs):
        th0 = jnp.asarray(theta_inits[r])
        init_thetas[r, :] = np.asarray(th0)

        _, kpet0, _ = make_prediction_and_kpet(
            th0, consts, x_nt, lai_365_j, x_maps, phi, awCap_base, wiltingp_base
        )
        kpet_init_runs[r, :] = np.asarray(kpet0)

        thf, tl, pl, rl, valid = train_one(
            th0, theta0, lower, upper, reg_const,
            consts, x_nt, lai_365_j, x_maps, y_j, metric_fn, phi,
            awCap_base, wiltingp_base,
        )

        if not valid:
            continue

        run_valid[r] = 1
        final_thetas[r, :] = np.asarray(thf)
        train_losses[r, :] = tl
        pred_losses[r, :]  = pl
        reg_losses[r, :]   = rl

        pred_f, kpet_f, _ = make_prediction_and_kpet(
            thf, consts, x_nt, lai_365_j, x_maps, phi, awCap_base, wiltingp_base
        )
        pred_sm_runs[r, :]    = np.asarray(pred_f)
        kpet_final_runs[r, :] = np.asarray(kpet_f)

    best_run = pick_best_run(pred_losses)

    return {
        "station":          station,
        "latitude":         st_lat,
        "longitude":        st_lon,
        "data_source":      data_source,
        "use_bias_corrected": int(use_bc),
        "startdate":        startdate,
        "enddate":          enddate,
        "target_var":       target_var,
        "forcing_tas_var":  tas_var,
        "forcing_pr_var":   pr_var,
        "best_run":         int(best_run),
        "run_valid":        run_valid,
        "initial_theta":    init_thetas,
        "final_theta":      final_thetas,
        "constants":        np.asarray(consts, dtype=np.float64),
        "train_loss":       train_losses,
        "pred_loss":        pred_losses,
        "reg_loss":         reg_losses,
        "pred_sm":          pred_sm_runs,
        "target_sm":        y,
        "lai_used":         lai_used_time,
        "kpet_used_init":   kpet_init_runs,
        "kpet_used_final":  kpet_final_runs,
        "time_coord":       time_coord,
        "awCap_base":       awCap_base,
        "wiltingp_base":    wiltingp_base,
        "reg_const":        reg_const,
    }

# ============================================================
# MAIN
# ============================================================
def main():
    ds_inputs   = xr.open_dataset(INPUTS_NC)
    ds_hist_raw = xr.open_dataset(HIST_RAW_NC)
    ds_hist_bc  = xr.open_dataset(HIST_BC_NC)

    scenarios = build_scenarios(ds_inputs, ds_hist_raw)
    n_scenarios = len(scenarios)
    n_metrics   = len(METRICS_TO_RUN)
    n_total     = n_scenarios * n_metrics

    # --------------------------------------------------------
    # Decode task ID
    # Layout: task_id = scenario_idx * n_metrics + metric_idx
    # --------------------------------------------------------
    if len(sys.argv) > 1:
        task_id = int(sys.argv[1])
    elif "SLURM_ARRAY_TASK_ID" in os.environ:
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    else:
        raise ValueError("Provide task index as argv[1] or set SLURM_ARRAY_TASK_ID")

    if task_id < 0 or task_id >= n_total:
        raise IndexError(
            f"task_id={task_id} out of range 0..{n_total - 1} "
            f"({n_scenarios} scenarios x {n_metrics} metrics)"
        )

    scenario_id = task_id // n_metrics
    metric_idx  = task_id  % n_metrics
    metric      = METRICS_TO_RUN[metric_idx]
    reg_const   = choose_reg_const(metric)
    metric_fn   = ERROR_FNS[metric]

    # Per-metric output directory
    out_dir = f"{DATA_DIR}/Calibration/AllStations/calibration_outputs_{metric}_gridmet"
    os.makedirs(out_dir, exist_ok=True)

    scenario = scenarios[scenario_id]
    print(
        f"task_id={task_id} | scenario_id={scenario_id}/{n_scenarios - 1} "
        f"| metric={metric} ({metric_idx + 1}/{n_metrics})"
    )
    print(f"Scenario: {scenario}")

    tag = (f"{scenario['station']}_{scenario['data_source']}"f"_{'bc' if scenario['use_bias_corrected'] else 'raw'}"f"_{scenario['startdate'].replace('-', '')}")

    nc_path = os.path.join(out_dir, f"scenario_{scenario_id:04d}_{tag}.nc")

    if os.path.exists(nc_path):
        try:
            xr.open_dataset(nc_path).close()
            print(f"Skipping existing valid file: {nc_path}")
            return
        except:
            print(f"Corrupt file detected. Recomputing: {nc_path}")

    out = run_one_scenario(
        ds_inputs, ds_hist_raw, ds_hist_bc, scenario,
        metric=metric, metric_fn=metric_fn, reg_const=reg_const,
    )
    nc_path = save_outputs(out, scenario_id, out_dir=out_dir, metric=metric)
    print(f"Saved: {nc_path}")

    ds_inputs.close()
    ds_hist_raw.close()
    ds_hist_bc.close()


if __name__ == "__main__":
    main()