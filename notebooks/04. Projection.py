#!/usr/bin/env python3

#Projection files have too many tasks for keeling to handle in a single batch job. Might need to loop.

import os
import glob
import sys
import numpy as np
import pandas as pd
import xarray as xr
import jax.numpy as jnp

# ============================================================
# PATHS
# ============================================================
main_dir = "/data/keeling/a/tahsina2/Alam_et_al_2026" #Update with yours
LOCA2_DIR  = "/data/keeling/a/tahsina2/a/LOCA2" #Update with yours
OUT_DIR    = "/data/keeling/a/tahsina2/a/260425_Projection" #Update with yours
DATA_DIR = f"{main_dir}/data"
SRC_DIR  = f"{main_dir}/src"

# ============================================================
# IMPORT MODEL FUNCTIONS
# ============================================================

sys.path.append(SRC_DIR)

from water_balance_jax import (
    wbm_jax,
    construct_Kpet_gen,
    construct_Kpet_crop,
)

# ============================================================
# PATHS
# ============================================================
INPUTS_NC  = f"{DATA_DIR}/inputs_by_station_noleap.nc"
LAI_NC     = f"{DATA_DIR}/LAI_GLDAS_clima_NLDASgrid.nc"



os.makedirs(OUT_DIR, exist_ok=True)

START_PROJ = "2023-01-01"
END_PROJ   = "2100-12-31"

# ============================================================
# CALIBRATION DIRECTORIES
# ============================================================
CALIB_DIRS = {
    "kgeprime":    f"{DATA_DIR}/Calibration/AllStations/calibration_outputs_kgeprime_gridmet",
    "rmse":        f"{DATA_DIR}/Calibration/AllStations/calibration_outputs_rmse_gridmet",
    "outer50rmse": f"{DATA_DIR}/Calibration/AllStations/calibration_outputs_outer50rmse_gridmet",
    "kge":         f"{DATA_DIR}/Calibration/AllStations/calibration_outputs_kge_gridmet",
}

METRIC_ORDER = ["kgeprime", "rmse", "outer50rmse", "kge"]

N_MODELS = 201
N_SCENARIOS = 170
N_METRICS = len(METRIC_ORDER)
TASKS_PER_MODEL = N_SCENARIOS * N_METRICS
N_TOTAL_TASKS = N_MODELS * TASKS_PER_MODEL

# ============================================================
# LAND-COVER / CROP CLASSES
# ============================================================
CROP_CLASSES = [
    "corn",
    "cotton",
    "rice",
    "sorghum",
    "soybeans",
    "wheat",
]

GENERIC_CLASSES = [
    "cropland_other",
    "evergreen_needleleaf",
    "evergreen_broadleaf",
    "deciduous_needleleaf",
    "deciduous_broadleaf",
    "mixed_forest",
    "woodland",
    "wooded_grassland",
    "closed_shurbland",
    "open_shrubland",
    "grassland",
    "barren",
    "urban",
]

# ============================================================
# TASK DECODER
# ============================================================
def get_loca2_models():
    zarrs = sorted(glob.glob(os.path.join(LOCA2_DIR, "*.zarr")))

    if len(zarrs) == 0:
        raise FileNotFoundError(f"No .zarr files found in {LOCA2_DIR}")

    if len(zarrs) != N_MODELS:
        print(f"WARNING: Expected {N_MODELS} LOCA2 models, found {len(zarrs)}")

    return zarrs


def decode_task_id(task_id, loca2_models):
    if task_id < 0 or task_id >= len(loca2_models) * TASKS_PER_MODEL:
        raise IndexError(
            f"task_id={task_id} out of range. "
            f"Valid range: 0 to {len(loca2_models) * TASKS_PER_MODEL - 1}"
        )

    model_idx = task_id // TASKS_PER_MODEL
    within_model_task = task_id % TASKS_PER_MODEL

    metric_idx = within_model_task // N_SCENARIOS
    scenario_id = within_model_task % N_SCENARIOS

    metric_name = METRIC_ORDER[metric_idx]
    zarr_path = loca2_models[model_idx]

    return zarr_path, model_idx, metric_name, metric_idx, scenario_id


def find_calibration_file(metric_name, scenario_id):
    calib_dir = CALIB_DIRS[metric_name]

    pattern = os.path.join(
        calib_dir,
        f"scenario_{scenario_id:04d}_*.nc"
    )

    matches = sorted(glob.glob(pattern))

    if len(matches) == 0:
        return None

    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple calibration files found for metric={metric_name}, "
            f"scenario_id={scenario_id}: {matches}"
        )

    return matches[0]


def scenario_label_from_attrs(attrs):
    src = str(attrs.get("data_source", "")).strip().lower()
    startdate = str(attrs.get("startdate", "")).strip()

    try:
        use_bc = int(attrs.get("use_bias_corrected", 0))
    except Exception:
        use_bc = 0

    if src == "insitu":
        if startdate == "2016-01-01":
            return "insitu7"
        return "insitu20"

    return f"{src}_{'bc' if use_bc else 'raw'}"

# ============================================================
# BASIC HELPERS
# ============================================================
def safe_scalar(ds, var, station):
    da = ds[var].sel(station=station)

    if "time" in da.dims:
        return float(da.isel(time=0).values)

    if "doy" in da.dims:
        return float(da.isel(doy=0).values)

    return float(da.values)


def to_theta_dict(ds, theta_name="final_theta_best"):
    param_names = [str(p) for p in ds["param"].values]
    theta_log = np.asarray(ds[theta_name].values, dtype=float)
    theta_vals = np.exp(theta_log)
    return dict(zip(param_names, theta_vals))

# ============================================================
# STATION METADATA
# ============================================================
def get_station_latlon(ds_inputs, station):
    lat = float(ds_inputs["latitude"].sel(station=station).values)
    lon = float(ds_inputs["longitude"].sel(station=station).values)
    return lat, lon


def get_station_lai(ds_lai, ds_inputs, station):
    station_lat = float(ds_inputs["latitude"].sel(station=station).values)
    station_lon = float(ds_inputs["longitude"].sel(station=station).values)

    da = ds_lai["LAI"].sel(lat=station_lat, lon=station_lon, method="nearest")
    lai = np.asarray(da.values, dtype=float).squeeze()

    if lai.ndim != 1:
        raise ValueError(
            f"LAI for station {station} is not 1D after extraction. "
            f"Shape={lai.shape}, dims={da.dims}"
        )

    if len(lai) != 365:
        raise ValueError(
            f"LAI climatology for station {station} must be length 365. "
            f"Got length={len(lai)}, shape={lai.shape}, dims={da.dims}"
        )

    return lai


def get_station_rootdepth(ds_inputs, station):
    return safe_scalar(ds_inputs, "rootDepth", station)


def get_station_initial_sm(ds_inputs, station):
    return safe_scalar(ds_inputs, "soilMoist_init", station)


def get_station_fraction(ds_inputs, station, class_name):
    if class_name == "wheat":
        frac = 0.0

        if "durum_wheat" in ds_inputs.data_vars:
            frac += safe_scalar(ds_inputs, "durum_wheat", station)

        if "spring_wheat" in ds_inputs.data_vars:
            frac += safe_scalar(ds_inputs, "spring_wheat", station)

        return frac

    if class_name not in ds_inputs.data_vars:
        return 0.0

    return safe_scalar(ds_inputs, class_name, station)

# ============================================================
# PARAMETER RECONSTRUCTION
# ============================================================
def reconstruct_wbm_params(ds_inputs, ds_scenario, station):
    theta = to_theta_dict(ds_scenario, "final_theta_best")
    attrs = ds_scenario.attrs

    awCap_base = float(attrs["awCap_base_from_training"])
    wiltingp_base = float(attrs["wiltingp_base_from_training"])

    clayfrac = safe_scalar(ds_inputs, "clayfrac", station)
    sandfrac = safe_scalar(ds_inputs, "sandfrac", station)
    siltfrac = safe_scalar(ds_inputs, "siltfrac", station)
    elev_std = safe_scalar(ds_inputs, "elev_std", station)
    rootDepth = get_station_rootdepth(ds_inputs, station)

    awCap = theta["awCap_scalar"] * awCap_base
    wiltingp = theta["wiltingp_scalar"] * wiltingp_base

    alpha = (
        1.0
        + theta["alpha_claycoef"] * clayfrac
        + theta["alpha_sandcoef"] * sandfrac
        + theta["alpha_siltcoef"] * siltfrac
    )

    betaHBV = (
        1.0
        + theta["betaHBV_claycoef"] * clayfrac
        + theta["betaHBV_sandcoef"] * sandfrac
        + theta["betaHBV_siltcoef"] * siltfrac
        + theta["betaHBV_elevcoef"] * elev_std
    )

    Ts = 0.0
    Tm = 0.0

    return np.array(
        [Ts, Tm, awCap, wiltingp, rootDepth, alpha, betaHBV],
        dtype=float
    )

# ============================================================
# BUILD WEIGHTED KPET
# ============================================================
def build_weighted_kpet(ds_lai, ds_inputs, ds_scenario, station):
    theta = to_theta_dict(ds_scenario, "final_theta_best")
    lai = get_station_lai(ds_lai, ds_inputs, station)

    Kpet_total = np.zeros(365, dtype=float)
    used_fraction_sum = 0.0

    for crop in CROP_CLASSES:
        frac = get_station_fraction(ds_inputs, station, crop)

        if frac <= 0:
            continue

        needed = [
            f"GS_start_{crop}",
            f"GS_end_{crop}",
            f"L_ini_{crop}",
            f"L_dev_{crop}",
            f"L_mid_{crop}",
            f"Kc_ini_{crop}",
            f"Kc_mid_{crop}",
            f"Kc_end_{crop}",
            f"Kmin_{crop}",
            f"Kmax_{crop}",
            f"c_lai_{crop}",
        ]

        if not all(k in theta for k in needed):
            continue

        L_ini = theta[f"L_ini_{crop}"]
        L_dev = theta[f"L_dev_{crop}"]
        L_mid = theta[f"L_mid_{crop}"]
        L_late = max(1.0 - (L_ini + L_dev + L_mid), 1e-6)

        kpet_crop = construct_Kpet_crop(
            GS_start=jnp.array(theta[f"GS_start_{crop}"]),
            GS_end=jnp.array(theta[f"GS_end_{crop}"]),
            L_ini=jnp.array(L_ini),
            L_dev=jnp.array(L_dev),
            L_mid=jnp.array(L_mid),
            L_late=jnp.array(L_late),
            Kc_ini=jnp.array(theta[f"Kc_ini_{crop}"]),
            Kc_mid=jnp.array(theta[f"Kc_mid_{crop}"]),
            Kc_end=jnp.array(theta[f"Kc_end_{crop}"]),
            Kmin=jnp.array(theta[f"Kmin_{crop}"]),
            Kmax=jnp.array(theta[f"Kmax_{crop}"]),
            c_lai=jnp.array(theta[f"c_lai_{crop}"]),
            lai=jnp.array(lai),
        )

        Kpet_total += frac * np.asarray(kpet_crop, dtype=float)
        used_fraction_sum += frac

    for cls in GENERIC_CLASSES:
        frac = get_station_fraction(ds_inputs, station, cls)

        if frac <= 0:
            continue

        needed = [
            f"Kmin_{cls}",
            f"Kmax_{cls}",
            f"c_lai_{cls}",
        ]

        if not all(k in theta for k in needed):
            continue

        kpet_cls = construct_Kpet_gen(
            Kmin=jnp.array(theta[f"Kmin_{cls}"]),
            Kmax=jnp.array(theta[f"Kmax_{cls}"]),
            c_lai=jnp.array(theta[f"c_lai_{cls}"]),
            lai=jnp.array(lai),
        )

        Kpet_total += frac * np.asarray(kpet_cls, dtype=float)
        used_fraction_sum += frac

    if used_fraction_sum <= 0:
        raise ValueError(f"No valid Kpet fractions found for station {station}")

    return Kpet_total, lai, used_fraction_sum

# ============================================================
# LOCA2 FORCING
# ============================================================
def extract_loca2_station_forcing(
    zarr_path,
    station_lat,
    station_lon,
    start_date=START_PROJ,
    end_date=END_PROJ,
):
    ds = xr.open_zarr(zarr_path)

    try:
        ds = ds.sel(time=slice(start_date, end_date))
        ds_pt = ds.sel(lat=station_lat, lon=station_lon, method="nearest")

        time = pd.to_datetime(ds_pt["time"].values)
        tas  = np.asarray(ds_pt["tas"].values, dtype=float)
        pr   = np.asarray(ds_pt["pr"].values, dtype=float)

    finally:
        ds.close()

    idx = pd.DatetimeIndex(time)
    keep = ~((idx.month == 2) & (idx.day == 29))

    time = time[keep]
    tas = tas[keep]
    pr = pr[keep]

    if len(time) % 365 != 0:
        raise ValueError(
            f"Noleap forcing length is not divisible by 365 for "
            f"{os.path.basename(zarr_path)}. Length={len(time)}"
        )

    return time, tas, pr

# ============================================================
# INITIAL STATES
# ============================================================
def get_initial_states(ds_inputs, station):
    Ws_init = get_station_initial_sm(ds_inputs, station)
    Wi_init = 0.0
    Sp_init = 0.0

    return Ws_init, Wi_init, Sp_init

# ============================================================
# RUN ONE PROJECTION
# ============================================================
def run_one_projection(
    ds_lai,
    ds_inputs,
    calib_file,
    zarr_path,
    station,
    metric_name,
    scenario_id,
    model_idx,
):
    ds_scenario = xr.open_dataset(calib_file)

    try:
        scenario_label = scenario_label_from_attrs(ds_scenario.attrs)

        model_name = os.path.basename(zarr_path).replace(".zarr", "")
        station_lat, station_lon = get_station_latlon(ds_inputs, station)
        phi = float(station_lat)

        time, tas, prcp = extract_loca2_station_forcing(
            zarr_path,
            station_lat,
            station_lon,
            start_date=START_PROJ,
            end_date=END_PROJ,
        )

        params = reconstruct_wbm_params(ds_inputs, ds_scenario, station)

        Kpet, lai, frac_sum = build_weighted_kpet(
            ds_lai,
            ds_inputs,
            ds_scenario,
            station,
        )

        Ws_init, Wi_init, Sp_init = get_initial_states(
            ds_inputs,
            station,
        )

        sim = wbm_jax(
            tas=jnp.array(tas, dtype=jnp.float32),
            prcp=jnp.array(prcp, dtype=jnp.float32),
            Kpet=jnp.array(Kpet, dtype=jnp.float32),
            Ws_init=jnp.array(Ws_init, dtype=jnp.float32),
            Wi_init=jnp.array(Wi_init, dtype=jnp.float32),
            Sp_init=jnp.array(Sp_init, dtype=jnp.float32),
            lai=jnp.array(lai, dtype=jnp.float32),
            phi=jnp.array(phi, dtype=jnp.float32),
            params=jnp.array(params, dtype=jnp.float32),
        )

        sim = np.asarray(sim, dtype=float)

        out = xr.Dataset(
            data_vars={
                "soil_moisture": (["time"], sim),
                "tas": (["time"], tas),
                "prcp": (["time"], prcp),
            },
            coords={
                "time": time,
            },
            attrs={
                "model_idx": int(model_idx),
                "scenario_id": int(scenario_id),
                "scenario_label": scenario_label,
                "station": station,
                "latitude": float(station_lat),
                "longitude": float(station_lon),
                "phi": float(phi),
                "loca2_model": model_name,
                "metric": metric_name,
                "data_source": str(ds_scenario.attrs.get("data_source", "")),
                "use_bias_corrected": int(ds_scenario.attrs.get("use_bias_corrected", 0)),
                "calibration_startdate": str(ds_scenario.attrs.get("startdate", "")),
                "calibration_enddate": str(ds_scenario.attrs.get("enddate", "")),
                "projection_start": START_PROJ,
                "projection_end": END_PROJ,
                "kpet_fraction_sum_used": float(frac_sum),
                "calibration_file": calib_file,
                "zarr_path": zarr_path,
            },
        )

        return out

    finally:
        ds_scenario.close()

# ============================================================
# SAVE
# ============================================================
def save_projection(ds_proj, out_dir, scenario_id):
    metric = ds_proj.attrs["metric"]
    model_name = ds_proj.attrs["loca2_model"]
    scenario_label = ds_proj.attrs["scenario_label"]
    station = ds_proj.attrs["station"]

    save_dir = os.path.join(
        out_dir,
        metric,
        model_name,
        f"scenario_{scenario_id:04d}_{scenario_label}",
    )

    os.makedirs(save_dir, exist_ok=True)

    out_file = os.path.join(
        save_dir,
        f"{station}_projection_2023_2100.nc",
    )

    if os.path.exists(out_file):
        print(f"Skipping existing projection: {out_file}")
        return out_file, True

    ds_proj.to_netcdf(out_file)
    return out_file, False

# ============================================================
# RUN ONE ARRAY TASK
# ============================================================
def run_task(task_id):
    loca2_models = get_loca2_models()

    zarr_path, model_idx, metric_name, metric_idx, scenario_id = decode_task_id(
        task_id,
        loca2_models,
    )

    model_name = os.path.basename(zarr_path).replace(".zarr", "")

    print("=" * 80)
    print("Projection task")
    print(f"Task ID:       {task_id}")
    print(f"Model index:   {model_idx}")
    print(f"Model:         {model_name}")
    print(f"Metric index:  {metric_idx}")
    print(f"Metric:        {metric_name}")
    print(f"Scenario ID:   {scenario_id:04d}")
    print("=" * 80)

    calib_file = find_calibration_file(metric_name, scenario_id)

    if calib_file is None:
        print(
            f"Calibration not ready yet. Skipping projection for "
            f"metric={metric_name}, scenario_id={scenario_id:04d}"
        )
        return

    with xr.open_dataset(calib_file) as ds_cal:
        station = str(ds_cal.attrs["station"]).strip()
        scenario_label = scenario_label_from_attrs(ds_cal.attrs)

    KEEP_RAW = {"vic_raw", "noah_raw", "mosaic_raw",}

    if scenario_label not in KEEP_RAW:
        print(f"Skipping task: {scenario_label}")
        return

    out_file = os.path.join(
        OUT_DIR,
        metric_name,
        model_name,
        f"scenario_{scenario_id:04d}_{scenario_label}",
        f"{station}_projection_2023_2100.nc",
    )

    if os.path.exists(out_file):
        print(f"Skipping existing projection: {out_file}")
        return

    ds_inputs = xr.open_dataset(INPUTS_NC)
    ds_lai = xr.open_dataset(LAI_NC)

    try:
        ds_proj = run_one_projection(
            ds_lai=ds_lai,
            ds_inputs=ds_inputs,
            calib_file=calib_file,
            zarr_path=zarr_path,
            station=station,
            metric_name=metric_name,
            scenario_id=scenario_id,
            model_idx=model_idx,
        )

        saved_path, skipped = save_projection(
            ds_proj,
            OUT_DIR,
            scenario_id=scenario_id,
        )

        if not skipped:
            print(f"Saved: {saved_path}")

        ds_proj.close()

    finally:
        ds_inputs.close()
        ds_lai.close()

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    if len(sys.argv) > 1:
        task_id = int(sys.argv[1])
    elif "SLURM_ARRAY_TASK_ID" in os.environ:
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    else:
        raise ValueError("Provide task_id as argv[1] or SLURM_ARRAY_TASK_ID")

    run_task(task_id)