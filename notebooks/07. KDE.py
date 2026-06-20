#!/usr/bin/env python3

import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

# ============================================================
# PATHS
# ============================================================
PROJ_ROOT = "/data/keeling/a/tahsina2/a/260425_Projection"
HIST_NC = "/data/keeling/a/tahsina2/Alam_et_al_2026/data/Historical_Inputs_biascorrected.nc"

TAG = "nldasraw_allstations"

OUT_DIR = (
    "/data/keeling/a/tahsina2/Alam_et_al_2026/data/"
    f"processed_daily_anomaly_kde_2090s_{TAG}"
)
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# SETTINGS
# ============================================================
METRICS = ["kgeprime", "rmse", "outer50rmse", "kge"]

GROUP_ORDER = [
    # "smap_raw",
    # "insitu7",
    "noah_raw",
    "vic_raw",
    "mosaic_raw",
]

GROUP_LABELS = {
    # "smap_raw": "SMAP raw",
    # "insitu7": "In-situ 2016-2022",
    "noah_raw": "Noah raw",
    "vic_raw": "VIC raw",
    "mosaic_raw": "Mosaic raw",
}

HIST_SOURCE_MAP = {
    # "smap_raw": ("SMAP_sm", 2016, 2022),
    # "insitu7": ("Insitu_sm", 2016, 2022),
    "noah_raw": ("NOAH_sm", 2016, 2022),
    "vic_raw": ("VIC_sm", 2016, 2022),
    "mosaic_raw": ("MOSAIC_sm", 2016, 2022),
}

PROJ_PERIOD = ("2090s", 2090, 2099)

SEASONS = ["full_year", "spring", "summer"]

EXCLUDE_STATIONS = set()

# ============================================================
# HELPERS
# ============================================================
def open_nc_safe(path):
    last_err = None

    for eng in ["netcdf4", "h5netcdf", "scipy"]:
        try:
            ds = xr.open_dataset(path, engine=eng)
            _ = ds.dims
            return ds
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Could not open {path}. Last error: {last_err}")


def get_station_list_from_hist(hist_nc):
    ds = open_nc_safe(hist_nc)

    try:
        if "station" not in ds.coords and "station" not in ds.variables:
            raise KeyError(f"'station' not found in {hist_nc}")

        stations = [
            str(s)
            for s in ds["station"].values
            if str(s) not in EXCLUDE_STATIONS
        ]

    finally:
        ds.close()

    return sorted(stations)


def get_period_mask(time_coord, start_year, end_year):
    return (time_coord.dt.year >= start_year) & (time_coord.dt.year <= end_year)


def get_season_mask(time_coord, season_name):
    month = time_coord.dt.month

    if season_name == "full_year":
        return xr.ones_like(month, dtype=bool)

    if season_name == "spring":
        return (month >= 3) & (month <= 5)

    if season_name == "summer":
        return (month >= 6) & (month <= 8)

    raise ValueError(f"Unknown season: {season_name}")


def drop_feb29(da):
    month = da["time"].dt.month
    day = da["time"].dt.day
    keep = ~((month == 2) & (day == 29))
    return da.where(keep, drop=True)


def historical_daily_climatology(sm_da, start_year, end_year):
    sm_da = drop_feb29(sm_da)

    mask = get_period_mask(sm_da["time"], start_year, end_year)
    da = sm_da.where(mask, drop=True)

    if da.sizes.get("time", 0) == 0:
        raise ValueError("No historical values found for daily climatology.")

    clim = da.groupby("time.dayofyear").mean("time", skipna=True)
    clim = clim.reindex(dayofyear=np.arange(1, 366))

    return clim


def anomaly_from_daily_climatology(sm_da, clim):
    sm_da = drop_feb29(sm_da)
    anom = sm_da.groupby("time.dayofyear") - clim
    return anom


def seasonal_values(da, start_year, end_year, season_name):
    mask = get_period_mask(da["time"], start_year, end_year)
    da = da.where(mask, drop=True)

    if da.sizes.get("time", 0) == 0:
        return np.array([], dtype=float)

    season_mask = get_season_mask(da["time"], season_name)
    da = da.where(season_mask, drop=True)

    if da.sizes.get("time", 0) == 0:
        return np.array([], dtype=float)

    vals = np.asarray(da.values, dtype=float).ravel()
    vals = vals[np.isfinite(vals)]

    return vals


def list_projection_files(metric, group, stations):
    proj_base_dir = os.path.join(PROJ_ROOT, metric)

    pattern = os.path.join(
        proj_base_dir,
        "*",
        f"scenario_*_{group}",
        "*_projection_2023_2100.nc"
    )

    files = sorted(glob.glob(pattern))
    rows = []

    for f in files:
        station = os.path.basename(f).replace("_projection_2023_2100.nc", "")

        if station not in stations:
            continue

        if station in EXCLUDE_STATIONS:
            continue

        parts = f.split(os.sep)
        model = parts[-3]
        scenario_dir = parts[-2]

        rows.append({
            "metric": metric,
            "model": model,
            "group": group,
            "scenario_dir": scenario_dir,
            "group_label": GROUP_LABELS[group],
            "station": station,
            "file": f,
        })

    return pd.DataFrame(rows)


# ============================================================
# HISTORICAL DAILY ANOMALY VALUES
# ============================================================
def process_historical_daily_anomaly_values(hist_nc, stations, metric):
    ds = open_nc_safe(hist_nc)
    records = []

    try:
        for group, (varname, start_year, end_year) in HIST_SOURCE_MAP.items():
            if varname not in ds.data_vars:
                raise KeyError(f"{varname} not found in {hist_nc}")

            for station in stations:
                sm = ds[varname].sel(station=station)

                clim = historical_daily_climatology(
                    sm,
                    start_year=start_year,
                    end_year=end_year
                )

                anom = anomaly_from_daily_climatology(sm, clim)

                for season_name in SEASONS:
                    vals = seasonal_values(
                        anom,
                        start_year=start_year,
                        end_year=end_year,
                        season_name=season_name
                    )

                    if vals.size == 0:
                        continue

                    for v in vals:
                        records.append({
                            "source": "historical",
                            "metric": metric,
                            "model": "historical",
                            "group": group,
                            "scenario_dir": "historical",
                            "group_label": GROUP_LABELS[group],
                            "station": station,
                            "period": "historical",
                            "season": season_name,
                            "soil_moisture_anomaly": float(v),
                        })

    finally:
        ds.close()

    out = pd.DataFrame(records)

    if out.empty:
        raise ValueError(f"No historical daily anomaly values created for metric={metric}.")

    return out


# ============================================================
# PROJECTION DAILY ANOMALY VALUES
# ============================================================
def process_projection_daily_anomaly_values_to_csv(hist_nc, stations, metric, out_csv):
    period_name, proj_start, proj_end = PROJ_PERIOD

    ds_hist = open_nc_safe(hist_nc)

    if os.path.exists(out_csv):
        os.remove(out_csv)

    first_write = True
    file_tables = []

    try:
        for group, (varname, hist_start, hist_end) in HIST_SOURCE_MAP.items():
            print(f"\nMetric={metric} | Group={group}")

            if varname not in ds_hist.data_vars:
                raise KeyError(f"{varname} not found in {hist_nc}")

            file_table = list_projection_files(metric, group, stations)
            file_tables.append(file_table)

            print(f"Projection files found: {len(file_table)}")

            if file_table.empty:
                continue

            clim_by_station = {}

            for station in stations:
                sm_hist = ds_hist[varname].sel(station=station)
                clim_by_station[station] = historical_daily_climatology(
                    sm_hist,
                    start_year=hist_start,
                    end_year=hist_end
                )

            for row in tqdm(
                file_table.itertuples(index=False),
                total=len(file_table),
                desc=f"{metric} {group} projection anomalies"
            ):
                ds_proj = None

                try:
                    ds_proj = open_nc_safe(row.file)

                    if "soil_moisture" not in ds_proj.data_vars:
                        raise KeyError(f"'soil_moisture' not found in {row.file}")

                    sm_proj = ds_proj["soil_moisture"]

                    clim = clim_by_station[row.station]
                    anom_proj = anomaly_from_daily_climatology(sm_proj, clim)

                    local_records = []

                    for season_name in SEASONS:
                        vals = seasonal_values(
                            anom_proj,
                            start_year=proj_start,
                            end_year=proj_end,
                            season_name=season_name
                        )

                        if vals.size == 0:
                            continue

                        for v in vals:
                            local_records.append({
                                "source": "projection",
                                "metric": metric,
                                "model": row.model,
                                "group": row.group,
                                "scenario_dir": row.scenario_dir,
                                "group_label": row.group_label,
                                "station": row.station,
                                "period": period_name,
                                "season": season_name,
                                "soil_moisture_anomaly": float(v),
                            })

                    if local_records:
                        df_chunk = pd.DataFrame(local_records)
                        df_chunk.to_csv(
                            out_csv,
                            mode="w" if first_write else "a",
                            header=first_write,
                            index=False
                        )
                        first_write = False

                except Exception as e:
                    print(f"Skipping {row.file}")
                    print(f"Error: {repr(e)}")

                finally:
                    if ds_proj is not None:
                        ds_proj.close()

    finally:
        ds_hist.close()

    if first_write:
        raise ValueError(f"No projection daily anomaly values created for metric={metric}.")

    if len(file_tables) == 0:
        return pd.DataFrame()

    return pd.concat(file_tables, ignore_index=True)


# ============================================================
# SAVE HELPERS
# ============================================================
def dataframe_to_obs_xarray(df):
    df = df.reset_index(drop=True).copy()
    df["obs"] = np.arange(len(df))

    ds = xr.Dataset(coords={"obs": df["obs"].values})

    for col in df.columns:
        if col == "obs":
            continue

        vals = df[col].values

        if df[col].dtype == object:
            ds[col] = ("obs", vals.astype(str))
        else:
            ds[col] = ("obs", vals)

    return ds


# ============================================================
# MAIN
# ============================================================
def build_daily_anomaly_kde_value_files():
    stations = get_station_list_from_hist(HIST_NC)

    print(f"Stations found ({len(stations)}):")
    print(stations)

    for metric in METRICS:
        print("\n" + "=" * 80)
        print(f"Processing metric: {metric}")
        print("=" * 80)

        metric_out_dir = os.path.join(OUT_DIR, metric)
        os.makedirs(metric_out_dir, exist_ok=True)

        file_table_csv = os.path.join(
            metric_out_dir,
            f"projection_file_index_daily_anomaly_2090s_{metric}_{TAG}.csv"
        )

        hist_csv = os.path.join(
            metric_out_dir,
            f"historical_daily_anomaly_values_full_spring_summer_{metric}_{TAG}.csv"
        )

        hist_nc = os.path.join(
            metric_out_dir,
            f"historical_daily_anomaly_values_full_spring_summer_{metric}_{TAG}.nc"
        )

        proj_csv = os.path.join(
            metric_out_dir,
            f"projection_daily_anomaly_values_2090s_full_spring_summer_{metric}_{TAG}.csv"
        )

        print("\nProcessing historical daily anomalies...")
        hist_df = process_historical_daily_anomaly_values(
            HIST_NC,
            stations,
            metric
        )

        hist_df.to_csv(hist_csv, index=False)

        hist_xr = dataframe_to_obs_xarray(hist_df)
        hist_xr.attrs["description"] = (
            "Historical daily soil moisture anomaly values for KDE. "
            "Anomaly = daily soil moisture minus station-specific historical daily climatology. "
            "Groups=noah_raw, vic_raw, mosaic_raw; seasons=full_year, spring, summer."
        )
        hist_xr.attrs["historical_source"] = HIST_NC
        hist_xr.attrs["stations_included"] = ",".join(stations)
        hist_xr.attrs["metric"] = metric
        hist_xr.attrs["projection_root"] = os.path.join(PROJ_ROOT, metric)
        hist_xr.attrs["historical_baseline"] = (
            "noah_raw: NOAH_sm 2016-2022; "
            "vic_raw: VIC_sm 2016-2022; "
            "mosaic_raw: MOSAIC_sm 2016-2022"
        )

        hist_xr.to_netcdf(hist_nc, engine="netcdf4")

        print("\nProcessing projection daily anomalies...")
        file_table = process_projection_daily_anomaly_values_to_csv(
            HIST_NC,
            stations,
            metric,
            proj_csv
        )

        file_table.to_csv(file_table_csv, index=False)

        print("\nSaved:")
        print(file_table_csv)
        print(hist_csv)
        print(hist_nc)
        print(proj_csv)

        print("\nProjection counts by group:")
        print(file_table.groupby("group").size().to_string())

        print("\nProjection counts by station:")
        print(file_table.groupby("station").size().to_string())

    print("\nAll metrics complete.")


if __name__ == "__main__":
    build_daily_anomaly_kde_value_files()