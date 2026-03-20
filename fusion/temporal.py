"""
PSE temporal alignment for multi-source fusion.

Aligns datasets from different sources to a common time axis before merging.
Different sources have different time steps:
  - ERA5:           hourly
  - Open-Meteo:     hourly
  - Sentinel-2:     scene acquisitions (~5-day revisit)
  - Global Solar Atlas: monthly climatology (no true time axis)
  - World Bank:     annual
  - OSM:            current-state (timeless)

The alignment strategy:
  - Continuous time-series (ERA5, Open-Meteo): interpolate to common hourly grid
  - Snapshot sources (Sentinel-2): nearest-time join
  - Climatological sources (GSA, World Bank): broadcast to each time step
"""
from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr

log = logging.getLogger(__name__)

# Sources whose time dimension represents climatological months, not real timestamps
_CLIMATOLOGICAL_SOURCES = {"global_solar_atlas"}
# Sources with no meaningful time dimension
_TIMELESS_SOURCES = {"osm"}


def classify_time_axis(ds: xr.Dataset) -> Literal["hourly", "daily", "monthly_clim", "annual", "snapshot", "none"]:
    """
    Classify the temporal resolution of a dataset.

    Returns one of:
      - "none"         : no time dimension
      - "monthly_clim" : 12 integer month indices (climatology)
      - "annual"       : annual frequency
      - "daily"        : daily frequency
      - "hourly"       : hourly frequency
      - "snapshot"     : irregular / sparse (e.g. Sentinel-2 scenes)
    """
    if "time" not in ds.dims:
        return "none"

    time_vals = ds.time.values
    if len(time_vals) == 0:
        return "none"

    # Check for integer month index (GSA climatology)
    if time_vals.dtype in (np.int32, np.int64):
        if set(time_vals.tolist()).issubset(set(range(1, 13))):
            return "monthly_clim"

    try:
        times = pd.to_datetime(time_vals)
    except Exception:
        return "snapshot"

    if len(times) < 2:
        return "snapshot"

    diffs = (times[1:] - times[:-1]).total_seconds()
    median_diff = float(np.median(diffs))

    if median_diff < 2 * 3600:
        return "hourly"
    if median_diff < 36 * 3600:
        return "daily"
    if median_diff < 32 * 86400:
        return "monthly_clim"
    return "annual"


def align_to_common_axis(
    datasets: list[xr.Dataset],
    target_freq: str = "1h",
    temporal_start: pd.Timestamp | None = None,
    temporal_end: pd.Timestamp | None = None,
) -> list[xr.Dataset]:
    """
    Align a list of datasets to a shared time axis at *target_freq* frequency.

    Args:
        datasets:        Datasets to align (may have different time resolutions).
        target_freq:     pandas offset alias for the target time step.
        temporal_start:  Start of common axis (inferred from data if None).
        temporal_end:    End of common axis (inferred from data if None).

    Returns:
        List of aligned datasets in the same order as the input.
    """
    if not datasets:
        return []

    # Build common time axis
    real_time_datasets = [
        ds for ds in datasets
        if classify_time_axis(ds) in ("hourly", "daily", "snapshot")
        and "time" in ds.dims
        and not ds.time.values.dtype in (np.int32, np.int64)
    ]

    if real_time_datasets:
        all_starts = [pd.Timestamp(ds.time.values[0]) for ds in real_time_datasets]
        all_ends = [pd.Timestamp(ds.time.values[-1]) for ds in real_time_datasets]
        t_start = temporal_start or min(all_starts)
        t_end = temporal_end or max(all_ends)
        common_times = pd.date_range(t_start, t_end, freq=target_freq)
    else:
        common_times = None

    aligned = []
    for ds in datasets:
        cls = classify_time_axis(ds)

        if cls == "none":
            # No time dim — broadcast a single time step
            aligned.append(ds)
            continue

        if cls == "monthly_clim":
            # GSA-style: 12-step climatology
            # Broadcast each month's value to the matching real time steps
            if common_times is not None:
                aligned.append(_broadcast_monthly_clim(ds, common_times))
            else:
                aligned.append(ds)
            continue

        if cls == "annual":
            if common_times is not None:
                aligned.append(_broadcast_annual(ds, common_times))
            else:
                aligned.append(ds)
            continue

        # hourly / daily / snapshot → interpolate to common axis
        if common_times is None or "time" not in ds.dims:
            aligned.append(ds)
            continue

        try:
            aligned.append(ds.interp(time=common_times, method="linear"))
        except Exception:
            try:
                aligned.append(ds.interp(time=common_times, method="nearest"))
            except Exception:
                log.warning("[fusion/temporal] Could not align dataset to common axis")
                aligned.append(ds)

    return aligned


def _broadcast_monthly_clim(ds: xr.Dataset, common_times: pd.DatetimeIndex) -> xr.Dataset:
    """
    Expand a 12-step climatology dataset to a full time axis by
    mapping each timestamp to its matching month.
    """
    months = ds.time.values  # integer month indices 1-12
    new_vars: dict[str, xr.DataArray] = {}

    for var in ds.data_vars:
        da = ds[var]
        if "time" not in da.dims:
            new_vars[var] = da
            continue

        time_idx = da.dims.index("time")
        out_shape = list(da.values.shape)
        out_shape[time_idx] = len(common_times)
        out_arr = np.full(out_shape, np.nan, dtype=np.float32)

        for i, ts in enumerate(common_times):
            month = ts.month
            m_indices = np.where(months == month)[0]
            if len(m_indices) > 0:
                src_idx = m_indices[0]
                idx_tuple = [slice(None)] * len(out_shape)
                idx_tuple[time_idx] = i
                src_tuple = [slice(None)] * len(da.values.shape)
                src_tuple[time_idx] = src_idx
                out_arr[tuple(idx_tuple)] = da.values[tuple(src_tuple)]

        new_dims = list(da.dims)
        new_coords = {k: v for k, v in da.coords.items() if k != "time"}
        new_coords["time"] = common_times
        new_vars[var] = xr.DataArray(out_arr, dims=new_dims, coords=new_coords)

    return xr.Dataset(new_vars, attrs=ds.attrs)


def _broadcast_annual(ds: xr.Dataset, common_times: pd.DatetimeIndex) -> xr.Dataset:
    """
    Broadcast annual values to each time step in common_times by matching year.
    """
    try:
        ds_years = pd.to_datetime(ds.time.values).year
    except Exception:
        return ds

    new_vars: dict[str, xr.DataArray] = {}
    for var in ds.data_vars:
        da = ds[var]
        if "time" not in da.dims:
            new_vars[var] = da
            continue

        time_idx = da.dims.index("time")
        out_shape = list(da.values.shape)
        out_shape[time_idx] = len(common_times)
        out_arr = np.full(out_shape, np.nan, dtype=np.float32)

        for i, ts in enumerate(common_times):
            year = ts.year
            y_indices = np.where(ds_years == year)[0]
            if len(y_indices) > 0:
                src_idx = y_indices[0]
                idx_tuple = [slice(None)] * len(out_shape)
                idx_tuple[time_idx] = i
                src_tuple = [slice(None)] * len(da.values.shape)
                src_tuple[time_idx] = src_idx
                out_arr[tuple(idx_tuple)] = da.values[tuple(src_tuple)]

        new_dims = list(da.dims)
        new_coords = {k: v for k, v in da.coords.items() if k != "time"}
        new_coords["time"] = common_times
        new_vars[var] = xr.DataArray(out_arr, dims=new_dims, coords=new_coords)

    return xr.Dataset(new_vars, attrs=ds.attrs)
