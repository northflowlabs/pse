"""
PSE temporal query helpers — slicing, alignment, and aggregation.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr

from pse.connectors.base import TemporalBounds


def clip_to_bounds(ds: xr.Dataset, temporal: TemporalBounds) -> xr.Dataset:
    """Slice dataset to the given time range (inclusive)."""
    return ds.sel(
        time=slice(
            pd.Timestamp(temporal.start),
            pd.Timestamp(temporal.end),
        )
    )


def align_to_common_times(
    datasets: list[xr.Dataset],
    method: Literal["nearest", "linear"] = "linear",
    freq: str = "1h",
) -> list[xr.Dataset]:
    """
    Interpolate all datasets to a shared, regular time axis.

    Useful before fusing data from sources with different time steps.

    Args:
        datasets:  List of datasets to align.
        method:    Interpolation method.
        freq:      Target frequency (pandas offset alias, e.g. '1h', '3h', '1D').

    Returns:
        List of datasets resampled to the common time axis.
    """
    if not datasets:
        return []

    # Build common time index spanning all datasets
    all_starts = [pd.Timestamp(ds.time.values[0]) for ds in datasets]
    all_ends = [pd.Timestamp(ds.time.values[-1]) for ds in datasets]
    common_start = min(all_starts)
    common_end = max(all_ends)

    common_times = pd.date_range(common_start, common_end, freq=freq)

    aligned = []
    for ds in datasets:
        # Only interpolate if the dataset has a time dimension
        if "time" not in ds.dims or len(ds.time) < 2:
            aligned.append(ds)
            continue
        try:
            ds_interp = ds.interp(time=common_times, method=method)
            aligned.append(ds_interp)
        except Exception:
            # Fall back to nearest-neighbour if linear fails (e.g. non-numeric vars)
            ds_interp = ds.interp(time=common_times, method="nearest")
            aligned.append(ds_interp)

    return aligned


def aggregate(
    ds: xr.Dataset,
    freq: str,
    method: Literal["mean", "sum", "max", "min"] = "mean",
) -> xr.Dataset:
    """
    Temporally aggregate a dataset to a coarser frequency.

    Args:
        ds:     Source dataset (must have a 'time' dimension).
        freq:   Target frequency, e.g. '1D' (daily), '1ME' (monthly).
        method: Aggregation method.

    Returns:
        Aggregated dataset.
    """
    resampler = ds.resample(time=freq)
    agg_fn = getattr(resampler, method)
    return agg_fn()


def fill_missing_times(ds: xr.Dataset, freq: str = "1h") -> xr.Dataset:
    """
    Reindex to a regular time axis and forward-fill short gaps (up to 3 steps).
    """
    start = pd.Timestamp(ds.time.values[0])
    end = pd.Timestamp(ds.time.values[-1])
    full_index = pd.date_range(start, end, freq=freq)
    return ds.reindex(time=full_index).ffill(dim="time", limit=3)


def latest_value(ds: xr.Dataset) -> xr.Dataset:
    """Return the last time step of a dataset."""
    return ds.isel(time=-1)
