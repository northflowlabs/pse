"""
PSE spatial grid utilities.

Provides a common coordinate system for fusing data from multiple sources.
All connectors produce data on their native grids; the fusion engine
reprojects everything onto a canonical regular lat/lon grid before merging.

H3 hexagonal grid support is present for future use (Sprint 3+).
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from pse.connectors.base import SpatialBounds


def make_regular_grid(
    spatial: SpatialBounds,
    resolution_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a regular lat/lon grid covering *spatial* at *resolution_m* metre spacing.

    Returns:
        (lats, lons) — 1-D numpy arrays of coordinate values.
    """
    deg_per_m = 1.0 / 111_000
    step = resolution_m * deg_per_m
    step = max(step, 0.001)  # minimum 0.001° (~111 m)

    lats = np.arange(spatial.min_lat, spatial.max_lat + step / 2, step)
    lons = np.arange(spatial.min_lon, spatial.max_lon + step / 2, step)

    return np.round(lats, 6), np.round(lons, 6)


def regrid_dataset(
    ds: xr.Dataset,
    target_lats: np.ndarray,
    target_lons: np.ndarray,
    method: str = "linear",
) -> xr.Dataset:
    """
    Interpolate *ds* to the target lat/lon grid.

    If the dataset's native grid is coarser than the target, this upsamples
    (produces smooth interpolation).  If finer, it downsamples.

    Datasets with non-standard dimensions (e.g. OSM single-cell) are
    broadcast to the target grid shape.
    """
    has_lat = "latitude" in ds.dims
    has_lon = "longitude" in ds.dims

    if not (has_lat and has_lon):
        # Dataset has no spatial dimensions (e.g. point timeseries) — return as-is
        return ds

    native_lats = ds.latitude.values
    native_lons = ds.longitude.values

    # If the native grid is already a single cell, broadcast to target grid
    if len(native_lats) == 1 and len(native_lons) == 1:
        return _broadcast_single_cell(ds, target_lats, target_lons)

    # Clip target to native extent (avoid extrapolation artefacts)
    clipped_lats = target_lats[
        (target_lats >= float(native_lats.min())) &
        (target_lats <= float(native_lats.max()))
    ]
    clipped_lons = target_lons[
        (target_lons >= float(native_lons.min())) &
        (target_lons <= float(native_lons.max()))
    ]

    if len(clipped_lats) == 0 or len(clipped_lons) == 0:
        # No overlap — return original
        return ds

    try:
        return ds.interp(
            latitude=clipped_lats,
            longitude=clipped_lons,
            method=method,
            kwargs={"fill_value": "extrapolate"},
        )
    except Exception:
        # Fallback to nearest-neighbour if linear fails
        return ds.interp(latitude=clipped_lats, longitude=clipped_lons, method="nearest")


def _broadcast_single_cell(
    ds: xr.Dataset,
    target_lats: np.ndarray,
    target_lons: np.ndarray,
) -> xr.Dataset:
    """Broadcast a 1×1 cell dataset to the full target grid."""
    new_vars = {}
    for var in ds.data_vars:
        da = ds[var]
        # Expand lat/lon dimensions by repeating the single value
        axes_to_expand = []
        if "latitude" in da.dims:
            axes_to_expand.append(da.dims.index("latitude"))
        if "longitude" in da.dims:
            axes_to_expand.append(da.dims.index("longitude"))

        arr = da.values
        # Build target shape by replacing 1s with target sizes
        new_shape = list(arr.shape)
        if "latitude" in da.dims:
            idx = list(da.dims).index("latitude")
            new_shape[idx] = len(target_lats)
        if "longitude" in da.dims:
            idx = list(da.dims).index("longitude")
            new_shape[idx] = len(target_lons)

        new_arr = np.broadcast_to(arr, new_shape).copy()
        new_vars[var] = (list(da.dims), new_arr)

    coords = dict(ds.coords)
    coords["latitude"] = target_lats
    coords["longitude"] = target_lons
    return xr.Dataset(new_vars, coords=coords, attrs=ds.attrs)
