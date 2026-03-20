"""
PSE spatial query helpers — clipping, reprojection, bounding-box operations.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import xarray as xr

from pse.connectors.base import SpatialBounds


def clip_to_bounds(ds: xr.Dataset, bounds: SpatialBounds) -> xr.Dataset:
    """
    Clip a Dataset to the given geographic bounding box.

    Assumes the Dataset has 'latitude' and 'longitude' coordinate arrays.
    """
    return ds.sel(
        latitude=slice(bounds.min_lat, bounds.max_lat),
        longitude=slice(bounds.min_lon, bounds.max_lon),
    )


def regrid_to_resolution(
    ds: xr.Dataset,
    target_resolution_m: float,
    bounds: Optional[SpatialBounds] = None,
) -> xr.Dataset:
    """
    Resample a gridded Dataset to a target spatial resolution using linear
    interpolation.

    Args:
        ds:                   Source dataset (must have latitude + longitude dims).
        target_resolution_m:  Desired grid spacing in metres.
        bounds:               If provided, the output grid is bounded to this region.

    Returns:
        Dataset resampled to the target grid.
    """
    deg_per_m = 1.0 / 111_000
    step = target_resolution_m * deg_per_m

    lat_min = float(ds.latitude.min()) if bounds is None else bounds.min_lat
    lat_max = float(ds.latitude.max()) if bounds is None else bounds.max_lat
    lon_min = float(ds.longitude.min()) if bounds is None else bounds.min_lon
    lon_max = float(ds.longitude.max()) if bounds is None else bounds.max_lon

    new_lats = np.arange(lat_min, lat_max + step / 2, step)
    new_lons = np.arange(lon_min, lon_max + step / 2, step)

    return ds.interp(latitude=new_lats, longitude=new_lons, method="linear")


def point_query(
    ds: xr.Dataset,
    lat: float,
    lon: float,
    method: str = "nearest",
) -> xr.Dataset:
    """
    Extract a timeseries at a single (lat, lon) point.

    Args:
        ds:     Gridded dataset.
        lat:    Target latitude.
        lon:    Target longitude.
        method: Interpolation method — 'nearest' or 'linear'.

    Returns:
        Dataset with latitude and longitude dimensions dropped (timeseries only).
    """
    return ds.sel(
        latitude=lat,
        longitude=lon,
        method=method,
    )


def bounding_box_area_km2(bounds: SpatialBounds) -> float:
    """
    Approximate area of a bounding box in km².
    Uses the equatorial approximation (good to ~1% for |lat| < 60°).
    """
    lat_km = (bounds.max_lat - bounds.min_lat) * 111.0
    avg_lat_rad = np.radians((bounds.min_lat + bounds.max_lat) / 2)
    lon_km = (bounds.max_lon - bounds.min_lon) * 111.0 * np.cos(avg_lat_rad)
    return abs(lat_km * lon_km)
