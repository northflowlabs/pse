"""
PSE query endpoints:

  GET  /api/v1/point   — timeseries at a single lat/lon
  GET  /api/v1/query   — gridded query over a bounding box
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Annotated, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import Response

from pse.api.middleware.auth import verify_api_key
from pse.api.middleware.rate_limit import rate_limit
from pse.connectors.base import SpatialBounds, TemporalBounds

router = APIRouter()

# ---------------------------------------------------------------------------
# /point
# ---------------------------------------------------------------------------

@router.get("/point")
async def point_query(
    request: Request,
    lat: Annotated[float, Query(ge=-90, le=90, description="Latitude (WGS84)")],
    lon: Annotated[float, Query(ge=-180, le=180, description="Longitude (WGS84)")],
    variables: Annotated[str, Query(description="Comma-separated PSE variable names")],
    time_start: Annotated[datetime, Query(description="Start of time range (ISO 8601 UTC)")],
    time_end: Annotated[datetime, Query(description="End of time range (ISO 8601 UTC)")],
    _key: str = Depends(verify_api_key),
    _rl: None = Depends(rate_limit),
):
    """
    Return a timeseries of requested variables at a single geographic point.

    Example::

        GET /api/v1/point
            ?lat=-6.2&lon=106.8
            &variables=temperature_2m,wind_speed_10m
            &time_start=2025-01-01T00:00:00Z
            &time_end=2025-01-07T00:00:00Z
    """
    engine = request.app.state.query_engine
    var_list = [v.strip() for v in variables.split(",") if v.strip()]

    if not var_list:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one variable is required.",
        )

    try:
        temporal = TemporalBounds(start=time_start, end=time_end)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

    try:
        ds = await engine.point_query(
            lat=lat,
            lon=lon,
            variables=var_list,
            temporal=temporal,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Data retrieval failed: {exc}",
        )

    return _dataset_to_response(ds, lat=lat, lon=lon)


# ---------------------------------------------------------------------------
# /query  (gridded bounding box)
# ---------------------------------------------------------------------------

@router.get("/query")
async def gridded_query(
    request: Request,
    variables: Annotated[str, Query(description="Comma-separated PSE variable names")],
    lat_min: Annotated[float, Query(ge=-90, le=90)],
    lat_max: Annotated[float, Query(ge=-90, le=90)],
    lon_min: Annotated[float, Query(ge=-180, le=180)],
    lon_max: Annotated[float, Query(ge=-180, le=180)],
    time_start: Annotated[datetime, Query(description="Start of time range (ISO 8601 UTC)")],
    time_end: Annotated[datetime, Query(description="End of time range (ISO 8601 UTC)")],
    resolution: Annotated[Optional[float], Query(gt=0, description="Target spatial resolution (metres)")] = None,
    _key: str = Depends(verify_api_key),
    _rl: None = Depends(rate_limit),
):
    """
    Return a gridded dataset over a bounding box.

    Example::

        GET /api/v1/query
            ?variables=temperature_2m,solar_ghi
            &lat_min=-8.5&lat_max=-6.0&lon_min=110.0&lon_max=115.0
            &time_start=2025-01-01T00:00:00Z&time_end=2025-03-01T00:00:00Z
    """
    engine = request.app.state.query_engine
    var_list = [v.strip() for v in variables.split(",") if v.strip()]

    if not var_list:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one variable is required.",
        )

    try:
        spatial = SpatialBounds(
            min_lat=lat_min, max_lat=lat_max,
            min_lon=lon_min, max_lon=lon_max,
        )
        temporal = TemporalBounds(start=time_start, end=time_end)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

    try:
        ds = await engine.query(
            variables=var_list,
            spatial=spatial,
            temporal=temporal,
            resolution_m=resolution,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Data retrieval failed: {exc}",
        )

    return _dataset_to_response(ds)


# ---------------------------------------------------------------------------
# Shared serialiser
# ---------------------------------------------------------------------------

def _dataset_to_response(ds, **extra_meta) -> dict:
    """
    Convert an xarray.Dataset to a JSON-serialisable dict.

    Structure::

        {
          "metadata": { ...provenance attrs + extra_meta },
          "coordinates": { "time": [...], "latitude": [...], "longitude": [...] },
          "variables": {
            "temperature_2m": { "values": [[...], ...], "units": "...", "dims": [...] }
          }
        }
    """
    import numpy as np
    import pandas as pd

    coords = {}
    for dim in ["time", "latitude", "longitude"]:
        if dim in ds.coords:
            raw = ds.coords[dim].values
            if dim == "time":
                coords[dim] = [
                    pd.Timestamp(t).isoformat() for t in raw
                ]
            else:
                coords[dim] = raw.tolist()

    variables = {}
    for var in ds.data_vars:
        da = ds[var]
        # Replace NaN with None for valid JSON
        vals = np.where(np.isnan(da.values), None, da.values)
        variables[var] = {
            "values": vals.tolist(),
            "dims": list(da.dims),
            "units": da.attrs.get("units", ""),
        }

    metadata = dict(ds.attrs)
    metadata.update(extra_meta)

    return {
        "metadata": metadata,
        "coordinates": coords,
        "variables": variables,
    }
