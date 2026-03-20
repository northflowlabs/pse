"""
POST /api/v1/fuse — multi-source quality-weighted fused query.

Unlike /api/v1/query (which fetches from a single connector per variable),
/fuse explicitly invokes the FusionEngine to fetch from ALL capable sources
simultaneously and return a quality-weighted blend.
"""
from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from pse.api.middleware.auth import verify_api_key
from pse.api.middleware.rate_limit import rate_limit
from pse.api.routes.query import _dataset_to_response
from pse.connectors.base import SpatialBounds, TemporalBounds

router = APIRouter()


class FuseRequest(BaseModel):
    variables: list[str] = Field(..., description="PSE canonical variable names")
    lat_min: float = Field(..., ge=-90, le=90)
    lat_max: float = Field(..., ge=-90, le=90)
    lon_min: float = Field(..., ge=-180, le=180)
    lon_max: float = Field(..., ge=-180, le=180)
    time_start: datetime = Field(..., description="ISO 8601 UTC")
    time_end: datetime = Field(..., description="ISO 8601 UTC")
    resolution_m: float = Field(5_000.0, gt=0, description="Target spatial resolution (metres)")
    prefer_recent: bool = Field(True, description="Weight more-recent sources higher")


@router.post("/fuse")
async def fused_query(
    body: FuseRequest,
    request: Request,
    _key: str = Depends(verify_api_key),
    _rl: None = Depends(rate_limit),
):
    """
    Multi-source quality-weighted fused query.

    Fetches from every connector capable of providing each requested variable,
    aligns them spatially and temporally, and returns a quality-weighted blend.

    Includes ``pse_fusion_weights`` and ``pse_source_quality_scores`` in the
    response metadata for full provenance.
    """
    fusion = request.app.state.fusion_engine

    try:
        spatial = SpatialBounds(
            min_lat=body.lat_min, max_lat=body.lat_max,
            min_lon=body.lon_min, max_lon=body.lon_max,
        )
        temporal = TemporalBounds(start=body.time_start, end=body.time_end)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

    try:
        ds = await fusion.query(
            variables=body.variables,
            spatial=spatial,
            temporal=temporal,
            resolution_m=body.resolution_m,
            prefer_recent=body.prefer_recent,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))

    return _dataset_to_response(ds)
