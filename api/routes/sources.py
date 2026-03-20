"""
GET /api/v1/sources — list available data sources and their metadata.
"""
from __future__ import annotations

from datetime import UTC

from fastapi import APIRouter, Depends, Request

from pse.api.middleware.auth import verify_api_key

router = APIRouter()


@router.get("/sources")
async def list_sources(
    request: Request,
    _key: str = Depends(verify_api_key),
):
    """Return all registered connectors with their variables and capabilities."""
    engine = request.app.state.query_engine
    return {
        "sources": [
            {
                "source_id": connector.source_id,
                "variables": connector.variables,
                "update_frequency_seconds": connector.update_frequency_seconds,
            }
            for connector in engine._connectors.values()
        ],
        "available_variables": engine.available_variables(),
    }


@router.get("/sources/{source_id}/status")
async def source_status(
    source_id: str,
    request: Request,
    _key: str = Depends(verify_api_key),
):
    """Return detailed status for a specific connector."""
    from datetime import datetime

    engine = request.app.state.query_engine
    connector = engine._connectors.get(source_id)
    if connector is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Source '{source_id}' not found.")

    try:
        latest_ts = await connector.get_latest_timestamp()
        lag = (datetime.now(UTC) - latest_ts.replace(tzinfo=UTC)).total_seconds()
        return {
            "source_id": source_id,
            "status": "ok",
            "latest_data_timestamp": latest_ts.isoformat(),
            "lag_seconds": round(lag, 0),
            "variables": connector.variables,
            "update_frequency_seconds": connector.update_frequency_seconds,
        }
    except Exception as exc:
        return {
            "source_id": source_id,
            "status": "error",
            "error": str(exc),
        }
