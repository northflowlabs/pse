"""
GET /api/v1/health — system health and connector freshness.
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, Request

from pse.api.middleware.auth import verify_api_key

if TYPE_CHECKING:
    from pse.query.engine import QueryEngine
    from pse.store.postgres import PSEDatabase

router = APIRouter()


@router.get("/health")
async def health(
    request: Request,
    _key: str = Depends(verify_api_key),
):
    """
    Return overall system health including:
    - Database reachability
    - Per-connector status (last fetch, latest data timestamp)
    - Cache statistics
    """
    engine: QueryEngine = request.app.state.query_engine
    db: PSEDatabase = request.app.state.db

    # Database ping
    db_ok = await db.ping()

    # Connector freshness — fetch latest timestamps in parallel
    connector_statuses = {}
    async def _check_connector(src_id, connector):
        try:
            ts = await asyncio.wait_for(connector.get_latest_timestamp(), timeout=5.0)
            lag = (datetime.now(UTC) - ts.replace(tzinfo=UTC)).total_seconds()
            return src_id, {
                "status": "ok",
                "latest_data_timestamp": ts.isoformat(),
                "lag_seconds": round(lag, 0),
                "variables": connector.variables,
                "update_frequency_seconds": connector.update_frequency_seconds,
            }
        except Exception as exc:
            return src_id, {
                "status": "error",
                "error": str(exc),
                "variables": connector.variables,
            }

    results = await asyncio.gather(
        *[_check_connector(sid, c) for sid, c in engine._connectors.items()]
    )
    for src_id, info in results:
        connector_statuses[src_id] = info

    # Cache stats
    cache_stats = engine._cache.stats()

    overall = "ok" if db_ok and all(
        s.get("status") == "ok" for s in connector_statuses.values()
    ) else "degraded"

    return {
        "status": overall,
        "timestamp": datetime.now(UTC).isoformat(),
        "database": {"status": "ok" if db_ok else "error"},
        "connectors": connector_statuses,
        "cache": cache_stats,
        "version": "0.1.0",
    }
