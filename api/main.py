"""
PSE FastAPI application entry point.

Startup sequence:
  1. Load settings from environment / .env
  2. Initialise database (create tables if needed)
  3. Build connector registry
  4. Create QueryEngine + Cache
  5. Mount routes
"""
from __future__ import annotations

import logging

import structlog
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from pse.api.middleware.cors import CORS_KWARGS
from pse.api.routes import fuse, health, query, sources
from pse.config.settings import settings
from pse.connectors.era5 import ERA5Connector
from pse.connectors.global_solar_atlas import GlobalSolarAtlasConnector
from pse.connectors.open_meteo import OpenMeteoConnector
from pse.connectors.osm import OSMConnector
from pse.connectors.sentinel2 import Sentinel2Connector
from pse.connectors.world_bank import WorldBankConnector
from pse.fusion.engine import FusionEngine
from pse.query.engine import QueryEngine
from pse.store.cache import PSECache
from pse.store.postgres import PSEDatabase

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=settings.log_level.upper())
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(
        logging.getLevelName(settings.log_level.upper())
    )
)
log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    app = FastAPI(
        title="PSE — Planetary Sensing Engine",
        description=(
            "Northflow Technologies AS — unified spatiotemporal data API. "
            "Ingests, normalises, and fuses satellite, weather, and sensor data."
        ),
        version="0.2.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS
    app.add_middleware(CORSMiddleware, **CORS_KWARGS)

    # ------------------------------------------------------------------ #
    # Startup / shutdown lifecycle                                         #
    # ------------------------------------------------------------------ #

    @app.on_event("startup")
    async def startup():
        log.info("PSE starting up", environment=settings.environment)

        # Database
        db = PSEDatabase(settings.database_url)
        await db.init()
        app.state.db = db

        # Connectors — all Sprint 2 sources registered here.
        # Connectors without credentials log a warning and are still registered;
        # they will return ConnectorError on fetch if the key is missing.
        connectors = {
            "open_meteo": OpenMeteoConnector(),
            "global_solar_atlas": GlobalSolarAtlasConnector(),
            "era5": ERA5Connector(),
            "sentinel2": Sentinel2Connector(),
            "osm": OSMConnector(),
            "world_bank": WorldBankConnector(),
        }

        # Query engine + cache
        cache = PSECache(default_ttl=3600.0)
        engine = QueryEngine(connectors=connectors, cache=cache)
        app.state.query_engine = engine

        # Fusion engine (wraps QueryEngine with multi-source alignment + blend)
        fusion = FusionEngine(engine)
        app.state.fusion_engine = fusion

        log.info(
            "PSE ready",
            connectors=list(connectors.keys()),
            variables=len(engine.available_variables()),
        )

    @app.on_event("shutdown")
    async def shutdown():
        await app.state.db.close()
        log.info("PSE shut down cleanly")

    # ------------------------------------------------------------------ #
    # Routes                                                               #
    # ------------------------------------------------------------------ #

    app.include_router(health.router, prefix="/api/v1", tags=["System"])
    app.include_router(query.router, prefix="/api/v1", tags=["Query"])
    app.include_router(fuse.router, prefix="/api/v1", tags=["Fusion"])
    app.include_router(sources.router, prefix="/api/v1", tags=["Sources"])

    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "service": "PSE — Planetary Sensing Engine",
            "version": "0.1.0",
            "docs": "/docs",
            "health": "/api/v1/health",
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "pse.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.environment == "development",
        log_level=settings.log_level.lower(),
    )
