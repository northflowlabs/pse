"""
PSE scheduled data ingestion.

Runs as a long-lived process (docker-compose scheduler service).
Fetches daily data from all configured connectors for a rolling 7-day window
and persists to the Zarr store.

Sprint 1: basic structure + Open-Meteo daily ingest.
Sprint 2: ERA5, Sentinel-2, OSM, World Bank.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from pse.config.settings import settings
from pse.connectors.base import SpatialBounds, TemporalBounds
from pse.connectors.era5 import ERA5Connector
from pse.connectors.global_solar_atlas import GlobalSolarAtlasConnector
from pse.connectors.open_meteo import OpenMeteoConnector
from pse.connectors.osm import OSMConnector
from pse.connectors.world_bank import WorldBankConnector
from pse.store.postgres import PSEDatabase
from pse.store.zarr_store import ZarrStore

logging.basicConfig(level=settings.log_level.upper())
log = structlog.get_logger(__name__)

# Default ingest region — adjust or make configurable per deployment.
# Covers Indonesia (Phase 1 FLUX focus).
DEFAULT_REGION = SpatialBounds(
    min_lat=-11.0, max_lat=6.0,
    min_lon=95.0, max_lon=141.0,
)
DEFAULT_VARIABLES_OPEN_METEO = [
    "temperature_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "solar_ghi",
    "surface_pressure",
    "cloud_cover",
    "relative_humidity",
]


async def ingest_open_meteo(db: PSEDatabase, zarr: ZarrStore) -> None:
    """Ingest yesterday's Open-Meteo data for the default region."""
    connector = OpenMeteoConnector()
    yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)
    temporal = TemporalBounds(
        start=datetime(yesterday.year, yesterday.month, yesterday.day),
        end=datetime(yesterday.year, yesterday.month, yesterday.day, 23, 59, 59),
    )

    log.info("Ingesting Open-Meteo", date=yesterday.isoformat())
    try:
        ds = await connector.fetch(
            variables=DEFAULT_VARIABLES_OPEN_METEO,
            spatial=DEFAULT_REGION,
            temporal=temporal,
            resolution=50_000,  # 50 km grid for regional ingest
        )
        zarr_path = zarr.write(ds, source_id="open_meteo", reference_date=yesterday)
        quality = await connector.get_quality(DEFAULT_REGION, temporal)

        async with db.session() as session:
            await db.record_ingest(
                session,
                source_id="open_meteo",
                variables=DEFAULT_VARIABLES_OPEN_METEO,
                min_lat=DEFAULT_REGION.min_lat,
                max_lat=DEFAULT_REGION.max_lat,
                min_lon=DEFAULT_REGION.min_lon,
                max_lon=DEFAULT_REGION.max_lon,
                time_start=temporal.start,
                time_end=temporal.end,
                zarr_path=zarr_path,
                quality_score=quality.overall_score,
                provenance=quality.provenance,
            )
            await db.update_source_status(
                session,
                "open_meteo",
                last_successful_fetch=datetime.now(timezone.utc),
                latest_data_timestamp=temporal.end,
                quality_score=quality.overall_score,
            )

        log.info("Open-Meteo ingest complete", date=yesterday.isoformat(), path=zarr_path)

    except Exception as exc:
        log.error("Open-Meteo ingest failed", error=str(exc))
        async with db.session() as session:
            await db.update_source_status(
                session, "open_meteo", last_error=str(exc)
            )


async def ingest_era5(db: PSEDatabase, zarr: ZarrStore) -> None:
    """Ingest ERA5 data for the last completed day (5-day lag)."""
    if not settings.era5_cds_api_key:
        log.info("ERA5 ingest skipped — ERA5_CDS_API_KEY not configured")
        return

    connector = ERA5Connector()
    from datetime import timedelta
    target_date = (datetime.now(timezone.utc) - timedelta(days=6)).date()
    temporal = TemporalBounds(
        start=datetime(target_date.year, target_date.month, target_date.day),
        end=datetime(target_date.year, target_date.month, target_date.day, 23, 59, 59),
    )
    era5_vars = [
        "temperature_2m", "wind_speed_10m", "wind_u_10m", "wind_v_10m",
        "solar_ghi", "surface_pressure", "precipitation",
    ]

    log.info("Ingesting ERA5", date=target_date.isoformat())
    try:
        ds = await connector.fetch(
            variables=era5_vars,
            spatial=DEFAULT_REGION,
            temporal=temporal,
            resolution=50_000,
        )
        zarr_path = zarr.write(ds, source_id="era5", reference_date=target_date)
        quality = await connector.get_quality(DEFAULT_REGION, temporal)

        async with db.session() as session:
            await db.record_ingest(
                session,
                source_id="era5",
                variables=era5_vars,
                min_lat=DEFAULT_REGION.min_lat, max_lat=DEFAULT_REGION.max_lat,
                min_lon=DEFAULT_REGION.min_lon, max_lon=DEFAULT_REGION.max_lon,
                time_start=temporal.start, time_end=temporal.end,
                zarr_path=zarr_path,
                quality_score=quality.overall_score,
                provenance=quality.provenance,
            )
            await db.update_source_status(
                session, "era5",
                last_successful_fetch=datetime.now(timezone.utc),
                latest_data_timestamp=temporal.end,
                quality_score=quality.overall_score,
            )
        log.info("ERA5 ingest complete", date=target_date.isoformat())

    except Exception as exc:
        log.error("ERA5 ingest failed", error=str(exc))
        async with db.session() as session:
            await db.update_source_status(session, "era5", last_error=str(exc))


async def ingest_osm_static(db: PSEDatabase, zarr: ZarrStore) -> None:
    """Refresh OSM infrastructure data for the default region (weekly)."""
    from datetime import timedelta
    connector = OSMConnector()
    temporal = TemporalBounds(
        start=datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0),
        end=datetime.now(timezone.utc),
    )
    osm_vars = ["power_line_density", "substation_count", "nearest_substation_lat",
                "nearest_substation_lon", "road_density", "settlement_count"]

    log.info("Ingesting OSM infrastructure")
    try:
        ds = await connector.fetch(variables=osm_vars, spatial=DEFAULT_REGION, temporal=temporal)
        today = datetime.now(timezone.utc).date()
        zarr_path = zarr.write(ds, source_id="osm", reference_date=today)
        quality = await connector.get_quality(DEFAULT_REGION, temporal)

        async with db.session() as session:
            await db.record_ingest(
                session,
                source_id="osm",
                variables=osm_vars,
                min_lat=DEFAULT_REGION.min_lat, max_lat=DEFAULT_REGION.max_lat,
                min_lon=DEFAULT_REGION.min_lon, max_lon=DEFAULT_REGION.max_lon,
                time_start=temporal.start, time_end=temporal.end,
                zarr_path=zarr_path,
                quality_score=quality.overall_score,
            )
            await db.update_source_status(
                session, "osm",
                last_successful_fetch=datetime.now(timezone.utc),
                quality_score=quality.overall_score,
            )
        log.info("OSM ingest complete")
    except Exception as exc:
        log.error("OSM ingest failed", error=str(exc))
        async with db.session() as session:
            await db.update_source_status(session, "osm", last_error=str(exc))


async def ingest_world_bank(db: PSEDatabase, zarr: ZarrStore) -> None:
    """Refresh World Bank indicators (monthly)."""
    from datetime import timedelta
    connector = WorldBankConnector()
    temporal = TemporalBounds(
        start=datetime(2015, 1, 1),
        end=datetime(datetime.now(timezone.utc).year - 1, 12, 31),
    )
    wb_vars = ["electricity_access_pct", "rural_electricity_access",
               "gdp_per_capita_usd", "population_total", "renewable_energy_pct"]

    log.info("Ingesting World Bank indicators")
    try:
        ds = await connector.fetch(variables=wb_vars, spatial=DEFAULT_REGION, temporal=temporal)
        today = datetime.now(timezone.utc).date()
        zarr_path = zarr.write(ds, source_id="world_bank", reference_date=today)
        quality = await connector.get_quality(DEFAULT_REGION, temporal)

        async with db.session() as session:
            await db.record_ingest(
                session,
                source_id="world_bank",
                variables=wb_vars,
                min_lat=DEFAULT_REGION.min_lat, max_lat=DEFAULT_REGION.max_lat,
                min_lon=DEFAULT_REGION.min_lon, max_lon=DEFAULT_REGION.max_lon,
                time_start=temporal.start, time_end=temporal.end,
                zarr_path=zarr_path,
                quality_score=quality.overall_score,
            )
            await db.update_source_status(
                session, "world_bank",
                last_successful_fetch=datetime.now(timezone.utc),
                quality_score=quality.overall_score,
            )
        log.info("World Bank ingest complete")
    except Exception as exc:
        log.error("World Bank ingest failed", error=str(exc))
        async with db.session() as session:
            await db.update_source_status(session, "world_bank", last_error=str(exc))


async def main() -> None:
    log.info("PSE scheduler starting")

    db = PSEDatabase(settings.database_url)
    await db.init()

    zarr = ZarrStore(settings.zarr_store_path)

    scheduler = AsyncIOScheduler(timezone="UTC")

    # Open-Meteo: daily at 04:00 UTC
    scheduler.add_job(
        ingest_open_meteo, trigger="cron", hour=4, minute=0, args=[db, zarr],
        id="open_meteo_daily", name="Open-Meteo daily ingest", replace_existing=True,
    )

    # ERA5: daily at 05:00 UTC (only runs if CDS key is configured)
    scheduler.add_job(
        ingest_era5, trigger="cron", hour=5, minute=0, args=[db, zarr],
        id="era5_daily", name="ERA5 daily ingest", replace_existing=True,
    )

    # OSM: weekly on Sunday at 02:00 UTC
    scheduler.add_job(
        ingest_osm_static, trigger="cron", day_of_week="sun", hour=2, minute=0, args=[db, zarr],
        id="osm_weekly", name="OSM weekly ingest", replace_existing=True,
    )

    # World Bank: monthly on the 1st at 03:00 UTC
    scheduler.add_job(
        ingest_world_bank, trigger="cron", day=1, hour=3, minute=0, args=[db, zarr],
        id="world_bank_monthly", name="World Bank monthly ingest", replace_existing=True,
    )

    scheduler.start()
    log.info("PSE scheduler running", jobs=[j.id for j in scheduler.get_jobs()])

    # Run immediately on startup
    await ingest_open_meteo(db, zarr)
    await ingest_osm_static(db, zarr)
    await ingest_world_bank(db, zarr)

    try:
        while True:
            await asyncio.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        await db.close()
        log.info("PSE scheduler stopped")


if __name__ == "__main__":
    asyncio.run(main())
