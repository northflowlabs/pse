"""
PSE PostgreSQL / PostGIS store — metadata, point observations, and spatial index.

Used for:
  - Tracking ingest history (what was fetched, when, for which region)
  - Storing point-style observations (weather stations, IoT sensors)
  - Fast spatial queries (nearest substation, polygon intersection)
  - API key management and usage metering

Requires:
  - PostgreSQL 14+ with the PostGIS extension enabled
  - asyncpg (async driver), SQLAlchemy 2 (ORM), GeoAlchemy2 (spatial types)
"""
from __future__ import annotations

import logging
from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

try:
    from geoalchemy2 import Geometry
    _HAS_GEOALCHEMY = True
except ImportError:
    _HAS_GEOALCHEMY = False
    Geometry = None  # type: ignore

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ORM Base + Models
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


class IngestRecord(Base):
    """Tracks every successful data fetch from a connector."""
    __tablename__ = "pse_ingest_records"

    id = Column(Integer, primary_key=True)
    source_id = Column(String(64), nullable=False, index=True)
    variables = Column(JSONB, nullable=False)    # list[str]
    min_lat = Column(Float, nullable=False)
    max_lat = Column(Float, nullable=False)
    min_lon = Column(Float, nullable=False)
    max_lon = Column(Float, nullable=False)
    time_start = Column(DateTime, nullable=False)
    time_end = Column(DateTime, nullable=False)
    fetched_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    zarr_path = Column(String(512), nullable=True)
    quality_score = Column(Float, nullable=True)
    provenance = Column(JSONB, nullable=True)

    __table_args__ = (
        Index("ix_ingest_source_fetched", "source_id", "fetched_at"),
    )


class PointObservation(Base):
    """
    A single scalar observation at a geographic point.

    Used for weather station readings, IoT sensor data, or any non-gridded
    point measurement.
    """
    __tablename__ = "pse_point_observations"

    id = Column(Integer, primary_key=True)
    source_id = Column(String(64), nullable=False, index=True)
    variable = Column(String(128), nullable=False)
    value = Column(Float, nullable=False)
    unit = Column(String(32), nullable=True)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    observed_at = Column(DateTime, nullable=False, index=True)
    metadata_ = Column("metadata", JSONB, nullable=True)

    # Optional PostGIS geometry column for fast spatial indexing
    if _HAS_GEOALCHEMY:
        geom = Column(
            Geometry(geometry_type="POINT", srid=4326),
            nullable=True,
        )

    __table_args__ = (
        Index("ix_obs_source_var_time", "source_id", "variable", "observed_at"),
    )


class DataSourceStatus(Base):
    """Latest known status / freshness for each data source connector."""
    __tablename__ = "pse_source_status"

    source_id = Column(String(64), primary_key=True)
    last_successful_fetch = Column(DateTime, nullable=True)
    last_error = Column(Text, nullable=True)
    last_error_at = Column(DateTime, nullable=True)
    latest_data_timestamp = Column(DateTime, nullable=True)
    quality_score = Column(Float, nullable=True)
    extra = Column(JSONB, nullable=True)


# ---------------------------------------------------------------------------
# Database session factory
# ---------------------------------------------------------------------------

class PSEDatabase:
    """
    Async database layer for PSE.

    Usage::

        db = PSEDatabase(settings.database_url)
        await db.init()

        async with db.session() as session:
            await db.record_ingest(session, ...)
    """

    def __init__(self, database_url: str):
        # asyncpg does not accept sslmode as a URL query parameter.
        # Strip it and pass SSL via connect_args instead.
        connect_args: dict = {}
        if "sslmode=require" in database_url or "ssl=true" in database_url.lower():
            connect_args["ssl"] = "require"
            database_url = (
                database_url
                .replace("?sslmode=require", "")
                .replace("&sslmode=require", "")
                .replace("?ssl=true", "")
                .replace("&ssl=true", "")
            )

        self._engine = create_async_engine(
            database_url,
            echo=False,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
            connect_args=connect_args,
        )
        self._session_factory = async_sessionmaker(
            self._engine, expire_on_commit=False
        )

    async def init(self) -> None:
        """Create all tables (idempotent — safe to call on every startup)."""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        log.info("PSE database initialised")

    def session(self) -> AsyncSession:
        """Return an async session context manager."""
        return self._session_factory()

    async def close(self) -> None:
        await self._engine.dispose()

    # ------------------------------------------------------------------
    # Ingest tracking
    # ------------------------------------------------------------------

    async def record_ingest(
        self,
        session: AsyncSession,
        source_id: str,
        variables: list[str],
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
        time_start: datetime,
        time_end: datetime,
        zarr_path: str | None = None,
        quality_score: float | None = None,
        provenance: dict | None = None,
    ) -> IngestRecord:
        record = IngestRecord(
            source_id=source_id,
            variables=variables,
            min_lat=min_lat,
            max_lat=max_lat,
            min_lon=min_lon,
            max_lon=max_lon,
            time_start=time_start,
            time_end=time_end,
            fetched_at=datetime.utcnow(),
            zarr_path=zarr_path,
            quality_score=quality_score,
            provenance=provenance,
        )
        session.add(record)
        await session.commit()
        await session.refresh(record)
        return record

    async def get_latest_ingest(
        self,
        session: AsyncSession,
        source_id: str,
    ) -> IngestRecord | None:
        stmt = (
            select(IngestRecord)
            .where(IngestRecord.source_id == source_id)
            .order_by(IngestRecord.fetched_at.desc())
            .limit(1)
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    # ------------------------------------------------------------------
    # Source status
    # ------------------------------------------------------------------

    async def update_source_status(
        self,
        session: AsyncSession,
        source_id: str,
        *,
        last_successful_fetch: datetime | None = None,
        last_error: str | None = None,
        latest_data_timestamp: datetime | None = None,
        quality_score: float | None = None,
    ) -> None:
        stmt = text("""
            INSERT INTO pse_source_status
                (source_id, last_successful_fetch, last_error, last_error_at,
                 latest_data_timestamp, quality_score)
            VALUES
                (:sid, :lsf, :lerr, :lerr_at, :ldt, :qs)
            ON CONFLICT (source_id) DO UPDATE SET
                last_successful_fetch = EXCLUDED.last_successful_fetch,
                last_error            = EXCLUDED.last_error,
                last_error_at         = EXCLUDED.last_error_at,
                latest_data_timestamp = EXCLUDED.latest_data_timestamp,
                quality_score         = EXCLUDED.quality_score
        """)
        await session.execute(stmt, {
            "sid": source_id,
            "lsf": last_successful_fetch,
            "lerr": last_error,
            "lerr_at": datetime.utcnow() if last_error else None,
            "ldt": latest_data_timestamp,
            "qs": quality_score,
        })
        await session.commit()

    async def get_all_source_statuses(
        self,
        session: AsyncSession,
    ) -> list[DataSourceStatus]:
        result = await session.execute(select(DataSourceStatus))
        return list(result.scalars().all())

    # ------------------------------------------------------------------
    # Point observations
    # ------------------------------------------------------------------

    async def insert_observation(
        self,
        session: AsyncSession,
        source_id: str,
        variable: str,
        value: float,
        lat: float,
        lon: float,
        observed_at: datetime,
        unit: str | None = None,
        metadata: dict | None = None,
    ) -> PointObservation:
        obs = PointObservation(
            source_id=source_id,
            variable=variable,
            value=value,
            unit=unit,
            lat=lat,
            lon=lon,
            observed_at=observed_at,
            metadata_=metadata,
        )
        if _HAS_GEOALCHEMY:
            obs.geom = f"SRID=4326;POINT({lon} {lat})"
        session.add(obs)
        await session.commit()
        await session.refresh(obs)
        return obs

    async def get_nearest_observations(
        self,
        session: AsyncSession,
        lat: float,
        lon: float,
        variable: str,
        radius_km: float = 50.0,
        limit: int = 10,
    ) -> list[PointObservation]:
        """Find the closest point observations within *radius_km* kilometres."""
        if _HAS_GEOALCHEMY:
            stmt = text("""
                SELECT * FROM pse_point_observations
                WHERE variable = :variable
                  AND ST_DWithin(
                        geom::geography,
                        ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography,
                        :radius_m
                      )
                ORDER BY geom::geography <-> ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography
                LIMIT :lim
            """)
            result = await session.execute(stmt, {
                "variable": variable,
                "lat": lat,
                "lon": lon,
                "radius_m": radius_km * 1000,
                "lim": limit,
            })
            return list(result.mappings().all())
        else:
            # Fallback: approximate great-circle filter using plain SQL
            stmt = (
                select(PointObservation)
                .where(PointObservation.variable == variable)
                .where(PointObservation.lat.between(lat - radius_km / 111, lat + radius_km / 111))
                .where(PointObservation.lon.between(lon - radius_km / 111, lon + radius_km / 111))
                .limit(limit)
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def ping(self) -> bool:
        """Return True if the database is reachable."""
        try:
            async with self._engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False
