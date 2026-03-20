# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-21

### Added

- Core `BaseConnector` abstract class with spatial/temporal bounds, quality metadata, and update-frequency contract
- 6 production data source connectors:
  - **ERA5** — ECMWF reanalysis via CDS API (temperature, wind u/v components, solar radiation, precipitation, surface pressure)
  - **Open-Meteo** — Free hourly weather forecast and historical data
  - **Global Solar Atlas** — Long-term GHI/DNI/DIF climatology at ~1 km resolution
  - **Sentinel-2** — CDSE-based optical imagery with NDVI, NDWI, and land-use classification
  - **OpenStreetMap** — Overpass API for power infrastructure, roads, and settlements
  - **World Bank** — Country-level socioeconomic indicators (electrification rate, GDP/capita, RE share)
- Quality-weighted fusion engine (`FusionEngine`) with:
  - Spatial regridding and temporal alignment
  - Quality-weighted merge across overlapping sources
  - Conflict detection with configurable thresholds
- Zarr-based array store with daily partitioning (local filesystem and S3/GCS compatible)
- PostgreSQL/PostGIS store for point observations, ingest records, and data source status
- SHA-256 keyed TTL+LRU cache layer
- FastAPI REST API with health, point query, bounding-box query, fusion, and source status endpoints
- JWT + API-key authentication middleware
- APScheduler-based data ingestion scheduler with configurable intervals
- `entrypoint.py` for Railway/cloud deployment (DATABASE_URL normalisation + PostGIS init)
- Docker Compose deployment stack (PSE API + PostgreSQL/PostGIS + scheduler)
- 100+ unit tests and integration tests against live APIs (ERA5/CDS and Open-Meteo)
- GitHub Actions CI across Python 3.11, 3.12, 3.13
- Apache 2.0 license
