# PSE Architecture

## Overview

PSE is structured as a layered service. Each layer has a single responsibility and communicates through well-defined interfaces.

```
┌─────────────────────────────────────────────────────────────┐
│                        REST API                             │
│         FastAPI  ·  JWT/API-key auth  ·  CORS               │
│  /health  /point  /query  /fuse  /sources                   │
├─────────────────────────────────────────────────────────────┤
│                     Query Engine                            │
│    Connector selection  ·  Parallel fetch  ·  Merge         │
├─────────────────────────────────────────────────────────────┤
│                    Fusion Engine                            │
│   Quality-weighted merge  ·  Spatial regrid  ·  Temporal    │
│   alignment  ·  Conflict detection                          │
├─────────────────────────────────────────────────────────────┤
│                      Connectors                             │
│  ERA5  ·  Open-Meteo  ·  Solar Atlas  ·  Sentinel-2         │
│  OpenStreetMap  ·  World Bank  ·  [extensible]              │
├─────────────────────────────────────────────────────────────┤
│                       Storage                               │
│  Zarr store (array data)  ·  PostgreSQL/PostGIS (metadata)  │
│  SHA-256 TTL+LRU cache                                      │
└─────────────────────────────────────────────────────────────┘
```

## Connectors

Each data source connector implements `BaseConnector`:

- **`fetch(variables, spatial, temporal)`** — returns an `xr.Dataset` in PSE variable naming, CF-convention coordinates
- **`get_quality(spatial, temporal)`** — returns `DataQuality(reliability, recency_hours)`
- **`get_latest_timestamp()`** — informs the scheduler when new data is available

Connectors are stateless and independently testable. Network calls are async and run in the FastAPI event loop.

## Fusion Engine

When multiple connectors provide the same variable over the same domain, the `FusionEngine`:

1. **Spatial alignment** — resamples all datasets onto a common grid
2. **Temporal alignment** — reindexes to a common time axis
3. **Quality weighting** — weights each source's contribution by `quality.reliability / quality.recency_hours`
4. **Conflict detection** — flags sources that disagree by more than a configurable threshold

The output is a single `xr.Dataset` with `pse_sources` and `pse_quality_weights` attributes.

## Storage

### Zarr Store

Array data is persisted in Zarr format, partitioned by `(source, variable, date)`. Zarr natively supports:
- Chunked, compressed array storage
- Lazy loading via Dask
- Direct mounting on S3/GCS via `fsspec`

### PostgreSQL / PostGIS

Relational metadata is stored in PostgreSQL with PostGIS for spatial indexing:
- `ingest_records` — log of every connector fetch
- `point_observations` — individual sensor observations
- `data_source_status` — per-connector health and freshness metrics

### Cache

A two-tier SHA-256 keyed cache sits in front of both stores:
- **L1** — in-process LRU cache (configurable max entries)
- **L2** — TTL-based invalidation (default 1 hour for forecast data, 24 hours for reanalysis)

## Scheduler

The `APScheduler`-based ingest scheduler runs as a separate process (`python -m pse.scheduler.ingest`). It:
1. Iterates over all registered connectors
2. Fetches the last N days of data for configured spatial regions
3. Writes to Zarr and PostgreSQL
4. Updates `DataSourceStatus` with freshness metadata
