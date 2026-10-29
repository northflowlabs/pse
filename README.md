<a href="https://northflow.no"><img src=".github/assets/northflow-mark.svg" alt="Northflow" width="40" height="40"></a>

# PSE

PSE (Planetary Sensing Engine) is an open source Python framework that fuses Earth observation and environmental data into one queryable API.

[Research page](https://research.northflow.no/systems/pse) · [Hosted API](https://pse.northflow.no/docs) · [Docs](docs/) · [API reference](docs/api-reference.md) · [Examples](examples/) · [Licence](#licence)

[![Tests](https://img.shields.io/github/actions/workflow/status/northflowlabs/pse/ci.yml?label=tests)](https://github.com/northflowlabs/pse/actions/workflows/ci.yml)
[![Licence: Apache 2.0](https://img.shields.io/badge/licence-Apache%202.0-0e6b63)](LICENSE)

## What PSE does

Earth observation data is spread across providers, formats, coordinate systems and time resolutions. Combining ERA5 reanalysis with Sentinel-2 imagery, OpenStreetMap infrastructure and World Bank statistics usually takes domain expertise and a lot of glue code.

PSE ingests these sources through one `BaseConnector` interface, normalises them into a single spatiotemporal model, blends overlapping sources by quality score, and serves the result through a REST API as standard `xarray.Dataset` objects that work with the Pangeo ecosystem.

```python
from pse.query.engine import QueryEngine
from pse.connectors.base import SpatialBounds, TemporalBounds
from datetime import datetime

engine = QueryEngine()

# Query solar irradiance and temperature for Bali, Indonesia
ds = await engine.query(
    variables=["solar_ghi", "temperature_2m", "wind_speed_10m"],
    spatial=SpatialBounds(min_lat=-8.8, max_lat=-8.0, min_lon=114.4, max_lon=115.7),
    temporal=TemporalBounds(start=datetime(2024, 1, 1), end=datetime(2024, 12, 31)),
)

print(ds)
# <xarray.Dataset>
# Dimensions:  (time: 365, latitude: 8, longitude: 13)
# Variables:   solar_ghi, temperature_2m, wind_speed_10m
# Attributes:  pse_sources: ['era5', 'open_meteo', 'global_solar_atlas']
```

## Features

- **Six connectors:** ERA5 (ECMWF), Open-Meteo, Global Solar Atlas, Sentinel-2 (Copernicus CDSE), OpenStreetMap and World Bank
- **Quality-weighted fusion:** where several sources provide the same variable, PSE blends them by quality score (reliability and recency)
- **Pangeo compatible:** all data flows through `xarray.Dataset` with CF-convention metadata and works with Dask and Zarr
- **Provenance:** every API response traces data back to its source, its transformations and its quality assessment
- **Extensible:** a new data source is one `BaseConnector` subclass, about 100 lines of Python
- **REST API:** FastAPI endpoints for point queries, bounding-box queries, multi-source fusion and export
- **Storage:** Zarr on local disk or S3, PostgreSQL and PostGIS for spatial metadata

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      PSE API  (:8001)                        │
│            FastAPI · QueryEngine · FusionEngine              │
├─────────────────────────────────────────────────────────────┤
│                       Connectors                             │
│  ERA5 · Open-Meteo · Solar Atlas · Sentinel-2 · OSM · WB    │
├─────────────────────────────────────────────────────────────┤
│               Fusion · Quality · Alignment                   │
│      Spatial regridding · Temporal alignment · QC merge      │
├─────────────────────────────────────────────────────────────┤
│                        Storage                               │
│          Zarr (array data) · PostgreSQL / PostGIS            │
│                  SHA-256 TTL+LRU cache                       │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Clone and install

```bash
git clone https://github.com/northflowlabs/pse.git
cd pse
pip install -r requirements.txt
```

### Configure credentials

```bash
cp .env.example .env
# Add your ERA5_CDS_API_KEY (free registration at cds.climate.copernicus.eu)
```

### Run the API

```bash
# Directly
uvicorn pse.api.main:app --host 0.0.0.0 --port 8001

# Or with Docker
docker build -t pse .
docker run -p 8001:8000 --env-file .env pse
```

### Query a point

```bash
curl "http://localhost:8001/api/v1/point?lat=-8.5&lon=115.2&variables=temperature_2m,solar_ghi&time_start=2025-01-01T00:00:00Z&time_end=2025-01-07T00:00:00Z"
```

## Data Sources

| Source | Variables | Resolution | Update Frequency | Key Required |
|--------|-----------|------------|-----------------|--------------|
| [ERA5](https://cds.climate.copernicus.eu/) | Temperature, wind, solar radiation, precipitation | 0.25° (~28 km) | Daily (5-day lag) | Yes (free) |
| [Open-Meteo](https://open-meteo.com/) | Temperature, wind, humidity, pressure, GHI | 0.25° | Hourly | No |
| [Global Solar Atlas](https://globalsolaratlas.info/) | GHI, DNI, DIF (long-term climatology) | ~1 km | Static | No |
| [Sentinel-2](https://dataspace.copernicus.eu/) | NDVI, NDWI, land-use classification | 10 to 60 m | 5-day revisit | Yes (free) |
| [OpenStreetMap](https://www.openstreetmap.org/) | Substations, power lines, roads, settlements | Vector | Weekly | No |
| [World Bank](https://data.worldbank.org/) | Electrification rate, GDP/capita, RE share | Country-level | Monthly | No |

## Writing a New Connector

To add a new data source, implement the `BaseConnector` interface:

```python
from pse.connectors.base import BaseConnector, SpatialBounds, TemporalBounds, DataQuality
import xarray as xr

class MyConnector(BaseConnector):
    @property
    def source_id(self) -> str:
        return "my_source"

    @property
    def variables(self) -> list[str]:
        return ["my_variable"]

    @property
    def update_frequency_seconds(self) -> int:
        return 86400  # daily

    async def fetch(self, variables, spatial, temporal, resolution=None) -> xr.Dataset:
        # Your data retrieval logic here
        ...

    async def get_quality(self, spatial, temporal) -> DataQuality:
        ...

    async def get_latest_timestamp(self):
        ...
```

See [docs/connectors.md](docs/connectors.md) for the full guide.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | System health, connector status, data freshness |
| `/api/v1/point` | GET | Point query: one lat/lon, several variables |
| `/api/v1/query` | GET | Bounding-box query: gridded data as JSON |
| `/api/v1/fuse` | POST | Multi-source quality-weighted fusion |
| `/api/v1/sources` | GET | List all available data sources |
| `/api/v1/sources/{id}/status` | GET | Detailed status for a single source |

## Related

- [climval](https://github.com/northflowlabs/climval) validates climate model output against reference data. `pip install climval`
- [Northflow Space and Research](https://research.northflow.no) describes the systems PSE belongs to.

## Testing

```bash
# Unit tests (no network, no API keys needed)
pytest -m "not integration"

# Integration tests (requires API keys in .env)
pytest -m integration

# Full suite with coverage
pytest --cov=pse --cov-report=html
```

## Citation

Pedersen, Tom Danny S. (2026). Northflow PSE: Planetary Sensing Engine.
GitHub. https://github.com/northflowlabs/pse. See [CITATION.cff](CITATION.cff).

## Contributing

We welcome contributions. See [CONTRIBUTING.md](.github/CONTRIBUTING.md).

Priority areas:
- New data source connectors (ocean, agriculture, air quality)
- Performance for large spatial extents
- More export formats (COG, STAC)
- Documentation

## Licence

PSE is licensed under the [Apache License 2.0](LICENSE).

Copyright 2026 Northflow Technologies AS
