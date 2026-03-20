<div align="center">
  <img src="docs/images/northflow-logo.png" alt="Northflow Technologies" width="80">

  # Northflow PSE

  ### Planetary Sensing Engine

  **Domain-agnostic spatiotemporal data fusion for Earth observation, climate science, and planetary intelligence.**

  [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
  [![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://python.org)
  [![Tests](https://img.shields.io/github/actions/workflow/status/northflowlabs/pse/ci.yml?label=tests)](https://github.com/northflowlabs/pse/actions)
  [![Pangeo](https://img.shields.io/badge/ecosystem-Pangeo-blue)](https://pangeo.io)

  [Documentation](docs/) · [API Reference](docs/api-reference.md) · [Examples](examples/) · [Contributing](.github/CONTRIBUTING.md)

</div>

---

**Northflow Technologies** builds institutional scientific discovery infrastructure for climate, Earth observation, and critical systems. PSE is our open-source data foundation — the sensing and data fusion layer of the Northflow platform.

## What is PSE?

PSE (Planetary Sensing Engine) is an open-source Python framework that ingests data from multiple Earth observation and environmental data sources, normalises them into a unified spatiotemporal model, and makes them queryable through a single API.

**The problem:** Earth observation data is fragmented across dozens of providers, formats, coordinate systems, and temporal resolutions. Combining ERA5 reanalysis with Sentinel-2 imagery, OpenStreetMap infrastructure data, and World Bank statistics requires deep domain expertise and thousands of lines of boilerplate code.

**PSE solves this** by providing a unified `BaseConnector` interface for any data source, a quality-weighted fusion engine that intelligently combines overlapping sources, and a REST API that serves fused data as standard `xarray.Dataset` objects — fully compatible with the Pangeo ecosystem.

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

## Key Features

- **6 built-in connectors** — ERA5 (ECMWF), Open-Meteo, Global Solar Atlas, Sentinel-2 (Copernicus CDSE), OpenStreetMap, World Bank
- **Quality-weighted fusion** — When multiple sources provide the same variable, PSE blends them using quality scores (reliability × recency), not majority vote
- **Pangeo-native** — All data flows through `xarray.Dataset` with CF-conventions metadata. Works with Dask, Zarr, and the full Pangeo stack
- **Full provenance** — Every API response traces data back to its original source, transformation history, and quality assessment
- **Extensible** — Add new data sources by implementing the `BaseConnector` abstract class (~100 lines of Python)
- **REST API** — FastAPI-based endpoints for point queries, bounding-box queries, multi-source fusion, and data export
- **Cloud-ready** — Zarr storage on local disk or S3. PostgreSQL/PostGIS for spatial metadata. Docker Compose for single-command deployment

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
# With Docker (includes PostgreSQL + PostGIS)
docker compose up

# Or directly
uvicorn pse.api.main:app --host 0.0.0.0 --port 8001
```

### Query a point

```bash
curl "http://localhost:8001/api/v1/point?lat=-8.5&lon=115.2&variables=temperature_2m,solar_ghi"
```

## Data Sources

| Source | Variables | Resolution | Update Frequency | Key Required |
|--------|-----------|------------|-----------------|--------------|
| [ERA5](https://cds.climate.copernicus.eu/) | Temperature, wind, solar radiation, precipitation | 0.25° (~28 km) | Daily (5-day lag) | Yes (free) |
| [Open-Meteo](https://open-meteo.com/) | Temperature, wind, humidity, pressure, GHI | 0.25° | Hourly | No |
| [Global Solar Atlas](https://globalsolaratlas.info/) | GHI, DNI, DIF (long-term climatology) | ~1 km | Static | No |
| [Sentinel-2](https://dataspace.copernicus.eu/) | NDVI, NDWI, land-use classification | 10–60 m | 5-day revisit | Yes (free) |
| [OpenStreetMap](https://www.openstreetmap.org/) | Substations, power lines, roads, settlements | Vector | Weekly | No |
| [World Bank](https://data.worldbank.org/) | Electrification rate, GDP/capita, RE share | Country-level | Monthly | No |

## Writing a New Connector

PSE is designed to be extended. To add a new data source, implement the `BaseConnector` interface:

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
| `/api/v1/point` | GET | Point query — single lat/lon, multiple variables |
| `/api/v1/query` | GET | Bounding-box query — gridded data as JSON |
| `/api/v1/fuse` | POST | Multi-source quality-weighted fusion |
| `/api/v1/sources` | GET | List all available data sources |
| `/api/v1/sources/{id}/status` | GET | Detailed status for a single source |

## Part of the Northflow Platform

PSE is one of several engines in Northflow's planetary intelligence infrastructure. Each engine serves a different function:

| Engine | Function | Status |
|--------|----------|--------|
| **PSE** | Planetary Sensing Engine — ingests, fuses, and serves Earth observation data | Open source (this repo) |
| **HGE** | Hypothesis Generation Engine — generates and ranks scientific hypotheses from data | Core platform |

Domain adapters connect to whichever engines they need:

- **FLUX** — Renewable energy siting and yield prediction · *built on PSE*
- **CERES** — Humanitarian crisis and famine early warning · *built on HGE* · [arXiv:2603.09425](https://arxiv.org/abs/2603.09425)
- **MARVIS** — Maritime domain intelligence · *built on HGE*
- **ClimVal** — Climate model validation · [PyPI](https://pypi.org/project/climval/)

As the platform matures, adapters will draw from multiple engines simultaneously — PSE for data, HGE for hypotheses.

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

If you use PSE in academic work, please cite:

```bibtex
@software{northflow_pse_2026,
  author    = {Pedersen, Tom Danny},
  title     = {{Northflow PSE}: Planetary Sensing Engine},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/northflowlabs/pse},
  license   = {Apache-2.0}
}
```

## Contributing

We welcome contributions. See [CONTRIBUTING.md](.github/CONTRIBUTING.md) for guidelines.

Priority areas:
- New data source connectors (ocean, agricultural, air quality)
- Performance optimisation for large spatial extents
- Additional export formats (COG, STAC)
- Documentation improvements

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

---

<div align="center">

**[Northflow Technologies AS](https://northflow.no)** · Stavanger, Norway

*Building the intelligence infrastructure for Planet Earth.*

</div>
