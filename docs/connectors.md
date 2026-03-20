# Writing a New PSE Connector

PSE connectors are the adapters between raw data source APIs and the unified PSE spatiotemporal data model. This guide walks through implementing a new connector from scratch.

## The BaseConnector Interface

Every connector implements `pse.connectors.base.BaseConnector`:

```python
from abc import ABC, abstractmethod
from datetime import datetime
from pse.connectors.base import SpatialBounds, TemporalBounds, DataQuality
import xarray as xr

class BaseConnector(ABC):

    @property
    @abstractmethod
    def source_id(self) -> str:
        """Unique identifier, e.g. 'my_source'."""

    @property
    @abstractmethod
    def variables(self) -> list[str]:
        """PSE variable names this source can provide."""

    @property
    @abstractmethod
    def update_frequency_seconds(self) -> int:
        """How often the source updates. Used by the scheduler."""

    @abstractmethod
    async def fetch(
        self,
        variables: list[str],
        spatial: SpatialBounds,
        temporal: TemporalBounds,
        resolution: float | None = None,
    ) -> xr.Dataset:
        """Fetch data and return a CF-conventions xarray Dataset."""

    @abstractmethod
    async def get_quality(
        self, spatial: SpatialBounds, temporal: TemporalBounds
    ) -> DataQuality:
        """Return a quality assessment for this source over this domain."""

    @abstractmethod
    async def get_latest_timestamp(self) -> datetime:
        """Return the timestamp of the most recent available data."""
```

## Step-by-Step Implementation

### 1. Create the connector file

```bash
touch connectors/my_source.py
```

### 2. Implement the class

```python
from __future__ import annotations
import logging
from datetime import datetime, timezone, timedelta
import httpx
import xarray as xr
import numpy as np
import pandas as pd
from pse.connectors.base import (
    BaseConnector, ConnectorError,
    SpatialBounds, TemporalBounds, DataQuality,
)

log = logging.getLogger(__name__)

class MySourceConnector(BaseConnector):

    @property
    def source_id(self) -> str:
        return "my_source"

    @property
    def variables(self) -> list[str]:
        return ["my_variable_1", "my_variable_2"]

    @property
    def update_frequency_seconds(self) -> int:
        return 3600  # hourly

    async def fetch(self, variables, spatial, temporal, resolution=None) -> xr.Dataset:
        self._validate_variables(variables)
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(
                    "https://api.my-source.example/data",
                    params={
                        "lat": (spatial.min_lat + spatial.max_lat) / 2,
                        "lon": (spatial.min_lon + spatial.max_lon) / 2,
                        "start": temporal.start.isoformat(),
                        "end": temporal.end.isoformat(),
                    },
                )
                response.raise_for_status()
                data = response.json()
        except Exception as exc:
            raise ConnectorError(f"[my_source] fetch failed: {exc}") from exc

        # Build xarray Dataset with CF conventions
        times = pd.to_datetime(data["timestamps"])
        ds = xr.Dataset(
            {"my_variable_1": ("time", np.array(data["values"]))},
            coords={"time": times},
            attrs={
                "pse_source": self.source_id,
                "pse_fetched_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        return ds

    async def get_quality(self, spatial, temporal) -> DataQuality:
        return DataQuality(reliability=0.85, recency_hours=1.0)

    async def get_latest_timestamp(self) -> datetime:
        return datetime.now(timezone.utc) - timedelta(hours=1)
```

### 3. Write the tests

```python
# tests/pse/test_connectors/test_my_source.py
import pytest
import numpy as np
import xarray as xr
from unittest.mock import patch, AsyncMock
from datetime import datetime
from pse.connectors.base import SpatialBounds, TemporalBounds
from connectors.my_source import MySourceConnector

BOUNDS = SpatialBounds(-8.9, -8.1, 114.8, 115.7)
RANGE  = TemporalBounds(datetime(2024, 1, 1), datetime(2024, 1, 7))

class TestMySourceUnit:
    def setup_method(self):
        self.connector = MySourceConnector()

    def test_source_id(self):
        assert self.connector.source_id == "my_source"

    @pytest.mark.asyncio
    async def test_fetch_returns_dataset(self):
        mock_response = {"timestamps": ["2024-01-01T00:00:00"], "values": [25.0]}
        with patch("httpx.AsyncClient.get", return_value=AsyncMock(...)):
            # mock and assert
            pass
```

### 4. Register the connector

In `api/main.py`, add your connector to the `connectors` dict in the startup handler:

```python
from connectors.my_source import MySourceConnector

connectors = {
    ...
    "my_source": MySourceConnector(),
}
```

## Dataset Conventions

PSE uses CF (Climate and Forecast) conventions for all datasets:

| Dimension | Name | Units |
|-----------|------|-------|
| Time | `time` | `datetime64[ns]` with UTC timezone |
| Latitude | `latitude` | degrees_north |
| Longitude | `longitude` | degrees_east |

Required dataset attributes:
- `pse_source`: the connector's `source_id`
- `pse_fetched_at`: ISO-8601 timestamp of fetch

## Variable Naming

PSE uses standardised variable names across all connectors:

| PSE name | Description | Units |
|----------|-------------|-------|
| `temperature_2m` | Air temperature at 2 m | °C |
| `wind_speed_10m` | Wind speed at 10 m | m/s |
| `wind_u_10m` | Eastward wind component at 10 m | m/s |
| `wind_v_10m` | Northward wind component at 10 m | m/s |
| `solar_ghi` | Global horizontal irradiance | W/m² |
| `solar_dni` | Direct normal irradiance | W/m² |
| `solar_dhi` | Diffuse horizontal irradiance | W/m² |
| `precipitation` | Total precipitation | mm/h |
| `surface_pressure` | Surface pressure | hPa |
| `relative_humidity` | Relative humidity | % |
| `ndvi` | Normalised Difference Vegetation Index | dimensionless |
