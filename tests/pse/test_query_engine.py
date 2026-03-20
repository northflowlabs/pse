"""
Tests for the PSE QueryEngine.
"""
from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pse.connectors.base import SpatialBounds, TemporalBounds
from pse.query.engine import QueryEngine
from pse.store.cache import PSECache

JAKARTA = SpatialBounds(min_lat=-6.3, max_lat=-6.1, min_lon=106.7, max_lon=106.9)
WEEK = TemporalBounds(start=datetime(2025, 1, 1), end=datetime(2025, 1, 7))


def _make_mock_connector(source_id: str, variables: list[str]) -> object:
    """Build a minimal mock connector."""
    from pse.connectors.base import BaseConnector, DataQuality

    class MockConnector(BaseConnector):
        @property
        def source_id(self): return source_id
        @property
        def variables(self): return variables
        @property
        def update_frequency_seconds(self): return 3600

        async def fetch(self, variables, spatial, temporal, resolution=None):
            times = pd.date_range(temporal.start, temporal.end, freq="1h")
            lats = np.linspace(spatial.min_lat, spatial.max_lat, 3)
            lons = np.linspace(spatial.min_lon, spatial.max_lon, 3)
            data = {
                var: (
                    ["time", "latitude", "longitude"],
                    np.random.rand(len(times), len(lats), len(lons)).astype(np.float32),
                )
                for var in variables
            }
            return xr.Dataset(data, coords={"time": times, "latitude": lats, "longitude": lons})

        async def get_quality(self, spatial, temporal):
            return DataQuality(0.99, 0.0, 11000, 0.88)

        async def get_latest_timestamp(self):
            return datetime.now(UTC)

    return MockConnector()


@pytest.fixture
def engine():
    c1 = _make_mock_connector("source_a", ["temperature_2m", "wind_speed_10m"])
    c2 = _make_mock_connector("source_b", ["solar_ghi", "solar_dni"])
    cache = PSECache(default_ttl=60.0)
    return QueryEngine(connectors={"source_a": c1, "source_b": c2}, cache=cache)


class TestQueryEngine:
    @pytest.mark.asyncio
    async def test_single_variable_fetch(self, engine):
        ds = await engine.query(
            variables=["temperature_2m"],
            spatial=JAKARTA,
            temporal=WEEK,
        )
        assert "temperature_2m" in ds.data_vars

    @pytest.mark.asyncio
    async def test_multi_source_fetch_merges(self, engine):
        ds = await engine.query(
            variables=["temperature_2m", "solar_ghi"],
            spatial=JAKARTA,
            temporal=WEEK,
        )
        assert "temperature_2m" in ds.data_vars
        assert "solar_ghi" in ds.data_vars

    @pytest.mark.asyncio
    async def test_unknown_variable_raises(self, engine):
        with pytest.raises(ValueError, match="No connector available"):
            await engine.query(
                variables=["made_up_variable"],
                spatial=JAKARTA,
                temporal=WEEK,
            )

    @pytest.mark.asyncio
    async def test_cache_hit_avoids_second_fetch(self, engine):
        connector = list(engine._connectors.values())[0]
        original_fetch = connector.fetch
        fetch_count = [0]

        async def counting_fetch(*args, **kwargs):
            fetch_count[0] += 1
            return await original_fetch(*args, **kwargs)

        connector.fetch = counting_fetch

        # First call — should fetch
        await engine.query(variables=["temperature_2m"], spatial=JAKARTA, temporal=WEEK)
        # Second identical call — should hit cache
        await engine.query(variables=["temperature_2m"], spatial=JAKARTA, temporal=WEEK)

        assert fetch_count[0] == 1, "Second identical query should have hit cache"

    @pytest.mark.asyncio
    async def test_point_query_drops_spatial_dims(self, engine):
        ds = await engine.point_query(
            lat=-6.2,
            lon=106.8,
            variables=["temperature_2m"],
            temporal=WEEK,
        )
        # After point extraction, latitude and longitude should be scalar coords, not dims
        assert "latitude" not in ds.dims
        assert "longitude" not in ds.dims

    def test_available_variables_lists_all(self, engine):
        avail = engine.available_variables()
        assert "temperature_2m" in avail
        assert "solar_ghi" in avail
        assert "source_a" in avail["temperature_2m"]
        assert "source_b" in avail["solar_ghi"]
