"""
Tests for the PSECache store.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest
import xarray as xr

from pse.connectors.base import SpatialBounds, TemporalBounds
from pse.store.cache import PSECache


def _dummy_ds(value: float = 25.0) -> xr.Dataset:
    return xr.Dataset(
        {"temperature_2m": (["time"], np.array([value], dtype=np.float32))},
        coords={"time": [np.datetime64("2025-01-01")]},
    )


@pytest.fixture
def cache():
    return PSECache(default_ttl=60.0, max_entries=4)


class TestPSECache:
    def test_get_miss_returns_none(self, cache):
        assert cache.get("nonexistent_key") is None

    def test_put_then_get(self, cache):
        ds = _dummy_ds()
        cache.put("k1", ds)
        result = cache.get("k1")
        assert result is not None
        assert "temperature_2m" in result.data_vars

    def test_expired_entry_returns_none(self, cache):
        import time
        ds = _dummy_ds()
        cache.put("k_exp", ds, ttl=0.01)
        time.sleep(0.05)
        assert cache.get("k_exp") is None

    def test_lru_eviction(self):
        cache = PSECache(default_ttl=3600, max_entries=2)
        cache.put("k1", _dummy_ds(1.0))
        cache.put("k2", _dummy_ds(2.0))
        # Access k1 to make k2 the LRU
        cache.get("k1")
        # Add k3 — should evict k2
        cache.put("k3", _dummy_ds(3.0))
        assert cache.get("k2") is None  # evicted
        assert cache.get("k1") is not None
        assert cache.get("k3") is not None

    def test_invalidate_by_source(self, cache):
        cache.put("open_meteo:abc", _dummy_ds())
        cache.put("open_meteo:def", _dummy_ds())
        cache.put("era5:xyz", _dummy_ds())

        removed = cache.invalidate("open_meteo")
        assert removed == 2
        assert cache.get("open_meteo:abc") is None
        assert cache.get("era5:xyz") is not None

    def test_build_key_is_deterministic(self):
        spatial = SpatialBounds(-6.3, -6.1, 106.7, 106.9)
        temporal = TemporalBounds(datetime(2025, 1, 1), datetime(2025, 1, 7))
        key1 = PSECache.build_key("open_meteo", ["temperature_2m", "wind_speed_10m"], spatial, temporal)
        key2 = PSECache.build_key("open_meteo", ["wind_speed_10m", "temperature_2m"], spatial, temporal)
        # Variable order should not matter
        assert key1 == key2

    def test_build_key_differs_by_source(self):
        spatial = SpatialBounds(-6.3, -6.1, 106.7, 106.9)
        temporal = TemporalBounds(datetime(2025, 1, 1), datetime(2025, 1, 7))
        key_a = PSECache.build_key("open_meteo", ["temperature_2m"], spatial, temporal)
        key_b = PSECache.build_key("era5", ["temperature_2m"], spatial, temporal)
        assert key_a != key_b

    def test_stats_reflect_contents(self, cache):
        cache.put("k1", _dummy_ds())
        cache.put("k2", _dummy_ds())
        stats = cache.stats()
        assert stats["entries"] == 2
        assert stats["max_entries"] == 4

    def test_clear(self, cache):
        cache.put("k1", _dummy_ds())
        cache.put("k2", _dummy_ds())
        cache.clear()
        assert cache.size == 0
        assert cache.get("k1") is None
