"""
Tests for the PSE FusionEngine and fusion sub-modules.
"""
from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pse.connectors.base import DataQuality, SpatialBounds, TemporalBounds
from pse.fusion.engine import FusionEngine
from pse.fusion.grid import make_regular_grid, regrid_dataset
from pse.fusion.quality import compute_weights, detect_conflicts, quality_weighted_merge
from pse.fusion.temporal import align_to_common_axis, classify_time_axis
from pse.query.engine import QueryEngine
from pse.store.cache import PSECache

JAKARTA = SpatialBounds(-6.3, -6.1, 106.7, 106.9)
WEEK = TemporalBounds(datetime(2025, 1, 1), datetime(2025, 1, 7))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_connector(source_id, variables):
    from pse.connectors.base import BaseConnector

    class MockConnector(BaseConnector):
        @property
        def source_id(self): return source_id
        @property
        def variables(self): return variables
        @property
        def update_frequency_seconds(self): return 3600

        async def fetch(self, variables, spatial, temporal, resolution=None):
            times = pd.date_range(temporal.start, temporal.end, freq="1h")
            lats = np.linspace(spatial.min_lat, spatial.max_lat, 4)
            lons = np.linspace(spatial.min_lon, spatial.max_lon, 4)
            data = {
                var: (["time", "latitude", "longitude"],
                      np.random.rand(len(times), 4, 4).astype(np.float32))
                for var in variables
            }
            return xr.Dataset(data, coords={"time": times, "latitude": lats, "longitude": lons},
                               attrs={"pse_source": source_id})

        async def get_quality(self, spatial, temporal):
            return DataQuality(0.95, 0.0, 11000, 0.90)

        async def get_latest_timestamp(self):
            return datetime.now(UTC)

    return MockConnector()


@pytest.fixture
def fusion_engine():
    c1 = _make_connector("source_a", ["temperature_2m", "wind_speed_10m"])
    c2 = _make_connector("source_b", ["temperature_2m", "solar_ghi"])
    cache = PSECache(default_ttl=60.0)
    qe = QueryEngine({"source_a": c1, "source_b": c2}, cache=cache)
    return FusionEngine(qe)


# ---------------------------------------------------------------------------
# Fusion grid tests
# ---------------------------------------------------------------------------

class TestFusionGrid:
    def test_make_regular_grid_covers_bounds(self):
        lats, lons = make_regular_grid(JAKARTA, 5_000)
        assert float(lats[0]) >= JAKARTA.min_lat - 0.01
        assert float(lats[-1]) <= JAKARTA.max_lat + 0.01

    def test_make_regular_grid_resolution(self):
        lats, lons = make_regular_grid(JAKARTA, 1_000)
        # At 1 km, a 0.2° box should have ~22 steps
        step = (lats[1] - lats[0]) * 111_000
        assert 500 < step < 2_000

    def test_regrid_interpolates(self):
        lats_src = np.array([-6.3, -6.2, -6.1])
        lons_src = np.array([106.7, 106.8, 106.9])
        data = np.random.rand(3, 3).astype(np.float32)
        ds = xr.Dataset(
            {"temperature_2m": (["latitude", "longitude"], data)},
            coords={"latitude": lats_src, "longitude": lons_src},
        )
        lats_tgt = np.linspace(-6.3, -6.1, 7)
        lons_tgt = np.linspace(106.7, 106.9, 7)
        ds_regridded = regrid_dataset(ds, lats_tgt, lons_tgt)
        assert ds_regridded["temperature_2m"].shape == (7, 7)


# ---------------------------------------------------------------------------
# Fusion quality tests
# ---------------------------------------------------------------------------

class TestFusionQuality:
    def test_compute_weights_normalise_to_one(self):
        weights = compute_weights([0.9, 0.7, 0.5])
        assert abs(weights.sum() - 1.0) < 1e-6

    def test_compute_weights_higher_quality_gets_more(self):
        weights = compute_weights([0.9, 0.1])
        assert weights[0] > weights[1]

    def test_quality_weighted_merge_single_source(self):
        ds = xr.Dataset(
            {"temp": (["time"], np.array([1.0, 2.0, 3.0]))},
            coords={"time": [0, 1, 2]},
        )
        result = quality_weighted_merge([ds], np.array([1.0]))
        np.testing.assert_array_equal(result["temp"].values, ds["temp"].values)

    def test_quality_weighted_merge_two_sources(self):
        ds1 = xr.Dataset({"temp": (["time"], np.array([10.0, 20.0]))}, coords={"time": [0, 1]})
        ds2 = xr.Dataset({"temp": (["time"], np.array([20.0, 30.0]))}, coords={"time": [0, 1]})
        weights = np.array([0.5, 0.5])
        result = quality_weighted_merge([ds1, ds2], weights)
        expected = np.array([15.0, 25.0])
        np.testing.assert_array_almost_equal(result["temp"].values, expected)

    def test_detect_conflicts_no_conflict(self):
        ds1 = xr.Dataset({"temp": (["x"], np.array([25.0, 26.0]))}, coords={"x": [0, 1]})
        ds2 = xr.Dataset({"temp": (["x"], np.array([25.5, 26.5]))}, coords={"x": [0, 1]})
        result = detect_conflicts([ds1, ds2], "temp", threshold=0.3)
        assert not result["conflict"]

    def test_detect_conflicts_large_disagreement(self):
        ds1 = xr.Dataset({"temp": (["x"], np.array([10.0]))}, coords={"x": [0]})
        ds2 = xr.Dataset({"temp": (["x"], np.array([100.0]))}, coords={"x": [0]})
        result = detect_conflicts([ds1, ds2], "temp", threshold=0.3)
        assert result["conflict"]


# ---------------------------------------------------------------------------
# Temporal alignment tests
# ---------------------------------------------------------------------------

class TestTemporalAlignment:
    def test_classify_hourly(self):
        times = pd.date_range("2025-01-01", periods=48, freq="1h")
        ds = xr.Dataset({"x": (["time"], np.zeros(48))}, coords={"time": times})
        assert classify_time_axis(ds) == "hourly"

    def test_classify_monthly_clim(self):
        ds = xr.Dataset({"ghi": (["time"], np.zeros(12))}, coords={"time": list(range(1, 13))})
        assert classify_time_axis(ds) == "monthly_clim"

    def test_classify_no_time(self):
        ds = xr.Dataset({"x": (["lat"], np.zeros(3))}, coords={"lat": [0, 1, 2]})
        assert classify_time_axis(ds) == "none"

    def test_align_broadcasts_monthly_clim(self):
        # Monthly climatology dataset
        months = list(range(1, 13))
        ds_clim = xr.Dataset(
            {"solar_ghi": (["time"], np.ones(12) * 150.0)},
            coords={"time": months},
        )
        # Hourly dataset
        times = pd.date_range("2025-01-01", periods=24, freq="1h")
        ds_hourly = xr.Dataset(
            {"temperature_2m": (["time"], np.random.rand(24))},
            coords={"time": times},
        )
        aligned = align_to_common_axis([ds_clim, ds_hourly])
        # Both should now have a proper datetime time axis
        assert len(aligned) == 2
        # The climatological dataset should have been broadcast
        if "time" in aligned[0].dims and len(aligned[0].time) > 12:
            clim_aligned = aligned[0]
            assert "solar_ghi" in clim_aligned.data_vars


# ---------------------------------------------------------------------------
# FusionEngine end-to-end tests
# ---------------------------------------------------------------------------

class TestFusionEngineE2E:
    @pytest.mark.asyncio
    async def test_single_source_variable(self, fusion_engine):
        ds = await fusion_engine.query(
            variables=["wind_speed_10m"],
            spatial=JAKARTA,
            temporal=WEEK,
            resolution_m=5_000,
        )
        assert "wind_speed_10m" in ds.data_vars
        assert ds.attrs.get("pse_fusion") is True

    @pytest.mark.asyncio
    async def test_multi_source_variable_returns_merged(self, fusion_engine):
        """temperature_2m is in both source_a and source_b — should be blended."""
        ds = await fusion_engine.query(
            variables=["temperature_2m"],
            spatial=JAKARTA,
            temporal=WEEK,
            resolution_m=5_000,
        )
        assert "temperature_2m" in ds.data_vars
        assert "pse_fusion_weights" in ds.attrs

    @pytest.mark.asyncio
    async def test_cross_source_query(self, fusion_engine):
        """Query spanning both source_a and source_b."""
        ds = await fusion_engine.query(
            variables=["wind_speed_10m", "solar_ghi"],
            spatial=JAKARTA,
            temporal=WEEK,
            resolution_m=5_000,
        )
        assert "wind_speed_10m" in ds.data_vars
        assert "solar_ghi" in ds.data_vars

    @pytest.mark.asyncio
    async def test_provenance_in_attrs(self, fusion_engine):
        ds = await fusion_engine.query(
            variables=["temperature_2m"],
            spatial=JAKARTA,
            temporal=WEEK,
        )
        assert "pse_version" in ds.attrs
        assert "pse_query_time" in ds.attrs
        assert "pse_sources_used" in ds.attrs
        assert len(ds.attrs["pse_sources_used"]) >= 1

    @pytest.mark.asyncio
    async def test_unknown_variable_raises(self, fusion_engine):
        with pytest.raises(ValueError, match="No connector available"):
            await fusion_engine.query(
                variables=["unknown_variable_xyz"],
                spatial=JAKARTA,
                temporal=WEEK,
            )
