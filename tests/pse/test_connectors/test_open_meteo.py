"""
Tests for the Open-Meteo connector.

Tests marked with @pytest.mark.integration make real HTTP calls.
Run unit tests only: pytest -m "not integration"
"""
from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
import xarray as xr

from pse.connectors.base import SpatialBounds, TemporalBounds
from pse.connectors.open_meteo import OpenMeteoConnector

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

JAKARTA = SpatialBounds(min_lat=-6.3, max_lat=-6.1, min_lon=106.7, max_lon=106.9)
JAN_WEEK = TemporalBounds(start=datetime(2025, 1, 1), end=datetime(2025, 1, 7))


@pytest.fixture
def connector():
    return OpenMeteoConnector()


def _make_mock_response(lat: float, lon: float, n_hours: int = 168) -> dict:
    """Build a minimal Open-Meteo JSON response for mocking."""
    import pandas as pd
    times = pd.date_range("2025-01-01", periods=n_hours, freq="1h")
    return {
        "latitude": lat,
        "longitude": lon,
        "hourly": {
            "time": [t.isoformat() for t in times],
            "temperature_2m": (np.random.uniform(25, 32, n_hours)).tolist(),
            "wind_speed_10m": (np.random.uniform(1, 8, n_hours)).tolist(),
        },
    }


# ---------------------------------------------------------------------------
# Unit tests (no network)
# ---------------------------------------------------------------------------

class TestOpenMeteoUnit:
    def test_source_id(self, connector):
        assert connector.source_id == "open_meteo"

    def test_variables_list_not_empty(self, connector):
        assert len(connector.variables) > 0
        assert "temperature_2m" in connector.variables
        assert "wind_speed_10m" in connector.variables
        assert "solar_ghi" in connector.variables

    def test_update_frequency(self, connector):
        assert connector.update_frequency_seconds == 3600

    def test_validate_variables_valid(self, connector):
        result = connector._validate_variables(["temperature_2m", "wind_speed_10m"])
        assert result == ["temperature_2m", "wind_speed_10m"]

    def test_validate_variables_unknown_raises(self, connector):
        with pytest.raises(ValueError, match="unknown variable"):
            connector._validate_variables(["not_a_variable"])

    def test_build_grid_single_point(self, connector):
        bounds = SpatialBounds(-6.25, -6.15, 106.75, 106.85)
        lats, lons = connector._build_grid(bounds, resolution_m=11_000)
        assert len(lats) >= 1
        assert len(lons) >= 1

    def test_build_grid_coarse(self, connector):
        # A 1° × 1° box at 100 km resolution → at least 1 point
        bounds = SpatialBounds(-7.0, -6.0, 106.0, 107.0)
        lats, lons = connector._build_grid(bounds, resolution_m=100_000)
        assert len(lats) >= 1
        assert len(lons) >= 1

    @pytest.mark.asyncio
    async def test_fetch_returns_correct_structure(self, connector):
        """Test that fetch assembles a correct Dataset from mocked HTTP."""
        mock_json = _make_mock_response(-6.2, 106.8)

        with patch.object(connector, "_fetch_point", new=AsyncMock(return_value=mock_json)):
            ds = await connector.fetch(
                variables=["temperature_2m", "wind_speed_10m"],
                spatial=JAKARTA,
                temporal=JAN_WEEK,
            )

        assert isinstance(ds, xr.Dataset)
        assert "temperature_2m" in ds.data_vars
        assert "wind_speed_10m" in ds.data_vars
        assert "time" in ds.dims
        assert "latitude" in ds.dims
        assert "longitude" in ds.dims

    @pytest.mark.asyncio
    async def test_fetch_adds_provenance_attrs(self, connector):
        mock_json = _make_mock_response(-6.2, 106.8)

        with patch.object(connector, "_fetch_point", new=AsyncMock(return_value=mock_json)):
            ds = await connector.fetch(
                variables=["temperature_2m"],
                spatial=JAKARTA,
                temporal=JAN_WEEK,
            )

        assert ds.attrs.get("pse_source") == "open_meteo"
        assert "pse_fetched_at" in ds.attrs
        assert "pse_spatial" in ds.attrs
        assert "pse_temporal" in ds.attrs

    @pytest.mark.asyncio
    async def test_fetch_temperature_range_realistic(self, connector):
        """Jakarta temperatures should be 20–40°C."""
        mock_json = _make_mock_response(-6.2, 106.8)

        with patch.object(connector, "_fetch_point", new=AsyncMock(return_value=mock_json)):
            ds = await connector.fetch(
                variables=["temperature_2m"],
                spatial=JAKARTA,
                temporal=JAN_WEEK,
            )

        mean_temp = float(ds["temperature_2m"].mean())
        assert 20 < mean_temp < 40, f"Mean temperature {mean_temp:.1f}°C out of tropical range"

    @pytest.mark.asyncio
    async def test_quality_returns_valid_scores(self, connector):
        quality = await connector.get_quality(JAKARTA, JAN_WEEK)

        assert 0 <= quality.completeness <= 1
        assert 0 <= quality.source_reliability <= 1
        assert quality.spatial_resolution > 0

    @pytest.mark.asyncio
    async def test_get_latest_timestamp_is_recent(self, connector):
        ts = await connector.get_latest_timestamp()
        now = datetime.now(UTC)
        # Should be within the last 2 hours
        lag = (now - ts.replace(tzinfo=UTC)).total_seconds()
        assert lag < 7200, f"latest_timestamp lag {lag}s is too large"


# ---------------------------------------------------------------------------
# Integration tests (real network — skipped in CI unless flag is set)
# ---------------------------------------------------------------------------

class TestOpenMeteoIntegration:
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_real_fetch_jakarta_temperature(self):
        """
        Integration test: fetch real temperature data for Jakarta.
        Requires internet access.
        """
        connector = OpenMeteoConnector()
        ds = await connector.fetch(
            variables=["temperature_2m"],
            spatial=JAKARTA,
            temporal=JAN_WEEK,
        )

        assert "temperature_2m" in ds.data_vars
        assert ds["temperature_2m"].dims == ("time", "latitude", "longitude")

        mean_temp = float(ds["temperature_2m"].mean())
        assert 20 < mean_temp < 40, f"Mean temperature {mean_temp:.1f}°C — not tropical?"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_real_fetch_solar_data(self):
        """Integration test: fetch solar radiation data."""
        connector = OpenMeteoConnector()
        ds = await connector.fetch(
            variables=["solar_ghi"],
            spatial=JAKARTA,
            temporal=JAN_WEEK,
        )

        assert "solar_ghi" in ds.data_vars
        assert float(ds["solar_ghi"].max()) > 0, "Solar GHI should be positive during daytime"
