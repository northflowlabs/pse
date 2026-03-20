"""
Tests for the Global Solar Atlas connector.
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
import xarray as xr

from pse.connectors.base import SpatialBounds, TemporalBounds
from pse.connectors.global_solar_atlas import GlobalSolarAtlasConnector

BALI = SpatialBounds(min_lat=-8.9, max_lat=-8.1, min_lon=114.8, max_lon=115.7)
YEAR_2025 = TemporalBounds(start=datetime(2025, 1, 1), end=datetime(2025, 12, 31))


@pytest.fixture
def connector():
    return GlobalSolarAtlasConnector()


def _make_mock_gsa_response(lat: float, lon: float) -> dict:
    """Simulate a GSA API JSON response."""
    monthly_ghi = [110, 120, 150, 160, 175, 165, 155, 160, 150, 140, 115, 105]
    annual_ghi = sum(monthly_ghi)
    return {
        "annual": {
            "GHI": annual_ghi,
            "DNI": 1543.1,
            "GTI": 1620.0,
            "DIF": 430.2,
            "PVOUT": 1540.0,
            "TEMP": 26.5,
            "WS": 3.2,
        },
        "monthly": {
            "GHI": monthly_ghi,
            "DNI": [100, 110, 140, 150, 165, 155, 145, 150, 140, 130, 105, 95],
            "GTI": [115, 125, 158, 170, 183, 173, 163, 168, 158, 148, 120, 110],
        },
    }


class TestGlobalSolarAtlasUnit:
    def test_source_id(self, connector):
        assert connector.source_id == "global_solar_atlas"

    def test_variables_include_solar(self, connector):
        assert "solar_ghi" in connector.variables
        assert "solar_dni" in connector.variables
        assert "solar_gti" in connector.variables
        assert "solar_pvout" in connector.variables

    def test_update_frequency(self, connector):
        assert connector.update_frequency_seconds == 86_400

    def test_validate_unknown_variable_raises(self, connector):
        with pytest.raises(ValueError):
            connector._validate_variables(["nonexistent_variable"])

    @pytest.mark.asyncio
    async def test_fetch_returns_12_monthly_steps(self, connector):
        mock_json = _make_mock_gsa_response(-8.5, 115.2)

        with patch.object(connector, "_fetch_point", new=AsyncMock(return_value=mock_json)):
            ds = await connector.fetch(
                variables=["solar_ghi", "solar_dni"],
                spatial=BALI,
                temporal=YEAR_2025,
            )

        assert isinstance(ds, xr.Dataset)
        # time dimension should have 12 monthly steps
        assert len(ds.coords["time"]) == 12
        assert "solar_ghi" in ds.data_vars
        assert "solar_dni" in ds.data_vars

    @pytest.mark.asyncio
    async def test_fetch_annual_variable_present(self, connector):
        mock_json = _make_mock_gsa_response(-8.5, 115.2)

        with patch.object(connector, "_fetch_point", new=AsyncMock(return_value=mock_json)):
            ds = await connector.fetch(
                variables=["solar_ghi"],
                spatial=BALI,
                temporal=YEAR_2025,
            )

        # Annual total should be a 2-D variable
        assert "solar_ghi_annual" in ds.data_vars
        assert "time" not in ds["solar_ghi_annual"].dims

    @pytest.mark.asyncio
    async def test_fetch_provenance_attrs(self, connector):
        mock_json = _make_mock_gsa_response(-8.5, 115.2)

        with patch.object(connector, "_fetch_point", new=AsyncMock(return_value=mock_json)):
            ds = await connector.fetch(
                variables=["solar_ghi"],
                spatial=BALI,
                temporal=YEAR_2025,
            )

        assert ds.attrs.get("pse_source") == "global_solar_atlas"

    @pytest.mark.asyncio
    async def test_quality_is_valid(self, connector):
        q = await connector.get_quality(BALI, YEAR_2025)
        assert 0 <= q.completeness <= 1
        assert 0 <= q.source_reliability <= 1
        assert q.spatial_resolution > 0

    @pytest.mark.asyncio
    async def test_ghi_values_reasonable_for_tropics(self, connector):
        """Bali monthly GHI should be in range 100–200 kWh/m²/month."""
        mock_json = _make_mock_gsa_response(-8.5, 115.2)

        with patch.object(connector, "_fetch_point", new=AsyncMock(return_value=mock_json)):
            ds = await connector.fetch(
                variables=["solar_ghi"],
                spatial=BALI,
                temporal=YEAR_2025,
            )

        ghi_values = ds["solar_ghi"].values.flatten()
        valid = ghi_values[~np.isnan(ghi_values)]
        assert len(valid) > 0
        assert float(valid.mean()) > 100, "Monthly GHI should exceed 100 kWh/m²"
        assert float(valid.max()) < 300, "Monthly GHI seems unrealistically high"


class TestGlobalSolarAtlasIntegration:
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_real_fetch_bali_solar(self):
        """Integration: real GSA API call for Bali.  Requires internet."""
        connector = GlobalSolarAtlasConnector()
        small_bounds = SpatialBounds(-8.6, -8.5, 115.1, 115.2)

        ds = await connector.fetch(
            variables=["solar_ghi"],
            spatial=small_bounds,
            temporal=YEAR_2025,
        )

        assert "solar_ghi" in ds.data_vars
        assert len(ds.coords["time"]) == 12
