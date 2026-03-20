"""
Tests for the World Bank connector.
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
import xarray as xr

from pse.connectors.base import SpatialBounds, TemporalBounds
from pse.connectors.world_bank import WorldBankConnector

INDONESIA = SpatialBounds(min_lat=-11.0, max_lat=6.0, min_lon=95.0, max_lon=141.0)
DECADE = TemporalBounds(start=datetime(2015, 1, 1), end=datetime(2024, 12, 31))


@pytest.fixture
def connector():
    return WorldBankConnector()


def _mock_wb_response(indicator: str, countries: list[str]) -> list:
    """Build a minimal World Bank API response."""
    metadata = {"page": 1, "pages": 1, "per_page": 1000, "total": 20}
    data_points = []
    for country in countries:
        for year in range(2015, 2025):
            data_points.append({
                "countryiso3code": country,
                "date": str(year),
                "value": 85.0 + year - 2015,  # Incrementing value
                "indicator": {"id": indicator},
            })
    return [metadata, data_points]


class TestWorldBankUnit:
    def test_source_id(self, connector):
        assert connector.source_id == "world_bank"

    def test_variables_include_core(self, connector):
        assert "electricity_access_pct" in connector.variables
        assert "rural_electricity_access" in connector.variables
        assert "gdp_per_capita_usd" in connector.variables
        assert "renewable_energy_pct" in connector.variables

    def test_update_frequency_monthly(self, connector):
        assert connector.update_frequency_seconds == 30 * 86_400

    def test_find_countries_indonesia(self, connector):
        countries = connector._find_countries(INDONESIA)
        assert "IDN" in countries

    def test_find_countries_no_match_returns_empty(self, connector):
        # Antarctic bbox — no countries defined
        bounds = SpatialBounds(-90.0, -80.0, 0.0, 10.0)
        countries = connector._find_countries(bounds)
        assert len(countries) == 0

    def test_find_countries_kenya(self, connector):
        kenya = SpatialBounds(min_lat=-5.0, max_lat=5.0, min_lon=34.0, max_lon=42.0)
        countries = connector._find_countries(kenya)
        assert "KEN" in countries

    def test_broadcast_to_grid(self, connector):
        country_data = {"IDN": {2020: 95.0, 2021: 96.0}}
        lats = [-8.0, 0.0]
        lons = [110.0, 120.0]
        arr = connector._broadcast_to_grid(country_data, [2020, 2021], ["IDN"], lats, lons)

        assert arr.shape == (2, 2, 2)  # time × lat × lon
        # All cells in Indonesia bbox should have the IDN value
        assert arr[0, 0, 0] == pytest.approx(95.0)
        assert arr[1, 0, 0] == pytest.approx(96.0)

    @pytest.mark.asyncio
    async def test_fetch_returns_valid_dataset(self, connector):
        mock_response = _mock_wb_response("EG.ELC.ACCS.ZS", ["IDN"])

        async def mock_fetch_indicator(var, countries, year_start, year_end):
            return {"IDN": {y: 85.0 + y - 2015 for y in range(year_start, year_end + 1)}}

        with patch.object(connector, "_fetch_indicator",
                          side_effect=mock_fetch_indicator):
            ds = await connector.fetch(
                variables=["electricity_access_pct"],
                spatial=INDONESIA,
                temporal=DECADE,
            )

        assert isinstance(ds, xr.Dataset)
        assert "electricity_access_pct" in ds.data_vars
        assert "time" in ds.dims

    @pytest.mark.asyncio
    async def test_electricity_access_in_valid_range(self, connector):
        async def mock_fetch_indicator(var, countries, year_start, year_end):
            return {"IDN": {y: 95.0 for y in range(year_start, year_end + 1)}}

        with patch.object(connector, "_fetch_indicator",
                          side_effect=mock_fetch_indicator):
            ds = await connector.fetch(
                variables=["electricity_access_pct"],
                spatial=INDONESIA,
                temporal=DECADE,
            )

        vals = ds["electricity_access_pct"].values.flatten()
        valid = vals[~np.isnan(vals)]
        assert len(valid) > 0
        assert np.all((valid >= 0) & (valid <= 100)), "Access % must be 0-100"

    @pytest.mark.asyncio
    async def test_quality_valid(self, connector):
        q = await connector.get_quality(INDONESIA, DECADE)
        assert 0 <= q.completeness <= 1
        assert 0 <= q.source_reliability <= 1


class TestWorldBankIntegration:
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_real_indonesia_electricity_access(self):
        """Integration: real World Bank API call."""
        connector = WorldBankConnector()
        recent = TemporalBounds(start=datetime(2019, 1, 1), end=datetime(2022, 12, 31))

        ds = await connector.fetch(
            variables=["electricity_access_pct"],
            spatial=INDONESIA,
            temporal=recent,
        )

        assert "electricity_access_pct" in ds.data_vars
        vals = ds["electricity_access_pct"].values.flatten()
        valid = vals[~np.isnan(vals)]
        assert len(valid) > 0
        assert float(valid.mean()) > 50, "Indonesia electricity access should exceed 50%"
