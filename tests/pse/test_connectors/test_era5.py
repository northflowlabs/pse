"""
Tests for the ERA5 connector.

Unit tests mock the CDS API (sync executor call); integration tests hit the
real CDS and are gated behind @pytest.mark.integration.
"""
from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from pse.connectors.base import SpatialBounds, TemporalBounds
from pse.connectors.era5 import ERA5Connector

BALI = SpatialBounds(min_lat=-8.9, max_lat=-8.1, min_lon=114.8, max_lon=115.7)
JAN_2025 = TemporalBounds(start=datetime(2025, 1, 1), end=datetime(2025, 1, 3))


@pytest.fixture
def connector():
    return ERA5Connector(api_key="test-key", api_url="https://cds.example.com/api")


def _make_mock_netcdf(variables: list[str], n_lat: int = 3, n_lon: int = 3, n_time: int = 48) -> bytes:
    """Build a synthetic NetCDF file in memory."""
    import tempfile
    from pathlib import Path

    import pandas as pd

    times = pd.date_range("2025-01-01", periods=n_time, freq="1h")
    lats = np.linspace(-8.9, -8.1, n_lat)
    lons = np.linspace(114.8, 115.7, n_lon)

    # Map PSE variable → NetCDF short name
    short_names = {
        "temperature_2m": "t2m",
        "wind_speed_10m": "u10",   # u-component for wind speed derivation
        "solar_ghi": "ssrd",
        "surface_pressure": "sp",
        "precipitation": "tp",
    }

    data_vars = {}
    for var in variables:
        short = short_names.get(var, var)
        data = np.random.rand(n_time, n_lat, n_lon).astype(np.float32) * 10 + 273.15
        data_vars[short] = (["time", "latitude", "longitude"], data)
        # Also add v10 for wind speed derivation
        if var == "wind_speed_10m":
            data_vars["v10"] = (
                ["time", "latitude", "longitude"],
                np.random.rand(n_time, n_lat, n_lon).astype(np.float32) * 5,
            )

    ds = xr.Dataset(data_vars, coords={"time": times, "latitude": lats, "longitude": lons})

    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        ds.to_netcdf(tmp.name)
        return Path(tmp.name).read_bytes()


class TestERA5Unit:
    def test_source_id(self, connector):
        assert connector.source_id == "era5"

    def test_variables_include_core(self, connector):
        assert "temperature_2m" in connector.variables
        assert "wind_speed_10m" in connector.variables
        assert "solar_ghi" in connector.variables
        assert "precipitation" in connector.variables

    def test_update_frequency(self, connector):
        assert connector.update_frequency_seconds == 86_400

    def test_validate_unknown_variable(self, connector):
        with pytest.raises(ValueError):
            connector._validate_variables(["not_a_real_variable"])

    def test_resolve_cds_vars_wind_speed_expands_to_components(self, connector):
        cds_vars = connector._resolve_cds_variables(["wind_speed_10m"])
        # Should include both u and v components
        assert "10m_u_component_of_wind" in cds_vars
        assert "10m_v_component_of_wind" in cds_vars

    def test_build_request_structure(self, connector):
        req = connector._build_request(
            {"2m_temperature"},
            BALI,
            JAN_2025,
        )
        assert req["product_type"] == "reanalysis"
        assert "2m_temperature" in req["variable"]
        assert req["data_format"] == "netcdf"
        assert req["area"] == [
            BALI.max_lat, BALI.min_lon, BALI.min_lat, BALI.max_lon
        ]
        assert "2025" in req["year"]

    @pytest.mark.asyncio
    async def test_fetch_returns_correct_structure(self, connector):
        nc_bytes = _make_mock_netcdf(["temperature_2m"])

        with patch.object(
            connector, "_retrieve_sync", return_value=nc_bytes
        ):
            ds = await connector.fetch(
                variables=["temperature_2m"],
                spatial=BALI,
                temporal=JAN_2025,
            )

        assert isinstance(ds, xr.Dataset)
        assert "temperature_2m" in ds.data_vars
        assert ds.attrs.get("pse_source") == "era5"

    @pytest.mark.asyncio
    async def test_fetch_wind_speed_derived(self, connector):
        """wind_speed_10m should be derived from u10 + v10."""
        nc_bytes = _make_mock_netcdf(["wind_speed_10m"])

        with patch.object(connector, "_retrieve_sync", return_value=nc_bytes):
            ds = await connector.fetch(
                variables=["wind_speed_10m"],
                spatial=BALI,
                temporal=JAN_2025,
            )

        assert "wind_speed_10m" in ds.data_vars
        vals = ds["wind_speed_10m"].values.flatten()
        valid = vals[~np.isnan(vals)]
        assert np.all(valid >= 0), "Wind speed must be non-negative"

    @pytest.mark.asyncio
    async def test_quality_returns_high_reliability(self, connector):
        quality = await connector.get_quality(BALI, JAN_2025)
        assert quality.source_reliability >= 0.95
        assert quality.completeness >= 0.99

    @pytest.mark.asyncio
    async def test_latest_timestamp_is_5_days_ago(self, connector):
        from datetime import timedelta
        ts = await connector.get_latest_timestamp()
        expected_lag = timedelta(days=5).total_seconds()
        actual_lag = (datetime.now(UTC) - ts.replace(tzinfo=UTC)).total_seconds()
        assert abs(actual_lag - expected_lag) < 86_400, "ERA5 lag should be ~5 days"


class TestERA5Integration:
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_real_fetch_bali_temperature(self):
        """Integration: submit a real CDS request. Requires ERA5_CDS_API_KEY."""
        connector = ERA5Connector()
        small_bounds = SpatialBounds(-8.6, -8.5, 115.1, 115.2)
        short_range = TemporalBounds(datetime(2023, 1, 1), datetime(2023, 1, 2))

        ds = await connector.fetch(
            variables=["temperature_2m"],
            spatial=small_bounds,
            temporal=short_range,
        )

        assert "temperature_2m" in ds.data_vars
        mean_temp_k = float(ds["temperature_2m"].mean())
        # Should be tropical ~298 K (25°C)
        assert 280 < mean_temp_k < 320, f"Temperature {mean_temp_k:.1f}K not in tropical range"
