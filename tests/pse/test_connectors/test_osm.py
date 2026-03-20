"""
Tests for the OSM connector.
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
import xarray as xr

from pse.connectors.base import SpatialBounds, TemporalBounds
from pse.connectors.osm import OSMConnector

EAST_JAVA = SpatialBounds(min_lat=-8.5, max_lat=-6.0, min_lon=110.0, max_lon=115.0)
NOW = TemporalBounds(start=datetime(2025, 1, 1), end=datetime(2025, 1, 2))


@pytest.fixture
def connector():
    return OSMConnector()


def _mock_power_response():
    return {
        "elements": [
            # A transmission line (way)
            {
                "type": "way",
                "id": 1,
                "tags": {"power": "line", "voltage": "150000"},
                "geometry": [
                    {"lat": -7.0, "lon": 111.0},
                    {"lat": -7.1, "lon": 111.5},
                    {"lat": -7.2, "lon": 112.0},
                ],
            },
            # A substation (node)
            {
                "type": "node",
                "id": 2,
                "lat": -7.5,
                "lon": 112.5,
                "tags": {"power": "substation", "voltage": "150000"},
            },
        ]
    }


def _mock_empty_response():
    return {"elements": []}


class TestOSMUnit:
    def test_source_id(self, connector):
        assert connector.source_id == "osm"

    def test_variables_list(self, connector):
        assert "power_line_density" in connector.variables
        assert "substation_count" in connector.variables
        assert "nearest_substation_lat" in connector.variables
        assert "road_density" in connector.variables

    def test_update_frequency_weekly(self, connector):
        assert connector.update_frequency_seconds == 604_800

    def test_bbox_str_format(self, connector):
        bbox = connector._bbox_str(EAST_JAVA)
        assert bbox == "-8.5,110.0,-6.0,115.0"

    def test_estimate_way_length_km(self):
        ways = [
            {
                "geometry": [
                    {"lat": 0.0, "lon": 0.0},
                    {"lat": 0.0, "lon": 1.0},   # ~111 km
                ]
            }
        ]
        length = OSMConnector._estimate_way_length_km(ways)
        assert 100 < length < 120, f"Expected ~111 km, got {length:.1f} km"

    def test_nearest_point_finds_closest(self):
        elements = [
            {"type": "node", "lat": 1.0, "lon": 1.0},
            {"type": "node", "lat": 5.0, "lon": 5.0},
        ]
        lat, lon = OSMConnector._nearest_point(elements, ref_lat=0.5, ref_lon=0.5)
        assert lat == 1.0 and lon == 1.0

    def test_nearest_point_empty_returns_nan(self):
        lat, lon = OSMConnector._nearest_point([], 0.0, 0.0)
        assert np.isnan(lat) and np.isnan(lon)

    def test_to_geojson_structure(self):
        elements = [
            {"type": "node", "id": 1, "lat": -7.0, "lon": 112.0, "tags": {"power": "substation"}},
        ]
        gj = OSMConnector._to_geojson(elements)
        assert gj["type"] == "FeatureCollection"
        assert len(gj["features"]) == 1
        assert gj["features"][0]["geometry"]["type"] == "Point"

    @pytest.mark.asyncio
    async def test_fetch_returns_valid_dataset(self, connector):
        with patch.object(connector, "_query_power_infrastructure",
                          new=AsyncMock(return_value=_mock_power_response())), \
             patch.object(connector, "_query_roads",
                          new=AsyncMock(return_value=_mock_empty_response())), \
             patch.object(connector, "_query_waterways",
                          new=AsyncMock(return_value=_mock_empty_response())), \
             patch.object(connector, "_query_settlements",
                          new=AsyncMock(return_value=_mock_empty_response())):

            ds = await connector.fetch(
                variables=["power_line_density", "substation_count",
                           "nearest_substation_lat", "nearest_substation_lon"],
                spatial=EAST_JAVA,
                temporal=NOW,
            )

        assert isinstance(ds, xr.Dataset)
        assert "substation_count" in ds.data_vars
        assert float(ds["substation_count"].values.flat[0]) == 1.0

    @pytest.mark.asyncio
    async def test_fetch_has_geojson_attrs(self, connector):
        with patch.object(connector, "_query_power_infrastructure",
                          new=AsyncMock(return_value=_mock_power_response())), \
             patch.object(connector, "_query_roads",
                          new=AsyncMock(return_value=_mock_empty_response())), \
             patch.object(connector, "_query_waterways",
                          new=AsyncMock(return_value=_mock_empty_response())), \
             patch.object(connector, "_query_settlements",
                          new=AsyncMock(return_value=_mock_empty_response())):

            ds = await connector.fetch(
                variables=["substation_count"],
                spatial=EAST_JAVA,
                temporal=NOW,
            )

        assert "osm_power_geojson" in ds.attrs
        assert "osm_substations" in ds.attrs

    @pytest.mark.asyncio
    async def test_quality_valid(self, connector):
        q = await connector.get_quality(EAST_JAVA, NOW)
        assert 0 <= q.completeness <= 1
        assert 0 <= q.source_reliability <= 1


class TestOSMIntegration:
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_real_overpass_query(self):
        """Integration: real Overpass API call for a small area."""
        connector = OSMConnector()
        small_bounds = SpatialBounds(-7.1, -7.0, 112.7, 112.8)

        ds = await connector.fetch(
            variables=["substation_count", "power_line_density"],
            spatial=small_bounds,
            temporal=NOW,
        )
        assert "substation_count" in ds.data_vars
