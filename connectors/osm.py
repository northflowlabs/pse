"""
OpenStreetMap connector — Overpass API.

Queries OSM for infrastructure features relevant to energy siting:
  - power_lines     : High-voltage transmission lines
  - substations     : Electrical substations (point + polygon)
  - roads           : Road network (access routes for construction)
  - waterways       : Rivers, canals (hydro potential + flood risk)
  - settlements     : Urban areas, villages (exclusion zones + demand)

OSM data is returned as GeoJSON feature collections and packed into an
xarray.Dataset with:
  - Coordinate arrays: latitude, longitude (feature centroids)
  - Data variables: feature-type indicator arrays
  - attrs: full GeoJSON FeatureCollection as JSON-encoded string

Overpass API: https://overpass-api.de/
QL guide:     https://wiki.openstreetmap.org/wiki/Overpass_API/Language_Guide

No authentication required.  Rate limit: be polite, cache aggressively.
"""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

import httpx
import numpy as np
import xarray as xr

from pse.connectors.base import (
    BaseConnector,
    ConnectorError,
    DataQuality,
    SpatialBounds,
    TemporalBounds,
)

log = logging.getLogger(__name__)

_OVERPASS_URL = "https://overpass-api.de/api/interpreter"
_TIMEOUT = 60
_RELIABILITY = 0.80

_PSE_VARIABLES = [
    "power_line_density",     # km of transmission lines per 100 km²
    "substation_count",       # number of substations in the region
    "nearest_substation_lat", # lat of nearest substation to region centre
    "nearest_substation_lon", # lon of nearest substation to region centre
    "road_density",           # km of roads per 100 km²
    "settlement_count",       # number of named settlements
    "waterway_density",       # km of waterways per 100 km²
]


class OSMConnector(BaseConnector):
    """
    PSE connector for OpenStreetMap data via the Overpass API.

    Returns a Dataset where spatial dimensions represent a coarse grid
    (or a single aggregate cell for the whole bounding box).
    The full GeoJSON feature data is stored in Dataset attributes.
    """

    def __init__(self, overpass_url: str = _OVERPASS_URL):
        self._url = overpass_url

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def source_id(self) -> str:
        return "osm"

    @property
    def variables(self) -> list[str]:
        return _PSE_VARIABLES

    @property
    def update_frequency_seconds(self) -> int:
        return 604_800  # OSM updates are continuous but weekly sync is sufficient

    # ------------------------------------------------------------------
    # Core fetch
    # ------------------------------------------------------------------

    async def fetch(
        self,
        variables: list[str],
        spatial: SpatialBounds,
        temporal: TemporalBounds,
        resolution: float | None = None,
    ) -> xr.Dataset:
        """
        Fetch OSM infrastructure features for the bounding box.

        The temporal argument is accepted for interface compatibility but
        ignored — OSM data is current-state only.
        """
        self._validate_variables(variables)

        # Run all Overpass queries in parallel
        import asyncio
        power_data, road_data, waterway_data, settlement_data = await asyncio.gather(
            self._query_power_infrastructure(spatial),
            self._query_roads(spatial),
            self._query_waterways(spatial),
            self._query_settlements(spatial),
        )

        ds = self._build_dataset(
            variables, spatial, power_data, road_data, waterway_data, settlement_data
        )
        ds.attrs.update(self._base_attrs(variables, spatial, temporal))
        return ds

    # ------------------------------------------------------------------
    # Quality + freshness
    # ------------------------------------------------------------------

    async def get_quality(
        self,
        spatial: SpatialBounds,
        temporal: TemporalBounds,
    ) -> DataQuality:
        return DataQuality(
            completeness=0.85,  # OSM coverage varies by region; good in SE Asia
            temporal_lag=0.0,   # Real-time data
            spatial_resolution=10.0,  # OSM nodes can be sub-metre
            source_reliability=_RELIABILITY,
            provenance={
                "source": self.source_id,
                "api": self._url,
                "license": "ODbL 1.0 (openstreetmap.org/copyright)",
                "note": "Coverage quality varies by country and region",
            },
        )

    async def get_latest_timestamp(self) -> datetime:
        return datetime.now(UTC)

    # ------------------------------------------------------------------
    # Overpass queries
    # ------------------------------------------------------------------

    async def _query_overpass(self, ql: str) -> dict:
        """Execute an Overpass QL query and return the JSON result."""
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.post(
                    self._url,
                    data={"data": ql},
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPError as exc:
            raise ConnectorError(f"[osm] Overpass query failed: {exc}") from exc
        except Exception as exc:
            raise ConnectorError(f"[osm] Unexpected error: {exc}") from exc

    def _bbox_str(self, s: SpatialBounds) -> str:
        """Overpass bbox format: south,west,north,east"""
        return f"{s.min_lat},{s.min_lon},{s.max_lat},{s.max_lon}"

    async def _query_power_infrastructure(self, spatial: SpatialBounds) -> dict:
        """Query power lines, substations, and generators."""
        bbox = self._bbox_str(spatial)
        ql = f"""
[out:json][timeout:{_TIMEOUT}];
(
  way["power"="line"]({bbox});
  way["power"="cable"]({bbox});
  node["power"="substation"]({bbox});
  way["power"="substation"]({bbox});
  node["power"="generator"]({bbox});
);
out geom;
"""
        try:
            return await self._query_overpass(ql)
        except ConnectorError as exc:
            log.warning("[osm] Power infrastructure query failed: %s", exc)
            return {"elements": []}

    async def _query_roads(self, spatial: SpatialBounds) -> dict:
        """Query major and secondary roads."""
        bbox = self._bbox_str(spatial)
        ql = f"""
[out:json][timeout:{_TIMEOUT}];
(
  way["highway"~"^(motorway|trunk|primary|secondary|tertiary)$"]({bbox});
);
out geom;
"""
        try:
            return await self._query_overpass(ql)
        except ConnectorError as exc:
            log.warning("[osm] Roads query failed: %s", exc)
            return {"elements": []}

    async def _query_waterways(self, spatial: SpatialBounds) -> dict:
        """Query rivers and canals."""
        bbox = self._bbox_str(spatial)
        ql = f"""
[out:json][timeout:{_TIMEOUT}];
(
  way["waterway"~"^(river|canal|stream)$"]({bbox});
);
out geom;
"""
        try:
            return await self._query_overpass(ql)
        except ConnectorError as exc:
            log.warning("[osm] Waterways query failed: %s", exc)
            return {"elements": []}

    async def _query_settlements(self, spatial: SpatialBounds) -> dict:
        """Query named settlements (cities, towns, villages)."""
        bbox = self._bbox_str(spatial)
        ql = f"""
[out:json][timeout:{_TIMEOUT}];
(
  node["place"~"^(city|town|village|hamlet)$"]({bbox});
);
out;
"""
        try:
            return await self._query_overpass(ql)
        except ConnectorError as exc:
            log.warning("[osm] Settlements query failed: %s", exc)
            return {"elements": []}

    # ------------------------------------------------------------------
    # Dataset assembly
    # ------------------------------------------------------------------

    def _build_dataset(
        self,
        variables: list[str],
        spatial: SpatialBounds,
        power_data: dict,
        road_data: dict,
        waterway_data: dict,
        settlement_data: dict,
    ) -> xr.Dataset:
        """
        Assemble OSM features into an xarray.Dataset.

        Since OSM data is not naturally gridded, we:
        1. Compute aggregate metrics over the whole bounding box.
        2. Return them on a coarse 1×1 grid (single cell representing the bbox).

        FLUX consumers use the raw GeoJSON (stored in attrs) for proximity
        calculations (e.g. "distance to nearest substation").
        """
        from pse.query.spatial import bounding_box_area_km2
        area_km2 = bounding_box_area_km2(spatial)

        power_elements = power_data.get("elements", [])
        road_elements = road_data.get("elements", [])
        waterway_elements = waterway_data.get("elements", [])
        settlement_elements = settlement_data.get("elements", [])

        # Compute metrics
        power_line_km = self._estimate_way_length_km(
            [e for e in power_elements if e.get("type") == "way"]
        )
        road_km = self._estimate_way_length_km(road_elements)
        waterway_km = self._estimate_way_length_km(waterway_elements)

        substations = [
            e for e in power_elements
            if e.get("tags", {}).get("power") == "substation"
        ]
        substation_count = len(substations)

        # Nearest substation to region centre
        centre_lat, centre_lon = spatial.center_lat, spatial.center_lon
        nearest_lat, nearest_lon = self._nearest_point(substations, centre_lat, centre_lon)

        metric_map = {
            "power_line_density":     power_line_km / area_km2 * 100 if area_km2 > 0 else 0.0,
            "substation_count":       float(substation_count),
            "nearest_substation_lat": nearest_lat,
            "nearest_substation_lon": nearest_lon,
            "road_density":           road_km / area_km2 * 100 if area_km2 > 0 else 0.0,
            "settlement_count":       float(len(settlement_elements)),
            "waterway_density":       waterway_km / area_km2 * 100 if area_km2 > 0 else 0.0,
        }

        # Single-cell (1×1) grid at bbox centre
        data_vars = {
            var: (["latitude", "longitude"], np.array([[metric_map.get(var, np.nan)]], dtype=np.float32))
            for var in variables
        }

        ds = xr.Dataset(
            data_vars,
            coords={
                "latitude": [centre_lat],
                "longitude": [centre_lon],
                "time": np.datetime64("now", "s"),
            },
        )

        # Store raw GeoJSON as attributes for FLUX proximity analysis
        ds.attrs["osm_power_geojson"] = json.dumps(self._to_geojson(power_elements))
        ds.attrs["osm_road_geojson"] = json.dumps(self._to_geojson(road_elements))
        ds.attrs["osm_settlements_geojson"] = json.dumps(self._to_geojson(settlement_elements))
        ds.attrs["osm_substations"] = json.dumps([
            {
                "lat": self._element_lat(e),
                "lon": self._element_lon(e),
                "tags": e.get("tags", {}),
            }
            for e in substations
        ])

        return ds

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_way_length_km(ways: list[dict]) -> float:
        """Rough length estimate using Euclidean haversine-free approximation."""
        total_km = 0.0
        for way in ways:
            geometry = way.get("geometry", [])
            if len(geometry) < 2:
                continue
            for i in range(len(geometry) - 1):
                dlat = (geometry[i + 1]["lat"] - geometry[i]["lat"]) * 111.0
                dlon = (geometry[i + 1]["lon"] - geometry[i]["lon"]) * 111.0 * np.cos(
                    np.radians((geometry[i]["lat"] + geometry[i + 1]["lat"]) / 2)
                )
                total_km += np.sqrt(dlat ** 2 + dlon ** 2)
        return total_km

    @staticmethod
    def _nearest_point(
        elements: list[dict],
        ref_lat: float,
        ref_lon: float,
    ) -> tuple[float, float]:
        """Return (lat, lon) of the element closest to (ref_lat, ref_lon)."""
        if not elements:
            return np.nan, np.nan

        best_dist = float("inf")
        best_lat, best_lon = np.nan, np.nan
        for el in elements:
            lat = OSMConnector._element_lat(el)
            lon = OSMConnector._element_lon(el)
            if np.isnan(lat) or np.isnan(lon):
                continue
            d = (lat - ref_lat) ** 2 + (lon - ref_lon) ** 2
            if d < best_dist:
                best_dist = d
                best_lat, best_lon = lat, lon
        return best_lat, best_lon

    @staticmethod
    def _element_lat(el: dict) -> float:
        if el.get("type") == "node":
            return el.get("lat", np.nan)
        bounds = el.get("bounds")
        if bounds:
            return (bounds["minlat"] + bounds["maxlat"]) / 2
        geom = el.get("geometry", [])
        if geom:
            return np.mean([g["lat"] for g in geom])
        return np.nan

    @staticmethod
    def _element_lon(el: dict) -> float:
        if el.get("type") == "node":
            return el.get("lon", np.nan)
        bounds = el.get("bounds")
        if bounds:
            return (bounds["minlon"] + bounds["maxlon"]) / 2
        geom = el.get("geometry", [])
        if geom:
            return np.mean([g["lon"] for g in geom])
        return np.nan

    @staticmethod
    def _to_geojson(elements: list[dict]) -> dict:
        """Convert Overpass elements to a minimal GeoJSON FeatureCollection."""
        features = []
        for el in elements:
            if el.get("type") == "node":
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [el.get("lon"), el.get("lat")]},
                    "properties": el.get("tags", {}),
                })
            elif el.get("type") == "way":
                coords = [[g["lon"], g["lat"]] for g in el.get("geometry", [])]
                if coords:
                    features.append({
                        "type": "Feature",
                        "geometry": {"type": "LineString", "coordinates": coords},
                        "properties": el.get("tags", {}),
                    })
        return {"type": "FeatureCollection", "features": features}
