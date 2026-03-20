"""
World Bank connector — World Bank Open Data API.

Provides country-level socioeconomic and energy access indicators:
  - electricity_access_pct    : Population with access to electricity (%)
  - rural_electricity_access  : Rural electricity access (%)
  - urban_electricity_access  : Urban electricity access (%)
  - gdp_per_capita_usd        : GDP per capita (current USD)
  - population_total          : Total population
  - renewable_energy_pct      : Renewables share of total energy (%)
  - co2_emissions_per_capita  : CO₂ emissions per capita (metric tons)

The World Bank API returns annual time-series by country (ISO3 code).
We broadcast country-level values to all grid cells within the country bbox
so the data can be spatially joined with site assessments.

API docs: https://datahelpdesk.worldbank.org/knowledgebase/articles/889392
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

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

_BASE_URL = "https://api.worldbank.org/v2"
_RELIABILITY = 0.90

# PSE canonical name → World Bank indicator code
_INDICATOR_MAP: dict[str, str] = {
    "electricity_access_pct":   "EG.ELC.ACCS.ZS",
    "rural_electricity_access": "EG.ELC.ACCS.RU.ZS",
    "urban_electricity_access": "EG.ELC.ACCS.UR.ZS",
    "gdp_per_capita_usd":       "NY.GDP.PCAP.CD",
    "population_total":         "SP.POP.TOTL",
    "renewable_energy_pct":     "EG.FEC.RNEW.ZS",
    "co2_emissions_per_capita": "EN.ATM.CO2E.PC",
    "energy_intensity":         "EG.EGY.PRIM.PP.KD",
    "fossil_fuel_energy_pct":   "EG.USE.COMM.FO.ZS",
}

# Country ISO3 → approximate bounding box mapping
# Used to find which country a (lat, lon) falls in.
# Covers the Phase 1 target countries (Indonesia, Kenya, Vietnam) plus neighbours.
_COUNTRY_BBOXES: dict[str, tuple[float, float, float, float]] = {
    # ISO3: (min_lat, max_lat, min_lon, max_lon)
    "IDN": (-11.0, 6.0, 95.0, 141.0),
    "KEN": (-4.7, 4.6, 33.9, 41.9),
    "VNM": (8.4, 23.4, 102.1, 109.5),
    "PHL": (4.6, 21.1, 116.9, 126.6),
    "TZA": (-11.7, -0.9, 29.3, 40.4),
    "ETH": (3.4, 14.9, 32.9, 47.9),
    "NGA": (4.3, 13.9, 2.7, 14.7),
    "GHA": (4.7, 11.2, -3.3, 1.2),
    "MOZ": (-26.9, -10.5, 30.2, 40.8),
    "ZMB": (-18.1, -8.2, 21.9, 33.7),
    "BGD": (20.7, 26.6, 88.0, 92.7),
    "PAK": (23.7, 37.1, 60.9, 77.8),
    "MMR": (9.8, 28.5, 92.2, 101.2),
    "KHM": (10.4, 14.7, 102.3, 107.6),
    "LAO": (13.9, 22.5, 100.1, 107.6),
    "THA": (5.6, 20.5, 97.4, 105.6),
}


class WorldBankConnector(BaseConnector):
    """
    PSE connector for World Bank Open Data.

    Country-level annual indicators are fetched and returned as a Dataset
    where:
    - time dimension = calendar years in the requested range
    - latitude/longitude = coarse grid covering the matched country bounding box
    """

    def __init__(self, base_url: str = _BASE_URL):
        self._url = base_url.rstrip("/")

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def source_id(self) -> str:
        return "world_bank"

    @property
    def variables(self) -> list[str]:
        return sorted(_INDICATOR_MAP.keys())

    @property
    def update_frequency_seconds(self) -> int:
        return 30 * 86_400  # monthly check (underlying data is annual)

    # ------------------------------------------------------------------
    # Core fetch
    # ------------------------------------------------------------------

    async def fetch(
        self,
        variables: list[str],
        spatial: SpatialBounds,
        temporal: TemporalBounds,
        resolution: Optional[float] = None,
    ) -> xr.Dataset:
        """
        Fetch World Bank indicators for countries overlapping the spatial bounds.
        """
        self._validate_variables(variables)

        # Find matching countries
        countries = self._find_countries(spatial)
        if not countries:
            log.warning("[world_bank] No known country found for bounds %s", spatial.to_dict())
            countries = []  # return empty dataset below

        year_start = temporal.start.year
        year_end = temporal.end.year

        # Fetch all indicators for all countries in parallel
        tasks = [
            self._fetch_indicator(var, countries, year_start, year_end)
            for var in variables
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Build time axis (annual)
        years = list(range(year_start, year_end + 1))
        import pandas as pd
        times = pd.to_datetime([f"{y}-01-01" for y in years])

        # Build lat/lon grid from the union of matched country bboxes
        lats, lons = self._country_grid(countries, resolution)

        if not lats or not lons:
            # Fallback to a single centre-point grid
            lats = [spatial.center_lat]
            lons = [spatial.center_lon]

        n_time, n_lat, n_lon = len(times), len(lats), len(lons)

        data_vars = {}
        for var, result in zip(variables, results):
            if isinstance(result, Exception):
                log.warning("[world_bank] %s fetch failed: %s", var, result)
                arr = np.full((n_time, n_lat, n_lon), np.nan, dtype=np.float32)
            else:
                # result is a dict: {country_iso3: {year: value}}
                arr = self._broadcast_to_grid(result, years, countries, lats, lons)

            data_vars[var] = (["time", "latitude", "longitude"], arr)

        ds = xr.Dataset(
            data_vars,
            coords={
                "time": times,
                "latitude": lats,
                "longitude": lons,
            },
        )
        ds.attrs["world_bank_countries"] = json_encode(countries)
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
        latest = await self.get_latest_timestamp()
        lag = (datetime.now(timezone.utc) - latest).total_seconds()
        return DataQuality(
            completeness=0.85,
            temporal_lag=lag,
            spatial_resolution=100_000,  # country-level granularity
            source_reliability=_RELIABILITY,
            provenance={
                "source": self.source_id,
                "api": self._url,
                "license": "CC BY 4.0",
                "note": "Annual country-level indicators broadcast to spatial grid",
            },
        )

    async def get_latest_timestamp(self) -> datetime:
        """World Bank typically lags 1-2 years behind present."""
        from datetime import timedelta
        latest_year = datetime.now(timezone.utc).year - 1
        return datetime(latest_year, 1, 1, tzinfo=timezone.utc)

    # ------------------------------------------------------------------
    # Private: API fetch
    # ------------------------------------------------------------------

    async def _fetch_indicator(
        self,
        pse_var: str,
        countries: list[str],
        year_start: int,
        year_end: int,
    ) -> dict[str, dict[int, Optional[float]]]:
        """
        Fetch a single indicator for a list of countries.

        Returns: {country_iso3: {year: value}}
        """
        indicator = _INDICATOR_MAP[pse_var]
        country_str = ";".join(c.lower() for c in countries) if countries else "all"
        url = (
            f"{self._url}/country/{country_str}/indicator/{indicator}"
            f"?format=json&per_page=1000&date={year_start}:{year_end}"
        )

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                payload = resp.json()
        except httpx.HTTPError as exc:
            raise ConnectorError(f"[world_bank] HTTP error for {indicator}: {exc}") from exc

        # World Bank API returns [metadata, [datapoints]]
        if not isinstance(payload, list) or len(payload) < 2:
            raise ConnectorError(f"[world_bank] Unexpected response format for {indicator}")

        data_points = payload[1] or []
        result: dict[str, dict[int, Optional[float]]] = {c: {} for c in countries}

        for dp in data_points:
            country_iso3 = dp.get("countryiso3code", "").upper()
            if country_iso3 not in result:
                continue
            year = dp.get("date")
            value = dp.get("value")
            if year:
                try:
                    result[country_iso3][int(year)] = float(value) if value is not None else None
                except (ValueError, TypeError):
                    pass

        return result

    # ------------------------------------------------------------------
    # Private: spatial helpers
    # ------------------------------------------------------------------

    def _find_countries(self, spatial: SpatialBounds) -> list[str]:
        """Return ISO3 codes for countries whose bboxes overlap the query bounds."""
        matches = []
        for iso3, (mn_lat, mx_lat, mn_lon, mx_lon) in _COUNTRY_BBOXES.items():
            if (mn_lat <= spatial.max_lat and mx_lat >= spatial.min_lat and
                    mn_lon <= spatial.max_lon and mx_lon >= spatial.min_lon):
                matches.append(iso3)
        return matches

    def _country_grid(
        self, countries: list[str], resolution: Optional[float]
    ) -> tuple[list[float], list[float]]:
        """Build a 2-3 point grid for each country in the list."""
        if not countries:
            return [], []
        lats_all: set[float] = set()
        lons_all: set[float] = set()
        for iso3 in countries:
            mn_lat, mx_lat, mn_lon, mx_lon = _COUNTRY_BBOXES[iso3]
            mid_lat = (mn_lat + mx_lat) / 2
            mid_lon = (mn_lon + mx_lon) / 2
            lats_all.update([round(mn_lat, 2), round(mid_lat, 2), round(mx_lat, 2)])
            lons_all.update([round(mn_lon, 2), round(mid_lon, 2), round(mx_lon, 2)])
        return sorted(lats_all), sorted(lons_all)

    def _broadcast_to_grid(
        self,
        country_data: dict[str, dict[int, Optional[float]]],
        years: list[int],
        countries: list[str],
        lats: list[float],
        lons: list[float],
    ) -> np.ndarray:
        """
        Broadcast country-level annual values to a (time × lat × lon) array.

        Each grid cell is assigned the value for the country whose bbox it
        falls within (first match).
        """
        n_time, n_lat, n_lon = len(years), len(lats), len(lons)
        arr = np.full((n_time, n_lat, n_lon), np.nan, dtype=np.float32)

        for i_lat, lat in enumerate(lats):
            for i_lon, lon in enumerate(lons):
                # Find the country for this cell
                cell_country = None
                for iso3 in countries:
                    mn_lat, mx_lat, mn_lon, mx_lon = _COUNTRY_BBOXES[iso3]
                    if mn_lat <= lat <= mx_lat and mn_lon <= lon <= mx_lon:
                        cell_country = iso3
                        break

                if cell_country is None:
                    continue

                year_data = country_data.get(cell_country, {})
                for i_t, year in enumerate(years):
                    val = year_data.get(year)
                    if val is not None:
                        arr[i_t, i_lat, i_lon] = val

        return arr


def json_encode(obj) -> str:
    import json
    return json.dumps(obj)
