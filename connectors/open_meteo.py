"""
Open-Meteo connector — free weather API, no key required.

Provides historical reanalysis (ERA5-Land via archive API) and short-range
forecasts.  Ideal for development/testing and as a low-latency fallback for
real-time data.

API docs: https://open-meteo.com/en/docs
Archive:  https://open-meteo.com/en/docs/historical-weather-api
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import httpx
import numpy as np
import pandas as pd
import xarray as xr

from .base import (
    BaseConnector,
    ConnectorError,
    DataQuality,
    SpatialBounds,
    TemporalBounds,
)

# ---------------------------------------------------------------------------
# Variable mapping  —  PSE canonical name  →  Open-Meteo parameter name
# ---------------------------------------------------------------------------
_VAR_MAP: dict[str, str] = {
    "temperature_2m":            "temperature_2m",
    "relative_humidity":         "relative_humidity_2m",
    "dew_point_2m":              "dew_point_2m",
    "precipitation":             "precipitation",
    "rain":                      "rain",
    "snowfall":                  "snowfall",
    "wind_speed_10m":            "wind_speed_10m",
    "wind_direction_10m":        "wind_direction_10m",
    "wind_gusts_10m":            "wind_gusts_10m",
    "surface_pressure":          "surface_pressure",
    "cloud_cover":               "cloud_cover",
    "cloud_cover_low":           "cloud_cover_low",
    "cloud_cover_mid":           "cloud_cover_mid",
    "cloud_cover_high":          "cloud_cover_high",
    "solar_shortwave_radiation":  "shortwave_radiation",
    "solar_direct_radiation":     "direct_radiation",
    "solar_diffuse_radiation":    "diffuse_radiation",
    "solar_ghi":                  "shortwave_radiation",        # alias
    "solar_dni":                  "direct_radiation",           # alias
    "evapotranspiration":        "et0_fao_evapotranspiration",
    "soil_temperature_0cm":      "soil_temperature_0cm",
    "soil_moisture_0_1cm":       "soil_moisture_0_to_1cm",
    "visibility":                "visibility",
    "vapor_pressure_deficit":    "vapor_pressure_deficit",
}

_HOURLY_VARS = set(_VAR_MAP.keys())

# Resolution: Open-Meteo uses ERA5-Land at ~11 km globally
_NATIVE_RESOLUTION_M = 11_000
_RELIABILITY = 0.88

# Open-Meteo archive endpoint (ERA5-based historical)
_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
# Live forecast (last 3 days + 16 days ahead)
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Cutoff: data older than ~5 days goes through archive API
_ARCHIVE_CUTOFF_DAYS = 5


class OpenMeteoConnector(BaseConnector):
    """
    PSE connector for Open-Meteo (https://open-meteo.com/).

    Transparently routes requests to the archive or forecast API depending
    on the requested time range, and assembles results into a spatially
    gridded xarray.Dataset by querying a grid of lat/lon points in parallel.
    """

    def __init__(
        self,
        archive_url: str = _ARCHIVE_URL,
        forecast_url: str = _FORECAST_URL,
        max_concurrent: int = 10,
    ):
        self._archive_url = archive_url
        self._forecast_url = forecast_url
        self._sem = asyncio.Semaphore(max_concurrent)

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def source_id(self) -> str:
        return "open_meteo"

    @property
    def variables(self) -> list[str]:
        return sorted(_HOURLY_VARS)

    @property
    def update_frequency_seconds(self) -> int:
        return 3600  # hourly

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
        Fetch hourly data for a bounding box by querying a grid of points in
        parallel and assembling into a 3-D (time × lat × lon) Dataset.
        """
        self._validate_variables(variables)
        om_vars = self._to_om_vars(variables)

        # Build a lat/lon grid.  Open-Meteo native res is ~11 km; we match it
        # or use a coarser grid for large areas to limit API calls.
        lats, lons = self._build_grid(spatial, resolution or _NATIVE_RESOLUTION_M)

        # Query all grid points concurrently
        tasks = [
            self._fetch_point(lat, lon, om_vars, temporal)
            for lat in lats
            for lon in lons
        ]
        point_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Assemble into 3-D dataset
        ds = self._assemble_grid(point_results, lats, lons, variables, temporal)
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
        lag = (datetime.now(UTC) - latest).total_seconds()
        return DataQuality(
            completeness=0.99,  # Open-Meteo ERA5-Land has essentially global coverage
            temporal_lag=lag,
            spatial_resolution=_NATIVE_RESOLUTION_M,
            source_reliability=_RELIABILITY,
            provenance={
                "source": self.source_id,
                "model": "ERA5-Land (archive) / GFS (forecast)",
                "archive_url": self._archive_url,
                "forecast_url": self._forecast_url,
            },
        )

    async def get_latest_timestamp(self) -> datetime:
        """
        Open-Meteo forecast data is updated hourly.
        We approximate the latest available time as (now - 1 hour), UTC.
        """
        now = datetime.now(UTC)
        return now.replace(minute=0, second=0, microsecond=0)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _to_om_vars(self, pse_variables: list[str]) -> list[str]:
        """Convert PSE canonical variable names to Open-Meteo parameter names."""
        seen: set[str] = set()
        om_vars: list[str] = []
        for v in pse_variables:
            om_v = _VAR_MAP[v]
            if om_v not in seen:
                seen.add(om_v)
                om_vars.append(om_v)
        return om_vars

    def _build_grid(
        self, spatial: SpatialBounds, resolution_m: float
    ) -> tuple[list[float], list[float]]:
        """
        Build a regular lat/lon grid covering spatial at approximately
        resolution_m metres spacing.

        Degrees per metre varies by latitude; we use a rough equatorial
        approximation (111 km/degree).
        """
        deg_per_m = 1.0 / 111_000
        step = max(resolution_m * deg_per_m, 0.01)  # minimum 0.01° (~1 km)

        lats = list(np.arange(spatial.min_lat, spatial.max_lat + step / 2, step))
        lons = list(np.arange(spatial.min_lon, spatial.max_lon + step / 2, step))

        # Always include at least the centre point
        if not lats:
            lats = [spatial.center_lat]
        if not lons:
            lons = [spatial.center_lon]

        return [round(lat, 6) for lat in lats], [round(lon, 6) for lon in lons]

    def _select_url(self, temporal: TemporalBounds) -> str:
        """Route to archive or forecast API based on recency of data."""
        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=_ARCHIVE_CUTOFF_DAYS)
        if pd.Timestamp(temporal.start, tz="UTC") < cutoff:
            return self._archive_url
        return self._forecast_url

    async def _fetch_point(
        self,
        lat: float,
        lon: float,
        om_vars: list[str],
        temporal: TemporalBounds,
    ) -> dict:
        """Fetch hourly timeseries for a single lat/lon point."""
        url = self._select_url(temporal)
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join(om_vars),
            "start_date": temporal.start.strftime("%Y-%m-%d"),
            "end_date": temporal.end.strftime("%Y-%m-%d"),
            "timezone": "UTC",
        }

        async with self._sem:
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.get(url, params=params)
                    resp.raise_for_status()
                    return resp.json()
            except httpx.HTTPError as exc:
                raise ConnectorError(
                    f"[open_meteo] HTTP error for ({lat}, {lon}): {exc}"
                ) from exc
            except Exception as exc:
                raise ConnectorError(
                    f"[open_meteo] Unexpected error for ({lat}, {lon}): {exc}"
                ) from exc

    def _assemble_grid(
        self,
        point_results: list,
        lats: list[float],
        lons: list[float],
        pse_variables: list[str],
        temporal: TemporalBounds,
    ) -> xr.Dataset:
        """
        Turn a flat list of per-point JSON responses into a 3-D
        (time × latitude × longitude) xarray.Dataset.
        """
        # Collect successful results; re-raise if everything failed
        successful = []
        for r in point_results:
            if isinstance(r, Exception):
                continue  # skip failed points; fill with NaN
            successful.append(r)

        if not successful:
            raise ConnectorError("[open_meteo] All point fetches failed.")

        # Build time index from first successful result
        first = successful[0]
        times = pd.to_datetime(first["hourly"]["time"])

        n_time = len(times)
        n_lat = len(lats)
        n_lon = len(lons)

        # Build reverse map: om_var → pse var (first match wins for aliases)
        om_to_pse: dict[str, str] = {}
        for pse_v in pse_variables:
            om_v = _VAR_MAP[pse_v]
            if om_v not in om_to_pse:
                om_to_pse[om_v] = pse_v

        # Allocate NaN arrays  (time × lat × lon)
        arrays: dict[str, np.ndarray] = {
            pse_v: np.full((n_time, n_lat, n_lon), np.nan, dtype=np.float32)
            for pse_v in pse_variables
        }

        result_iter = iter(point_results)
        for i_lat, _ in enumerate(lats):
            for i_lon, _ in enumerate(lons):
                result = next(result_iter)
                if isinstance(result, Exception):
                    continue  # leave as NaN
                hourly = result.get("hourly", {})
                for om_v, pse_v in om_to_pse.items():
                    raw = hourly.get(om_v)
                    if raw is not None:
                        arrays[pse_v][:, i_lat, i_lon] = np.array(
                            raw, dtype=np.float32
                        )

        data_vars = {
            pse_v: (["time", "latitude", "longitude"], arrays[pse_v])
            for pse_v in pse_variables
        }

        return xr.Dataset(
            data_vars,
            coords={
                "time": times,
                "latitude": lats,
                "longitude": lons,
            },
        )
