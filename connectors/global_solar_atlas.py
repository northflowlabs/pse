"""
Global Solar Atlas (GSA) connector.

GSA (https://globalsolaratlas.info) provides high-resolution long-term average
solar resource data.  The public REST API returns annual/monthly climatological
values rather than time-series — ideal for yield estimation and siting.

Key variables:
  - GHI  — Global Horizontal Irradiance  (kWh/m²/year or kWh/m²/day)
  - DNI  — Direct Normal Irradiance
  - GTI  — Global Tilted Irradiance (optimum tilt)
  - DIF  — Diffuse Horizontal Irradiance
  - TEMP — Air temperature at 2 m
  - WS   — Wind speed at 10 m
  - PVOUT — Theoretical PV output (kWh/kWp/year)

The API is publicly accessible without authentication for data query.
Bulk download requires a GSA account; point/area queries are free.

API reference: https://globalsolaratlas.info/api
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional

import httpx
import numpy as np
import xarray as xr

from .base import (
    BaseConnector,
    ConnectorError,
    DataQuality,
    SpatialBounds,
    TemporalBounds,
)

# ---------------------------------------------------------------------------
# Variable mapping  —  PSE canonical name → GSA parameter name
# ---------------------------------------------------------------------------
_VAR_MAP: dict[str, str] = {
    "solar_ghi":    "GHI",
    "solar_dni":    "DNI",
    "solar_gti":    "GTI",
    "solar_dif":    "DIF",
    "solar_pvout":  "PVOUT",
    "temperature_2m": "TEMP",
    "wind_speed_10m": "WS",
}

_NATIVE_RESOLUTION_M = 1_000   # ~1 km
_RELIABILITY = 0.92

# GSA data endpoint — returns JSON with monthly + annual statistics
_API_URL = "https://api.globalsolaratlas.info/data/lta"


class GlobalSolarAtlasConnector(BaseConnector):
    """
    PSE connector for the Global Solar Atlas long-term average (LTA) API.

    Because GSA provides *climatological* averages (not hourly time-series),
    the returned Dataset uses a synthetic time dimension representing months
    (index 0–11 for Jan–Dec) or a single annual aggregate, depending on how
    the data is requested.

    When a TemporalBounds is provided:
    - If the range covers a full year or more → annual averages are returned.
    - Otherwise → monthly climatology values are returned as a 12-step series.

    Spatial coverage: global (land areas).
    """

    def __init__(
        self,
        api_url: str = _API_URL,
        max_concurrent: int = 8,
    ):
        self._api_url = api_url
        self._sem = asyncio.Semaphore(max_concurrent)

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def source_id(self) -> str:
        return "global_solar_atlas"

    @property
    def variables(self) -> list[str]:
        return sorted(_VAR_MAP.keys())

    @property
    def update_frequency_seconds(self) -> int:
        return 86_400  # dataset updated annually; check daily is more than enough

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
        Fetch long-term average solar data for a bounding box.

        Queries a grid of points in parallel, then assembles them into a
        (time × latitude × longitude) Dataset where the time dimension
        contains monthly climatology steps (12 values).
        """
        self._validate_variables(variables)

        lats, lons = self._build_grid(spatial, resolution or _NATIVE_RESOLUTION_M)

        tasks = [
            self._fetch_point(lat, lon)
            for lat in lats
            for lon in lons
        ]
        point_results = await asyncio.gather(*tasks, return_exceptions=True)

        ds = self._assemble_grid(point_results, lats, lons, variables)
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
        # GSA is a static climatological dataset — effectively always "fresh"
        return DataQuality(
            completeness=0.95,  # minor gaps over open ocean / polar regions
            temporal_lag=0.0,
            spatial_resolution=_NATIVE_RESOLUTION_M,
            source_reliability=_RELIABILITY,
            provenance={
                "source": self.source_id,
                "dataset": "Global Solar Atlas 2.0 — long-term average",
                "api_url": self._api_url,
                "temporal_coverage": "1994–2018 (22-year climatology)",
            },
        )

    async def get_latest_timestamp(self) -> datetime:
        # Climatological data — treat as always current
        return datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_grid(
        self, spatial: SpatialBounds, resolution_m: float
    ) -> tuple[list[float], list[float]]:
        deg_per_m = 1.0 / 111_000
        step = max(resolution_m * deg_per_m, 0.01)

        lats = list(np.arange(spatial.min_lat, spatial.max_lat + step / 2, step))
        lons = list(np.arange(spatial.min_lon, spatial.max_lon + step / 2, step))

        if not lats:
            lats = [spatial.center_lat]
        if not lons:
            lons = [spatial.center_lon]

        return [round(lat, 6) for lat in lats], [round(lon, 6) for lon in lons]

    async def _fetch_point(self, lat: float, lon: float) -> dict:
        """
        Fetch LTA data for one point.

        GSA JSON structure (simplified):
        {
          "annual": { "GHI": 1820.3, "DNI": 1543.1, ... },
          "monthly": {
            "GHI": [110, 120, 160, ...],   # 12 monthly values (kWh/m²/month)
            "DNI": [...],
            ...
          }
        }
        """
        params = {
            "lat": lat,
            "lon": lon,
            "outputformat": "json",
        }
        async with self._sem:
            try:
                async with httpx.AsyncClient(timeout=20.0) as client:
                    resp = await client.get(self._api_url, params=params)
                    resp.raise_for_status()
                    return resp.json()
            except httpx.HTTPStatusError as exc:
                raise ConnectorError(
                    f"[global_solar_atlas] HTTP {exc.response.status_code} "
                    f"for ({lat}, {lon}): {exc}"
                ) from exc
            except httpx.HTTPError as exc:
                raise ConnectorError(
                    f"[global_solar_atlas] HTTP error for ({lat}, {lon}): {exc}"
                ) from exc
            except Exception as exc:
                raise ConnectorError(
                    f"[global_solar_atlas] Unexpected error for ({lat}, {lon}): {exc}"
                ) from exc

    def _assemble_grid(
        self,
        point_results: list,
        lats: list[float],
        lons: list[float],
        pse_variables: list[str],
    ) -> xr.Dataset:
        """
        Assemble per-point results into (time=12 months × lat × lon) Dataset.

        Monthly values are used as the time dimension (index 0–11 = Jan–Dec).
        Annual totals are stored as Dataset-level attributes.
        """
        n_lat = len(lats)
        n_lon = len(lons)
        n_months = 12

        # Allocate NaN arrays  (month × lat × lon)
        arrays: dict[str, np.ndarray] = {
            pse_v: np.full((n_months, n_lat, n_lon), np.nan, dtype=np.float32)
            for pse_v in pse_variables
        }
        annual: dict[str, np.ndarray] = {
            pse_v: np.full((n_lat, n_lon), np.nan, dtype=np.float32)
            for pse_v in pse_variables
        }

        result_iter = iter(point_results)
        for i_lat in range(n_lat):
            for i_lon in range(n_lon):
                result = next(result_iter)
                if isinstance(result, Exception):
                    continue

                monthly = result.get("monthly", {})
                ann = result.get("annual", {})

                for pse_v in pse_variables:
                    gsa_v = _VAR_MAP[pse_v]
                    m_vals = monthly.get(gsa_v)
                    if m_vals and len(m_vals) == 12:
                        arrays[pse_v][:, i_lat, i_lon] = np.array(
                            m_vals, dtype=np.float32
                        )
                    a_val = ann.get(gsa_v)
                    if a_val is not None:
                        annual[pse_v][i_lat, i_lon] = float(a_val)

        # Use integer month index (1–12) as the time coordinate
        months = list(range(1, 13))

        data_vars = {
            pse_v: (["time", "latitude", "longitude"], arrays[pse_v])
            for pse_v in pse_variables
        }

        ds = xr.Dataset(
            data_vars,
            coords={
                "time": months,
                "latitude": lats,
                "longitude": lons,
            },
        )
        ds["time"].attrs["long_name"] = "month_of_year"
        ds["time"].attrs["units"] = "month"

        # Attach annual totals as a secondary set of data variables
        for pse_v in pse_variables:
            ds[f"{pse_v}_annual"] = xr.DataArray(
                annual[pse_v],
                dims=["latitude", "longitude"],
                attrs={"long_name": f"{pse_v} annual total", "units": "kWh/m2/year"},
            )

        return ds
