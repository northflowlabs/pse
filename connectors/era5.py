"""
ERA5 Reanalysis connector — Copernicus Climate Data Store (CDS).

ERA5 is ECMWF's state-of-the-art global reanalysis at ~31 km / 1-hour resolution,
covering 1940–present.  It is the gold standard for historical climate data and the
highest-reliability source in the PSE connector registry.

CDS API workflow:
  1. Submit a retrieval request  →  receive a job ID / URL
  2. Poll until status == "completed"
  3. Download the result file (NetCDF by default)
  4. Open with xarray, normalise to PSE conventions

Authentication: set ERA5_CDS_API_KEY and ERA5_CDS_API_URL in environment (or .cdsapirc).

CDS API reference: https://cds.climate.copernicus.eu/api
ERA5 variable catalogue: https://confluence.ecmwf.int/display/CKB/ERA5
"""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
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

# ---------------------------------------------------------------------------
# Variable mapping  —  PSE canonical  →  CDS / ERA5 short name
# ---------------------------------------------------------------------------
_VAR_MAP: dict[str, str] = {
    "temperature_2m":           "2m_temperature",
    "dew_point_2m":             "2m_dewpoint_temperature",
    "relative_humidity":        "relative_humidity",             # derived
    "wind_speed_10m":           "10m_u_component_of_wind",      # combined with v
    "wind_u_10m":               "10m_u_component_of_wind",
    "wind_v_10m":               "10m_v_component_of_wind",
    "wind_speed_100m":          "100m_u_component_of_wind",
    "wind_u_100m":              "100m_u_component_of_wind",
    "wind_v_100m":              "100m_v_component_of_wind",
    "surface_pressure":         "surface_pressure",
    "mean_sea_level_pressure":  "mean_sea_level_pressure",
    "precipitation":            "total_precipitation",
    "solar_ghi":                "surface_solar_radiation_downwards",
    "solar_direct_radiation":   "total_sky_direct_solar_radiation_at_surface",
    "cloud_cover":              "total_cloud_cover",
    "snowfall":                 "snowfall",
    "soil_temperature_0cm":     "soil_temperature_level_1",
    "evapotranspiration":       "potential_evaporation",
    "boundary_layer_height":    "boundary_layer_height",
    "wind_gusts_10m":           "instantaneous_10m_wind_gust",
    "sea_surface_temperature":  "sea_surface_temperature",
}

# ERA5 variables that require combining u + v components into speed
_WIND_SPEED_COMBOS = {
    "wind_speed_10m":  ("wind_u_10m", "wind_v_10m"),
    "wind_speed_100m": ("wind_u_100m", "wind_v_100m"),
}

_NATIVE_RESOLUTION_M = 27_800   # ~0.25° ERA5 native grid
_RELIABILITY = 0.97

# CDS API endpoints
_DEFAULT_CDS_URL = "https://cds.climate.copernicus.eu/api"
_DATASET = "reanalysis-era5-single-levels"

# Maximum area in degrees² that we'll request in a single CDS job
# (larger requests take very long to queue)
_MAX_AREA_DEG2 = 100.0

# Poll interval and timeout for CDS job completion
_POLL_INTERVAL = 10   # seconds
_POLL_TIMEOUT = 600   # seconds (10 minutes)


class ERA5Connector(BaseConnector):
    """
    PSE connector for ERA5 reanalysis via the Copernicus CDS API.

    Handles the async retrieve-and-poll workflow, decompresses the NetCDF
    response, and normalises variable names and units to PSE conventions.

    Requires:
      - ERA5_CDS_API_KEY environment variable (or cdsapirc file)
      - ERA5_CDS_API_URL environment variable (optional, defaults to public CDS)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
    ):
        self._api_key = api_key or os.environ.get("ERA5_CDS_API_KEY", "")
        self._api_url = (api_url or os.environ.get("ERA5_CDS_API_URL", _DEFAULT_CDS_URL)).rstrip("/")

        if not self._api_key:
            log.warning(
                "[era5] ERA5_CDS_API_KEY not set — ERA5 fetches will fail. "
                "Register at https://cds.climate.copernicus.eu/ to obtain a key."
            )

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def source_id(self) -> str:
        return "era5"

    @property
    def variables(self) -> list[str]:
        return sorted(_VAR_MAP.keys())

    @property
    def update_frequency_seconds(self) -> int:
        return 86_400  # ERA5 near-real-time: ~5-day lag; reanalysis: daily updates

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
        Submit a CDS retrieval request and wait for the result.

        For large spatial/temporal extents, the CDS may queue requests for
        several minutes.  This method will wait up to _POLL_TIMEOUT seconds.
        """
        self._validate_variables(variables)

        # Resolve which CDS variables to request (expand wind speed to u+v)
        cds_vars = self._resolve_cds_variables(variables)

        # Build the CDS request payload
        request = self._build_request(cds_vars, spatial, temporal)

        log.info("[era5] Submitting CDS request: %s", list(cds_vars))

        # Submit + poll + download in a thread-pool executor (cdsapi is sync)
        try:
            nc_bytes = await asyncio.get_event_loop().run_in_executor(
                None, self._retrieve_sync, request
            )
        except Exception as exc:
            raise ConnectorError(f"[era5] CDS retrieval failed: {exc}") from exc

        # Parse NetCDF bytes into xarray Dataset
        ds = self._parse_netcdf(nc_bytes, variables, cds_vars)
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
        lag = (datetime.now(timezone.utc) - latest.replace(tzinfo=timezone.utc)).total_seconds()
        return DataQuality(
            completeness=0.999,
            temporal_lag=lag,
            spatial_resolution=_NATIVE_RESOLUTION_M,
            source_reliability=_RELIABILITY,
            provenance={
                "source": self.source_id,
                "dataset": _DATASET,
                "model": "ERA5 reanalysis (ECMWF)",
                "api_url": self._api_url,
                "temporal_coverage": "1940-01-01 to present (~5-day lag)",
            },
        )

    async def get_latest_timestamp(self) -> datetime:
        """ERA5 near-real-time has ~5 day lag. Return approximate latest."""
        from datetime import timedelta
        now = datetime.now(timezone.utc)
        return (now - timedelta(days=5)).replace(hour=0, minute=0, second=0, microsecond=0)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_cds_variables(self, pse_variables: list[str]) -> set[str]:
        """
        Expand PSE variable names to CDS names, including u+v components
        needed to derive wind speed.
        """
        cds_vars: set[str] = set()
        for var in pse_variables:
            # For derived wind speed, request both u and v components
            if var in _WIND_SPEED_COMBOS:
                for component_var in _WIND_SPEED_COMBOS[var]:
                    cds_vars.add(_VAR_MAP[component_var])
            else:
                cds_vars.add(_VAR_MAP[var])
        return cds_vars

    def _build_request(
        self,
        cds_vars: set[str],
        spatial: SpatialBounds,
        temporal: TemporalBounds,
    ) -> dict:
        """Build the CDS API request dictionary."""
        import pandas as pd

        # Generate list of dates in the range
        dates = pd.date_range(temporal.start, temporal.end, freq="1D")
        years = sorted(set(str(d.year) for d in dates))
        months = sorted(set(f"{d.month:02d}" for d in dates))
        days = sorted(set(f"{d.day:02d}" for d in dates))

        # ERA5 native resolution is 0.25°; ensure the bounding box spans at
        # least one grid cell in each direction so MARS doesn't reject the job.
        _MIN_SPAN = 0.25
        north = spatial.max_lat
        south = spatial.min_lat
        west  = spatial.min_lon
        east  = spatial.max_lon
        if (north - south) < _MIN_SPAN:
            mid = (north + south) / 2
            north, south = mid + _MIN_SPAN / 2, mid - _MIN_SPAN / 2
        if (east - west) < _MIN_SPAN:
            mid = (east + west) / 2
            east, west = mid + _MIN_SPAN / 2, mid - _MIN_SPAN / 2

        return {
            "product_type": "reanalysis",
            "variable": sorted(cds_vars),
            "year": years,
            "month": months,
            "day": days,
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": [north, west, south, east],
            "data_format": "netcdf",
        }

    def _retrieve_sync(self, request: dict) -> bytes:
        """
        Synchronous CDS retrieval (runs in executor to avoid blocking event loop).

        Uses the cdsapi library which handles authentication, job submission,
        polling, and download transparently.
        """
        import cdsapi

        client = cdsapi.Client(
            url=self._api_url,
            key=self._api_key,
            quiet=True,
            verify=True,
            retry_max=5,
        )

        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            client.retrieve(_DATASET, request, tmp_path)
            return Path(tmp_path).read_bytes()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def _parse_netcdf(
        self,
        nc_bytes: bytes,
        pse_variables: list[str],
        cds_vars: set[str],
    ) -> xr.Dataset:
        """
        Parse CDS NetCDF bytes and rename/derive variables to PSE conventions.
        """
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            tmp.write(nc_bytes)
            tmp_path = tmp.name

        try:
            raw = xr.open_dataset(tmp_path, engine="netcdf4")
            raw.load()   # force into memory so we can close the file handle
            raw.close()  # release handle before unlink (required on Windows)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        # Rename CDS coordinates to PSE conventions
        rename_map = {}
        if "latitude" not in raw.dims and "lat" in raw.dims:
            rename_map["lat"] = "latitude"
        if "longitude" not in raw.dims and "lon" in raw.dims:
            rename_map["lon"] = "longitude"
        if rename_map:
            raw = raw.rename(rename_map)

        # Build output with PSE variable names
        data_vars = {}
        for pse_v in pse_variables:
            if pse_v in _WIND_SPEED_COMBOS:
                # Derive wind speed from u + v components
                u_var, v_var = _WIND_SPEED_COMBOS[pse_v]
                u_cds = _VAR_MAP[u_var]
                v_cds = _VAR_MAP[v_var]
                # CDS NetCDF short names (e.g. "u10", "v10")
                u_da = self._find_var(raw, u_cds)
                v_da = self._find_var(raw, v_cds)
                if u_da is not None and v_da is not None:
                    speed = np.sqrt(u_da ** 2 + v_da ** 2)
                    data_vars[pse_v] = speed.rename(pse_v)
            else:
                cds_name = _VAR_MAP[pse_v]
                da = self._find_var(raw, cds_name)
                if da is not None:
                    data_vars[pse_v] = da.rename(pse_v)

        return xr.Dataset(data_vars)

    @staticmethod
    def _find_var(ds: xr.Dataset, cds_name: str) -> Optional[xr.DataArray]:
        """
        Find a variable in an xarray Dataset by CDS long name or common short name.
        CDS NetCDF files use short names (e.g. 't2m', 'u10') not the API long names.
        """
        # Direct match first
        if cds_name in ds:
            return ds[cds_name]

        # Match by long_name attribute
        for var in ds.data_vars:
            attrs = ds[var].attrs
            if attrs.get("long_name", "").lower().replace(" ", "_") == cds_name.lower().replace(" ", "_"):
                return ds[var]
            if attrs.get("standard_name", "") == cds_name:
                return ds[var]

        # Known short-name mappings for common ERA5 variables
        _SHORT_NAMES = {
            "2m_temperature":                             "t2m",
            "2m_dewpoint_temperature":                    "d2m",
            "10m_u_component_of_wind":                   "u10",
            "10m_v_component_of_wind":                   "v10",
            "100m_u_component_of_wind":                  "u100",
            "100m_v_component_of_wind":                  "v100",
            "surface_pressure":                           "sp",
            "mean_sea_level_pressure":                    "msl",
            "total_precipitation":                        "tp",
            "surface_solar_radiation_downwards":          "ssrd",
            "total_cloud_cover":                          "tcc",
            "snowfall":                                   "sf",
            "soil_temperature_level_1":                   "stl1",
            "potential_evaporation":                      "pev",
            "boundary_layer_height":                      "blh",
            "instantaneous_10m_wind_gust":                "i10fg",
            "sea_surface_temperature":                    "sst",
            "total_sky_direct_solar_radiation_at_surface": "fdir",
        }
        short = _SHORT_NAMES.get(cds_name)
        if short and short in ds:
            return ds[short]

        log.warning("[era5] Could not find variable '%s' in NetCDF output", cds_name)
        return None
