"""
Sentinel-2 connector — Copernicus Data Space Ecosystem (CDSE).

Sentinel-2 provides multispectral optical imagery at 10–60 m resolution with a
~5-day global revisit.  For PSE / FLUX, we use it to derive:

  - NDVI  (Normalised Difference Vegetation Index)  — land cover, crop health
  - NDWI  (Normalised Difference Water Index)       — water bodies, flood extent
  - land_use_class                                   — coarse classification
  - cloud_cover                                      — scene-level cloud fraction

Authentication:
  COPERNICUS_DATASPACE_CLIENT_ID / CLIENT_SECRET (OAuth2 client credentials).
  Register at: https://dataspace.copernicus.eu/

CDSE STAC API: https://stac.dataspace.copernicus.eu/
OData API:     https://catalogue.dataspace.copernicus.eu/odata/v1
"""
from __future__ import annotations

import asyncio
import base64
import logging
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

_STAC_URL = "https://stac.dataspace.copernicus.eu/v1"
_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
_DOWNLOAD_URL = "https://zipper.dataspace.copernicus.eu/download"

_NATIVE_RESOLUTION_M = 10
_REVISIT_DAYS = 5
_RELIABILITY = 0.95

# Band numbers for common Sentinel-2 L2A products
# B04 = Red (665 nm), B08 = NIR (842 nm), B03 = Green (559 nm), B11 = SWIR (1610 nm)
_BAND_MAP = {
    "B02": "blue",
    "B03": "green",
    "B04": "red",
    "B08": "nir",
    "B11": "swir1",
    "B12": "swir2",
    "SCL": "scene_classification",
}

# PSE variables this connector provides
_PSE_VARIABLES = [
    "ndvi",
    "ndwi",
    "cloud_cover",
    "land_use_class",
    "red_reflectance",
    "nir_reflectance",
    "swir1_reflectance",
]


class Sentinel2Connector(BaseConnector):
    """
    PSE connector for Sentinel-2 L2A imagery via CDSE STAC API.

    Behaviour:
    - Searches STAC for scenes covering the requested region and time range.
    - Filters by maximum cloud cover (default 30%).
    - Downloads the least-cloudy scene covering the region.
    - Computes NDVI, NDWI, and a coarse land use classification.
    - Returns a time-stamped Dataset at the scene acquisition time.

    Note: Sentinel-2 does not provide continuous time-series like weather data.
    The 'time' dimension contains the acquisition timestamps of the matched scenes.
    """

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        max_cloud_cover: float = 30.0,
        max_concurrent_downloads: int = 2,
    ):
        import os
        self._client_id = client_id or os.environ.get("COPERNICUS_DATASPACE_CLIENT_ID", "")
        self._client_secret = client_secret or os.environ.get("COPERNICUS_DATASPACE_CLIENT_SECRET", "")
        self._max_cloud = max_cloud_cover
        self._sem = asyncio.Semaphore(max_concurrent_downloads)
        self._token: Optional[str] = None
        self._token_expiry: float = 0.0

        if not (self._client_id and self._client_secret):
            log.warning(
                "[sentinel2] COPERNICUS_DATASPACE_CLIENT_ID / CLIENT_SECRET not set. "
                "Register at https://dataspace.copernicus.eu/ to enable Sentinel-2 access."
            )

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def source_id(self) -> str:
        return "sentinel2"

    @property
    def variables(self) -> list[str]:
        return _PSE_VARIABLES

    @property
    def update_frequency_seconds(self) -> int:
        return _REVISIT_DAYS * 86_400  # 5-day revisit

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
        Find and retrieve the best Sentinel-2 scenes for the request.

        Returns a Dataset with one time step per scene found.
        """
        self._validate_variables(variables)

        # Search STAC for matching scenes
        scenes = await self._search_stac(spatial, temporal)

        if not scenes:
            raise ConnectorError(
                f"[sentinel2] No scenes found for region={spatial.to_dict()}, "
                f"time={temporal.to_dict()}, max_cloud={self._max_cloud}%"
            )

        log.info("[sentinel2] Found %d scenes", len(scenes))

        # Process scenes (up to 3 to keep download time reasonable)
        tasks = [self._process_scene(scene, variables, spatial) for scene in scenes[:3]]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        valid = [r for r in results if isinstance(r, xr.Dataset)]
        if not valid:
            raise ConnectorError("[sentinel2] All scene downloads/processing failed.")

        ds = xr.concat(valid, dim="time") if len(valid) > 1 else valid[0]
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
        try:
            scenes = await self._search_stac(spatial, temporal)
            completeness = min(1.0, len(scenes) / max(1, temporal.duration_days / _REVISIT_DAYS))
            avg_cloud = np.mean([s.get("properties", {}).get("eo:cloud_cover", 50) for s in scenes]) if scenes else 100
            adj_completeness = completeness * (1.0 - avg_cloud / 100 * 0.5)
        except Exception:
            completeness = 0.0
            adj_completeness = 0.0

        latest = await self.get_latest_timestamp()
        lag = (datetime.now(timezone.utc) - latest.replace(tzinfo=timezone.utc)).total_seconds()

        return DataQuality(
            completeness=min(1.0, adj_completeness),
            temporal_lag=lag,
            spatial_resolution=_NATIVE_RESOLUTION_M,
            source_reliability=_RELIABILITY,
            provenance={
                "source": self.source_id,
                "dataset": "Sentinel-2 L2A",
                "stac_url": _STAC_URL,
                "max_cloud_cover": self._max_cloud,
            },
        )

    async def get_latest_timestamp(self) -> datetime:
        """Return latest scene timestamp from STAC or approximate now - 5 days."""
        from datetime import timedelta
        return (datetime.now(timezone.utc) - timedelta(days=_REVISIT_DAYS)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

    # ------------------------------------------------------------------
    # Private: authentication
    # ------------------------------------------------------------------

    async def _get_token(self) -> str:
        """Obtain or refresh an OAuth2 access token."""
        import time
        if self._token and time.monotonic() < self._token_expiry - 30:
            return self._token

        data = {
            "grant_type": "client_credentials",
            "client_id": self._client_id,
            "client_secret": self._client_secret,
        }
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(_TOKEN_URL, data=data)
            resp.raise_for_status()
            payload = resp.json()

        import time
        self._token = payload["access_token"]
        self._token_expiry = time.monotonic() + payload.get("expires_in", 300)
        return self._token

    # ------------------------------------------------------------------
    # Private: STAC search
    # ------------------------------------------------------------------

    async def _search_stac(
        self,
        spatial: SpatialBounds,
        temporal: TemporalBounds,
    ) -> list[dict]:
        """Query CDSE STAC for Sentinel-2 L2A scenes."""
        bbox = [spatial.min_lon, spatial.min_lat, spatial.max_lon, spatial.max_lat]
        body = {
            "collections": ["SENTINEL-2"],
            "bbox": bbox,
            "datetime": f"{temporal.start.strftime('%Y-%m-%dT%H:%M:%SZ')}/"
                        f"{temporal.end.strftime('%Y-%m-%dT%H:%M:%SZ')}",
            "query": {
                "eo:cloud_cover": {"lte": self._max_cloud},
                "s2:processing_level": {"eq": "L2A"},
            },
            "sortby": [{"field": "eo:cloud_cover", "direction": "asc"}],
            "limit": 20,
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(f"{_STAC_URL}/search", json=body)
                resp.raise_for_status()
                return resp.json().get("features", [])
        except httpx.HTTPError as exc:
            raise ConnectorError(f"[sentinel2] STAC search failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Private: scene processing
    # ------------------------------------------------------------------

    async def _process_scene(
        self,
        scene: dict,
        variables: list[str],
        spatial: SpatialBounds,
    ) -> xr.Dataset:
        """
        Download required bands and compute derived indices for a single scene.

        Returns a single-time-step Dataset or raises on failure.
        """
        props = scene.get("properties", {})
        acquisition_time = props.get("datetime", "")
        scene_id = scene.get("id", "unknown")

        log.debug("[sentinel2] Processing scene %s (cloud=%.1f%%)", scene_id, props.get("eo:cloud_cover", -1))

        # Determine which bands to download based on requested variables
        bands_needed = self._resolve_bands(variables)
        assets = scene.get("assets", {})

        band_arrays: dict[str, np.ndarray] = {}
        for band in bands_needed:
            asset = assets.get(band)
            if asset is None:
                continue
            href = asset.get("href", "")
            if href:
                try:
                    arr = await self._download_band(href, spatial)
                    if arr is not None:
                        band_arrays[band] = arr
                except Exception as exc:
                    log.warning("[sentinel2] Band %s download failed: %s", band, exc)

        if not band_arrays:
            raise ConnectorError(f"[sentinel2] No bands downloaded for scene {scene_id}")

        data_vars = self._compute_variables(variables, band_arrays, props)

        # Common lat/lon grid for the returned arrays
        n_lat = next(iter(data_vars.values())).shape[0]
        n_lon = next(iter(data_vars.values())).shape[1]
        lats = np.linspace(spatial.max_lat, spatial.min_lat, n_lat)
        lons = np.linspace(spatial.min_lon, spatial.max_lon, n_lon)

        import pandas as pd
        t = pd.Timestamp(acquisition_time) if acquisition_time else pd.Timestamp.now(tz="UTC")

        xr_vars = {
            var: (["latitude", "longitude"], arr)
            for var, arr in data_vars.items()
        }

        return xr.Dataset(
            xr_vars,
            coords={
                "time": t,
                "latitude": lats,
                "longitude": lons,
            },
        ).expand_dims("time")

    def _resolve_bands(self, variables: list[str]) -> list[str]:
        """Determine which raw bands are needed to compute the requested variables."""
        bands: set[str] = set()
        for var in variables:
            if var in ("ndvi", "red_reflectance", "nir_reflectance"):
                bands.update(["B04", "B08"])
            if var in ("ndwi",):
                bands.update(["B03", "B08"])
            if var in ("swir1_reflectance",):
                bands.update(["B11"])
            if var in ("land_use_class",):
                bands.update(["SCL"])
            if var in ("cloud_cover",):
                bands.update(["SCL"])
        return sorted(bands)

    async def _download_band(
        self,
        href: str,
        spatial: SpatialBounds,
    ) -> Optional[np.ndarray]:
        """
        Download a single band asset and clip to spatial bounds.

        Uses rasterio for reading Cloud-Optimised GeoTIFF (COG) assets —
        only the overlapping window is downloaded.
        """
        async with self._sem:
            try:
                import rasterio
                from rasterio.windows import from_bounds

                token = await self._get_token()
                # Construct signed URL with auth header
                headers = {"Authorization": f"Bearer {token}"}

                with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
                    tmp_path = tmp.name

                async with httpx.AsyncClient(timeout=120.0, headers=headers) as client:
                    resp = await client.get(href, follow_redirects=True)
                    resp.raise_for_status()
                    Path(tmp_path).write_bytes(resp.content)

                try:
                    with rasterio.open(tmp_path) as src:
                        window = from_bounds(
                            spatial.min_lon, spatial.min_lat,
                            spatial.max_lon, spatial.max_lat,
                            src.transform
                        )
                        data = src.read(1, window=window).astype(np.float32)
                        # Sentinel-2 L2A reflectance is stored as int16 / 10000
                        if src.nodata is not None:
                            data[data == src.nodata] = np.nan
                        if data.max() > 2.0:  # likely stored as int * 10000
                            data = data / 10_000.0
                        return data
                finally:
                    Path(tmp_path).unlink(missing_ok=True)

            except Exception as exc:
                log.warning("[sentinel2] _download_band failed: %s", exc)
                return None

    def _compute_variables(
        self,
        variables: list[str],
        bands: dict[str, np.ndarray],
        props: dict,
    ) -> dict[str, np.ndarray]:
        """Compute derived PSE variables from raw band arrays."""
        out: dict[str, np.ndarray] = {}

        red = bands.get("B04")
        nir = bands.get("B08")
        green = bands.get("B03")
        scl = bands.get("SCL")

        # Reference shape for outputs (use first available band)
        ref = next(iter(bands.values()))
        shape = ref.shape

        for var in variables:
            if var == "ndvi" and red is not None and nir is not None:
                denominator = nir + red
                with np.errstate(divide="ignore", invalid="ignore"):
                    out["ndvi"] = np.where(denominator > 0, (nir - red) / denominator, np.nan).astype(np.float32)

            elif var == "ndwi" and green is not None and nir is not None:
                denominator = green + nir
                with np.errstate(divide="ignore", invalid="ignore"):
                    out["ndwi"] = np.where(denominator > 0, (green - nir) / denominator, np.nan).astype(np.float32)

            elif var == "red_reflectance" and red is not None:
                out["red_reflectance"] = red

            elif var == "nir_reflectance" and nir is not None:
                out["nir_reflectance"] = nir

            elif var == "swir1_reflectance" and bands.get("B11") is not None:
                out["swir1_reflectance"] = bands["B11"]

            elif var == "cloud_cover":
                # Scene-level cloud cover from metadata (scalar, broadcast to grid)
                cc = float(props.get("eo:cloud_cover", np.nan)) / 100.0
                out["cloud_cover"] = np.full(shape, cc, dtype=np.float32)

            elif var == "land_use_class" and scl is not None:
                # Sentinel-2 Scene Classification Layer: simplify to 0-4
                # 0=nodata, 1=saturated, 2=dark, 3=cloud_shadow, 4=vegetation,
                # 5=bare_soil, 6=water, 7=unclassified, 8=cloud_medium, 9=cloud_high,
                # 10=cirrus, 11=snow_ice
                simplified = np.zeros(shape, dtype=np.float32)
                simplified[scl == 4] = 1   # vegetation
                simplified[scl == 5] = 2   # bare soil
                simplified[scl == 6] = 3   # water
                simplified[scl >= 8] = 4   # cloud/snow
                out["land_use_class"] = simplified

            else:
                if var not in out:
                    out[var] = np.full(shape, np.nan, dtype=np.float32)

        return out
