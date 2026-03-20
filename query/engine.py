"""
PSE Query Engine — orchestrates multi-source data retrieval and caching.

The QueryEngine is the central coordinator:
  1. Receives a query (variables, spatial, temporal, resolution).
  2. Maps each variable to one or more capable connectors.
  3. Checks the cache; fetches from connectors only for cache misses.
  4. Returns a unified xarray.Dataset with full provenance.

This module is intentionally kept simple for Sprint 1 — single-source queries
work end-to-end.  The full multi-source fusion engine (FusionEngine) is
implemented separately in pse/fusion/ and layered on top.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

import xarray as xr

from pse.connectors.base import BaseConnector, SpatialBounds, TemporalBounds
from pse.query.spatial import clip_to_bounds, point_query, regrid_to_resolution
from pse.query.temporal import clip_to_bounds as temporal_clip
from pse.store.cache import PSECache

log = logging.getLogger(__name__)


class QueryEngine:
    """
    High-level query interface for PSE.

    Instantiate once at application startup, passing all available connectors
    and a shared cache instance.

    Example::

        engine = QueryEngine(
            connectors={
                "open_meteo": OpenMeteoConnector(),
                "global_solar_atlas": GlobalSolarAtlasConnector(),
            },
            cache=PSECache(default_ttl=3600),
        )

        ds = await engine.query(
            variables=["temperature_2m", "solar_ghi"],
            spatial=SpatialBounds(-6.3, -6.1, 106.7, 106.9),
            temporal=TemporalBounds(datetime(2025,1,1), datetime(2025,1,7)),
        )
    """

    def __init__(
        self,
        connectors: dict[str, BaseConnector],
        cache: Optional[PSECache] = None,
    ):
        self._connectors = connectors
        self._cache = cache or PSECache()

        # Build reverse index: variable → list[connector]
        self._var_index: dict[str, list[BaseConnector]] = {}
        for connector in connectors.values():
            for var in connector.variables:
                self._var_index.setdefault(var, []).append(connector)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def query(
        self,
        variables: list[str],
        spatial: SpatialBounds,
        temporal: TemporalBounds,
        resolution_m: Optional[float] = None,
    ) -> xr.Dataset:
        """
        Retrieve data for the requested variables, region, and time range.

        Variables are routed to the best available connector (first that
        declares the variable).  Cache is checked before fetching.

        Args:
            variables:    PSE canonical variable names.
            spatial:      Bounding box.
            temporal:     Time range.
            resolution_m: Target grid resolution in metres.

        Returns:
            Merged xarray.Dataset covering all requested variables.

        Raises:
            ValueError: If any variable has no registered connector.
        """
        # Group variables by connector (pick the first capable one for now)
        connector_map: dict[str, list[str]] = {}  # connector source_id → [vars]
        missing = []
        for var in variables:
            candidates = self._var_index.get(var, [])
            if not candidates:
                missing.append(var)
                continue
            chosen = candidates[0]  # TODO Sprint 2: use quality-weighted selection
            connector_map.setdefault(chosen.source_id, []).append(var)

        if missing:
            raise ValueError(
                f"No connector available for variable(s): {missing}. "
                f"Available variables: {sorted(self._var_index)}"
            )

        # Fetch from each connector (cache-aware, in parallel)
        fetch_tasks = [
            self._fetch_with_cache(
                connector=self._connectors[src_id],
                variables=vars_,
                spatial=spatial,
                temporal=temporal,
                resolution=resolution_m,
            )
            for src_id, vars_ in connector_map.items()
        ]
        datasets = await asyncio.gather(*fetch_tasks)

        # Merge all per-connector datasets into one
        if len(datasets) == 1:
            return datasets[0]

        return xr.merge(datasets, join="outer")

    async def point_query(
        self,
        lat: float,
        lon: float,
        variables: list[str],
        temporal: TemporalBounds,
    ) -> xr.Dataset:
        """
        Retrieve a timeseries at a single geographic point.

        Internally queries a small bounding box (0.2° × 0.2°) and extracts
        the nearest grid cell to *lat*, *lon*.
        """
        # Use a small bounding box centred on the point
        delta = 0.1
        spatial = SpatialBounds(
            min_lat=lat - delta,
            max_lat=lat + delta,
            min_lon=lon - delta,
            max_lon=lon + delta,
        )
        ds = await self.query(variables, spatial, temporal)
        return point_query(ds, lat, lon, method="nearest")

    def available_variables(self) -> dict[str, list[str]]:
        """Return a map of variable → list of source_ids that provide it."""
        return {
            var: [c.source_id for c in connectors]
            for var, connectors in self._var_index.items()
        }

    def connector_status(self) -> dict[str, dict]:
        """Return a summary of all registered connectors."""
        return {
            src_id: {
                "source_id": c.source_id,
                "variables": c.variables,
                "update_frequency_seconds": c.update_frequency_seconds,
            }
            for src_id, c in self._connectors.items()
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _fetch_with_cache(
        self,
        connector: BaseConnector,
        variables: list[str],
        spatial: SpatialBounds,
        temporal: TemporalBounds,
        resolution: Optional[float],
    ) -> xr.Dataset:
        key = PSECache.build_key(
            connector.source_id, variables, spatial, temporal, resolution
        )
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        log.info(
            "Fetching %s from %s (spatial=%s, temporal=%s)",
            variables,
            connector.source_id,
            spatial.to_dict(),
            temporal.to_dict(),
        )
        ds = await connector.fetch(variables, spatial, temporal, resolution)

        # Cache with TTL = connector's update frequency
        self._cache.put(key, ds, ttl=float(connector.update_frequency_seconds))
        return ds
