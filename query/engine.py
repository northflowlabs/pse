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

import xarray as xr

from pse.connectors.base import BaseConnector, SpatialBounds, TemporalBounds
from pse.query.spatial import point_query
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
        cache: PSECache | None = None,
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
        resolution_m: float | None = None,
    ) -> xr.Dataset:
        """
        Retrieve data for the requested variables, region, and time range.

        Variables are routed to the primary capable connector first.  If a
        connector fails, the engine falls back to the next available connector
        for each affected variable, only raising if ALL connectors fail for a
        given variable.

        Args:
            variables:    PSE canonical variable names.
            spatial:      Bounding box.
            temporal:     Time range.
            resolution_m: Target grid resolution in metres.

        Returns:
            Merged xarray.Dataset covering all requested variables.
            ``ds.attrs["pse_connector_results"]`` contains per-variable
            success/failure metadata.

        Raises:
            ValueError: If any variable has no registered connector.
            RuntimeError: If all connectors fail for any variable.
        """
        # Build per-variable candidate lists and primary-connector groups
        var_candidates: dict[str, list[BaseConnector]] = {}
        primary_groups: dict[str, list[str]] = {}  # source_id → [vars]
        missing: list[str] = []

        for var in variables:
            candidates = self._var_index.get(var, [])
            if not candidates:
                missing.append(var)
                continue
            var_candidates[var] = candidates
            primary_groups.setdefault(candidates[0].source_id, []).append(var)

        if missing:
            raise ValueError(
                f"No connector available for variable(s): {missing}. "
                f"Available variables: {sorted(self._var_index)}"
            )

        # Phase 1 — try each primary-connector group in parallel
        async def _try_group(
            src_id: str, vars_: list[str]
        ) -> tuple[str, list[str], xr.Dataset | None, Exception | None]:
            try:
                ds = await self._fetch_with_cache(
                    self._connectors[src_id], vars_, spatial, temporal, resolution_m
                )
                return src_id, vars_, ds, None
            except Exception as exc:  # noqa: BLE001
                return src_id, vars_, None, exc

        group_results = await asyncio.gather(
            *[_try_group(sid, vrs) for sid, vrs in primary_groups.items()]
        )

        # Phase 2 — collect successes; queue failed variables for fallback
        datasets: list[xr.Dataset] = []
        connector_succeeded: dict[str, str] = {}        # var → source_id
        connector_failed: dict[str, list[str]] = {}     # var → [source_ids]
        needs_fallback: list[str] = []

        for src_id, vars_, ds, err in group_results:
            if err is None:
                datasets.append(ds)
                for v in vars_:
                    connector_succeeded[v] = src_id
            else:
                log.warning(
                    "Connector %s failed for %s — will try alternatives. Error: %s",
                    src_id, vars_, err,
                )
                for v in vars_:
                    connector_failed.setdefault(v, []).append(src_id)
                    needs_fallback.append(v)

        # Phase 3 — sequential per-variable fallback for failed groups
        for var in needs_fallback:
            already_tried = set(connector_failed.get(var, []))
            alternatives = [
                c for c in var_candidates[var]
                if c.source_id not in already_tried
            ]
            succeeded = False
            for connector in alternatives:
                try:
                    ds = await self._fetch_with_cache(
                        connector, [var], spatial, temporal, resolution_m
                    )
                    datasets.append(ds)
                    connector_succeeded[var] = connector.source_id
                    log.info(
                        "Variable '%s' succeeded with fallback connector %s",
                        var, connector.source_id,
                    )
                    succeeded = True
                    break
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "Fallback connector %s also failed for '%s': %s",
                        connector.source_id, var, exc,
                    )
                    connector_failed.setdefault(var, []).append(connector.source_id)

            if not succeeded:
                raise RuntimeError(
                    f"All connectors failed for variable '{var}'. "
                    f"Tried: {connector_failed.get(var, [])}"
                )

        # Merge all per-connector datasets
        merged = datasets[0] if len(datasets) == 1 else xr.merge(datasets, join="outer")

        # Attach connector provenance metadata
        merged.attrs["pse_connector_results"] = {
            "succeeded": connector_succeeded,
            "failed": connector_failed,
        }
        return merged

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
        resolution: float | None,
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
