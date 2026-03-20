"""
PSE Fusion Engine — multi-source spatiotemporal data fusion.

The FusionEngine is the heart of PSE.  When a query spans multiple connectors
it:

  1. Identifies which connectors provide which variables.
  2. Fetches data from each connector in parallel (cache-aware via QueryEngine).
  3. Aligns all datasets to a common spatial grid (regular lat/lon at the
     requested resolution).
  4. Aligns all datasets to a common time axis.
  5. Resolves conflicts using quality-weighted blending.
  6. Returns a single unified xarray.Dataset with full provenance.

The FusionEngine wraps the QueryEngine — it adds the alignment and merge
steps on top of the QueryEngine's per-source fetch logic.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime

import xarray as xr

from pse.connectors.base import SpatialBounds, TemporalBounds
from pse.fusion.grid import make_regular_grid, regrid_dataset
from pse.fusion.quality import compute_weights, detect_conflicts, quality_weighted_merge
from pse.fusion.temporal import align_to_common_axis
from pse.query.engine import QueryEngine

log = logging.getLogger(__name__)

_PSE_VERSION = "0.2.0"


class FusionEngine:
    """
    Multi-source spatiotemporal data fusion engine.

    Usage::

        fusion = FusionEngine(query_engine)

        ds = await fusion.query(
            variables=["temperature_2m", "solar_ghi", "substation_count"],
            spatial=SpatialBounds(-6.3, -6.1, 106.7, 106.9),
            temporal=TemporalBounds(datetime(2025, 1, 1), datetime(2025, 1, 7)),
            resolution_m=5000,
        )
    """

    def __init__(
        self,
        query_engine: QueryEngine,
        prefer_recent: bool = True,
        conflict_threshold: float = 0.3,
    ):
        self._qe = query_engine
        self._prefer_recent = prefer_recent
        self._conflict_threshold = conflict_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def query(
        self,
        variables: list[str],
        spatial: SpatialBounds,
        temporal: TemporalBounds,
        resolution_m: float = 5_000.0,
        prefer_recent: bool | None = None,
    ) -> xr.Dataset:
        """
        Execute a fused multi-source query.

        Args:
            variables:    PSE canonical variable names.
            spatial:      Geographic bounding box.
            temporal:     Time range.
            resolution_m: Target spatial resolution in metres.
            prefer_recent: Override the engine-level prefer_recent setting.

        Returns:
            Fused xarray.Dataset with provenance metadata.
        """
        prefer_recent = prefer_recent if prefer_recent is not None else self._prefer_recent

        # --- 1. Group variables by connector ---
        connector_map: dict[str, list[str]] = {}  # source_id → [vars]
        missing = []
        for var in variables:
            candidates = self._qe._var_index.get(var, [])
            if not candidates:
                missing.append(var)
                continue
            # All connectors that provide this variable (not just the first)
            for c in candidates:
                connector_map.setdefault(c.source_id, [])
                if var not in connector_map[c.source_id]:
                    connector_map[c.source_id].append(var)

        if missing:
            raise ValueError(
                f"No connector available for: {missing}. "
                f"Registered variables: {sorted(self._qe._var_index)}"
            )

        log.info(
            "[fusion] Query: vars=%s, sources=%s, resolution=%dm",
            variables, list(connector_map), int(resolution_m)
        )

        # --- 2. Fetch from each connector in parallel (cache-aware) ---
        fetch_tasks = [
            self._qe._fetch_with_cache(
                connector=self._qe._connectors[src_id],
                variables=vars_,
                spatial=spatial,
                temporal=temporal,
                resolution=resolution_m,
            )
            for src_id, vars_ in connector_map.items()
        ]
        raw_datasets: list[xr.Dataset | Exception] = await asyncio.gather(
            *fetch_tasks, return_exceptions=True
        )

        # Collect successful fetches and their metadata
        valid_datasets: list[xr.Dataset] = []
        quality_scores: list[float] = []
        recency_scores: list[float] = []
        sources_used: list[str] = []
        errors: dict[str, str] = {}

        for src_id, result in zip(connector_map.keys(), raw_datasets, strict=False):
            if isinstance(result, Exception):
                log.warning("[fusion] Connector '%s' failed: %s", src_id, result)
                errors[src_id] = str(result)
                continue

            valid_datasets.append(result)
            sources_used.append(src_id)

            # Fetch quality score for weighting
            try:
                connector = self._qe._connectors[src_id]
                quality = await connector.get_quality(spatial, temporal)
                quality_scores.append(quality.overall_score)
                # Recency: invert lag (0 lag = 1.0, 7-day lag = 0.0)
                recency = max(0.0, 1.0 - quality.temporal_lag / (7 * 86_400))
                recency_scores.append(recency)
            except Exception:
                quality_scores.append(0.5)
                recency_scores.append(0.5)

        if not valid_datasets:
            raise RuntimeError(
                f"All connectors failed. Errors: {errors}"
            )

        if len(valid_datasets) == 1:
            fused = valid_datasets[0]
        else:
            # --- 3. Spatial alignment ---
            target_lats, target_lons = make_regular_grid(spatial, resolution_m)
            spatially_aligned = [
                regrid_dataset(ds, target_lats, target_lons)
                for ds in valid_datasets
            ]

            # --- 4. Temporal alignment ---
            import pandas as pd
            t_start = pd.Timestamp(temporal.start)
            t_end = pd.Timestamp(temporal.end)
            temporally_aligned = align_to_common_axis(
                spatially_aligned,
                temporal_start=t_start,
                temporal_end=t_end,
            )

            # --- 5. Detect conflicts (log only, don't raise) ---
            shared_vars = set(variables)
            for ds in temporally_aligned[1:]:
                shared_vars &= set(ds.data_vars)
            for var in shared_vars:
                conflict = detect_conflicts(temporally_aligned, var, self._conflict_threshold)
                if conflict.get("conflict"):
                    log.warning("[fusion] Conflict detected for '%s': %s", var, conflict)

            # --- 6. Quality-weighted merge ---
            weights = compute_weights(quality_scores, prefer_recent, recency_scores)
            fused = quality_weighted_merge(temporally_aligned, weights)

        # --- 7. Provenance ---
        fused.attrs.update(self._build_provenance(
            variables, sources_used, quality_scores, errors, spatial, temporal, resolution_m
        ))

        return fused

    async def point_query(
        self,
        lat: float,
        lon: float,
        variables: list[str],
        temporal: TemporalBounds,
        resolution_m: float = 1_000.0,
    ) -> xr.Dataset:
        """Fused query at a single point (small bbox + nearest-cell extraction)."""
        from pse.query.spatial import point_query as extract_point
        delta = 0.1
        spatial = SpatialBounds(
            min_lat=lat - delta, max_lat=lat + delta,
            min_lon=lon - delta, max_lon=lon + delta,
        )
        ds = await self.query(variables, spatial, temporal, resolution_m)
        return extract_point(ds, lat, lon, method="nearest")

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _build_provenance(
        self,
        variables: list[str],
        sources_used: list[str],
        quality_scores: list[float],
        errors: dict,
        spatial: SpatialBounds,
        temporal: TemporalBounds,
        resolution_m: float,
    ) -> dict:
        return {
            "pse_version": _PSE_VERSION,
            "pse_fusion": True,
            "pse_query_time": datetime.now(UTC).isoformat(),
            "pse_sources_used": sources_used,
            "pse_source_quality_scores": {
                src: round(q, 4)
                for src, q in zip(sources_used, quality_scores, strict=False)
            },
            "pse_variables": variables,
            "pse_spatial": spatial.to_dict(),
            "pse_temporal": temporal.to_dict(),
            "pse_resolution_m": resolution_m,
            "pse_errors": errors if errors else None,
        }
