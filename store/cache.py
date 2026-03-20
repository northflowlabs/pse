"""
PSE in-memory cache layer — TTL-aware caching for connector fetch results.

Sits between the query engine and connectors to avoid redundant API calls.
Cache keys encode: source_id + variables + spatial bounds + temporal bounds +
resolution.

This is a lightweight process-local cache.  For multi-process deployments,
replace with Redis via an identical interface.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Optional

import xarray as xr

from pse.connectors.base import SpatialBounds, TemporalBounds

log = logging.getLogger(__name__)


@dataclass
class _CacheEntry:
    dataset: xr.Dataset
    stored_at: float    # time.monotonic()
    ttl: float          # seconds


class PSECache:
    """
    Simple in-process TTL cache for xarray.Dataset objects.

    Not thread-safe in the traditional sense — but since the PSE API runs
    in a single asyncio event loop, concurrent coroutines won't race on dict
    mutations (GIL + cooperative scheduling).
    """

    def __init__(self, default_ttl: float = 3600.0, max_entries: int = 256):
        """
        Args:
            default_ttl:  Default time-to-live in seconds.
            max_entries:  Maximum number of entries before LRU eviction.
        """
        self._default_ttl = default_ttl
        self._max_entries = max_entries
        self._store: dict[str, _CacheEntry] = {}
        self._access_order: list[str] = []   # LRU tracking (front = oldest)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[xr.Dataset]:
        """Return cached dataset, or None if missing / expired."""
        entry = self._store.get(key)
        if entry is None:
            return None
        if time.monotonic() - entry.stored_at > entry.ttl:
            self._evict(key)
            return None
        # Refresh LRU position
        self._touch(key)
        log.debug("Cache hit: %s", key[:48])
        return entry.dataset

    def put(
        self,
        key: str,
        dataset: xr.Dataset,
        ttl: Optional[float] = None,
    ) -> None:
        """Insert or replace a cache entry."""
        if len(self._store) >= self._max_entries and key not in self._store:
            self._evict_lru()

        self._store[key] = _CacheEntry(
            dataset=dataset,
            stored_at=time.monotonic(),
            ttl=ttl if ttl is not None else self._default_ttl,
        )
        self._touch(key)
        log.debug("Cache put: %s", key[:48])

    def invalidate(self, source_id: str) -> int:
        """Remove all entries associated with a given source_id.  Returns count."""
        keys_to_remove = [k for k in self._store if k.startswith(source_id + ":")]
        for k in keys_to_remove:
            self._evict(k)
        log.info("Cache invalidated %d entries for source '%s'", len(keys_to_remove), source_id)
        return len(keys_to_remove)

    def clear(self) -> None:
        """Flush all entries."""
        self._store.clear()
        self._access_order.clear()

    @property
    def size(self) -> int:
        return len(self._store)

    def stats(self) -> dict:
        now = time.monotonic()
        return {
            "entries": self.size,
            "max_entries": self._max_entries,
            "default_ttl": self._default_ttl,
            "expired": sum(
                1 for e in self._store.values()
                if now - e.stored_at > e.ttl
            ),
        }

    # ------------------------------------------------------------------
    # Key building
    # ------------------------------------------------------------------

    @staticmethod
    def build_key(
        source_id: str,
        variables: list[str],
        spatial: SpatialBounds,
        temporal: TemporalBounds,
        resolution: Optional[float] = None,
    ) -> str:
        """
        Build a deterministic cache key from query parameters.
        Uses a short SHA-256 digest to keep keys compact.
        """
        payload = {
            "source": source_id,
            "vars": sorted(variables),
            "spatial": spatial.to_dict(),
            "temporal": temporal.to_dict(),
            "resolution": resolution,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()[:20]
        return f"{source_id}:{digest}"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _touch(self, key: str) -> None:
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

    def _evict(self, key: str) -> None:
        self._store.pop(key, None)
        if key in self._access_order:
            self._access_order.remove(key)

    def _evict_lru(self) -> None:
        if self._access_order:
            oldest = self._access_order[0]
            log.debug("Cache LRU evict: %s", oldest[:48])
            self._evict(oldest)
