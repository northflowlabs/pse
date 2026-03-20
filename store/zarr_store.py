"""
PSE Zarr store — cloud-optimised array storage for gridded datasets.

Datasets are written once and read many times.  The store organises data by:

  {root}/{source_id}/{variable}/{YYYY-MM-DD}/

Inside each leaf directory sits a Zarr group produced from an xarray.Dataset.
Supports both local filesystem and S3 backends via fsspec.
"""
from __future__ import annotations

import contextlib
import logging
from datetime import date
from pathlib import Path

import fsspec
import xarray as xr
import zarr

log = logging.getLogger(__name__)


def _resolve_store(path_or_url: str) -> fsspec.spec.AbstractFileSystem:
    """Return an fsspec filesystem + root path for *path_or_url*."""
    if path_or_url.startswith("s3://"):
        import s3fs
        fs = s3fs.S3FileSystem(anon=False)
    else:
        fs = fsspec.filesystem("file")
    return fs


class ZarrStore:
    """
    Thin wrapper around Zarr for storing and retrieving PSE gridded datasets.

    All writes are append-idempotent: writing the same data twice does not
    produce duplicates (Zarr groups are overwritten at the day level).
    """

    def __init__(self, root: str):
        """
        Args:
            root: Local path or S3 URL root, e.g.
                  '/data/zarr'  or  's3://northflow-pse-data/zarr'
        """
        self._root = root.rstrip("/")
        self._fs = _resolve_store(root)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(
        self,
        ds: xr.Dataset,
        source_id: str,
        reference_date: date | None = None,
    ) -> str:
        """
        Persist *ds* to the store.

        Args:
            ds:             Dataset to write (must have a 'time' coordinate).
            source_id:      E.g. 'open_meteo', 'era5'.
            reference_date: Date key for the partition.  Defaults to the first
                            date found in ds['time'].

        Returns:
            The store path where the data was written.
        """
        if reference_date is None:
            first_time = ds["time"].values[0] if "time" in ds.coords else None
            if first_time is not None:
                import pandas as pd
                reference_date = pd.Timestamp(first_time).date()
            else:
                from datetime import date as date_cls
                reference_date = date_cls.today()

        store_path = self._partition_path(source_id, reference_date)
        log.info("Writing Zarr partition: %s", store_path)

        ds.to_zarr(store_path, mode="w", consolidated=True)
        return store_path

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read(
        self,
        source_id: str,
        reference_date: date,
    ) -> xr.Dataset | None:
        """
        Load a stored Dataset for *source_id* on *reference_date*.

        Returns None if the partition does not exist.
        """
        store_path = self._partition_path(source_id, reference_date)
        try:
            ds = xr.open_zarr(store_path, consolidated=True)
            return ds
        except (FileNotFoundError, zarr.errors.GroupNotFoundError, KeyError):
            return None

    def read_range(
        self,
        source_id: str,
        start: date,
        end: date,
    ) -> xr.Dataset | None:
        """
        Load and concatenate daily Zarr partitions from *start* to *end*
        (inclusive).  Missing days are silently skipped.

        Returns None if no partitions are found.
        """
        import pandas as pd

        datasets = []
        for d in pd.date_range(start, end, freq="D"):
            ds = self.read(source_id, d.date())
            if ds is not None:
                datasets.append(ds)

        if not datasets:
            return None

        return xr.concat(datasets, dim="time")

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_dates(self, source_id: str) -> list[date]:
        """Return all available partition dates for *source_id*."""
        source_root = f"{self._root}/{source_id}"
        try:
            entries = self._fs.ls(source_root, detail=False)
        except FileNotFoundError:
            return []

        dates: list[date] = []
        for entry in entries:
            name = Path(entry).name
            with contextlib.suppress(ValueError):
                dates.append(date.fromisoformat(name))
        return sorted(dates)

    def exists(self, source_id: str, reference_date: date) -> bool:
        """Check whether a partition exists without loading it."""
        store_path = self._partition_path(source_id, reference_date)
        return self._fs.exists(store_path)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _partition_path(self, source_id: str, d: date) -> str:
        return f"{self._root}/{source_id}/{d.isoformat()}"
