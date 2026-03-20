"""
PSE Base Connector — abstract interface all data source connectors must implement.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime

import xarray as xr


@dataclass
class SpatialBounds:
    """Geographic bounding box (WGS84 / EPSG:4326)."""
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float

    def __post_init__(self):
        if self.min_lat >= self.max_lat:
            raise ValueError(f"min_lat ({self.min_lat}) must be < max_lat ({self.max_lat})")
        if self.min_lon >= self.max_lon:
            raise ValueError(f"min_lon ({self.min_lon}) must be < max_lon ({self.max_lon})")
        if not (-90 <= self.min_lat <= 90 and -90 <= self.max_lat <= 90):
            raise ValueError("Latitudes must be in [-90, 90]")
        if not (-180 <= self.min_lon <= 180 and -180 <= self.max_lon <= 180):
            raise ValueError("Longitudes must be in [-180, 180]")

    @property
    def center_lat(self) -> float:
        return (self.min_lat + self.max_lat) / 2

    @property
    def center_lon(self) -> float:
        return (self.min_lon + self.max_lon) / 2

    def to_dict(self) -> dict:
        return {
            "min_lat": self.min_lat,
            "max_lat": self.max_lat,
            "min_lon": self.min_lon,
            "max_lon": self.max_lon,
        }


@dataclass
class TemporalBounds:
    """Time range for a query."""
    start: datetime
    end: datetime

    def __post_init__(self):
        if self.start >= self.end:
            raise ValueError(f"start ({self.start}) must be before end ({self.end})")

    @property
    def duration_days(self) -> float:
        return (self.end - self.start).total_seconds() / 86400

    def to_dict(self) -> dict:
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
        }


@dataclass
class DataQuality:
    """Quality metadata for a connector's data over a specific region/period."""
    completeness: float          # 0–1, fraction of non-null values
    temporal_lag: float          # seconds since last update
    spatial_resolution: float    # native resolution in metres
    source_reliability: float    # 0–1, historical reliability score
    provenance: dict = field(default_factory=dict)  # full provenance metadata

    def __post_init__(self):
        for name, val in [
            ("completeness", self.completeness),
            ("source_reliability", self.source_reliability),
        ]:
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {val}")

    @property
    def overall_score(self) -> float:
        """Simple composite quality score 0–1."""
        recency_score = max(0.0, 1.0 - self.temporal_lag / (7 * 86400))  # decays over a week
        return (self.completeness * 0.4 + self.source_reliability * 0.4 + recency_score * 0.2)

    def to_dict(self) -> dict:
        return {
            "completeness": self.completeness,
            "temporal_lag_seconds": self.temporal_lag,
            "spatial_resolution_m": self.spatial_resolution,
            "source_reliability": self.source_reliability,
            "overall_score": self.overall_score,
            "provenance": self.provenance,
        }


class ConnectorError(Exception):
    """Raised when a connector fails to fetch or assess data."""


class BaseConnector(ABC):
    """Abstract base class for all PSE data source connectors.

    Subclasses must implement:
      - source_id (property)
      - variables (property)
      - update_frequency_seconds (property)
      - fetch()
      - get_quality()
      - get_latest_timestamp()
    """

    # ------------------------------------------------------------------ #
    # Identity                                                             #
    # ------------------------------------------------------------------ #

    @property
    @abstractmethod
    def source_id(self) -> str:
        """Unique identifier for this data source (e.g. 'open_meteo', 'era5')."""

    @property
    @abstractmethod
    def variables(self) -> list[str]:
        """All variable names this connector can provide."""

    @property
    @abstractmethod
    def update_frequency_seconds(self) -> int:
        """How often the upstream source publishes new data (seconds)."""

    # ------------------------------------------------------------------ #
    # Core operations                                                      #
    # ------------------------------------------------------------------ #

    @abstractmethod
    async def fetch(
        self,
        variables: list[str],
        spatial: SpatialBounds,
        temporal: TemporalBounds,
        resolution: float | None = None,
    ) -> xr.Dataset:
        """Fetch data from this source.

        Args:
            variables: Subset of self.variables to retrieve.
            spatial:   Geographic bounding box.
            temporal:  Time range.
            resolution: Target spatial resolution in metres (connector may
                        return its native resolution if reprojection is not
                        supported).

        Returns:
            xarray.Dataset with:
              - Coordinates: latitude, longitude, time
              - Data variables: as requested (units in CF conventions)
              - Attributes:
                  pse_source      — self.source_id
                  pse_fetched_at  — UTC ISO timestamp of this fetch
                  pse_variables   — list of variables returned
                  pse_spatial     — bounding box dict
                  pse_temporal    — time range dict

        Raises:
            ConnectorError: on any fetch failure.
        """

    @abstractmethod
    async def get_quality(
        self,
        spatial: SpatialBounds,
        temporal: TemporalBounds,
    ) -> DataQuality:
        """Assess data quality for the requested region and period."""

    @abstractmethod
    async def get_latest_timestamp(self) -> datetime:
        """Return the most recent data timestamp available from this source."""

    # ------------------------------------------------------------------ #
    # Helpers available to all connectors                                  #
    # ------------------------------------------------------------------ #

    def _validate_variables(self, requested: list[str]) -> list[str]:
        """Validate requested variables against what this connector supports."""
        unknown = set(requested) - set(self.variables)
        if unknown:
            raise ValueError(
                f"[{self.source_id}] unknown variable(s): {sorted(unknown)}. "
                f"Supported: {self.variables}"
            )
        return requested

    def _base_attrs(
        self,
        variables: list[str],
        spatial: SpatialBounds,
        temporal: TemporalBounds,
    ) -> dict:
        """Build the standard PSE provenance attributes for a Dataset."""
        return {
            "pse_source": self.source_id,
            "pse_fetched_at": datetime.now(UTC).isoformat(),
            "pse_variables": variables,
            "pse_spatial": spatial.to_dict(),
            "pse_temporal": temporal.to_dict(),
        }
