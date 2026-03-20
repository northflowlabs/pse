"""
Tests for the BaseConnector data classes — SpatialBounds, TemporalBounds, DataQuality.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from pse.connectors.base import DataQuality, SpatialBounds, TemporalBounds


class TestSpatialBounds:
    def test_valid_construction(self):
        b = SpatialBounds(-6.3, -6.1, 106.7, 106.9)
        assert b.min_lat == -6.3
        assert b.center_lat == pytest.approx(-6.2)
        assert b.center_lon == pytest.approx(106.8)

    def test_rejects_inverted_lat(self):
        with pytest.raises(ValueError, match="min_lat"):
            SpatialBounds(-6.1, -6.3, 106.7, 106.9)

    def test_rejects_inverted_lon(self):
        with pytest.raises(ValueError, match="min_lon"):
            SpatialBounds(-6.3, -6.1, 106.9, 106.7)

    def test_rejects_out_of_range_lat(self):
        with pytest.raises(ValueError, match="Latitudes"):
            SpatialBounds(-91.0, -6.1, 106.7, 106.9)

    def test_rejects_out_of_range_lon(self):
        with pytest.raises(ValueError, match="Longitudes"):
            SpatialBounds(-6.3, -6.1, -181.0, 106.9)

    def test_to_dict(self):
        b = SpatialBounds(-6.3, -6.1, 106.7, 106.9)
        d = b.to_dict()
        assert d["min_lat"] == -6.3
        assert d["max_lat"] == -6.1
        assert d["min_lon"] == 106.7
        assert d["max_lon"] == 106.9


class TestTemporalBounds:
    def test_valid_construction(self):
        t = TemporalBounds(datetime(2025, 1, 1), datetime(2025, 1, 7))
        assert t.duration_days == pytest.approx(6.0)

    def test_rejects_start_after_end(self):
        with pytest.raises(ValueError, match="before end"):
            TemporalBounds(datetime(2025, 1, 7), datetime(2025, 1, 1))

    def test_rejects_equal_start_end(self):
        t = datetime(2025, 1, 1)
        with pytest.raises(ValueError):
            TemporalBounds(t, t)

    def test_to_dict(self):
        t = TemporalBounds(datetime(2025, 1, 1), datetime(2025, 1, 7))
        d = t.to_dict()
        assert "2025-01-01" in d["start"]
        assert "2025-01-07" in d["end"]


class TestDataQuality:
    def test_valid_construction(self):
        q = DataQuality(
            completeness=0.98,
            temporal_lag=300.0,
            spatial_resolution=11_000,
            source_reliability=0.88,
        )
        assert 0 < q.overall_score <= 1

    def test_rejects_completeness_out_of_range(self):
        with pytest.raises(ValueError):
            DataQuality(
                completeness=1.5,
                temporal_lag=0,
                spatial_resolution=1000,
                source_reliability=0.9,
            )

    def test_rejects_reliability_out_of_range(self):
        with pytest.raises(ValueError):
            DataQuality(
                completeness=0.9,
                temporal_lag=0,
                spatial_resolution=1000,
                source_reliability=-0.1,
            )

    def test_overall_score_decreases_with_age(self):
        fresh = DataQuality(0.99, 0.0, 1000, 0.95)
        stale = DataQuality(0.99, 10 * 86400, 1000, 0.95)
        assert fresh.overall_score > stale.overall_score

    def test_to_dict(self):
        q = DataQuality(0.95, 3600, 1000, 0.90, provenance={"model": "ERA5"})
        d = q.to_dict()
        assert "completeness" in d
        assert "overall_score" in d
        assert d["provenance"]["model"] == "ERA5"
