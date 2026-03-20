# API Reference

Base URL: `https://pse.northflow.no/api/v1`

All endpoints return JSON unless otherwise noted. Authentication via `Authorization: Bearer <jwt>` or `X-API-Key: <key>` header (configurable; development mode bypasses auth).

---

## Health

### `GET /health`

Returns system status, connector freshness, and storage availability.

**Response**
```json
{
  "status": "healthy",
  "environment": "production",
  "connectors": {
    "era5": {"status": "ok", "latest": "2026-03-20T00:00:00Z"},
    "open_meteo": {"status": "ok", "latest": "2026-03-21T06:00:00Z"}
  },
  "zarr_store": {"status": "ok", "path": "/data/zarr"},
  "database": {"status": "ok"}
}
```

---

## Point Query

### `GET /point`

Query all available variables for a single lat/lon point.

**Parameters**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `lat` | float | Yes | Latitude (-90 to 90) |
| `lon` | float | Required | Longitude (-180 to 180) |
| `variables` | string | No | Comma-separated list (default: all) |
| `start` | ISO-8601 | No | Start datetime (default: 7 days ago) |
| `end` | ISO-8601 | No | End datetime (default: now) |

**Example**
```bash
curl "https://pse.northflow.no/api/v1/point?lat=-8.5&lon=115.2&variables=temperature_2m,solar_ghi"
```

---

## Bounding-Box Query

### `GET /query`

Gridded query over a spatial bounding box.

**Parameters**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `min_lat` | float | Yes | Southern bound |
| `max_lat` | float | Yes | Northern bound |
| `min_lon` | float | Yes | Western bound |
| `max_lon` | float | Yes | Eastern bound |
| `variables` | string | No | Comma-separated list |
| `start` | ISO-8601 | No | Start datetime |
| `end` | ISO-8601 | No | End datetime |
| `resolution` | float | No | Output resolution in degrees |

---

## Multi-Source Fusion

### `POST /fuse`

Explicitly request quality-weighted fusion across specified sources.

**Request body**
```json
{
  "variables": ["temperature_2m", "solar_ghi"],
  "spatial": {"min_lat": -8.9, "max_lat": -8.1, "min_lon": 114.8, "max_lon": 115.7},
  "temporal": {"start": "2024-01-01T00:00:00Z", "end": "2024-12-31T00:00:00Z"},
  "sources": ["era5", "open_meteo"]
}
```

---

## Sources

### `GET /sources`

List all registered connectors with their supported variables and status.

### `GET /sources/{source_id}/status`

Detailed status for a single connector including last fetch time, coverage, and quality metrics.
