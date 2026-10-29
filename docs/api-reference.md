# API Reference

Base URL: `https://pse.northflow.no/api/v1`

All endpoints return JSON. Authentication uses the `X-API-Key: <key>` header. In development mode (`ENVIRONMENT=development`, the default) every request is accepted without a key.

---

## Health

### `GET /health`

Returns database reachability, per-connector freshness and cache statistics. `status` is `ok` when the database and every connector respond, otherwise `degraded`.

**Response**
```json
{
  "status": "ok",
  "timestamp": "2026-10-29T21:31:07Z",
  "database": {"status": "ok"},
  "connectors": {
    "open_meteo": {
      "status": "ok",
      "latest_data_timestamp": "2026-10-29T21:00:00+00:00",
      "lag_seconds": 1867.0,
      "variables": ["temperature_2m", "..."],
      "update_frequency_seconds": 3600
    }
  },
  "cache": {"entries": 2, "max_entries": 256, "default_ttl": 3600.0, "expired": 2},
  "version": "0.1.0"
}
```

---

## Point Query

### `GET /point`

Timeseries of the requested variables at a single lat/lon point.

**Parameters**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `lat` | float | Yes | Latitude (-90 to 90) |
| `lon` | float | Yes | Longitude (-180 to 180) |
| `variables` | string | Yes | Comma-separated PSE variable names |
| `time_start` | ISO-8601 | Yes | Start of time range (UTC) |
| `time_end` | ISO-8601 | Yes | End of time range (UTC) |

**Example**
```bash
curl "https://pse.northflow.no/api/v1/point?lat=-8.5&lon=115.2&variables=temperature_2m,solar_ghi&time_start=2025-01-01T00:00:00Z&time_end=2025-01-07T00:00:00Z"
```

---

## Bounding-Box Query

### `GET /query`

Gridded query over a spatial bounding box.

**Parameters**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `variables` | string | Yes | Comma-separated PSE variable names |
| `lat_min` | float | Yes | Southern bound |
| `lat_max` | float | Yes | Northern bound |
| `lon_min` | float | Yes | Western bound |
| `lon_max` | float | Yes | Eastern bound |
| `time_start` | ISO-8601 | Yes | Start of time range (UTC) |
| `time_end` | ISO-8601 | Yes | End of time range (UTC) |
| `resolution` | float | No | Target spatial resolution in metres |

---

## Multi-Source Fusion

### `POST /fuse`

Fetches the variables from every source that provides them, aligns them spatially and temporally, and returns a quality-weighted blend.

**Request body**
```json
{
  "variables": ["temperature_2m", "solar_ghi"],
  "lat_min": -8.9,
  "lat_max": -8.1,
  "lon_min": 114.8,
  "lon_max": 115.7,
  "time_start": "2024-01-01T00:00:00Z",
  "time_end": "2024-01-31T00:00:00Z",
  "resolution_m": 5000,
  "prefer_recent": true
}
```

`resolution_m` (default 5000) and `prefer_recent` (default `true`) are optional.

---

## Sources

### `GET /sources`

Lists the registered connectors with their variables and update frequency, plus the full list of available variables.

### `GET /sources/{source_id}/status`

Status of a single connector: latest data timestamp, lag in seconds, variables and update frequency.
