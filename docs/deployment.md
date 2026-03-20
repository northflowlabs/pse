# Deployment Guide

## Docker Compose (recommended for local / single-server)

```bash
# Clone and configure
git clone https://github.com/northflowlabs/pse.git
cd pse
cp .env.example .env
# Fill in ERA5_CDS_API_KEY and other credentials

# Start the stack (PSE API + PostgreSQL/PostGIS)
docker compose up -d

# Verify
curl http://localhost:8001/api/v1/health
```

The stack includes:
- `pse-api` — FastAPI service on port 8001
- `postgres` — PostgreSQL 16 + PostGIS 3.4 on port 5432
- `scheduler` — Background ingest daemon (optional, `--profile scheduler`)

---

## Railway (managed cloud)

PSE ships with a `railway.toml` for zero-config Railway deployment.

### Prerequisites

1. [Railway CLI](https://docs.railway.app/develop/cli): `npm i -g @railway/cli`
2. A Railway account and project

### Steps

```bash
railway login
cd pse
railway init
```

In the Railway dashboard:
1. **Add a PostgreSQL plugin** — Railway injects `DATABASE_URL` automatically
2. **Add a Volume** — Mount at `/data/zarr` for persistent Zarr storage
3. **Set environment variables:**
   - `ERA5_CDS_API_KEY` — from Copernicus CDS
   - `JWT_SECRET_KEY` — `openssl rand -hex 32`
   - `ZARR_STORE_PATH` — `/data/zarr`
   - `ENVIRONMENT` — `production`
4. **Set Root Directory** → `pse`

```bash
railway up
```

### Custom domain

In Railway dashboard → service → **Custom Domain** → add `pse.northflow.no`.

Add a CNAME in your DNS:
```
pse.northflow.no  CNAME  <your-app>.up.railway.app
```

Railway provisions TLS automatically.

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DATABASE_URL` | Yes | — | PostgreSQL connection string |
| `ZARR_STORE_PATH` | Yes | `/data/zarr` | Local path for Zarr storage |
| `ZARR_STORE_URL` | No | — | S3/GCS URL (overrides path) |
| `ERA5_CDS_API_KEY` | No | — | Copernicus CDS API key |
| `COPERNICUS_DATASPACE_CLIENT_ID` | No | — | Sentinel-2 CDSE client ID |
| `COPERNICUS_DATASPACE_CLIENT_SECRET` | No | — | Sentinel-2 CDSE client secret |
| `JWT_SECRET_KEY` | Yes (prod) | `change-me` | JWT signing secret |
| `ENVIRONMENT` | No | `development` | `development` or `production` |
| `LOG_LEVEL` | No | `INFO` | Python log level |
