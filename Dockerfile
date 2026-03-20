FROM python:3.12-slim

# System dependencies for rasterio, geopandas, PostGIS
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev \
    gdal-bin \
    libproj-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Railway overrides CMD via railway.toml startCommand ("python entrypoint.py").
# The bare uvicorn CMD is kept as a sensible default for local docker-compose.
CMD ["uvicorn", "pse.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
