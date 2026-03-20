# Data Sources

## ERA5 — ECMWF Reanalysis

**Source:** [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/)
**Connector:** `pse.connectors.era5.ERA5Connector`
**Key required:** Yes (free registration)

ERA5 is the fifth generation ECMWF atmospheric reanalysis of the global climate. It provides hourly estimates of atmospheric variables with global coverage from 1940 to present, at 0.25° (~28 km) horizontal resolution.

PSE variables: `temperature_2m`, `wind_speed_10m`, `wind_u_10m`, `wind_v_10m`, `solar_ghi`, `precipitation`, `surface_pressure`

Register at: https://cds.climate.copernicus.eu/

---

## Open-Meteo — Free Weather API

**Source:** [Open-Meteo](https://open-meteo.com/)
**Connector:** `pse.connectors.open_meteo.OpenMeteoConnector`
**Key required:** No

Open-Meteo is a free, open-source weather API that provides historical and forecast data globally. It aggregates multiple NWP models (ECMWF IFS, GFS, ICON) and provides hourly data at 0.25° resolution.

PSE variables: `temperature_2m`, `wind_speed_10m`, `solar_ghi`, `relative_humidity`, `surface_pressure`

---

## Global Solar Atlas — Long-Term Solar Climatology

**Source:** [World Bank / Solargis](https://globalsolaratlas.info/)
**Connector:** `pse.connectors.global_solar_atlas.GlobalSolarAtlasConnector`
**Key required:** No

The Global Solar Atlas provides long-term averages of solar irradiance at ~1 km resolution globally. It is the reference dataset for solar resource assessment and is used by the World Bank's energy investment programmes.

PSE variables: `solar_ghi`, `solar_dni`, `solar_dhi`

---

## Sentinel-2 — Copernicus Optical Imagery

**Source:** [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/)
**Connector:** `pse.connectors.sentinel2.Sentinel2Connector`
**Key required:** Yes (free registration)

Sentinel-2 is a twin-satellite ESA mission providing high-resolution (10–60 m) multispectral imagery with a 5-day global revisit cycle. PSE uses it for land-use classification, vegetation monitoring (NDVI), and water body detection (NDWI).

PSE variables: `ndvi`, `ndwi`, `land_use`

---

## OpenStreetMap — Infrastructure Data

**Source:** [OpenStreetMap Overpass API](https://overpass-api.de/)
**Connector:** `pse.connectors.osm.OSMConnector`
**Key required:** No

OpenStreetMap provides volunteered geographic information on infrastructure, land use, roads, buildings, and utilities. PSE uses the Overpass API to query power infrastructure (substations, transmission lines), transport networks, and settlement locations.

PSE variables: `nearest_substation_km`, `grid_capacity_mw`, `road_access_km`

---

## World Bank — Socioeconomic Indicators

**Source:** [World Bank Open Data](https://data.worldbank.org/)
**Connector:** `pse.connectors.world_bank.WorldBankConnector`
**Key required:** No

The World Bank provides country-level development indicators updated monthly. PSE uses these for energy access and renewable energy penetration context in site assessments.

PSE variables: `electrification_rate`, `gdp_per_capita`, `renewable_energy_share`
