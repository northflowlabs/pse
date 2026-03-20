"""
Northflow PSE — Quick Start Example

Query temperature and solar irradiance for a location in Bali, Indonesia.
No API keys required (uses Open-Meteo and Global Solar Atlas).

Usage:
    pip install -r requirements.txt
    python examples/quickstart.py
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

# Add repo root to path so 'pse' is importable when running from examples/
sys.path.insert(0, str(Path(__file__).parent.parent))

from pse.connectors.base import SpatialBounds, TemporalBounds
from pse.connectors.open_meteo import OpenMeteoConnector


async def main() -> None:
    connector = OpenMeteoConnector()

    # Bali, Indonesia
    spatial = SpatialBounds(min_lat=-8.8, max_lat=-8.0, min_lon=114.4, max_lon=115.7)
    temporal = TemporalBounds(start=datetime(2025, 1, 1), end=datetime(2025, 1, 7))

    print("Fetching data from Open-Meteo...")
    ds = await connector.fetch(
        variables=["temperature_2m", "solar_ghi"],
        spatial=spatial,
        temporal=temporal,
    )

    print(ds)
    print(f"\nMean temperature : {float(ds['temperature_2m'].mean()):.1f} °C")
    print(f"Mean solar GHI   : {float(ds['solar_ghi'].mean()):.1f} W/m²")
    print(f"Data source      : {ds.attrs.get('pse_source', 'open_meteo')}")
    print(f"Time steps       : {ds.dims.get('time', '—')}")


if __name__ == "__main__":
    asyncio.run(main())
