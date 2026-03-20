"""
Northflow PSE — Multi-Source Fusion Example

Fetches solar irradiance from Open-Meteo and Global Solar Atlas, then
uses PSE's quality-weighted FusionEngine to merge them into a single
best-estimate dataset.

Usage:
    pip install -r requirements.txt
    python examples/multi_source_fusion.py
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pse.connectors.base import SpatialBounds, TemporalBounds
from pse.connectors.open_meteo import OpenMeteoConnector
from pse.connectors.global_solar_atlas import GlobalSolarAtlasConnector
from pse.fusion.engine import FusionEngine


async def main() -> None:
    # Nairobi, Kenya
    spatial = SpatialBounds(min_lat=-1.5, max_lat=-1.1, min_lon=36.6, max_lon=37.0)
    temporal = TemporalBounds(start=datetime(2024, 6, 1), end=datetime(2024, 6, 30))

    # Fetch from two independent sources in parallel
    print("Fetching from Open-Meteo and Global Solar Atlas...")
    open_meteo = OpenMeteoConnector()
    solar_atlas = GlobalSolarAtlasConnector()

    ds_meteo, ds_atlas = await asyncio.gather(
        open_meteo.fetch(["solar_ghi"], spatial, temporal),
        solar_atlas.fetch(["solar_ghi"], spatial, temporal),
    )

    q_meteo = await open_meteo.get_quality(spatial, temporal)
    q_atlas = await solar_atlas.get_quality(spatial, temporal)

    print(f"Open-Meteo quality  : reliability={q_meteo.reliability:.2f}")
    print(f"Solar Atlas quality : reliability={q_atlas.reliability:.2f}")

    # Fuse with quality-weighted merge
    engine = FusionEngine()
    fused = engine.fuse(
        datasets=[ds_meteo, ds_atlas],
        qualities=[q_meteo, q_atlas],
        variable="solar_ghi",
    )

    print(f"\nFused GHI (mean) : {float(fused['solar_ghi'].mean()):.1f} W/m²")
    print(f"Sources used     : {fused.attrs.get('pse_sources', ['open_meteo', 'global_solar_atlas'])}")


if __name__ == "__main__":
    asyncio.run(main())
