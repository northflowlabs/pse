"""
PSE Railway entrypoint.

Runs before uvicorn:
  1. Normalises DATABASE_URL  (Railway provides postgresql://, app needs +asyncpg)
  2. Creates PostGIS extensions (idempotent, safe to re-run on every deploy)

Then exec-replaces itself with uvicorn so signals are forwarded correctly.
"""
from __future__ import annotations

import asyncio
import os
import sys


async def _init_postgis() -> None:
    import asyncpg  # already a dependency via SQLAlchemy asyncpg driver

    raw_url = os.environ["DATABASE_URL"]
    # asyncpg.connect() needs postgresql://, not postgresql+asyncpg://
    connect_url = raw_url.replace("postgresql+asyncpg://", "postgresql://", 1)

    try:
        conn = await asyncpg.connect(connect_url)
        await conn.execute("CREATE EXTENSION IF NOT EXISTS postgis")
        await conn.execute("CREATE EXTENSION IF NOT EXISTS postgis_topology")
        await conn.close()
        print("[entrypoint] PostGIS ready", flush=True)
    except Exception as exc:  # noqa: BLE001
        # Log but don't abort — PostGIS may already be enabled by the provider
        print(f"[entrypoint] PostGIS init warning (proceeding): {exc}", file=sys.stderr, flush=True)


def _normalise_database_url() -> None:
    """Railway injects postgresql://… — SQLAlchemy asyncpg needs postgresql+asyncpg://…"""
    url = os.environ.get("DATABASE_URL", "")
    if url.startswith("postgresql://") and "+asyncpg" not in url:
        os.environ["DATABASE_URL"] = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        print("[entrypoint] DATABASE_URL normalised to asyncpg driver", flush=True)


if __name__ == "__main__":
    _normalise_database_url()
    asyncio.run(_init_postgis())

    port = os.environ.get("PORT", "8000")
    # exec-replace this process so uvicorn receives SIGTERM directly
    os.execvp(
        "uvicorn",
        ["uvicorn", "pse.api.main:app", "--host", "0.0.0.0", "--port", port],
    )
