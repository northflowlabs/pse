"""
PSE API key authentication middleware.

API keys are passed via the X-API-Key header and checked against PSE_API_KEYS, a
comma-separated list set in the environment. In development mode (ENVIRONMENT=development)
a missing key is allowed through so the API can be tested without setup. Outside
development, a server with no PSE_API_KEYS configured refuses every request.
"""
from __future__ import annotations

import os
import secrets

from fastapi import HTTPException, Request, status
from fastapi.security import APIKeyHeader

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

_DEV_MODE = os.getenv("ENVIRONMENT", "development") == "development"

# The valid keys, read once at start.
_KEYS = tuple(k.strip() for k in os.getenv("PSE_API_KEYS", "").split(",") if k.strip())


def _known(key: str) -> bool:
    # Compare against every key so the time taken does not reveal which one matched.
    hit = False
    for k in _KEYS:
        hit |= secrets.compare_digest(key.encode(), k.encode())
    return hit


async def verify_api_key(request: Request) -> str | None:
    """
    FastAPI dependency that validates the X-API-Key header.

    Returns the raw key on success; raises HTTP 401 on failure.
    In development mode, passes through without validation.
    """
    if _DEV_MODE:
        return request.headers.get("X-API-Key")

    key = request.headers.get("X-API-Key")
    if not key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header.",
        )
    if not _known(key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )
    return key
