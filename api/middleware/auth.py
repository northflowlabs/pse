"""
PSE API key authentication middleware.

API keys are passed via the X-API-Key header.  In development mode (ENVIRONMENT=development)
a missing key is allowed through so the API can be tested without setup.
"""
from __future__ import annotations

import os
import secrets

from fastapi import HTTPException, Request, status
from fastapi.security import APIKeyHeader

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# In dev, any key (or no key) is accepted.
_DEV_MODE = os.getenv("ENVIRONMENT", "development") == "development"

# Hardcoded dev key — override with a real key store in production.
_DEV_KEY = "pse-dev-key-northflow"


async def verify_api_key(request: Request) -> str | None:
    """
    FastAPI dependency that validates the X-API-Key header.

    Returns the raw key on success; raises HTTP 401 on failure.
    In development mode, passes through without validation.
    """
    if _DEV_MODE:
        return _DEV_KEY

    key = request.headers.get("X-API-Key")
    if not key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header.",
        )

    # In a real deployment, validate against a key store / database.
    # Here we accept the dev key as a fallback for early testing.
    if not secrets.compare_digest(key, _DEV_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )
    return key
