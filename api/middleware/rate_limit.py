"""
Simple in-process rate limiter for the PSE API.

Uses a fixed-window counter keyed on client IP (or API key).
For production, replace with a Redis-backed sliding-window implementation.
"""
from __future__ import annotations

import time
from collections import defaultdict

from fastapi import HTTPException, Request, status

# Default: 60 requests per minute per client
_DEFAULT_LIMIT = 60
_DEFAULT_WINDOW = 60  # seconds

_counters: dict[str, list[float]] = defaultdict(list)


def _client_id(request: Request) -> str:
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"key:{api_key}"
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return f"ip:{forwarded.split(',')[0].strip()}"
    return f"ip:{request.client.host}"


async def rate_limit(
    request: Request,
    limit: int = _DEFAULT_LIMIT,
    window: int = _DEFAULT_WINDOW,
) -> None:
    """
    FastAPI dependency that enforces a per-client request rate limit.

    Raises HTTP 429 if the limit is exceeded.
    """
    cid = _client_id(request)
    now = time.monotonic()
    window_start = now - window

    # Prune old timestamps
    _counters[cid] = [t for t in _counters[cid] if t > window_start]

    if len(_counters[cid]) >= limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {limit} requests per {window}s.",
            headers={"Retry-After": str(window)},
        )

    _counters[cid].append(now)
