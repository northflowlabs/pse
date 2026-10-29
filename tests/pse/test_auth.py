"""
Tests for the X-API-Key check.
"""
from __future__ import annotations

import importlib

import pytest
from fastapi import HTTPException
from starlette.requests import Request


def _request(key: str | None = None) -> Request:
    headers = [(b"x-api-key", key.encode())] if key else []
    return Request({"type": "http", "method": "GET", "path": "/", "headers": headers})


def _auth(monkeypatch, environment: str, keys: str = ""):
    monkeypatch.setenv("ENVIRONMENT", environment)
    monkeypatch.setenv("PSE_API_KEYS", keys)
    from pse.api.middleware import auth

    return importlib.reload(auth)


async def test_development_lets_requests_through(monkeypatch):
    auth = _auth(monkeypatch, "development")
    assert await auth.verify_api_key(_request()) is None


async def test_production_accepts_a_configured_key(monkeypatch):
    auth = _auth(monkeypatch, "production", "first, second")
    assert await auth.verify_api_key(_request("second")) == "second"


@pytest.mark.parametrize("key", [None, "wrong", "pse-dev-key-northflow"])
async def test_production_refuses_missing_or_unknown_keys(monkeypatch, key):
    auth = _auth(monkeypatch, "production", "first")
    with pytest.raises(HTTPException) as e:
        await auth.verify_api_key(_request(key))
    assert e.value.status_code == 401


async def test_production_without_keys_refuses_everything(monkeypatch):
    auth = _auth(monkeypatch, "production")
    with pytest.raises(HTTPException):
        await auth.verify_api_key(_request("anything"))
