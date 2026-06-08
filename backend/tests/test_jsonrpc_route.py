"""Tests for the JSON-RPC API endpoint."""

from __future__ import annotations

import asyncio

from src.api.app import create_app
from src.api.routes.jsonrpc import jsonrpc_endpoint


class _FakeCommandPublisher:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.calls: list[dict] = []

    async def send(self, payload: dict) -> dict:
        self.calls.append(dict(payload))
        return dict(self.response)


class _FakeRequest:
    def __init__(self, app, body: dict) -> None:
        self.app = app
        self._body = body

    async def json(self) -> dict:
        return dict(self._body)


def test_jsonrpc_health_returns_ok():
    app = create_app()
    request = _FakeRequest(
        app,
        {"jsonrpc": "2.0", "method": "health", "params": {}, "id": 1},
    )

    response = asyncio.run(jsonrpc_endpoint(request))

    assert response.status_code == 200
    assert response.body == b'{"jsonrpc":"2.0","result":{"ok":true},"id":1}'


def test_jsonrpc_detect_once_proxies_internal_command():
    app = create_app()
    publisher = _FakeCommandPublisher(
        {"x": 12.34, "y": 56.78, "u": -15.0, "state": "retry_1"}
    )
    app.state.command_publisher = publisher
    request = _FakeRequest(
        app,
        {"jsonrpc": "2.0", "method": "detect_once", "params": {}, "id": "abc"},
    )

    response = asyncio.run(jsonrpc_endpoint(request))

    assert response.status_code == 200
    assert publisher.calls == [{"action": "detect_once"}]
    assert (
        response.body
        == b'{"jsonrpc":"2.0","result":{"x":12.34,"y":56.78,"u":-15.0,"state":"retry_1"},"id":"abc"}'
    )
