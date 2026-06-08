"""Minimal JSON-RPC 2.0 endpoint for external automation scripts."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from .events import _handle_detect_once

router = APIRouter()


def _success(result: dict[str, Any], rpc_id: Any) -> JSONResponse:
    return JSONResponse(
        {
            "jsonrpc": "2.0",
            "result": result,
            "id": rpc_id,
        }
    )


def _error(code: int, message: str, rpc_id: Any) -> JSONResponse:
    return JSONResponse(
        {
            "jsonrpc": "2.0",
            "error": {
                "code": code,
                "message": message,
            },
            "id": rpc_id,
        }
    )


@router.post("/jsonrpc")
async def jsonrpc_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception as exc:  # noqa: BLE001
        return _error(-32700, f"Parse error: {exc}", None)

    if not isinstance(body, dict):
        return _error(-32600, "Invalid Request", None)

    rpc_id = body.get("id")
    if body.get("jsonrpc") != "2.0":
        return _error(-32600, "Invalid Request", rpc_id)

    method = body.get("method")
    if not isinstance(method, str):
        return _error(-32600, "Invalid Request", rpc_id)

    if method == "health":
        return _success({"ok": True}, rpc_id)

    if method == "detect_once":
        result = await _handle_detect_once(request.app)
        if result.get("type") == "error":
            return _error(-32000, str(result.get("message") or "detect_once failed"), rpc_id)
        return _success(result, rpc_id)

    return _error(-32601, f"Method not found: {method}", rpc_id)
