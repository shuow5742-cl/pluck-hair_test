"""WebSocket endpoint for streaming detection events."""

from fastapi import APIRouter, FastAPI, WebSocket, WebSocketDisconnect

router = APIRouter()


@router.websocket("/ws/events")
async def websocket_events(websocket: WebSocket):
    app = websocket.app
    ws_manager = getattr(app.state, "ws_manager", None)
    if ws_manager is None:
        await websocket.close(code=1011)
        return

    await ws_manager.connect(websocket)

    try:
        while True:
            # Receive command from frontend
            data = await websocket.receive_json()

            # Handle request_target
            if data.get("action") == "request_target":
                response = await _handle_request_target(app)
                await websocket.send_json(response)

            # Handle detect_once
            elif data.get("action") == "detect_once":
                response = await _handle_detect_once(app)
                await websocket.send_json(response)

            # Handle pick_done
            elif data.get("action") == "pick_done":
                track_id = data.get("track_id")
                response = await _handle_pick_done(app, track_id)
                await websocket.send_json(response)

            # Handle reset
            elif data.get("action") == "reset":
                response = await _handle_reset(app)
                await websocket.send_json(response)

    except WebSocketDisconnect:
        pass
    finally:
        await ws_manager.disconnect(websocket)


async def _handle_request_target(app: FastAPI) -> dict:
    """Handle request_target command via comm signal channel."""
    cmd_publisher = getattr(app.state, "command_publisher", None)
    if not cmd_publisher:
        return {"type": "error", "message": "Command publisher not configured"}

    return await cmd_publisher.send({"action": "request_target"})


async def _handle_detect_once(app: FastAPI) -> dict:
    """Handle detect_once command via comm signal channel."""
    cmd_publisher = getattr(app.state, "command_publisher", None)
    if not cmd_publisher:
        return {"type": "error", "message": "Command publisher not configured"}

    return await cmd_publisher.send({"action": "detect_once"})


async def _handle_pick_done(app: FastAPI, track_id: int) -> dict:
    """Handle pick_done command via comm signal channel."""
    cmd_publisher = getattr(app.state, "command_publisher", None)
    if not cmd_publisher:
        return {"type": "error", "message": "Command publisher not configured"}

    return await cmd_publisher.send({"action": "pick_done", "track_id": track_id})


async def _handle_reset(app: FastAPI) -> dict:
    """Handle reset command via comm signal channel."""
    cmd_publisher = getattr(app.state, "command_publisher", None)
    if not cmd_publisher:
        return {"type": "error", "message": "Command publisher not configured"}

    return await cmd_publisher.send({"action": "reset"})
