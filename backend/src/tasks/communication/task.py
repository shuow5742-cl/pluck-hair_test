"""Communication task for comm signal handling (SideTask, event-driven)."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import logging
import threading
from typing import Callable, Optional

from autoweaver.comm import CommSideTask, CommSignalBase

logger = logging.getLogger(__name__)


class CommunicationTask(CommSideTask):
    """Handle comm signal commands via EventBus.

    Extends CommSideTask with pluck-specific business logic:
    request_target / pick_done / reset commands and event handlers.
    """

    name = "communication"

    def __init__(
        self,
        comm_signal: CommSignalBase,
        *,
        confirmation_timeout_sec: float = 5.0,
    ) -> None:
        super().__init__(transport=comm_signal)
        self._confirmation_timeout_sec = confirmation_timeout_sec
        self._target_response: Optional[dict] = None
        self._pick_result: Optional[dict] = None
        self._pick_result_event = threading.Event()
        self._active_track_id: Optional[int] = None
        self._unsubscribers: list[Callable[[], None]] = []

    def subscribe(self) -> None:
        self._unsubscribers.append(
            self._event_bus.subscribe("COMM:TARGET_RESPONSE", self._on_target_response)
        )
        self._unsubscribers.append(
            self._event_bus.subscribe("TASK:PICK_RESULT", self._on_pick_result)
        )

    def close(self) -> None:
        for unsub in self._unsubscribers:
            unsub()
        self._unsubscribers.clear()
        super().close()

    def handle_message(self, message: dict) -> Optional[dict]:
        request_id = message.get("request_id")
        action = message.get("action")
        if not request_id:
            logger.warning("Comm message missing request_id: %s", message)
            return None
        response = self._handle_action(action, message)
        return {"request_id": request_id, **response}

    # -- Event handlers --

    def _on_target_response(self, _event: str, data: dict) -> None:
        self._target_response = data.get("payload", {})

    def _on_pick_result(self, _event: str, data: dict) -> None:
        payload = data.get("payload", {})
        pick_result = self._normalize_pick_result(payload.get("pick_result"))
        if pick_result is not None:
            self._pick_result = pick_result
            self._pick_result_event.set()
            try:
                self.send({"type": "pick_result", **pick_result})
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to send pick result to PLC: %s", exc)

    # -- Business logic --

    def _handle_action(self, action: Optional[str], message: dict) -> dict:
        try:
            if action == "request_target":
                return self._cmd_request_target()
            if action == "detect_once":
                return self._cmd_detect_once()
            if action == "pick_done":
                track_id = message.get("track_id")
                return self._cmd_pick_done(track_id)
            if action == "reset":
                return self._cmd_reset()
            return {"type": "error", "message": f"Unknown action: {action}"}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Comm action failed: %s", exc)
            return {"type": "error", "message": str(exc)}

    def _cmd_request_target(self) -> dict:
        resp = self._request_target_response()
        if resp is None:
            return {"type": "error", "message": "No targets available"}
        if resp.get("type") == "target":
            self._active_track_id = resp.get("track_id")
        return resp

    def _cmd_detect_once(self) -> dict:
        if self._active_track_id is not None:
            self._await_confirmation(self._active_track_id)
            self._active_track_id = None

        target = self._request_target_response()
        if target is None or target.get("type") != "target":
            self._active_track_id = None
            return {"state": "no_target"}

        self._active_track_id = target.get("track_id")
        return self._build_detect_once_result(target)

    def _cmd_pick_done(self, track_id: Optional[int]) -> dict:
        if track_id is None:
            return {"type": "error", "message": "track_id is required"}
        self.broadcast("COMM:PICK_DONE", {"source": self.name, "payload": {"track_id": track_id}})
        return {"type": "ack", "status": "ok", "track_id": track_id}

    def _cmd_reset(self) -> dict:
        self._active_track_id = None
        self._target_response = None
        self._pick_result = None
        self._pick_result_event.clear()
        self.broadcast("COMM:RESET", {"source": self.name, "payload": {}})
        return {"type": "ack", "status": "reset"}

    def _request_target_response(self) -> Optional[dict]:
        self._target_response = None
        self.broadcast("COMM:REQUEST_TARGET", {"source": self.name, "payload": {}})
        return self._target_response

    def _await_confirmation(self, track_id: int) -> dict:
        self._pick_result = None
        self._pick_result_event.clear()
        self.broadcast(
            "COMM:PICK_DONE",
            {"source": self.name, "payload": {"track_id": track_id}},
        )
        completed = self._pick_result_event.wait(timeout=self._confirmation_timeout_sec)
        if not completed or self._pick_result is None:
            raise TimeoutError("Timed out waiting for confirmation frames")
        return self._pick_result

    @staticmethod
    def _normalize_pick_result(pick_result: object) -> Optional[dict]:
        if pick_result is None:
            return None
        if isinstance(pick_result, dict):
            return dict(pick_result)
        if is_dataclass(pick_result):
            return asdict(pick_result)
        return None

    @staticmethod
    def _build_detect_once_result(target: dict) -> dict:
        dispatch_state = target.get("dispatch_state")
        if not isinstance(dispatch_state, str):
            attempts = int(target.get("pick_attempts") or 1)
            if attempts <= 1:
                dispatch_state = "new_target"
            elif attempts == 2:
                dispatch_state = "retry_1"
            else:
                dispatch_state = "retry_2"

        return {
            "x": float(target["x"]),
            "y": float(target["y"]),
            "u": float(target.get("u") or 0.0),
            "state": dispatch_state,
        }
