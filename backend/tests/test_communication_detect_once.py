"""Tests for detect_once orchestration in CommunicationTask."""

from __future__ import annotations

from autoweaver.comm import CommSignalBase
from autoweaver.reactive import EventBus

from src.tasks.communication.task import CommunicationTask
from src.tasks.stabilized_detection.pick_process import PickResult


class _FakeTransport(CommSignalBase):
    def __init__(self) -> None:
        self.sent_messages: list[dict] = []

    def receive(self):
        return None

    def send(self, message: dict) -> None:
        self.sent_messages.append(dict(message))

    def close(self) -> None:
        return None


def test_detect_once_returns_new_target_then_retry_1():
    bus = EventBus()
    transport = _FakeTransport()
    comm = CommunicationTask(transport, confirmation_timeout_sec=0.2)

    target_responses = iter(
        [
            {
                "type": "target",
                "track_id": 1,
                "x": 10.5,
                "y": 20.5,
                "u": 30.0,
                "pick_attempts": 1,
                "dispatch_state": "new_target",
            },
            {
                "type": "target",
                "track_id": 1,
                "x": 11.0,
                "y": 21.0,
                "u": 31.0,
                "pick_attempts": 2,
                "dispatch_state": "retry_1",
            },
        ]
    )

    def on_request_target(_event: str, _data: dict) -> None:
        response = next(target_responses, None)
        if response is not None:
            bus.publish("COMM:TARGET_RESPONSE", {"payload": response})

    def on_pick_done(_event: str, data: dict) -> None:
        track_id = data["payload"]["track_id"]
        bus.publish(
            "TASK:PICK_RESULT",
            {
                "payload": {
                    "pick_result": PickResult(
                        success=False,
                        target_id=track_id,
                        message="not_disappeared",
                    )
                }
            },
        )

    bus.subscribe("COMM:REQUEST_TARGET", on_request_target)
    bus.subscribe("COMM:PICK_DONE", on_pick_done)

    try:
        comm.attach(bus)

        first = comm.handle_message({"request_id": "1", "action": "detect_once"})
        assert first == {
            "request_id": "1",
            "x": 10.5,
            "y": 20.5,
            "u": 30.0,
            "state": "new_target",
        }

        second = comm.handle_message({"request_id": "2", "action": "detect_once"})
        assert second == {
            "request_id": "2",
            "x": 11.0,
            "y": 21.0,
            "u": 31.0,
            "state": "retry_1",
        }
    finally:
        comm.close()
