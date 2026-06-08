"""Tests for lazy imports in src.comm."""

from __future__ import annotations

import importlib
import sys


def test_importing_src_comm_for_redis_control_client_does_not_import_autoweaver():
    for name in list(sys.modules):
        if name == "src.comm" or name.startswith("src.comm.") or name == "autoweaver" or name.startswith("autoweaver."):
            sys.modules.pop(name, None)

    module = importlib.import_module("src.comm")

    assert "autoweaver" not in sys.modules
    assert module.RedisControlClient.__name__ == "RedisControlClient"
    assert "autoweaver" not in sys.modules
