"""Communication module exports with lazy imports."""

__all__ = [
    "CommSignalBase",
    "ModbusAdapter",
    "RedisAdapter",
    "RedisControlClient",
]


def __getattr__(name: str):
    if name in {"CommSignalBase", "ModbusAdapter"}:
        from autoweaver.comm import CommSignalBase, ModbusAdapter

        exports = {
            "CommSignalBase": CommSignalBase,
            "ModbusAdapter": ModbusAdapter,
        }
        return exports[name]
    if name == "RedisAdapter":
        from src.comm.redis_adapter import RedisAdapter

        return RedisAdapter
    if name == "RedisControlClient":
        from src.comm.redis_control_client import RedisControlClient

        return RedisControlClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
