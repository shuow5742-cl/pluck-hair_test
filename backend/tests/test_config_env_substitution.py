"""Tests for environment variable substitution in YAML config."""

from __future__ import annotations

from src.config import AppConfig, _substitute_env_vars


def test_substitute_env_vars_supports_default_values():
    content = "scheduler:\n  show_preview: ${SCHEDULER_SHOW_PREVIEW:-true}\n"

    resolved = _substitute_env_vars(content)

    assert resolved == "scheduler:\n  show_preview: true\n"


def test_app_config_uses_default_when_env_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("SCHEDULER_SHOW_PREVIEW", raising=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "scheduler:\n  show_preview: ${SCHEDULER_SHOW_PREVIEW:-true}\n",
        encoding="utf-8",
    )

    config = AppConfig.from_yaml(str(config_path))

    assert config.scheduler.show_preview is True


def test_app_config_uses_env_override_when_present(tmp_path, monkeypatch):
    monkeypatch.setenv("SCHEDULER_SHOW_PREVIEW", "false")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "scheduler:\n  show_preview: ${SCHEDULER_SHOW_PREVIEW:-true}\n",
        encoding="utf-8",
    )

    config = AppConfig.from_yaml(str(config_path))

    assert config.scheduler.show_preview is False
