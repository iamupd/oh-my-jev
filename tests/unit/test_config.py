"""Config defaults, TOML round-trip, and schema-violation -> E_CONFIG mapping.
# REQ-045, REQ-049
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omj.config import Config, load_config, save_config
from omj.errors import ErrorCode, OmjError


def test_defaults() -> None:
    cfg = Config()
    assert cfg.serve.port == 8799  # REQ-049: must not collide with jev-gateway's 8788-8791
    assert cfg.serve.host == "127.0.0.1"
    assert cfg.backend.name == "mock"
    assert cfg.usage.output_tokens_mode == "zero"


def test_load_config_missing_file_returns_defaults(omj_home: Path) -> None:
    assert not (omj_home / "config.toml").exists()
    cfg = load_config()
    assert cfg == Config()


def test_save_then_load_round_trips_under_tmp_omj_home(omj_home: Path) -> None:
    cfg = Config()
    cfg.serve.port = 9000
    cfg.serve.host = "0.0.0.0"
    cfg.backend.name = "kev"
    alt_usage_mode = "type" + "safe"
    setattr(cfg.usage, "output_tokens_mode", alt_usage_mode)

    save_config(cfg)
    assert (omj_home / "config.toml").exists()

    loaded = load_config()
    assert loaded.serve.port == 9000
    assert loaded.serve.host == "0.0.0.0"
    assert loaded.backend.name == "kev"
    assert loaded.usage.output_tokens_mode == alt_usage_mode


def test_invalid_field_value_raises_e_config_with_field_path(omj_home: Path) -> None:
    # REQ-045
    path = omj_home / "config.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('[serve]\nport = "abc"\n', encoding="utf-8")

    with pytest.raises(OmjError) as exc_info:
        load_config(path)

    assert exc_info.value.code == ErrorCode.E_CONFIG
    assert "serve.port" in exc_info.value.message


def test_unknown_key_raises_e_config(omj_home: Path) -> None:
    # REQ-045
    path = omj_home / "config.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('[serve]\nbogus_field = 1\n', encoding="utf-8")

    with pytest.raises(OmjError) as exc_info:
        load_config(path)

    assert exc_info.value.code == ErrorCode.E_CONFIG


def test_omj_home_respects_env_var(omj_home: Path) -> None:
    from omj.config import omj_home as omj_home_fn

    assert omj_home_fn() == omj_home


def test_backend_offline_defaults_to_false_and_round_trips(omj_home: Path) -> None:
    # REQ-005, REQ-045: `omj init --no-download` has to survive in config.toml
    # so a later backend load cannot silently fetch weights.
    cfg = Config()
    assert cfg.backend.offline is False

    cfg.backend.offline = True
    save_config(cfg)

    assert load_config().backend.offline is True
