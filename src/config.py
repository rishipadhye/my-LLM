"""Load a YAML config file into an attribute-accessible object.

Usage:
    from config import load_config
    cfg = load_config("configs/tiny.yaml")
    print(cfg.hidden_size, cfg.optimizer)   # attribute access
    print(cfg["hidden_size"])               # item access also works
    print(cfg.as_dict())                    # back to a plain dict
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config:
    """Thin wrapper giving attribute + item access to a (possibly nested) dict."""

    def __init__(self, data: dict[str, Any]) -> None:
        for key, value in data.items():
            setattr(self, key, _wrap(value))

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def as_dict(self) -> dict[str, Any]:
        return {k: _unwrap(v) for k, v in vars(self).items()}

    def __repr__(self) -> str:
        return f"Config({self.as_dict()!r})"


def _wrap(value: Any) -> Any:
    if isinstance(value, dict):
        return Config(value)
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def _unwrap(value: Any) -> Any:
    if isinstance(value, Config):
        return value.as_dict()
    if isinstance(value, list):
        return [_unwrap(v) for v in value]
    return value


def load_config(path: str | Path) -> Config:
    """Read a YAML file and return it as a Config object."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Top-level YAML in {path} must be a mapping, got {type(data).__name__}")
    return Config(data)


if __name__ == "__main__":
    import sys

    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/tiny.yaml"
    cfg = load_config(cfg_path)
    for key, value in cfg.as_dict().items():
        print(f"{key:22} = {value}")
