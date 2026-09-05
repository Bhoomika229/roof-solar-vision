"""Lightweight YAML configuration loading with attribute-style access.

We deliberately avoid a heavier dependency (Hydra/OmegaConf) for a project this size — a
small recursive namespace over a plain dict is enough and keeps the dependency surface small.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterator

import yaml


class Config:
    """A dict-backed config object that supports both ``cfg.a.b`` and ``cfg["a"]["b"]`` access.

    Nested mappings are recursively wrapped in :class:`Config` on construction and un-wrapped
    back to plain dicts by :meth:`to_dict`, so the object round-trips cleanly through YAML.
    """

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        object.__setattr__(self, "_data", {})
        for key, value in (data or {}).items():
            self._data[key] = self._wrap(value)

    @staticmethod
    def _wrap(value: Any) -> Any:
        if isinstance(value, dict):
            return Config(value)
        if isinstance(value, list):
            return [Config._wrap(v) for v in value]
        return value

    @staticmethod
    def _unwrap(value: Any) -> Any:
        if isinstance(value, Config):
            return value.to_dict()
        if isinstance(value, list):
            return [Config._unwrap(v) for v in value]
        return value

    def to_dict(self) -> dict[str, Any]:
        return {k: self._unwrap(v) for k, v in self._data.items()}

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def __getattr__(self, item: str) -> Any:
        try:
            return self._data[item]
        except KeyError as exc:
            raise AttributeError(f"Config has no field '{item}'") from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self._data[key] = self._wrap(value)

    def __getitem__(self, item: str) -> Any:
        return self._data[item]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = self._wrap(value)

    def __contains__(self, item: str) -> bool:
        return item in self._data

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __repr__(self) -> str:
        return f"Config({self.to_dict()!r})"

    def merge(self, other: "Config | dict[str, Any]") -> "Config":
        """Return a new Config with ``other`` recursively merged on top of ``self``."""
        base = copy.deepcopy(self.to_dict())
        overlay = other.to_dict() if isinstance(other, Config) else other

        def _merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
            out = dict(a)
            for k, v in b.items():
                if k in out and isinstance(out[k], dict) and isinstance(v, dict):
                    out[k] = _merge(out[k], v)
                else:
                    out[k] = v
            return out

        return Config(_merge(base, overlay))


def load_config(path: str | Path) -> Config:
    """Load a YAML config file into a :class:`Config` object.

    Supports a single-level ``_base_: other.yaml`` key (resolved relative to ``path``'s
    directory) so model configs can inherit shared dataset/solar defaults without duplication.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    base_name = raw.pop("_base_", None)
    cfg = Config(raw)
    if base_name:
        base_cfg = load_config(path.parent / base_name)
        cfg = base_cfg.merge(cfg)
    return cfg


def save_config(cfg: Config | dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = cfg.to_dict() if isinstance(cfg, Config) else cfg
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
