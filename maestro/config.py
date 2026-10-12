"""Central configuration for Maestro.

Loads ``config/maestro.yaml`` (falling back to ``config/maestro.example.yaml``)
and ``.env``, with ``${VAR}`` interpolation against the environment, and exposes a
singleton ``cfg``. This is the single place every host/port/path/credential lives.

Run everything on one machine by pointing the service hosts at ``localhost`` in
``config/maestro.yaml``.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

# Repo root: explicit override, else two levels up from this file (maestro/config.py -> repo).
REPO_ROOT = Path(os.environ.get("MAESTRO_HOME") or Path(__file__).resolve().parent.parent)
os.environ.setdefault("MAESTRO_HOME", str(REPO_ROOT))


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (no hard dependency on python-dotenv)."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_dotenv(REPO_ROOT / ".env")

_VAR = re.compile(r"\$\{([^}]+)\}")


def _interp(obj):
    """Recursively expand ${VAR} using the process environment."""
    if isinstance(obj, str):
        return _VAR.sub(lambda m: os.environ.get(m.group(1), m.group(0)), obj)
    if isinstance(obj, dict):
        return {k: _interp(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_interp(v) for v in obj]
    return obj


class Attr(dict):
    """dict with attribute access; missing keys return None instead of raising."""

    def __getattr__(self, key):
        try:
            val = self[key]
        except KeyError:
            return None
        return Attr(val) if isinstance(val, dict) else val

    def __setattr__(self, key, val):
        self[key] = val


def _config_path() -> Path:
    user = REPO_ROOT / "config" / "maestro.yaml"
    return user if user.exists() else REPO_ROOT / "config" / "maestro.example.yaml"


def load() -> Attr:
    with open(_config_path()) as f:
        raw = yaml.safe_load(f) or {}
    return Attr(_interp(raw))


cfg = load()


def service_url(name: str, scheme: str = "http") -> str:
    """Base URL for a configured service, e.g. ``service_url('curobo')``."""
    svc = cfg.services[name] if cfg.services and name in cfg.services else None
    if not svc:
        raise KeyError(f"service '{name}' is not defined in {_config_path()}")
    return f"{scheme}://{svc['host']}:{svc['port']}"
