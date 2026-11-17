import os
from pathlib import Path


def data_dir() -> Path:
    base = os.environ.get("ML_INTERN_DATA_DIR")
    if base:
        return Path(base)
    xdg = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(xdg) / "ml-intern-mcp"


def cache_dir() -> Path:
    base = os.environ.get("ML_INTERN_CACHE_DIR")
    if base:
        return Path(base)
    xdg = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(xdg) / "ml-intern-mcp"
