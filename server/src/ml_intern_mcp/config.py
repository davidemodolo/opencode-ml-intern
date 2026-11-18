import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    pass


@dataclass
class HfJobsConfig:
    namespace: str | None = None
    poll_seconds: float = 30.0


@dataclass
class SkyPilotConfig:
    infra: list[str] = field(default_factory=list)
    use_spot: bool = False
    max_restarts_on_errors: int = 0
    poll_seconds: float = 60.0


@dataclass
class ServerConfig:
    hf_jobs: HfJobsConfig | None = None
    skypilot: SkyPilotConfig | None = None


BACKEND_SECTIONS = {"hf_jobs": HfJobsConfig, "skypilot": SkyPilotConfig}


def config_path() -> Path:
    explicit = os.environ.get("ML_INTERN_CONFIG")
    if explicit:
        return Path(explicit)
    xdg = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(xdg) / "ml-intern-mcp" / "config.toml"


def _section(name: str, cls: type, values: Any) -> Any:
    if not isinstance(values, dict):
        raise ConfigError(f"[backends.{name}] must be a table")
    allowed = {f.name for f in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ConfigError(f"unknown keys in [backends.{name}]: {', '.join(unknown)}")
    return cls(**values)


def load_config(path: Path) -> ServerConfig:
    """Reads the enabled remote backends; a missing file enables the local backend only."""
    if not path.exists():
        return ServerConfig()
    try:
        data = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"{path}: {error}") from error
    unknown = sorted(set(data) - {"backends"})
    if unknown:
        raise ConfigError(f"{path}: unknown sections {', '.join(unknown)}")
    backends = data.get("backends", {})
    unknown = sorted(set(backends) - set(BACKEND_SECTIONS))
    if unknown:
        raise ConfigError(
            f"{path}: unknown backends {', '.join(unknown)}; "
            f"available: {', '.join(BACKEND_SECTIONS)}"
        )
    return ServerConfig(
        **{
            name: _section(name, BACKEND_SECTIONS[name], values)
            for name, values in backends.items()
        }
    )
