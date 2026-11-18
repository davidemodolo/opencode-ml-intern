import pytest

from ml_intern_mcp.compute.store import JobStore
from ml_intern_mcp.config import (
    ConfigError,
    HfJobsConfig,
    SkyPilotConfig,
    config_path,
    load_config,
)
from ml_intern_mcp.server import build_backends


def test_a_missing_file_enables_the_local_backend_only(tmp_path):
    config = load_config(tmp_path / "missing.toml")
    assert config.hf_jobs is None and config.skypilot is None
    assert list(build_backends(JobStore(tmp_path), config)) == ["local"]


def test_sections_enable_remote_backends(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        '[backends.hf_jobs]\nnamespace = "org"\n\n'
        '[backends.skypilot]\ninfra = ["runpod", "vast"]\nuse_spot = true\n'
    )
    config = load_config(path)
    assert config.hf_jobs == HfJobsConfig(namespace="org")
    assert config.skypilot == SkyPilotConfig(infra=["runpod", "vast"], use_spot=True)
    assert list(build_backends(JobStore(tmp_path), config)) == ["local", "hf_jobs", "skypilot"]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("[backends.modal]\n", "unknown backends modal"),
        ("[backends.hf_jobs]\ntoken = 'x'\n", "unknown keys in \\[backends.hf_jobs\\]: token"),
        ("[other]\n", "unknown sections other"),
        ("[backends\n", "config.toml"),
    ],
)
def test_invalid_files_are_rejected(tmp_path, text, message):
    path = tmp_path / "config.toml"
    path.write_text(text)
    with pytest.raises(ConfigError, match=message):
        load_config(path)


def test_config_path_follows_the_environment(monkeypatch, tmp_path):
    monkeypatch.delenv("ML_INTERN_CONFIG", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert config_path() == tmp_path / "ml-intern-mcp" / "config.toml"
    monkeypatch.setenv("ML_INTERN_CONFIG", "/etc/mli.toml")
    assert str(config_path()) == "/etc/mli.toml"
