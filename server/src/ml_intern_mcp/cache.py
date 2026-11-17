import hashlib
import json
import time
from pathlib import Path


class ResultCache:
    """On-disk cache of successful tool results, keyed by tool name and arguments."""

    def __init__(self, root: Path, ttl_seconds: float, enabled: bool = True):
        self.root = root
        self.ttl = ttl_seconds
        self.enabled = enabled

    def _path(self, tool: str, arguments: dict) -> Path:
        key = json.dumps({"tool": tool, "arguments": arguments}, sort_keys=True, default=str)
        return self.root / tool / (hashlib.sha256(key.encode()).hexdigest() + ".json")

    def get(self, tool: str, arguments: dict) -> str | None:
        if not self.enabled:
            return None
        path = self._path(tool, arguments)
        if not path.exists() or time.time() - path.stat().st_mtime > self.ttl:
            return None
        return json.loads(path.read_text())["text"]

    def put(self, tool: str, arguments: dict, text: str) -> None:
        if not self.enabled:
            return
        path = self._path(tool, arguments)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"text": text}))
        tmp.replace(path)
