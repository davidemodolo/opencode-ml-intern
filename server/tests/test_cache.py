import threading

from ml_intern_mcp.cache import ResultCache


def test_concurrent_writers_of_the_same_key_do_not_fail(tmp_path):
    cache = ResultCache(tmp_path, ttl_seconds=60)
    errors = []

    def writer():
        for _ in range(200):
            try:
                cache.put("tool", {"a": 1}, "text")
            except OSError as error:
                errors.append(error)

    threads = [threading.Thread(target=writer) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert cache.get("tool", {"a": 1}) == "text"


def test_an_entry_removed_while_reading_is_a_miss(tmp_path, monkeypatch):
    cache = ResultCache(tmp_path, ttl_seconds=60)
    cache.put("tool", {"a": 1}, "text")

    def removed(self, *args, **kwargs):
        raise FileNotFoundError(self)

    monkeypatch.setattr(type(tmp_path), "read_text", removed)
    assert cache.get("tool", {"a": 1}) is None


def test_expired_entries_are_removed(tmp_path):
    cache = ResultCache(tmp_path, ttl_seconds=-1)
    cache.put("tool", {"a": 1}, "text")
    assert cache.get("tool", {"a": 1}) is None
    assert list((tmp_path / "tool").iterdir()) == []
