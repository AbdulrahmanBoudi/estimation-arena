import hashlib
import json

import pytest

from arena import models
from arena.run import anchors_to_play


def _setup(tmp_path, monkeypatch, served=b"weights", locked=b"weights"):
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"bots/x/net.npz": {
        "repo": "o/r", "revision": "abc", "file": "net.npz", "sha256": hashlib.sha256(locked).hexdigest()}}))
    source = tmp_path / "served"
    source.write_bytes(served)
    monkeypatch.setattr(models, "url_for", lambda entry: source.as_uri())
    return lock


def test_downloads_a_missing_file_once(tmp_path, monkeypatch):
    lock = _setup(tmp_path, monkeypatch)
    assert models.fetch(tmp_path, lock) == ["bots/x/net.npz"]
    assert (tmp_path / "bots/x/net.npz").read_bytes() == b"weights"
    assert models.fetch(tmp_path, lock) == []


def test_refuses_a_download_with_the_wrong_checksum(tmp_path, monkeypatch):
    lock = _setup(tmp_path, monkeypatch, served=b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        models.fetch(tmp_path, lock)
    assert list((tmp_path / "bots/x").iterdir()) == []


def test_refuses_a_local_file_that_differs(tmp_path, monkeypatch):
    lock = _setup(tmp_path, monkeypatch)
    (tmp_path / "bots/x").mkdir(parents=True)
    (tmp_path / "bots/x/net.npz").write_bytes(b"other")
    with pytest.raises(ValueError, match="differs"):
        models.fetch(tmp_path, lock)


def test_url_pins_the_exact_commit():
    entry = {"repo": "o/r", "revision": "abc", "file": "net.npz"}
    assert models.url_for(entry) == "https://huggingface.co/o/r/resolve/abc/net.npz"


ANCHORS = [{"name": "Random", "spec": "random"}, {"name": "Hybrid", "spec": "hybrid"}]


def _old(specs, bot_hash="h", engine="0.1.0"):
    return {"bot_hash": bot_hash, "engine": engine, "versus": [{"spec": s} for s in specs]}


def test_unchanged_bot_plays_nothing():
    assert anchors_to_play(ANCHORS, _old(["random", "hybrid"]), "h", "0.1.0") == []


def test_unchanged_bot_plays_only_a_new_anchor():
    assert anchors_to_play(ANCHORS, _old(["random"]), "h", "0.1.0") == [ANCHORS[1]]


@pytest.mark.parametrize("old", [None, _old(["random", "hybrid"], bot_hash="x"), _old(["random", "hybrid"], engine="9")])
def test_new_or_changed_bot_plays_everything(old):
    assert anchors_to_play(ANCHORS, old, "h", "0.1.0") == ANCHORS
