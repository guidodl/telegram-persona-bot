import pytest
from mcp_tools import store

def test_register_get_roundtrip():
    store.register("t1", user_id=7, image_bytes=b"x")
    entry = store.get("t1")
    assert entry["user_id"] == 7
    assert entry["image_bytes"] == b"x"
    assert entry["found_image_url"] is None

def test_set_and_pop_found_image_drops_entry():
    store.register("t2", user_id=7, image_bytes=None)
    store.set_found_image("t2", "http://img/1.jpg")
    assert store.pop_found_image("t2") == "http://img/1.jpg"
    with pytest.raises(KeyError):
        store.get("t2")

def test_pop_missing_turn_returns_none():
    assert store.pop_found_image("nope") is None
