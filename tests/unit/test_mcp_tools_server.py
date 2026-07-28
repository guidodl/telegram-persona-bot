import base64
from fastapi.testclient import TestClient
from mcp_tools.server import app
from mcp_tools import store

def test_post_turn_registers_and_get_found_image_pops():
    client = TestClient(app)
    b64 = base64.b64encode(b"jpeg").decode()
    r = client.post("/turns", json={"turn_id": "t1", "user_id": 7, "image_bytes_b64": b64})
    assert r.status_code == 204
    assert store.get("t1")["image_bytes"] == b"jpeg"
    store.set_found_image("t1", "http://img/1.jpg")
    r = client.get("/turns/t1/found_image")
    assert r.json() == {"found_image_url": "http://img/1.jpg"}
    # entry dropped after pop
    r2 = client.get("/turns/t1/found_image")
    assert r2.json() == {"found_image_url": None}

def test_post_turn_null_image():
    client = TestClient(app)
    r = client.post("/turns", json={"turn_id": "t2", "user_id": 7, "image_bytes_b64": None})
    assert r.status_code == 204
    assert store.get("t2")["image_bytes"] is None
