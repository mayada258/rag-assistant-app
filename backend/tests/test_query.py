from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_happy_path():
    response = client.post("/query", json={"question": "What is this document about?"})
    assert response.status_code == 200
    body = response.json()
    assert "answer" in body
    assert "sources" in body


def test_query_invalid_input():
    # missing required "question" field -> 422
    response = client.post("/query", json={})
    assert response.status_code == 422
