"""
tests/test_health.py
~~~~~~~~~~~~~~~~~~~~
Tests for the /health and / (root) endpoints.

These are the simplest possible tests — no mocking needed.
"""


def test_health_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_response_body(client):
    resp = client.get("/health")
    assert resp.json() == {"status": "healthy"}


def test_root_returns_200(client):
    resp = client.get("/")
    assert resp.status_code == 200


def test_root_response_contains_message(client):
    data = client.get("/").json()
    assert "message" in data
    assert isinstance(data["message"], str)
    assert len(data["message"]) > 0
