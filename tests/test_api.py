"""Tests for CountLoop FastAPI REST API."""

import pytest
from fastapi.testclient import TestClient

from countloop.serve import app

client = TestClient(app)


def test_api_root_and_health():
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    ui = client.get("/ui")
    assert ui.status_code == 200
    assert "<title>CountLoop Playground (TMLR 2026)</title>" in ui.text


def test_api_plan_endpoint():
    payload = {
        "prompt": "20 oranges in a wooden crate",
        "target_count": 20,
    }
    res = client.post("/api/v1/plan", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["target_count"] == 20
    assert data["num_instances"] == 20
    assert "graph" in data
    assert len(data["graph"]["objects"]) == 20


def test_api_generate_endpoint_mock():
    payload = {
        "prompt": "10 cups on a wooden table",
        "target_count": 10,
        "max_rounds": 2,
        "mock_mode": True,
    }
    res = client.post("/api/v1/generate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "task_id" in data
    assert data["target_count"] == 10
    assert "final_image_base64" in data
    assert len(data["final_image_base64"]) > 100
    assert data["iterations_run"] >= 1
    assert "history" in data
