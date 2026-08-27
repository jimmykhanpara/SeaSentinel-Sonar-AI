import pytest
from fastapi.testclient import TestClient

from sonar_debris.server.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_system_status(client):
    res = client.get("/api/system-status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert data["mode"] == "offline_edge_ready"
    assert "ghost_net" in data["supported_classes"]


def test_generate_sample_mission(client):
    res = client.post("/api/generate-sample?scenario=ghost_net_field&num_targets=4&conf_threshold=50")
    assert res.status_code == 200
    data = res.json()
    assert "mission_id" in data
    assert "report" in data
    assert data["report"]["summary"]["total_detections"] >= 0

    m_id = data["mission_id"]

    # Test export endpoints
    res_json = client.get(f"/api/export/{m_id}/json")
    assert res_json.status_code == 200

    res_csv = client.get(f"/api/export/{m_id}/csv")
    assert res_csv.status_code == 200

    res_geojson = client.get(f"/api/export/{m_id}/geojson")
    assert res_geojson.status_code == 200

    # Test feedback
    res_fb = client.post("/api/feedback", json={
        "mission_id": m_id,
        "detection_id": "test_det",
        "class_name": "ghost_net",
        "is_confirmed": True,
        "notes": "Verified by marine biologist"
    })
    assert res_fb.status_code == 200
