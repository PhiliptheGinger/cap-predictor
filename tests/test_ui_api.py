from fastapi.testclient import TestClient

from sentimental_cap_predictor.ui.api import app

client = TestClient(app)


def test_chat_endpoint():
    response = client.post("/chat", json={"message": "hello"})
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert data["response"].startswith("Echo:")


def test_metrics_endpoint():
    response = client.get("/metrics/main")
    assert response.status_code == 200
    assert response.json()["model"] == "main"


def test_asset_performance_endpoint():
    response = client.get("/assets/demo/performance")
    assert response.status_code == 200
    assert response.json()["asset"] == "demo"


def test_trace_endpoint():
    response = client.get("/trace/123")
    assert response.status_code == 200
    assert response.json()["prediction_id"] == "123"
