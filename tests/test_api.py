from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_openapi_schema():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/api/v1/predict" in paths
    assert "/api/v1/complaints" in paths
    assert "/api/v1/recalls" in paths
    assert "/api/v1/stats" in paths

def test_root_returns_200():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["version"] == "1.0.0"