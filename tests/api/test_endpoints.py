import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from diffusyn.interface.api import app

client = TestClient(app)

def test_health_check_via_status():
    # Mock celery AsyncResult to avoid Redis connection
    with patch("diffusyn.interface.api.celery_app.AsyncResult") as mock_result:
        mock_instance = MagicMock()
        mock_instance.status = "PENDING"
        mock_instance.ready.return_value = False
        mock_result.return_value = mock_instance
        
        response = client.get("/status/some-task-id")
        assert response.status_code == 200
        assert response.json()["status"] == "PENDING"

def test_download_not_found():
    response = client.get("/download/nonexistent.csv")
    assert response.status_code == 404