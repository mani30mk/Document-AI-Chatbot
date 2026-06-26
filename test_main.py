from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import main


@pytest.fixture
def client():
    return TestClient(main.app)


def test_embeddings_failure_returns_503_on_upload(client):
    with patch("main.get_embeddings", side_effect=RuntimeError("offline")):
        response = client.post(
            "/session/new",
            json={},
        )
        sid = response.json()["session_id"]

        upload_response = client.post(
            f"/upload/{sid}",
            files=[("files", ("sample.txt", b"hello world", "text/plain"))],
        )

    assert upload_response.status_code == 503
    assert "Embedding model" in upload_response.json()["detail"]
