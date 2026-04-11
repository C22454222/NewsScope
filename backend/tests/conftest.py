"""Shared pytest fixtures for the NewsScope test suite."""
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_supabase():
    """Patch the Supabase client used by the backend."""
    with patch("app.db.supabase.supabase") as mock:
        yield mock


@pytest.fixture
def client(mock_supabase):
    """FastAPI TestClient with Supabase mocked."""
    from app.main import app
    return TestClient(app)


@pytest.fixture
def auth_headers():
    """Bypass Firebase JWT validation and return a fake header."""
    with patch("app.routes.users.verify_id_token") as mock:
        mock.return_value = {"uid": "test_uid", "email": "test@example.com"}
        yield {"Authorization": "Bearer faketoken"}
