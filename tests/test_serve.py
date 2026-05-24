"""
Serve API tests — mock the entire retrieval package so no torch/faiss/numpy
is needed at test time. All heavy dependencies are intercepted via sys.modules
before serve.py is imported.
"""
import io
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Inject fake retrieval.* modules before serve.py is imported
# ---------------------------------------------------------------------------

def _make_fake_retrieval():
    fake_pkg = ModuleType("retrieval")
    fake_pipeline_mod = ModuleType("retrieval.pipeline")
    fake_model_mod = ModuleType("retrieval.model")
    fake_index_mod = ModuleType("retrieval.index")

    class FakePipeline:
        def __init__(self, *a, **kw):
            self.index = MagicMock()
            self.index.ntotal = 2000
            self.preprocess = lambda img: MagicMock()
            self.device = "cpu"

        def query(self, image, k=20):
            return [
                {"rank": i + 1, "index": i, "score": round(1.0 - i * 0.01, 4)}
                for i in range(k)
            ]

    fake_pipeline_mod.RetrievalPipeline = FakePipeline
    fake_model_mod.embed = MagicMock(return_value=MagicMock(squeeze=lambda: [0.0] * 256))
    fake_pkg.pipeline = fake_pipeline_mod

    sys.modules.setdefault("retrieval", fake_pkg)
    sys.modules.setdefault("retrieval.pipeline", fake_pipeline_mod)
    sys.modules.setdefault("retrieval.model", fake_model_mod)
    sys.modules.setdefault("retrieval.index", fake_index_mod)


_make_fake_retrieval()

# Now it is safe to import serve
import serve  # noqa: E402  (must come after sys.modules patch)
from fastapi.testclient import TestClient  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color=(128, 64, 192)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(autouse=True)
def inject_pipeline(monkeypatch):
    """Replace the module-level _pipeline with a fresh FakePipeline each test."""
    from tests.test_serve import sys  # noqa — already imported
    fake = sys.modules["retrieval.pipeline"].RetrievalPipeline()
    monkeypatch.setattr(serve, "_pipeline", fake)
    monkeypatch.setattr(serve, "API_KEY", "")


@pytest.fixture()
def client():
    return TestClient(serve.app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["index_size"] == 2000


def test_search_returns_ranked_results(client):
    r = client.post(
        "/search",
        files={"file": ("test.png", _png_bytes(), "image/png")},
        params={"k": 5},
    )
    assert r.status_code == 200
    body = r.json()
    assert "request_id" in body
    assert body["results"][0]["rank"] == 1
    assert body["results"][0]["score"] <= 1.0


def test_search_rejects_invalid_image(client):
    r = client.post(
        "/search",
        files={"file": ("bad.png", b"not-an-image", "image/png")},
    )
    assert r.status_code == 422


def test_auth_rejects_wrong_key(client, monkeypatch):
    monkeypatch.setattr(serve, "API_KEY", "secret")
    r = client.post(
        "/search",
        files={"file": ("test.png", _png_bytes(), "image/png")},
        headers={"x-api-key": "wrong"},
    )
    assert r.status_code == 401


def test_auth_passes_correct_key(client, monkeypatch):
    monkeypatch.setattr(serve, "API_KEY", "secret")
    r = client.post(
        "/search",
        files={"file": ("test.png", _png_bytes(), "image/png")},
        headers={"x-api-key": "secret"},
    )
    assert r.status_code == 200
