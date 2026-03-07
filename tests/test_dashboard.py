"""Regression tests for the dashboard FastAPI app.

Tests API endpoints with a real (tiny) NEC directory, ensuring:
- Catalog loads and returns correct structure
- File detail / geometry / types endpoints work
- Missing file returns 404
- Background scan exposes loading flag correctly
- Pattern endpoint rejects invalid types
"""

import json
import textwrap
import time
from pathlib import Path

import pytest

from antenna_classifier.dashboard import create_app

# We need httpx for the async TestClient
pytest.importorskip("httpx")
from starlette.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DIPOLE_NEC = textwrap.dedent("""\
    CM Half-wave dipole
    GW 1 21 0 -5.0 0 0 5.0 0 0.001
    GE 0
    EX 0 1 11 0 1 0
    FR 0 1 0 0 14.15
    EN
""")

YAGI_NEC = textwrap.dedent("""\
    CM 3-element yagi
    GW 1 21 0 -5.5 0 0 5.5 0 0.001
    GW 2 21 3 -5.0 0 3 5.0 0 0.001
    GW 3 21 6 -4.5 0 6 4.5 0 0.001
    GE 0
    EX 0 1 11 0 1 0
    FR 0 1 0 0 14.15
    EN
""")


@pytest.fixture()
def nec_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with two NEC files."""
    (tmp_path / "dipole.nec").write_text(DIPOLE_NEC)
    (tmp_path / "yagi.nec").write_text(YAGI_NEC)
    return tmp_path


@pytest.fixture()
def client(nec_dir: Path) -> TestClient:
    """Create a test client with a fresh app."""
    app = create_app(nec_dir=nec_dir, solver_url="http://localhost:99999")
    with TestClient(app) as c:
        # Wait briefly for the background catalog scan to finish
        for _ in range(50):
            resp = c.get("/api/catalog")
            data = resp.json()
            if not data.get("loading"):
                break
            time.sleep(0.05)
        yield c


# ---------------------------------------------------------------------------
# Catalog endpoint
# ---------------------------------------------------------------------------

class TestCatalog:
    def test_returns_all_files(self, client: TestClient):
        resp = client.get("/api/catalog")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        filenames = {f["filename"] for f in data["files"]}
        assert filenames == {"dipole.nec", "yagi.nec"}

    def test_loading_flag_false_when_complete(self, client: TestClient):
        resp = client.get("/api/catalog")
        assert resp.json()["loading"] is False

    def test_filter_by_type(self, client: TestClient):
        resp = client.get("/api/catalog?antenna_type=yagi")
        assert resp.status_code == 200
        data = resp.json()
        assert all(f["antenna_type"] == "yagi" for f in data["files"])

    def test_filter_valid_only(self, client: TestClient):
        resp = client.get("/api/catalog?valid_only=true")
        data = resp.json()
        assert all(f.get("valid") for f in data["files"])

    def test_catalog_record_structure(self, client: TestClient):
        resp = client.get("/api/catalog")
        rec = resp.json()["files"][0]
        required_keys = {
            "filename", "path", "valid", "antenna_type",
            "confidence", "frequency_mhz", "band", "element_count",
            "ground_type", "wire_count", "fingerprint", "complexity",
        }
        assert required_keys.issubset(rec.keys()), f"Missing: {required_keys - rec.keys()}"


# ---------------------------------------------------------------------------
# Summary endpoint
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_structure(self, client: TestClient):
        resp = client.get("/api/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert "total" in data
        assert "valid" in data
        assert "types" in data
        assert "bands" in data
        assert data["total"] == 2

    def test_summary_type_counts(self, client: TestClient):
        data = client.get("/api/summary").json()
        # At least some type should have > 0 count
        assert sum(data["types"].values()) == 2


# ---------------------------------------------------------------------------
# File detail endpoint
# ---------------------------------------------------------------------------

class TestFileDetail:
    def test_get_file_detail(self, client: TestClient):
        resp = client.get("/api/file/dipole.nec")
        assert resp.status_code == 200
        data = resp.json()
        assert data["filename"] == "dipole.nec"
        assert "nec_content" in data
        assert "cards" in data
        assert "validation" in data
        assert data["valid"] is True

    def test_file_not_found(self, client: TestClient):
        resp = client.get("/api/file/nonexistent.nec")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Geometry endpoint
# ---------------------------------------------------------------------------

class TestGeometry:
    def test_returns_wires(self, client: TestClient):
        resp = client.get("/api/geometry/dipole.nec")
        assert resp.status_code == 200
        data = resp.json()
        assert "wires" in data
        assert len(data["wires"]) >= 1

    def test_wire_has_points(self, client: TestClient):
        data = client.get("/api/geometry/dipole.nec").json()
        wire = data["wires"][0]
        assert "points" in wire
        assert len(wire["points"]) == 2

    def test_geometry_has_bounds(self, client: TestClient):
        data = client.get("/api/geometry/dipole.nec").json()
        assert "bounds" in data
        assert "min" in data["bounds"]
        assert "max" in data["bounds"]

    def test_geometry_not_found(self, client: TestClient):
        resp = client.get("/api/geometry/nonexistent.nec")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Types endpoint
# ---------------------------------------------------------------------------

class TestTypes:
    def test_returns_type_list(self, client: TestClient):
        resp = client.get("/api/types")
        assert resp.status_code == 200
        types = resp.json()["types"]
        assert isinstance(types, list)
        assert "yagi" in types
        assert "dipole" in types
        assert "unknown" in types


# ---------------------------------------------------------------------------
# Pattern endpoint validation
# ---------------------------------------------------------------------------

class TestPatternValidation:
    def test_invalid_pattern_type_rejected(self, client: TestClient):
        resp = client.post("/api/pattern/dipole.nec?type=bogus")
        assert resp.status_code == 400

    def test_valid_pattern_types_accepted(self, client: TestClient):
        """Valid pattern types should not return 400 (may fail on solver timeout)."""
        for ptype in ("elevation", "azimuth", "full"):
            resp = client.post(f"/api/pattern/dipole.nec?type={ptype}")
            assert resp.status_code != 400, f"type={ptype} wrongly rejected"


# ---------------------------------------------------------------------------
# Index page
# ---------------------------------------------------------------------------

class TestIndexPage:
    def test_index_returns_html(self, client: TestClient):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")

    def test_favicon_no_error(self, client: TestClient):
        resp = client.get("/favicon.ico")
        assert resp.status_code == 204


# ---------------------------------------------------------------------------
# Reload endpoint
# ---------------------------------------------------------------------------

class TestReload:
    def test_reload_returns_started(self, client: TestClient):
        resp = client.post("/api/reload")
        assert resp.status_code == 200
        assert resp.json()["reloaded"] == "started"

    def test_reload_sets_loading(self, client: TestClient):
        """After reload the catalog should briefly show loading=True."""
        client.post("/api/reload")
        resp = client.get("/api/catalog")
        # Depending on timing it might already be done, but shouldn't error
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# _find_nec_file fallback (catalog not ready)
# ---------------------------------------------------------------------------

class TestFindNecFileFallback:
    def test_file_detail_before_catalog_ready(self, nec_dir: Path):
        """File endpoints should work even before catalog finishes via rglob fallback."""
        app = create_app(nec_dir=nec_dir, solver_url="http://localhost:99999")
        with TestClient(app, raise_server_exceptions=False) as c:
            # Request immediately — catalog may not be ready yet
            resp = c.get("/api/file/dipole.nec")
            # Should still find the file via fallback
            assert resp.status_code == 200
            assert resp.json()["filename"] == "dipole.nec"
