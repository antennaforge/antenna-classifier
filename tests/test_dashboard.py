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
    import os
    # Redirect user NEC dir to temp path so tests don't need /data
    os.environ["USER_NEC_DIR"] = str(nec_dir / "user_nec_files")
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


# ---------------------------------------------------------------------------
# Catalog filter combinations
# ---------------------------------------------------------------------------

class TestCatalogFilters:
    def test_filter_by_band(self, client: TestClient):
        """Band filter should return only matching files."""
        # Get all bands from summary first
        bands = client.get("/api/summary").json()["bands"]
        if not bands:
            pytest.skip("No bands detected in test data")
        band = next(iter(bands))
        resp = client.get(f"/api/catalog?band={band}")
        assert resp.status_code == 200
        data = resp.json()
        assert all(f.get("band") == band for f in data["files"])

    def test_combined_type_and_valid_filter(self, client: TestClient):
        """Multiple filters should compose — type + valid_only together."""
        resp = client.get("/api/catalog?antenna_type=dipole&valid_only=true")
        assert resp.status_code == 200
        for f in resp.json()["files"]:
            assert f["antenna_type"] == "dipole"
            assert f["valid"] is True

    def test_filter_nonexistent_type_returns_empty(self, client: TestClient):
        """Filtering by a type that doesn't exist should return zero results."""
        resp = client.get("/api/catalog?antenna_type=nonexistent_antenna_xyz")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["files"] == []

    def test_filter_nonexistent_band_returns_empty(self, client: TestClient):
        resp = client.get("/api/catalog?band=999GHz")
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    def test_filtered_total_matches_file_count(self, client: TestClient):
        """The 'total' field should match the length of 'files'."""
        resp = client.get("/api/catalog?valid_only=true")
        data = resp.json()
        assert data["total"] == len(data["files"])


# ---------------------------------------------------------------------------
# HTML regression guards — verifies critical JS patterns in served HTML
# ---------------------------------------------------------------------------

class TestHTMLRegressions:
    """Guard against regressions in client-side JS patterns.

    These tests fetch the served index.html and assert that key code
    patterns we've fixed are present. If someone reverts a fix, these
    will catch it.
    """

    @pytest.fixture()
    def html(self, client: TestClient) -> str:
        resp = client.get("/")
        assert resp.status_code == 200
        return resp.text

    def test_refresh_catalog_uses_apply_filters(self, html: str):
        """refreshCatalog must call applyFilters(), not renderFileList(catalog).

        Regression: refreshCatalog previously called renderFileList(catalog)
        which bypassed active filters and reset the sidebar every 3 seconds.
        """
        # Find the refreshCatalog function body
        idx = html.find("function refreshCatalog")
        assert idx != -1, "refreshCatalog function not found in HTML"
        # Check within the next ~600 chars (the function body)
        body = html[idx:idx + 600]
        assert "applyFilters()" in body, (
            "refreshCatalog must call applyFilters() to preserve user filters"
        )
        assert "renderFileList(catalog)" not in body, (
            "refreshCatalog must NOT call renderFileList(catalog) — bypasses filters"
        )

    def test_degenerate_gain_check_in_force_pattern(self, html: str):
        """forcePattern must guard against degenerate gain data (all -999.99).

        Regression: files with all gains = -999.99 produced Infinity
        surfacecolor values that hung Plotly, leaving the spinner forever.
        """
        idx = html.find("async function forcePattern")
        assert idx != -1, "forcePattern function not found"
        # Need a large window — the gain check is after fetch/response handling
        body = html[idx:idx + 1500]
        assert "validGains" in body, (
            "forcePattern must filter validGains before rendering"
        )

    def test_degenerate_gain_check_in_render_simulation(self, html: str):
        """renderSimulation initial path must check for degenerate gain data."""
        idx = html.find("function renderSimulation")
        assert idx != -1, "renderSimulation function not found"
        body = html[idx:idx + 5000]
        # Should check gains before calling renderRadiationPattern
        assert "g > -900" in body, (
            "renderSimulation must check for degenerate gain data (g > -900)"
        )

    def test_spinner_cleared_before_plotly_render(self, html: str):
        """forcePattern must explicitly clear spinner before Plotly.newPlot.

        Regression: spinner persisted because Plotly render was slow and
        the spinner div was not cleared before the render call.
        """
        idx = html.find("async function forcePattern")
        assert idx != -1
        body = html[idx:idx + 1500]
        assert "clear spinner" in body.lower() or "innerHTML = ''" in body, (
            "forcePattern must clear spinner before Plotly render"
        )

    def test_currents_button_outside_viewer3d_container(self, html: str):
        """Currents button must be a sibling of viewer3d-container, not a child.

        Regression: load3DView does container.innerHTML='' which destroyed
        any child elements. The button and legend must be in a wrapper div.
        """
        # Look for the wrapper pattern: position:relative div containing
        # viewer3d-container AND btn-currents as siblings
        idx = html.find("position:relative")
        assert idx != -1, "wrapper div with position:relative not found"
        # The btn-currents should NOT be inside viewer3d-container
        v3d_start = html.find('id="viewer3d-container"')
        assert v3d_start != -1
        # Find the closing </div> of viewer3d-container
        v3d_close = html.find("</div>", v3d_start)
        v3d_block = html[v3d_start:v3d_close]
        assert "btn-currents" not in v3d_block, (
            "btn-currents must NOT be inside viewer3d-container (innerHTML wipe destroys it)"
        )

    def test_legend_high_before_low(self, html: str):
        """Current legend must show High at top (red) and Low at bottom (blue).

        Regression: labels were placed left/right of a vertical bar instead
        of top/bottom matching the gradient direction.
        """
        idx = html.find('id="currents-legend"')
        assert idx != -1, "currents-legend not found"
        legend = html[idx:idx + 800]
        # High should appear before Low in the DOM (top before bottom)
        high_pos = legend.find(">High<")
        low_pos = legend.find(">Low<")
        assert high_pos != -1 and low_pos != -1, "High/Low labels not found in legend"
        assert high_pos < low_pos, (
            "High must appear before Low in DOM (top of gradient = High)"
        )
        # Should use flex-direction:column for vertical layout
        assert "flex-direction:column" in legend, (
            "Legend must use flex-direction:column for vertical layout"
        )
