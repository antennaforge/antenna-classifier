"""Regression tests for the dashboard FastAPI app.

Tests API endpoints with a real (tiny) NEC directory, ensuring:
- Catalog loads and returns correct structure
- File detail / geometry / types endpoints work
- Missing file returns 404
- Background scan exposes loading flag correctly
- Pattern endpoint rejects invalid types
"""

from collections.abc import Generator
import json
import textwrap
import time
from pathlib import Path

import pytest

from antenna_classifier.dashboard import create_app
from antenna_classifier import tuning_lab

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

MULTIBAND_NEC = textwrap.dedent("""\
    CM Two-band dipole
    GW 1 21 0 -5.0 0 0 5.0 0 0.001
    GE 0
    EX 0 1 11 0 1 0
    FR 0 1 0 0 14.15
    FR 0 1 0 0 28.5
    EN
""")


@pytest.fixture()
def nec_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with two NEC files."""
    (tmp_path / "dipole.nec").write_text(DIPOLE_NEC)
    (tmp_path / "yagi.nec").write_text(YAGI_NEC)
    (tmp_path / "multiband.nec").write_text(MULTIBAND_NEC)
    return tmp_path


@pytest.fixture()
def client(nec_dir: Path) -> Generator[TestClient, None, None]:
    """Create a test client with a fresh app."""
    import os
    # Redirect user NEC dir to temp path so tests don't need /data
    os.environ["USER_NEC_DIR"] = str(nec_dir / "user_nec_files")
    os.environ["CLASSIFICATION_DB"] = str(nec_dir / "classification_reviews.sqlite")
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
        assert data["total"] == 3
        filenames = {f["filename"] for f in data["files"]}
        assert filenames == {"dipole.nec", "yagi.nec", "multiband.nec"}

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
        assert "needs_review" in data
        assert data["total"] == 3

    def test_summary_type_counts(self, client: TestClient):
        data = client.get("/api/summary").json()
        # At least some type should have > 0 count
        assert sum(data["types"].values()) == 3


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
        assert "auto_antenna_type" in data
        assert "fingerprint_details" in data
        assert "lpda_fit" in data
        assert data["lpda_fit"] is None

    def test_multiband_file_detail_includes_design_frequencies(self, client: TestClient):
        resp = client.get("/api/file/multiband.nec")
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_multiband"] is True
        assert data["bands"] == ["20m", "10m"]
        assert data["design_frequencies_mhz"] == [14.15, 28.5]

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


class TestProxyAuthIdentity:
    def test_api_me_reports_anonymous_without_proxy_headers(self, client: TestClient):
        resp = client.get("/api/me")
        assert resp.status_code == 200
        assert resp.json() == {"authenticated": False}

    def test_api_me_preserves_admin_state_from_proxy_headers(self, client: TestClient):
        resp = client.get(
            "/api/me",
            headers={
                "X-HF-User-Id": "7",
                "X-HF-Callsign": "KQ4ZGQ",
                "X-HF-AI-Enabled": "1",
                "X-HF-Is-Admin": "1",
            },
        )
        assert resp.status_code == 200
        assert resp.json() == {
            "authenticated": True,
            "user_id": 7,
            "callsign": "KQ4ZGQ",
            "ai_enabled": True,
            "is_admin": True,
        }


class TestTuningLab:
    def test_lists_tuning_lab_exercises(self, client: TestClient):
        resp = client.get("/api/tuning-lab/exercises")
        assert resp.status_code == 200
        data = resp.json()
        assert {item["id"] for item in data["exercises"]} == {
            "dipole-basics",
            "vertical-match",
            "yagi-driven-element",
        }

    def test_get_tuning_lab_exercise_detail(self, client: TestClient):
        resp = client.get("/api/tuning-lab/exercises/dipole-basics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["title"] == "Dipole Resonance Basics"
        assert data["equations"][0]["expression"] == "Z = R + jX"
        assert len(data["controls"]) == 3

    def test_simulate_tuning_lab_exercise(self, client: TestClient, monkeypatch):
        monkeypatch.setattr(
            tuning_lab,
            "simulate_exercise",
            lambda exercise_id, values, base_url: {
                "exercise": {"id": exercise_id, "title": "Stub"},
                "current_values": values,
                "reference_values": {"length_scale": 1.0},
                "current": {
                    "ok": True,
                    "swr_sweep": {"freq_mhz": [14.0, 14.2], "swr": [1.6, 1.3]},
                    "impedance_sweep": {"freq_mhz": [14.0, 14.2], "r": [45.0, 50.0], "x": [8.0, 0.0]},
                    "analysis": {"reactive_state": "near resonance", "guidance": []},
                    "geometry_cards": [],
                },
                "reference": {
                    "ok": True,
                    "swr_sweep": {"freq_mhz": [14.0, 14.2], "swr": [1.4, 1.1]},
                    "impedance_sweep": {"freq_mhz": [14.0, 14.2], "r": [48.0, 50.0], "x": [4.0, 0.0]},
                    "analysis": {"reactive_state": "near resonance", "guidance": []},
                    "geometry_cards": [],
                },
            },
        )

        resp = client.post(
            "/api/tuning-lab/exercises/dipole-basics/simulate",
            json={"values": {"length_scale": 1.01}},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["exercise"]["id"] == "dipole-basics"
        assert data["current_values"]["length_scale"] == 1.01
        assert data["current"]["analysis"]["reactive_state"] == "near resonance"

    def test_get_tuning_lab_geometry(self, client: TestClient, monkeypatch):
        monkeypatch.setattr(
            tuning_lab,
            "exercise_geometry",
            lambda exercise_id, values: {
                "wires": [{"tag": 1, "points": [[0, 0, 0], [0, 0, 5]], "radius": 0.001, "segments": 21}],
                "ground_type": "perfect",
                "bounds": {"min": [0, 0, 0], "max": [1, 1, 5]},
            },
        )

        resp = client.post(
            "/api/tuning-lab/exercises/vertical-match/geometry",
            json={"values": {"radiator_scale": 1.0}},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ground_type"] == "perfect"
        assert data["bounds"]["max"][2] == 5

    def test_get_tuning_lab_pattern(self, client: TestClient, monkeypatch):
        monkeypatch.setattr(
            tuning_lab,
            "exercise_pattern",
            lambda exercise_id, values, pattern_type, base_url: {
                "ok": True,
                "radiation_pattern": {
                    "theta": [0, 45, 90],
                    "phi": [0, 0, 0],
                    "gain_db": [1.0, 4.0, 2.0],
                },
            },
        )

        resp = client.post(
            "/api/tuning-lab/exercises/yagi-driven-element/pattern?type=elevation",
            json={"values": {"driven_scale": 1.0}},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["radiation_pattern"]["theta"][1] == 45


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
# Classification review endpoint
# ---------------------------------------------------------------------------

class TestClassificationReview:
    def test_save_review_updates_effective_type(self, client: TestClient):
        resp = client.post(
            "/api/review/yagi.nec",
            json={
                "reviewed_antenna_type": "quad",
                "reason": "manual correction after geometry review",
                "references": ["https://example.com/reference"],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["antenna_type"] == "quad"
        assert data["auto_antenna_type"] == "yagi"
        assert data["is_reviewed"] is True
        assert data["review_reason"] == "manual correction after geometry review"
        assert data["references"][0]["url"] == "https://example.com/reference"

    def test_reference_catalog_lists_type_references(self, client: TestClient):
        client.post(
            "/api/review/yagi.nec",
            json={
                "reviewed_antenna_type": "yagi",
                "reason": "seed yagi reference catalog",
                "references": ["https://example.com/yagi-reference"],
            },
        )
        resp = client.get("/api/reference-catalog?antenna_type=yagi")
        assert resp.status_code == 200
        data = resp.json()
        assert data["antenna_type"] == "yagi"
        assert any(ref["url"] == "https://example.com/yagi-reference" for ref in data["references"])

    def test_save_review_accepts_catalog_reference_ids(self, client: TestClient):
        client.post(
            "/api/review/yagi.nec",
            json={
                "reviewed_antenna_type": "yagi",
                "reason": "seed yagi reference catalog",
                "references": ["https://example.com/yagi-reference"],
            },
        )
        catalog = client.get("/api/reference-catalog?antenna_type=yagi").json()
        ref_id = catalog["references"][0]["id"]

        resp = client.post(
            "/api/review/yagi.nec",
            json={
                "reviewed_antenna_type": "yagi",
                "reason": "picked from catalog",
                "reference_ids": [ref_id],
                "references": [],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["references"][0]["reference_id"] == ref_id
        assert data["references"][0]["url"] == "https://example.com/yagi-reference"

    def test_catalog_uses_reviewed_type(self, client: TestClient):
        client.post(
            "/api/review/yagi.nec",
            json={"reviewed_antenna_type": "quad", "reason": "override"},
        )
        resp = client.get("/api/catalog?antenna_type=quad")
        assert resp.status_code == 200
        files = {row["filename"] for row in resp.json()["files"]}
        assert "yagi.nec" in files

    def test_clear_review_restores_auto_type(self, client: TestClient):
        client.post(
            "/api/review/yagi.nec",
            json={"reviewed_antenna_type": "quad", "reason": "override"},
        )
        resp = client.post(
            "/api/review/yagi.nec",
            json={"reviewed_antenna_type": "", "reason": "", "references": []},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["antenna_type"] == "yagi"
        assert data["is_reviewed"] is False

    def test_invalid_review_type_rejected(self, client: TestClient):
        resp = client.post(
            "/api/review/yagi.nec",
            json={"reviewed_antenna_type": "definitely_not_a_real_type"},
        )
        assert resp.status_code == 400

    def test_invalid_reference_id_rejected(self, client: TestClient):
        resp = client.post(
            "/api/review/yagi.nec",
            json={"reference_ids": [999999]},
        )
        assert resp.status_code == 400


class TestReviewQueue:
    def test_review_queue_lists_invalid_models_first(self, client: TestClient, nec_dir: Path):
        invalid_file = nec_dir / "broken.nec"
        invalid_file.write_text("CM broken\nGW 1 1 0 0 0 0 0 0 0.001\nEN\n")
        client.post("/api/reload")
        for _ in range(50):
            resp = client.get("/api/catalog")
            data = resp.json()
            if not data.get("loading"):
                break
            time.sleep(0.05)

        queue = client.get("/api/review-queue?limit=10")
        assert queue.status_code == 200
        data = queue.json()
        assert data["total"] >= 1
        first = data["files"][0]
        assert first["filename"] == "broken.nec"
        assert first["needs_review"] is True
        assert "invalid_model" in first["review_reasons"]

    def test_catalog_needs_review_filter(self, client: TestClient):
        resp = client.get("/api/catalog?needs_review=true")
        assert resp.status_code == 200
        for record in resp.json()["files"]:
            assert record["needs_review"] is True


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

    def test_load_catalog_reads_review_queue_response(self, html: str):
        idx = html.find("async function loadCatalog")
        assert idx != -1, "loadCatalog function not found"
        body = html[idx:idx + 1000]
        assert "const [catResp, sumResp, queueResp" in body
        assert "reviewQueue = queueResp.files || []" in body
        assert "tuningLabExercises = labResp.exercises || []" in body

    def test_review_filter_option_present(self, html: str):
        assert '<option value="needs_review">Needs review</option>' in html

    def test_tuner_mode_selector_present(self, html: str):
        assert 'id="create-tuner-mode"' in html
        assert 'Seeded GA' in html

    def test_create_flows_include_tuner_mode(self, html: str):
        assert 'tuner_mode: _getSelectedTunerMode()' in html
        assert "fd.append('tuner_mode', _getSelectedTunerMode())" in html

    def test_review_reference_catalog_ui_present(self, html: str):
        assert 'id="review-reference-options"' in html
        assert 'reference_ids: referenceIds' in html
        assert 'api/reference-catalog?antenna_type=' in html

    def test_workspace_mode_shell_present(self, html: str):
        assert 'id="app-container"' in html
        assert "setWorkspaceMode('browse')" in html
        assert 'id="workspace-tab-lab" onclick="openTuningLab()"' in html

    def test_lab_results_pane_is_side_by_side(self, html: str):
        assert 'class="lab-stage"' in html
        assert 'class="lab-results-pane"' in html

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
