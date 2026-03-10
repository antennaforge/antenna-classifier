"""
UI Structure Regression Tests — Antenna Classifier
====================================================
Guards the structural integrity of the classifier's single-page
dashboard (index.html), static assets, and platform integration.

These tests catch the class of bugs where:
- Escaped backticks (\\`) appear inside ${...} expression contexts,
  causing JS SyntaxError in browsers (see commit fixing lines 1204-1219)
- Platform nav files (platform.css, platform-nav.js) are missing from
  the classifier's own static dir, breaking standalone mode
- Static asset references in HTML don't resolve to real files
- Template literal nesting is broken by improper escaping
- viewer3d.js module import reference goes stale
- renderPlatformNav / renderDetail / renderUserAntennaDetail functions
  are accidentally removed or renamed

These tests require NO running server — they parse the static files directly.
"""

import os
import re

import pytest

STATIC_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'src', 'antenna_classifier', 'static')
)
INDEX_HTML = os.path.join(STATIC_DIR, 'index.html')
HAMFEEDS_PLATFORM_CSS = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), '..', '..', 'hamfeeds', 'static', 'css', 'platform.css'
    )
)
HAMFEEDS_PLATFORM_NAV = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), '..', '..', 'hamfeeds', 'static', 'js', 'platform-nav.js'
    )
)


def _read(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def _read_index():
    return _read(INDEX_HTML)


def _extract_script_blocks(html):
    """Return list of inline <script> contents (not src= externals)."""
    return re.findall(r'<script(?:\s[^>]*)?>(?!.*?src=)(.*?)</script>', html, re.DOTALL)


# ═══════════════════════════════════════════════════════════════════════
#  Section 1: Static Asset Presence
# ═══════════════════════════════════════════════════════════════════════

class TestStaticAssets:
    """Ensure every file referenced in index.html actually exists on disk."""

    def test_index_html_exists(self):
        assert os.path.isfile(INDEX_HTML), "index.html missing from static dir"

    def test_viewer3d_js_exists(self):
        assert os.path.isfile(os.path.join(STATIC_DIR, 'viewer3d.js')), \
            "viewer3d.js missing — 3D viewer module will fail to load"

    def test_platform_css_exists(self):
        path = os.path.join(STATIC_DIR, 'css', 'platform.css')
        assert os.path.isfile(path), \
            "css/platform.css missing — top nav bar will be unstyled"

    def test_platform_nav_js_exists(self):
        path = os.path.join(STATIC_DIR, 'js', 'platform-nav.js')
        assert os.path.isfile(path), \
            "js/platform-nav.js missing — top nav bar will not render"

    def test_html_references_platform_css(self):
        html = _read_index()
        assert '__ROOT_PATH__/static/css/platform.css' in html, \
            "index.html should reference platform.css through the injected root path"

    def test_html_references_platform_nav_js(self):
        html = _read_index()
        assert '__ROOT_PATH__/static/js/platform-nav.js' in html, \
            "index.html should reference platform-nav.js through the injected root path"

    def test_platform_css_matches_hamfeeds_copy_when_available(self):
        if not os.path.isfile(HAMFEEDS_PLATFORM_CSS):
            pytest.skip("hamfeeds workspace copy not available")
        css = _read(os.path.join(STATIC_DIR, 'css', 'platform.css'))
        hf_css = _read(HAMFEEDS_PLATFORM_CSS)
        assert css == hf_css, (
            "classifier platform.css is out of sync with hamfeeds/static/css/platform.css"
        )


# ═══════════════════════════════════════════════════════════════════════
#  Section 2: JavaScript Syntax Integrity
# ═══════════════════════════════════════════════════════════════════════

class TestJsSyntaxIntegrity:
    """Catch template-literal escaping bugs that break the page."""

    def test_no_escaped_backticks_in_script(self):
        """Escaped backticks (\\`) inside ${...} expressions cause
        SyntaxError. They should never appear in inline <script> blocks.
        Nested template literals work naturally via ${...} expression
        context — no escaping needed."""
        html = _read_index()
        scripts = _extract_script_blocks(html)
        for i, script in enumerate(scripts):
            assert '\\`' not in script, (
                f"Script block {i} contains escaped backtick (\\`). "
                "Use regular backtick inside ${{...}} for nested template literals."
            )

    def test_no_escaped_template_expressions_in_script(self):
        """\\${...} inside a nested template literal outputs literal text
        instead of interpolating. All ${...} inside inner template literals
        (within ${...} expression contexts) should use unescaped ${."""
        html = _read_index()
        scripts = _extract_script_blocks(html)
        for i, script in enumerate(scripts):
            # Find \${ that is NOT at the start of a template literal
            # (i.e. preceded by something other than a backtick)
            matches = re.findall(r'(?<!`)\\\$\{', script)
            assert len(matches) == 0, (
                f"Script block {i} contains escaped template expression (\\${{...}}). "
                "This outputs literal text instead of evaluating the expression."
            )


# ═══════════════════════════════════════════════════════════════════════
#  Section 3: Critical Function Presence
# ═══════════════════════════════════════════════════════════════════════

class TestCriticalFunctions:
    """Verify essential JS functions are defined in index.html."""

    @pytest.mark.parametrize("func_name", [
        "renderDetail",
        "renderUserAntennaDetail",
        "openTuningLab",
        "renderPlatformNav",       # from platform-nav.js, called in index.html
        "loadUser3D",
        "loadSimulation",
    ])
    def test_function_referenced(self, func_name):
        html = _read_index()
        # renderPlatformNav is called, not defined, in index.html
        assert func_name in html, \
            f"'{func_name}' not found in index.html — function may have been removed"

    def test_renderPlatformNav_called_on_boot(self):
        html = _read_index()
        assert 'renderPlatformNav(' in html, \
            "renderPlatformNav() is not called — top nav will not appear"

    def test_boot_passes_explicit_platform_nav_api_base(self):
        html = _read_index()
        assert 'renderPlatformNav({ apiBase: BASE_URL, currentPath: PLATFORM_NAV_PATH, cssHref: `${BASE_URL}/static/css/platform.css` });' in html, \
            "renderPlatformNav() should use the resolved BASE_URL for both auth and stylesheet loading"

    def test_base_url_prefers_browser_path_over_root_path(self):
        html = _read_index()
        assert "const IS_ROOT_REQUEST = window.location.pathname === '/' || window.location.pathname === '';" in html, \
            "BASE_URL logic should explicitly detect direct root-page loads"
        assert "const BASE_URL = IS_ROOT_REQUEST ? '' : (LOCATION_BASE_URL || CONFIGURED_BASE_URL);" in html, \
            "BASE_URL should stay empty on root-page loads instead of inheriting the injected ROOT_PATH"

    def test_simulation_stats_distinguish_design_freq_and_swr_dip(self):
        html = _read_index()
        assert "label: 'Design Freq'" in html, \
            "Simulation stats should show the NEC FR/design frequency explicitly"
        assert "label: 'SWR Dip'" in html, \
            "Simulation stats should label the sweep minimum as SWR Dip, not Frequency"

    def test_simulation_ui_shows_sweep_dips_section(self):
        html = _read_index()
        assert 'id="sim-sweep-insights"' in html, \
            "Simulation UI should include a dedicated sweep-dips section"
        assert 'Sweep Dips' in html, \
            "Simulation UI should expose sweep dip reporting text"

    def test_tuning_lab_ui_present(self):
        html = _read_index()
        assert 'Tuning Lab' in html, \
            "Dashboard should expose the Tuning Lab workspace"
        assert '/api/tuning-lab/exercises' in html, \
            "Tuning Lab UI should fetch exercise metadata from the backend"
        assert 'lab-plot-swr' in html, \
            "Tuning Lab UI should include the SWR overlay plot container"
        assert "scheduleTuningExerciseRun('live')" in html, \
            "Tuning Lab controls should trigger debounced live reruns as sliders move"
        assert 'data-lab-live-status' in html, \
            "Tuning Lab should expose visible live-status feedback while the user tunes"
        assert 'Resonance Error' in html, \
            "Tuning Lab should expose telemetry cards that update with each live sweep"
        assert 'renderTuningNotesModal()' in html, \
            "Tuning Lab should open its notes content on demand instead of leaving all guidance inline"
        assert 'Concepts Introduced' in html, \
            "Tuning Lab should preserve the educational concepts inside the notes workflow"
        assert 'RF Vocabulary' in html, \
            "Tuning Lab should expose an in-context glossary for matching terminology"
        assert 'Reactance' in html and 'Inductance' in html and 'Capacitive' in html and 'Inductive' in html, \
            "Tuning Notes should define the core impedance and matching terms used by the coaching UI"
        assert 'Tuning Progression' in html, \
            "Tuning Lab should preserve the stepwise tuning order inside the notes workflow"
        assert 'Tuning Notes' in html, \
            "Tuning Lab should expose an on-demand notes modal for educational reference material"
        assert 'tuning-notes-modal' in html, \
            "Tuning Lab should include a dedicated modal container for the notes workflow"
        assert 'analysis?.next_action' in html, \
            "Tuning Lab should consume backend next-action guidance for staged exercises"
        assert 'reveal_analysis_flag' in html, \
            "Tuning Lab controls should support analysis-driven reveal rules for final-stage lessons"
        assert 'data-lab-next-action' in html, \
            "Tuning Lab should mark the next control to emphasize"
        assert 'Next step:' in html, \
            "Tuning Lab should surface a visible next-step status cue"
        assert 'lab-control.next-action' in html, \
            "Tuning Lab should style the next staged control distinctly"
        assert 'lab-control.useful-match' in html, \
            "Tuning Lab should style an already-useful hairpin control distinctly from the next-action state"
        assert 'lab-control-feedback' in html, \
            "Tuning Lab should merge live reference delta feedback into the relevant control cards"
        assert 'analysis?.hairpin_useful_match' in html, \
            "Tuning Lab should consume backend useful-match state for the hairpin control"
        assert 'Already useful: SWR < 1.5, R within 10 Ω of 50, and X within ±5 Ω.' in html, \
            "Tuning Lab should explain why the hairpin is already in a usable operating zone"
        assert "control.type === 'checkbox'" in html, \
            "Tuning Lab should support checkbox controls for staged Yagi assembly"
        assert 'reveal_control_id' in html, \
            "Tuning Lab controls should support dependency-based reveal rules"
        assert 'exercise.progression_steps' in html, \
            "Tuning Lab should render exercise-defined stepwise progression metadata"
        assert 'analysis?.progression_snapshot' in html, \
            "Tuning Lab should render live Yagi progression deltas from the backend analysis"
        assert 'tuningControlFeedbackMap(analysis?.progression_snapshot || [])' in html, \
            "Tuning Lab should translate progression snapshots into per-control inline feedback"
        assert 'progression_snapshot' in html, \
            "Tuning Lab UI should render backend-supplied live progression deltas"
        assert 'function shouldShowTuningControl(control)' in html, \
            "Tuning Lab should include helper logic for conditionally revealing fine-tuning controls"
        assert 'reveal_reactance_abs_max' in html, \
            "Tuning Lab control metadata should include conditional reveal thresholds"
        assert 'Math.abs(reactance) <= revealMax' in html, \
            "Tuning Lab should reveal fine-trim controls once reactance is close enough to zero"
        assert 'unlocks once |X| is within' in html, \
            "Tuning Lab should explain when the fine-trim control becomes available"
        assert 'Stop trimming' in html, \
            "Tuning Lab should surface a stop-trimming indicator when the model is already close enough"
        assert 'analysis?.stop_trimming' in html, \
            "Tuning Lab should key the stop indicator off the analysis payload"
        assert 'lab-viewer3d-container' in html, \
            "Tuning Lab should include an embedded 3D viewer container beneath the metrics"
        assert 'loadTuningLab3DView' in html, \
            "Tuning Lab should initialize the shared 3D viewer for exercise geometry"
        assert '/api/tuning-lab/exercises/${encodeURIComponent(activeTuningExercise.id)}/geometry' in html, \
            "Tuning Lab should fetch live exercise geometry through a dedicated endpoint"
        assert 'lab-plot-pattern' in html, \
            "Tuning Lab should include a dedicated radiation-pattern plot container"
        assert 'setTuningLabPatternType' in html, \
            "Tuning Lab should allow switching between elevation and azimuth pattern views"
        assert '/api/tuning-lab/exercises/${encodeURIComponent(activeTuningExercise.id)}/pattern?type=${tuningLabPatternType}' in html, \
            "Tuning Lab should fetch elevation and azimuth pattern cuts through a dedicated endpoint"
        assert "patternType: tuningLabPatternType" in html, \
            "Tuning Lab should pass the requested pattern type into the shared renderer"
        assert 'id="lab-dr-range"' in html, \
            "Tuning Lab should use its own dynamic-range selector instead of borrowing the main simulation control"
        assert 'reRenderTuningLabPattern()' in html, \
            "Tuning Lab should re-render its own pattern plot without falling back to the main simulation plot path"
        assert 'plotEl?.dataset.patternType' in html, \
            "Shared pattern rendering should preserve the last explicit pattern type on the target plot"
        assert "requestedPatternType === 'azimuth'" in html, \
            "Explicit azimuth requests should force the shared renderer into polar mode"
        assert "type: 'scatterpolar'" in html, \
            "2D pattern cuts should render with Plotly polar traces"
        assert 'lab-stat .secondary' in html, \
            "Tuning Lab geometry cards should support a smaller secondary line for feet values"
        assert 'card.secondary_value' in html, \
            "Tuning Lab geometry cards should render optional feet readouts under meter values"
        assert 'Peak Gain' in html, \
            "Tuning Lab should expose a 3D peak-gain metric card for directive antennas"
        assert 'Front-To-Back' in html, \
            "Tuning Lab should expose a front-to-back metric card for directive antennas"
        assert 'id="workspace-tab-lab" onclick="openTuningLab()"' in html, \
            "Tuning Lab should remain reachable from the top workspace tab"
        assert 'id="tuning-lab-launch"' not in html, \
            "Catalog sidebar should not contain a duplicate Tuning Lab launch button"
        assert 'class="summary-lab-preview"' not in html, \
            "Catalog welcome view should not contain duplicate Tuning Lab preview cards"
        assert 'class="info-btn" onclick="openTuningLab()"' not in html, \
            "Catalog welcome view should not contain a duplicate Tuning Lab info button"

    def test_workspace_mode_tabs_present(self):
        html = _read_index()
        assert "Catalog / My Antennas" in html, \
            "Dashboard should expose a browse workspace tab"
        assert "workspace-tab-lab" in html, \
            "Dashboard should expose a dedicated lab workspace tab"
        assert 'onclick="openTuningLab()"' in html, \
            "The Tuning Lab workspace tab should load lab content, not just toggle visibility"

    def test_lab_uses_side_by_side_stage_layout(self):
        html = _read_index()
        assert 'class="lab-stage"' in html, \
            "Tuning Lab should use a side-by-side stage layout for controls and results"
        assert 'lab-controls-pane' in html, \
            "Tuning Lab should isolate a focused controls pane from the live feedback pane"
        assert 'lab-results-pane' in html, \
            "Tuning Lab should render a dedicated results pane beside the controls"
        assert 'position: sticky' in html, \
            "Tuning Lab should keep the live feedback pane pinned beside the controls while tuning"
        assert '.lab-list { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; }' in html, \
            "Tuning Lab exercise cards should render in a top strip under the hero instead of a left column"
        assert 'Every knob move auto-runs the exercise after a brief pause' in html, \
            "Tuning Lab should tell the user that slider changes refresh the overlays automatically"

    def test_lab_prioritizes_goals_and_overlays_in_layout(self):
        html = _read_index()
        lab_markup = html[html.index('view.innerHTML = `'):]
        assert 'exercise.success_criteria || []' in lab_markup, \
            "Tuning Lab should keep success criteria visible in the controls pane so the target outcome leads the interaction"
        assert lab_markup.index('Controls') < lab_markup.index('Tuning Notes'), \
            "Tuning Lab should pair the notes entry point directly with the controls header"
        assert html.index('renderTuningNotesModal()') < html.index('Concepts Introduced'), \
            "Tuning Lab should move educational framing behind the notes modal instead of leaving it inline"
        assert 'Coach Notes' in html and 'Core Equations' in html, \
            "Tuning Lab should preserve both coaching guidance and equations within the new notes workflow"
        assert lab_markup.index('SWR Overlay') < lab_markup.index('R At Target'), \
            "Tuning Lab should place the overlay plots above the metric cards on the right"


# ═══════════════════════════════════════════════════════════════════════
#  Section 4: Platform Nav JS Module Integrity
# ═══════════════════════════════════════════════════════════════════════

class TestPlatformNavModule:
    """Validate platform-nav.js has required exports and structure."""

    def _read_nav_js(self):
        return _read(os.path.join(STATIC_DIR, 'js', 'platform-nav.js'))

    def test_defines_renderPlatformNav(self):
        js = self._read_nav_js()
        assert 'renderPlatformNav' in js, \
            "platform-nav.js does not define renderPlatformNav"

    def test_nav_items_include_antennas(self):
        js = self._read_nav_js()
        assert '/antenna/' in js, \
            "platform-nav.js missing Antennas nav item (/antenna/)"

    def test_nav_items_include_home(self):
        js = self._read_nav_js()
        assert "href: '/'" in js or "href='/'" in js, \
            "platform-nav.js missing home nav item"

    def test_nav_fetches_api_me(self):
        js = self._read_nav_js()
        assert '/api/me' in js, \
            "platform-nav.js does not fetch /api/me for auth state"

    def test_platform_nav_matches_hamfeeds_copy_when_available(self):
        if not os.path.isfile(HAMFEEDS_PLATFORM_NAV):
            pytest.skip("hamfeeds workspace copy not available")
        js = self._read_nav_js()
        hf_js = _read(HAMFEEDS_PLATFORM_NAV)
        assert js == hf_js, (
            "classifier platform-nav.js is out of sync with hamfeeds/static/js/platform-nav.js"
        )


# ═══════════════════════════════════════════════════════════════════════
#  Section 5: Viewer 3D Module Integrity
# ═══════════════════════════════════════════════════════════════════════

class TestViewer3dModule:
    """Validate viewer3d.js references and structure."""

    def test_html_imports_viewer3d(self):
        html = _read_index()
        assert 'viewer3d.js' in html, \
            "index.html does not reference viewer3d.js"

    def test_viewer3d_exports_class_or_function(self):
        js = _read(os.path.join(STATIC_DIR, 'viewer3d.js'))
        has_export = 'export' in js or 'class Viewer3D' in js or 'function' in js
        assert has_export, "viewer3d.js appears empty or has no exports"

    def test_pattern_fetch_uses_full_type(self):
        """3D radiation pattern fetch should use ?type=full for full hemisphere."""
        html = _read_index()
        assert 'type=full' in html, \
            "index.html should fetch pattern data with ?type=full for 3D hemisphere"
