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
        assert '/static/css/platform.css' in html, \
            "index.html does not reference platform.css"

    def test_html_references_platform_nav_js(self):
        html = _read_index()
        assert '/static/js/platform-nav.js' in html, \
            "index.html does not reference platform-nav.js"


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
