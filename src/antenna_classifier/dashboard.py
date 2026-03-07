"""Antenna Classifier Dashboard — web UI combining classifier, simulator, and analysis.

Run with:
    antenna-classifier dashboard --nec-dir /path/to/nec_files [--port 8501]

Or via Docker:
    docker compose up dashboard
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from . import classifier, parser, validator
from .fingerprint import fingerprint as make_fingerprint
from .simulator import simulate, DEFAULT_URL as DEFAULT_SOLVER_URL

# ---------------------------------------------------------------------------
# Lazy import FastAPI so the rest of the package works without it installed
# ---------------------------------------------------------------------------
try:
    from fastapi import FastAPI, Query, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse, Response
    from fastapi.staticfiles import StaticFiles
except ImportError:
    FastAPI = None  # type: ignore[assignment,misc]


def create_app(
    nec_dir: str | Path | None = None,
    solver_url: str | None = None,
) -> Any:
    """Build and return the FastAPI application."""
    if FastAPI is None:
        print("FastAPI is required: pip install fastapi uvicorn", file=sys.stderr)
        sys.exit(1)

    nec_dir = Path(nec_dir or os.getenv("NEC_DIR", "."))
    solver_url = solver_url or os.getenv("NEC_SOLVER_URL", DEFAULT_SOLVER_URL)
    root_path = os.getenv("ROOT_PATH", "")

    app = FastAPI(
        title="Antenna Classifier Dashboard",
        version="0.3.0",
    )

    # ---- Static files (frontend) ----
    static_dir = Path(__file__).parent / "static"
    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # ---- Cache of scanned files ----
    _catalog: list[dict] = []
    _catalog_index: dict[str, dict] = {}  # filename -> record

    def _ensure_catalog() -> None:
        if _catalog:
            return
        for p in sorted(nec_dir.rglob("*.nec")):
            try:
                parsed = parser.parse_file(p)
                val = validator.validate(parsed)
                cls = classifier.classify(parsed)
                fp = make_fingerprint(parsed)
                rec = {
                    "filename": p.name,
                    "path": str(p),
                    "relative_path": str(p.relative_to(nec_dir)) if p.is_relative_to(nec_dir) else p.name,
                    "valid": val.valid,
                    "antenna_type": cls.antenna_type,
                    "confidence": round(cls.confidence, 2),
                    "frequency_mhz": cls.frequency_mhz,
                    "band": cls.band,
                    "element_count": cls.element_count,
                    "ground_type": cls.ground_type,
                    "wire_count": len(parsed.wire_cards),
                    "fingerprint": fp.signature,
                    "complexity": round(fp.complexity_score, 3),
                    "evidence": cls.evidence,
                    "subtypes": cls.subtypes,
                    "errors": len(val.errors),
                    "warnings": len(val.warnings),
                }
                _catalog.append(rec)
                _catalog_index[p.name] = rec
            except Exception as exc:
                _catalog.append({
                    "filename": p.name,
                    "path": str(p),
                    "relative_path": str(p.relative_to(nec_dir)) if p.is_relative_to(nec_dir) else p.name,
                    "valid": False,
                    "antenna_type": "error",
                    "confidence": 0.0,
                    "error": str(exc),
                })

    # ---- API Endpoints ----

    @app.get("/favicon.ico")
    async def favicon():
        return Response(status_code=204)

    @app.get("/", response_class=HTMLResponse)
    async def index():
        html_path = static_dir / "index.html"
        if html_path.exists():
            html = html_path.read_text().replace(
                '"__ROOT_PATH__"', json.dumps(root_path),
            )
            return HTMLResponse(html)
        return HTMLResponse("<h1>Antenna Classifier Dashboard</h1><p>Static files not found.</p>")

    @app.get("/api/catalog")
    async def get_catalog(
        antenna_type: str | None = Query(None),
        valid_only: bool = Query(False),
        band: str | None = Query(None),
    ):
        _ensure_catalog()
        results = _catalog
        if valid_only:
            results = [r for r in results if r.get("valid")]
        if antenna_type:
            results = [r for r in results if r.get("antenna_type") == antenna_type]
        if band:
            results = [r for r in results if r.get("band") == band]
        return JSONResponse({
            "total": len(_catalog),
            "filtered": len(results),
            "files": results,
        })

    @app.get("/api/summary")
    async def get_summary():
        _ensure_catalog()
        type_counts: dict[str, int] = {}
        band_counts: dict[str, int] = {}
        valid_count = 0
        for rec in _catalog:
            atype = rec.get("antenna_type", "unknown")
            type_counts[atype] = type_counts.get(atype, 0) + 1
            b = rec.get("band") or "unknown"
            band_counts[b] = band_counts.get(b, 0) + 1
            if rec.get("valid"):
                valid_count += 1
        return JSONResponse({
            "total": len(_catalog),
            "valid": valid_count,
            "invalid": len(_catalog) - valid_count,
            "types": dict(sorted(type_counts.items(), key=lambda x: -x[1])),
            "bands": dict(sorted(band_counts.items(), key=lambda x: -x[1])),
        })

    @app.get("/api/file/{filename:path}")
    async def get_file_detail(filename: str):
        _ensure_catalog()
        rec = _catalog_index.get(filename)
        if not rec:
            raise HTTPException(404, f"File not found: {filename}")

        # Re-parse for full detail
        p = Path(rec["path"])
        parsed = parser.parse_file(p)
        val = validator.validate(parsed)
        cls = classifier.classify(parsed)
        fp = make_fingerprint(parsed)

        # Read raw NEC content (first 200 lines max for display)
        raw_lines = p.read_text(errors="replace").splitlines()[:200]

        return JSONResponse({
            **rec,
            "nec_content": "\n".join(raw_lines),
            "cards": [
                {"type": c.card_type, "line": c.line_number, "raw": c.raw}
                for c in parsed.cards[:100]
            ],
            "validation": [
                {"severity": i.severity.value, "message": i.message, "line": i.line}
                for i in val.issues
            ],
            "evidence": cls.evidence,
        })

    @app.get("/api/geometry/{filename:path}")
    async def get_geometry(filename: str):
        """Return 3D geometry data for the viewer."""
        _ensure_catalog()
        rec = _catalog_index.get(filename)
        if not rec:
            raise HTTPException(404, f"File not found: {filename}")
        p = Path(rec["path"])
        parsed = parser.parse_file(p)
        from .visualizer import extract_geometry
        return JSONResponse(extract_geometry(parsed))

    @app.post("/api/simulate/{filename:path}")
    async def run_simulation(filename: str):
        _ensure_catalog()
        rec = _catalog_index.get(filename)
        if not rec:
            raise HTTPException(404, f"File not found: {filename}")

        p = Path(rec["path"])
        result = simulate(p, base_url=solver_url)
        return JSONResponse(result.to_dict())

    @app.post("/api/pattern/{filename:path}")
    async def run_pattern(filename: str, type: str = "elevation"):
        """Run a forced radiation pattern (elevation, azimuth, or full 3D)."""
        _ensure_catalog()
        rec = _catalog_index.get(filename)
        if not rec:
            raise HTTPException(404, f"File not found: {filename}")
        if type not in ("elevation", "azimuth", "full"):
            raise HTTPException(400, f"Invalid pattern type: {type}")
        p = Path(rec["path"])
        from .simulator import simulate_pattern
        result = simulate_pattern(p, base_url=solver_url, force_pattern=type)
        return JSONResponse(result.to_dict())

    @app.post("/api/sweep/{filename:path}")
    async def run_sweep(filename: str):
        """Run frequency sweep (SWR + impedance across ±15% of design freq)."""
        _ensure_catalog()
        rec = _catalog_index.get(filename)
        if not rec:
            raise HTTPException(404, f"File not found: {filename}")

        p = Path(rec["path"])
        # Scale sweep points inversely with model complexity
        wires = rec.get("wire_count", 10)
        n_pts = 21 if wires <= 10 else 11 if wires <= 30 else 7
        from .simulator import simulate_sweep
        result = simulate_sweep(p, base_url=solver_url, n_points=n_pts)
        return JSONResponse(result.to_dict())

    @app.get("/api/types")
    async def get_types():
        """List all known antenna types."""
        return JSONResponse({"types": classifier.ANTENNA_TYPES})

    @app.post("/api/reload")
    async def reload_catalog():
        """Clear and rescan the catalog."""
        _catalog.clear()
        _catalog_index.clear()
        _ensure_catalog()
        return JSONResponse({"reloaded": len(_catalog)})

    return app


def run_dashboard(
    nec_dir: str | Path,
    port: int = 8501,
    host: str = "0.0.0.0",
    solver_url: str | None = None,
):
    """Start the dashboard server."""
    try:
        import uvicorn
    except ImportError:
        print("uvicorn is required: pip install uvicorn", file=sys.stderr)
        sys.exit(1)

    app = create_app(nec_dir=nec_dir, solver_url=solver_url)
    print(f"\n  Antenna Classifier Dashboard")
    print(f"  NEC files: {nec_dir}")
    print(f"  Solver:    {solver_url or DEFAULT_SOLVER_URL}")
    print(f"  URL:       http://{host}:{port}\n")
    uvicorn.run(app, host=host, port=port)
