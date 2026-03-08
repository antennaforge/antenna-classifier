"""Antenna Classifier Dashboard — web UI combining classifier, simulator, and analysis.

Run with:
    antenna-classifier dashboard --nec-dir /path/to/nec_files [--port 8501]

Or via Docker:
    docker compose up dashboard
"""

from __future__ import annotations

import json
import os
import queue
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

from . import classifier, parser, validator
from .fingerprint import fingerprint as make_fingerprint
from .simulator import simulate, DEFAULT_URL as DEFAULT_SOLVER_URL

# ---------------------------------------------------------------------------
# Lazy import FastAPI so the rest of the package works without it installed
# ---------------------------------------------------------------------------
try:
    from fastapi import FastAPI, Query, HTTPException, Request
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
    _catalog_ready = False
    _catalog_lock = threading.Lock()

    def _build_catalog() -> None:
        """Scan and classify all NEC files (runs in background thread)."""
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
                with _catalog_lock:
                    _catalog.append(rec)
                    _catalog_index[p.name] = rec
            except Exception as exc:
                with _catalog_lock:
                    _catalog.append({
                        "filename": p.name,
                        "path": str(p),
                        "relative_path": str(p.relative_to(nec_dir)) if p.is_relative_to(nec_dir) else p.name,
                        "valid": False,
                        "antenna_type": "error",
                        "confidence": 0.0,
                        "error": str(exc),
                    })
        nonlocal _catalog_ready
        _catalog_ready = True

    @app.on_event("startup")
    async def _start_catalog_scan():
        threading.Thread(target=_build_catalog, daemon=True).start()

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
        with _catalog_lock:
            results = list(_catalog)
        if valid_only:
            results = [r for r in results if r.get("valid")]
        if antenna_type:
            results = [r for r in results if r.get("antenna_type") == antenna_type]
        if band:
            results = [r for r in results if r.get("band") == band]
        return JSONResponse({
            "total": len(results),
            "filtered": len(results),
            "files": results,
            "loading": not _catalog_ready,
        })

    @app.get("/api/summary")
    async def get_summary():
        with _catalog_lock:
            snapshot = list(_catalog)
        type_counts: dict[str, int] = {}
        band_counts: dict[str, int] = {}
        valid_count = 0
        for rec in snapshot:
            atype = rec.get("antenna_type", "unknown")
            type_counts[atype] = type_counts.get(atype, 0) + 1
            b = rec.get("band") or "unknown"
            band_counts[b] = band_counts.get(b, 0) + 1
            if rec.get("valid"):
                valid_count += 1
        return JSONResponse({
            "total": len(snapshot),
            "valid": valid_count,
            "invalid": len(snapshot) - valid_count,
            "types": dict(sorted(type_counts.items(), key=lambda x: -x[1])),
            "bands": dict(sorted(band_counts.items(), key=lambda x: -x[1])),
        })

    def _find_nec_file(filename: str) -> Path:
        """Look up a NEC file by name — use catalog index or fall back to glob."""
        with _catalog_lock:
            rec = _catalog_index.get(filename)
        if rec:
            return Path(rec["path"])
        # Catalog may still be loading — search the filesystem directly
        for p in nec_dir.rglob(filename):
            if p.is_file():
                return p
        raise HTTPException(404, f"File not found: {filename}")

    @app.get("/api/file/{filename:path}")
    async def get_file_detail(filename: str):
        p = _find_nec_file(filename)

        # Re-parse for full detail
        parsed = parser.parse_file(p)
        val = validator.validate(parsed)
        cls = classifier.classify(parsed)
        fp = make_fingerprint(parsed)

        # Use catalog record as base if available, otherwise build one
        with _catalog_lock:
            rec = _catalog_index.get(filename) or {}
        base = rec or {
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

        # Read raw NEC content (first 200 lines max for display)
        raw_lines = p.read_text(errors="replace").splitlines()[:200]

        return JSONResponse({
            **base,
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
        p = _find_nec_file(filename)
        parsed = parser.parse_file(p)
        from .visualizer import extract_geometry
        return JSONResponse(extract_geometry(parsed))

    @app.post("/api/simulate/{filename:path}")
    async def run_simulation(filename: str):
        p = _find_nec_file(filename)
        result = simulate(p, base_url=solver_url)
        return JSONResponse(result.to_dict())

    @app.post("/api/pattern/{filename:path}")
    async def run_pattern(filename: str, type: str = "elevation"):
        """Run a forced radiation pattern (elevation, azimuth, or full 3D)."""
        p = _find_nec_file(filename)
        if type not in ("elevation", "azimuth", "full"):
            raise HTTPException(400, f"Invalid pattern type: {type}")
        from .simulator import simulate_pattern
        result = simulate_pattern(p, base_url=solver_url, force_pattern=type)
        return JSONResponse(result.to_dict())

    @app.post("/api/sweep/{filename:path}")
    async def run_sweep(filename: str):
        """Run frequency sweep (SWR + impedance across ±15% of design freq)."""
        p = _find_nec_file(filename)

        # Scale sweep points inversely with model complexity
        parsed = parser.parse_file(p)
        wires = len(parsed.wire_cards)
        n_pts = 21 if wires <= 10 else 11 if wires <= 30 else 7
        from .simulator import simulate_sweep
        result = simulate_sweep(p, base_url=solver_url, n_points=n_pts)
        return JSONResponse(result.to_dict())

    @app.post("/api/currents/{filename:path}")
    async def run_currents(filename: str):
        """Extract per-segment structure currents for 3D heat-map overlay."""
        p = _find_nec_file(filename)
        from .simulator import simulate_currents
        result = simulate_currents(p, base_url=solver_url)
        return JSONResponse(result)

    @app.get("/api/types")
    async def get_types():
        """List all known antenna types."""
        return JSONResponse({"types": classifier.ANTENNA_TYPES})

    @app.post("/api/reload")
    async def reload_catalog():
        """Clear and rescan the catalog."""
        nonlocal _catalog_ready
        with _catalog_lock:
            _catalog.clear()
            _catalog_index.clear()
            _catalog_ready = False
        threading.Thread(target=_build_catalog, daemon=True).start()
        return JSONResponse({"reloaded": "started"})

    # ------------------------------------------------------------------
    # Hamfeeds user identity from trusted proxy headers
    # ------------------------------------------------------------------

    def _get_hf_user(request: "Request") -> dict | None:
        """Extract hamfeeds user from trusted proxy headers."""
        uid = request.headers.get("X-HF-User-Id")
        if not uid:
            return None
        return {
            "user_id": int(uid),
            "callsign": request.headers.get("X-HF-Callsign", ""),
            "ai_enabled": request.headers.get("X-HF-AI-Enabled") == "1",
        }

    @app.get("/api/me")
    async def get_current_user(request: "Request"):
        """Return the current user's identity from proxy headers."""
        hf_user = _get_hf_user(request)
        if not hf_user:
            return JSONResponse({"authenticated": False})
        return JSONResponse({"authenticated": True, **hf_user})

    # ------------------------------------------------------------------
    # My Antenna endpoints — AI generation + Postgres storage
    # ------------------------------------------------------------------
    from .storage import ensure_table, list_antennas, get_antenna, create_antenna, delete_antenna
    from .nec_generator import generate_nec_from_form, generate_nec_from_pdf

    # User NEC directory for simulation (written temp files)
    _user_nec_dir = Path(os.getenv("USER_NEC_DIR", "/data/user_nec_files"))
    _user_nec_dir.mkdir(parents=True, exist_ok=True)

    @app.on_event("startup")
    async def _init_user_antennas_table():
        try:
            ensure_table()
        except Exception as exc:
            print(f"[my-antennas] DB table init skipped: {exc}", file=sys.stderr)

    @app.get("/api/my-antennas")
    async def list_my_antennas(request: "Request"):
        """List user-generated antennas (no NEC content)."""
        hf_user = _get_hf_user(request)
        if not hf_user:
            raise HTTPException(401, "Authentication required")
        try:
            rows = list_antennas(owner_user_id=hf_user["user_id"])
            # Serialise datetimes
            for r in rows:
                for k in ("created_at", "updated_at"):
                    if hasattr(r.get(k), "isoformat"):
                        r[k] = r[k].isoformat()
            return JSONResponse({"antennas": rows})
        except Exception as exc:
            return JSONResponse({"antennas": [], "error": str(exc)})

    @app.get("/api/my-antennas/{antenna_id}")
    async def get_my_antenna(antenna_id: int, request: "Request"):
        """Get a single user antenna including NEC content."""
        hf_user = _get_hf_user(request)
        if not hf_user:
            raise HTTPException(401, "Authentication required")
        row = get_antenna(antenna_id, owner_user_id=hf_user["user_id"])
        if not row:
            raise HTTPException(404, "Antenna not found")
        for k in ("created_at", "updated_at"):
            if hasattr(row.get(k), "isoformat"):
                row[k] = row[k].isoformat()
        if isinstance(row.get("metadata"), dict):
            pass  # already a dict
        return JSONResponse(row)

    @app.post("/api/my-antennas/generate")
    async def generate_my_antenna_form(request: "Request"):
        """Generate NEC via AI from form data and save to Postgres."""
        hf_user = _get_hf_user(request)
        if not hf_user:
            raise HTTPException(401, "Authentication required")
        if not hf_user["ai_enabled"]:
            raise HTTPException(403, "AI features not enabled for your account. Contact admin.")
        body = await request.json()
        name = body.get("name", "").strip()
        antenna_type = body.get("antenna_type", "dipole")
        frequency_mhz = float(body.get("frequency_mhz", 14.0))
        ground_type = body.get("ground_type", "free_space")
        description = body.get("description", "")

        if not name:
            raise HTTPException(400, "Name is required")

        try:
            result = generate_nec_from_form(
                antenna_type=antenna_type,
                frequency_mhz=frequency_mhz,
                ground_type=ground_type,
                description=description,
                pipeline=True,
            )
        except RuntimeError as exc:
            raise HTTPException(503, str(exc))
        except Exception as exc:
            raise HTTPException(500, f"AI generation failed: {exc}")

        nec_content = result["nec_content"]

        # Classify the generated NEC to get band
        try:
            tmp = tempfile.NamedTemporaryFile(
                suffix=".nec", dir=str(_user_nec_dir), delete=False, mode="w",
            )
            tmp.write(nec_content)
            tmp.close()
            parsed = parser.parse_file(Path(tmp.name))
            cls = classifier.classify(parsed)
            band = cls.band
        except Exception:
            band = None

        row = create_antenna(
            name=name,
            description=description,
            antenna_type=result.get("classified_type", antenna_type),
            frequency_mhz=frequency_mhz,
            band=band,
            ground_type=ground_type,
            nec_content=nec_content,
            source="form",
            metadata={
                "model": result.get("model"),
                "usage": result.get("usage"),
                "iterations": result.get("iterations"),
                "classified_type": result.get("classified_type"),
                "confidence": result.get("confidence"),
                "refinement_log": result.get("refinement_log"),
                "steps": result.get("steps"),
                "goal_verdict": result.get("goal_verdict"),
                "buildability": result.get("buildability"),
            },
            owner_user_id=hf_user["user_id"],
        )
        for k in ("created_at", "updated_at"):
            if hasattr(row.get(k), "isoformat"):
                row[k] = row[k].isoformat()
        return JSONResponse(row, status_code=201)

    # ---- SSE streaming generate endpoint ----

    # Map pipeline step names → OODA phase index (0-3)
    _STEP_PHASE = {
        "classify": 0, "extract": 0,       # Observe
        "generate": 1, "validate": 1,      # Orient
        "convert": 2, "simulate": 2,       # Decide
        "tune": 3, "feedback": 3,          # Act
    }

    @app.post("/api/my-antennas/generate-stream")
    async def generate_my_antenna_stream(request: "Request"):
        """SSE stream: emit OODA step events during NEC generation."""
        from starlette.responses import StreamingResponse

        hf_user = _get_hf_user(request)
        if not hf_user:
            raise HTTPException(401, "Authentication required")
        if not hf_user["ai_enabled"]:
            raise HTTPException(403, "AI features not enabled")
        body = await request.json()
        name = body.get("name", "").strip()
        antenna_type = body.get("antenna_type", "dipole")
        frequency_mhz = float(body.get("frequency_mhz", 14.0))
        ground_type = body.get("ground_type", "free_space")
        description = body.get("description", "")
        if not name:
            raise HTTPException(400, "Name is required")

        q: queue.Queue[dict | None] = queue.Queue()

        def _on_step(step_log: Any) -> None:
            phase = _STEP_PHASE.get(step_log.name, -1)
            q.put({
                "step": step_log.step,
                "name": step_log.name,
                "status": step_log.status,
                "detail": step_log.detail,
                "phase": phase,
            })

        def _run() -> None:
            try:
                result = generate_nec_from_form(
                    antenna_type=antenna_type,
                    frequency_mhz=frequency_mhz,
                    ground_type=ground_type,
                    description=description,
                    pipeline=True,
                    on_step=_on_step,
                )
                q.put({"_result": result})
            except Exception as exc:
                q.put({"_error": str(exc)})
            finally:
                q.put(None)  # sentinel

        t = threading.Thread(target=_run, daemon=True)
        t.start()

        async def _event_stream():
            import asyncio
            while True:
                try:
                    msg = q.get(timeout=0.1)
                except queue.Empty:
                    continue
                if msg is None:
                    break
                if "_result" in msg:
                    # Pipeline complete — save antenna and emit done
                    try:
                        result = msg["_result"]
                        nec_content = result["nec_content"]
                        try:
                            tmp = tempfile.NamedTemporaryFile(
                                suffix=".nec", dir=str(_user_nec_dir),
                                delete=False, mode="w",
                            )
                            tmp.write(nec_content)
                            tmp.close()
                            parsed = parser.parse_file(Path(tmp.name))
                            cls = classifier.classify(parsed)
                            band = cls.band
                        except Exception:
                            band = None
                        row = create_antenna(
                            name=name,
                            description=description,
                            antenna_type=result.get("classified_type", antenna_type),
                            frequency_mhz=frequency_mhz,
                            band=band,
                            ground_type=ground_type,
                            nec_content=nec_content,
                            source="form",
                            metadata={
                                "model": result.get("model"),
                                "usage": result.get("usage"),
                                "iterations": result.get("iterations"),
                                "classified_type": result.get("classified_type"),
                                "confidence": result.get("confidence"),
                                "refinement_log": result.get("refinement_log"),
                                "steps": result.get("steps"),
                                "goal_verdict": result.get("goal_verdict"),
                                "buildability": result.get("buildability"),
                            },
                            owner_user_id=hf_user["user_id"],
                        )
                        for k in ("created_at", "updated_at"):
                            if hasattr(row.get(k), "isoformat"):
                                row[k] = row[k].isoformat()
                        yield f"data: {json.dumps({'event': 'done', 'antenna': row})}\n\n"
                    except Exception as exc:
                        yield f"data: {json.dumps({'event': 'error', 'detail': str(exc)})}\n\n"
                elif "_error" in msg:
                    yield f"data: {json.dumps({'event': 'error', 'detail': msg['_error']})}\n\n"
                else:
                    yield f"data: {json.dumps({'event': 'step', **msg})}\n\n"
                await asyncio.sleep(0)

        return StreamingResponse(
            _event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.delete("/api/my-antennas/{antenna_id}")
    async def delete_my_antenna(antenna_id: int, request: "Request"):
        hf_user = _get_hf_user(request)
        if not hf_user:
            raise HTTPException(401, "Authentication required")
        ok = delete_antenna(antenna_id, owner_user_id=hf_user["user_id"])
        if not ok:
            raise HTTPException(404, "Antenna not found")
        return JSONResponse({"deleted": True})

    # ---- Simulate / view a user antenna by writing temp NEC file ----

    def _write_user_nec(antenna_id: int, owner_user_id: int | None = None) -> Path:
        """Write user antenna NEC to a temp file and return the path."""
        row = get_antenna(antenna_id, owner_user_id=owner_user_id)
        if not row:
            raise HTTPException(404, "Antenna not found")
        p = _user_nec_dir / f"user_{antenna_id}.nec"
        p.write_text(row["nec_content"])
        return p

    @app.get("/api/my-antennas/{antenna_id}/geometry")
    async def user_antenna_geometry(antenna_id: int, request: "Request"):
        hf_user = _get_hf_user(request)
        if not hf_user:
            raise HTTPException(401, "Authentication required")
        p = _write_user_nec(antenna_id, owner_user_id=hf_user["user_id"])
        parsed = parser.parse_file(p)
        from .visualizer import extract_geometry
        return JSONResponse(extract_geometry(parsed))

    @app.post("/api/my-antennas/{antenna_id}/simulate")
    async def user_antenna_simulate(antenna_id: int, request: "Request"):
        hf_user = _get_hf_user(request)
        if not hf_user:
            raise HTTPException(401, "Authentication required")
        p = _write_user_nec(antenna_id, owner_user_id=hf_user["user_id"])
        result = simulate(p, base_url=solver_url)
        return JSONResponse(result.to_dict())

    @app.post("/api/my-antennas/{antenna_id}/pattern")
    async def user_antenna_pattern(antenna_id: int, request: "Request", type: str = "elevation"):
        hf_user = _get_hf_user(request)
        if not hf_user:
            raise HTTPException(401, "Authentication required")
        p = _write_user_nec(antenna_id, owner_user_id=hf_user["user_id"])
        if type not in ("elevation", "azimuth", "full"):
            raise HTTPException(400, f"Invalid pattern type: {type}")
        from .simulator import simulate_pattern
        result = simulate_pattern(p, base_url=solver_url, force_pattern=type)
        return JSONResponse(result.to_dict())

    @app.post("/api/my-antennas/{antenna_id}/sweep")
    async def user_antenna_sweep(antenna_id: int, request: "Request"):
        hf_user = _get_hf_user(request)
        if not hf_user:
            raise HTTPException(401, "Authentication required")
        p = _write_user_nec(antenna_id, owner_user_id=hf_user["user_id"])
        parsed = parser.parse_file(p)
        wires = len(parsed.wire_cards)
        n_pts = 21 if wires <= 10 else 11 if wires <= 30 else 7
        from .simulator import simulate_sweep
        result = simulate_sweep(p, base_url=solver_url, n_points=n_pts)
        return JSONResponse(result.to_dict())

    @app.post("/api/my-antennas/{antenna_id}/currents")
    async def user_antenna_currents(antenna_id: int, request: "Request"):
        hf_user = _get_hf_user(request)
        if not hf_user:
            raise HTTPException(401, "Authentication required")
        p = _write_user_nec(antenna_id, owner_user_id=hf_user["user_id"])
        from .simulator import simulate_currents
        result = simulate_currents(p, base_url=solver_url)
        return JSONResponse(result)

    @app.post("/api/my-antennas/upload-pdf")
    async def generate_from_pdf(request: "Request"):
        """Upload PDF, extract text, generate NEC via AI, save to Postgres."""
        hf_user = _get_hf_user(request)
        if not hf_user:
            raise HTTPException(401, "Authentication required")
        if not hf_user["ai_enabled"]:
            raise HTTPException(403, "AI features not enabled for your account. Contact admin.")
        from starlette.datastructures import UploadFile
        form = await request.form()
        pdf_file = form.get("pdf")
        if not pdf_file or not hasattr(pdf_file, "read"):
            raise HTTPException(400, "No PDF file uploaded")

        name = form.get("name", "").strip() or "PDF Upload"
        extra = form.get("description", "").strip()
        hint_type = form.get("antenna_type", "").strip()

        pdf_bytes = await pdf_file.read()
        if len(pdf_bytes) > 10 * 1024 * 1024:
            raise HTTPException(400, "PDF too large (max 10 MB)")

        try:
            result = generate_nec_from_pdf(pdf_bytes, extra_instructions=extra,
                                           antenna_type=hint_type,
                                           pipeline=True)
        except RuntimeError as exc:
            raise HTTPException(503, str(exc))
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        except Exception as exc:
            raise HTTPException(500, f"AI generation failed: {exc}")

        nec_content = result["nec_content"]

        # Use classification from the refinement loop if available,
        # otherwise fall back to a fresh classify pass
        antenna_type = result.get("classified_type", "")
        band = None
        freq = None
        try:
            tmp = tempfile.NamedTemporaryFile(
                suffix=".nec", dir=str(_user_nec_dir), delete=False, mode="w",
            )
            tmp.write(nec_content)
            tmp.close()
            parsed = parser.parse_file(Path(tmp.name))
            cls = classifier.classify(parsed)
            if not antenna_type:
                antenna_type = cls.antenna_type
            band = cls.band
            freq = cls.frequency_mhz
        except Exception:
            antenna_type = antenna_type or "unknown"

        row = create_antenna(
            name=name,
            description=extra,
            antenna_type=antenna_type,
            frequency_mhz=freq,
            band=band,
            nec_content=nec_content,
            source="pdf",
            metadata={
                "model": result.get("model"),
                "usage": result.get("usage"),
                "pdf_text_preview": result.get("pdf_text", "")[:500],
                "iterations": result.get("iterations"),
                "classified_type": result.get("classified_type"),
                "confidence": result.get("confidence"),
                "refinement_log": result.get("refinement_log"),
                "steps": result.get("steps"),
                "goal_verdict": result.get("goal_verdict"),
                "buildability": result.get("buildability"),
            },
            owner_user_id=hf_user["user_id"],
        )
        for k in ("created_at", "updated_at"):
            if hasattr(row.get(k), "isoformat"):
                row[k] = row[k].isoformat()
        return JSONResponse(row, status_code=201)

    @app.post("/api/my-antennas/from-url")
    async def generate_from_url(request: "Request"):
        """Fetch a web page, extract antenna data, generate NEC via AI, save to Postgres."""
        hf_user = _get_hf_user(request)
        if not hf_user:
            raise HTTPException(401, "Authentication required")
        if not hf_user["ai_enabled"]:
            raise HTTPException(403, "AI features not enabled for your account. Contact admin.")
        from .nec_generator import generate_nec_from_url

        body = await request.json()
        url = body.get("url", "").strip()
        if not url:
            raise HTTPException(400, "URL is required")
        if not url.startswith(("http://", "https://")):
            raise HTTPException(400, "Only http:// and https:// URLs are supported")

        name = body.get("name", "").strip() or "URL Import"
        extra = body.get("description", "").strip()
        hint_type = body.get("antenna_type", "").strip()

        try:
            result = generate_nec_from_url(url, extra_instructions=extra,
                                           antenna_type=hint_type,
                                           pipeline=True)
        except RuntimeError as exc:
            raise HTTPException(503, str(exc))
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        except Exception as exc:
            raise HTTPException(500, f"AI generation failed: {exc}")

        nec_content = result["nec_content"]

        antenna_type = result.get("classified_type", "")
        band = None
        freq = None
        try:
            tmp = tempfile.NamedTemporaryFile(
                suffix=".nec", dir=str(_user_nec_dir), delete=False, mode="w",
            )
            tmp.write(nec_content)
            tmp.close()
            parsed = parser.parse_file(Path(tmp.name))
            cls = classifier.classify(parsed)
            if not antenna_type:
                antenna_type = cls.antenna_type
            band = cls.band
            freq = cls.frequency_mhz
        except Exception:
            antenna_type = antenna_type or "unknown"

        row = create_antenna(
            name=name,
            description=extra or url,
            antenna_type=antenna_type,
            frequency_mhz=freq,
            band=band,
            nec_content=nec_content,
            source="url",
            metadata={
                "model": result.get("model"),
                "usage": result.get("usage"),
                "url": url,
                "url_text_preview": result.get("url_text", "")[:500],
                "iterations": result.get("iterations"),
                "classified_type": result.get("classified_type"),
                "confidence": result.get("confidence"),
                "refinement_log": result.get("refinement_log"),
                "steps": result.get("steps"),
                "goal_verdict": result.get("goal_verdict"),
                "buildability": result.get("buildability"),
            },
            owner_user_id=hf_user["user_id"],
        )
        for k in ("created_at", "updated_at"):
            if hasattr(row.get(k), "isoformat"):
                row[k] = row[k].isoformat()
        return JSONResponse(row, status_code=201)

    @app.post("/api/my-antennas/surprise")
    async def generate_surprise(request: "Request"):
        """Pick a random antenna type + band and generate via AI."""
        import random as _random

        hf_user = _get_hf_user(request)
        if not hf_user:
            raise HTTPException(401, "Authentication required")
        if not hf_user["ai_enabled"]:
            raise HTTPException(403, "AI features not enabled for your account. Contact admin.")

        # Good types for surprise — types that reliably produce working NEC
        # HF-only pool (10 m – 160 m)
        _SURPRISE_POOL = [
            ("dipole", 14.175, "20 m Dipole"),
            ("dipole", 7.1, "40 m Dipole"),
            ("dipole", 3.6, "80 m Dipole"),
            ("dipole", 1.85, "160 m Dipole"),
            ("dipole", 28.4, "10 m Dipole"),
            ("inverted_v", 14.175, "20 m Inverted V"),
            ("inverted_v", 7.1, "40 m Inverted V"),
            ("inverted_v", 3.6, "80 m Inverted V"),
            ("vertical", 14.175, "20 m Vertical"),
            ("vertical", 28.4, "10 m Vertical"),
            ("vertical", 7.1, "40 m Vertical"),
            ("yagi", 28.4, "10 m 3-Element Yagi"),
            ("yagi", 21.2, "15 m 3-Element Yagi"),
            ("yagi", 14.175, "20 m 3-Element Yagi"),
            ("moxon", 28.4, "10 m Moxon Rectangle"),
            ("moxon", 21.2, "15 m Moxon Rectangle"),
            ("moxon", 14.175, "20 m Moxon Rectangle"),
            ("quad", 21.2, "15 m Cubical Quad"),
            ("quad", 14.175, "20 m Cubical Quad"),
            ("end_fed", 7.1, "40 m End-Fed Half-Wave"),
            ("end_fed", 14.175, "20 m End-Fed Half-Wave"),
            ("delta_loop", 14.175, "20 m Delta Loop"),
            ("delta_loop", 28.4, "10 m Delta Loop"),
            ("magnetic_loop", 14.175, "20 m Magnetic Loop"),
            ("magnetic_loop", 21.2, "15 m Magnetic Loop"),
            ("half_square", 14.175, "20 m Half Square"),
            ("bobtail_curtain", 14.175, "20 m Bobtail Curtain"),
            ("bobtail_curtain", 7.1, "40 m Bobtail Curtain"),
        ]
        pick = _random.choice(_SURPRISE_POOL)
        antenna_type, freq, nice_name = pick

        try:
            result = generate_nec_from_form(
                antenna_type=antenna_type,
                frequency_mhz=freq,
                ground_type="free_space",
                description=f"Classic {nice_name} — surprise design!",
                pipeline=True,
            )
        except RuntimeError as exc:
            raise HTTPException(503, str(exc))
        except Exception as exc:
            raise HTTPException(500, f"AI generation failed: {exc}")

        nec_content = result["nec_content"]
        band = None
        try:
            tmp = tempfile.NamedTemporaryFile(
                suffix=".nec", dir=str(_user_nec_dir), delete=False, mode="w",
            )
            tmp.write(nec_content)
            tmp.close()
            parsed = parser.parse_file(Path(tmp.name))
            cls = classifier.classify(parsed)
            band = cls.band
        except Exception:
            pass

        row = create_antenna(
            name=f"🎲 {nice_name}",
            description=f"Surprise design! Classic {nice_name}.",
            antenna_type=result.get("classified_type", antenna_type),
            frequency_mhz=freq,
            band=band,
            ground_type="free_space",
            nec_content=nec_content,
            source="surprise",
            metadata={
                "model": result.get("model"),
                "usage": result.get("usage"),
                "iterations": result.get("iterations"),
                "classified_type": result.get("classified_type"),
                "confidence": result.get("confidence"),
                "refinement_log": result.get("refinement_log"),
                "steps": result.get("steps"),
                "goal_verdict": result.get("goal_verdict"),
                "buildability": result.get("buildability"),
                "surprise_pick": nice_name,
            },
            owner_user_id=hf_user["user_id"],
        )
        for k in ("created_at", "updated_at"):
            if hasattr(row.get(k), "isoformat"):
                row[k] = row[k].isoformat()
        return JSONResponse(row, status_code=201)

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
