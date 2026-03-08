#!/usr/bin/env python3
"""End-to-end test for the structured 7-step NEC generation pipeline.

Exercises the pipeline on real PDFs, URLs, and form specs to verify
that the decomposed approach (classify → extract → generate → validate →
convert → simulate → feedback) works on actual antenna documents.

Usage:
  python scripts/test_structured_pipeline.py --pdf pdfs/Cebik2a.pdf
  python scripts/test_structured_pipeline.py --form
  python scripts/test_structured_pipeline.py --all-pdfs
  python scripts/test_structured_pipeline.py --all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def _section(title: str) -> None:
    print(f"\n--- {title} ---")


def _print_pipeline_result(pr) -> None:
    """Pretty-print a PipelineResult object."""
    from antenna_classifier.nec_pipeline import PipelineResult

    # Steps log
    _section("Pipeline Steps")
    for s in pr.steps:
        print(f"  [{s.step}] {s.name:10s}  {s.status:5s}  {s.detail}")
    print(f"  Iterations: {pr.iterations}")

    # Extracted concepts
    _section("Extracted Concepts")
    if pr.concepts:
        c = pr.concepts
        print(f"  Type:       {c.antenna_type}")
        print(f"  Freq:       {c.freq_mhz} MHz")
        print(f"  Gain:       {c.gain_dbi}")
        print(f"  F/B:        {c.fb_db}")
        print(f"  SWR:        {c.max_swr}")
        print(f"  Ground:     {c.ground_type}")
        print(f"  Elements:   {c.elements}")
        print(f"  Wire dia:   {c.wire_dia_mm} mm")
        print(f"  Height:     {c.height_m} m")

    # NEC content
    _section("NEC Content (first 30 lines)")
    nec = pr.nec_content or "(none)"
    for i, line in enumerate(nec.splitlines()[:30], 1):
        print(f"  {i:3d}| {line}")
    total = len(nec.splitlines())
    if total > 30:
        print(f"  ... ({total - 30} more lines)")

    # Classification
    _section("Reverse Classification")
    print(f"  Type:       {pr.classified_type}")
    print(f"  Confidence: {pr.confidence:.2f}")

    # Goals
    _section("Goal Verdict")
    gv = pr.goal_verdict
    if gv:
        print(f"  Score:    {gv.get('score', 0):.2f}")
        print(f"  Passed:   {gv.get('checks_passed', '?')}/{gv.get('checks_total', '?')}")
        for chk in gv.get("checks", [])[:8]:
            mark = "PASS" if chk.get("passed") else "FAIL"
            print(f"    [{mark}] {chk.get('name', '?')}: "
                  f"{chk.get('actual', '?')} (target: {chk.get('target', '?')})")
    else:
        print("  (no goal verdict)")

    # Buildability
    _section("Buildability")
    bld = pr.buildability
    if bld:
        print(f"  Score: {bld.get('score', 0):.0f}/100")
    else:
        print("  (no buildability report)")

    # Token usage
    _section("Token Usage")
    print(f"  Prompt:     {pr.usage.get('prompt_tokens', 0)}")
    print(f"  Completion: {pr.usage.get('completion_tokens', 0)}")


# ---------------------------------------------------------------------------
# PDF tests
# ---------------------------------------------------------------------------

PDF_TESTS = [
    {"path": "pdfs/Cebik2a.pdf", "antenna_type": "yagi",
     "note": "Cebik 2-element Yagi design article"},
    {"path": "pdfs/9107030.pdf", "antenna_type": "quad",
     "note": "Quad loop antenna paper"},
    {"path": "pdfs/Ceecom_10-11m_Moxon_user.pdf", "antenna_type": "moxon",
     "note": "Ceecom Moxon user manual"},
    {"path": "pdfs/3 BAND QUAD LOOP.pdf", "antenna_type": "quad",
     "note": "Multi-band quad loop"},
    {"path": "pdfs/POTA-PERformer-Antenna-by-KJ6ER-2023-03-1-1.pdf", "antenna_type": "",
     "note": "POTA field antenna (type unknown, let pipeline classify)"},
    {"path": "pdfs/20m-elevated-vertical-g8ode-iss-1-31.pdf", "antenna_type": "vertical",
     "note": "Elevated vertical for 20m"},
]


def test_pdf(pdf_path: str, antenna_type: str = "", note: str = "") -> dict:
    """Run the structured pipeline on a single PDF."""
    from antenna_classifier.nec_pipeline import pipeline_from_pdf

    label = os.path.basename(pdf_path)
    if note:
        label += f" ({note})"
    _banner(f"PDF: {label}")
    if antenna_type:
        print(f"  Type hint: {antenna_type}")

    t0 = time.time()
    try:
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        pr = pipeline_from_pdf(
            pdf_bytes,
            antenna_type=antenna_type,
        )
        elapsed = time.time() - t0

        _print_pipeline_result(pr)
        print(f"\n  Elapsed: {elapsed:.1f}s")

        # Determine if the NEC passed nec2c
        sim_steps = [s for s in pr.steps if s.name == "simulate"]
        sim_ok = any(s.status == "ok" for s in sim_steps)

        return {
            "test": os.path.basename(pdf_path),
            "note": note,
            "status": "ok",
            "nec2c_pass": sim_ok,
            "classified_type": pr.classified_type,
            "confidence": pr.confidence,
            "iterations": pr.iterations,
            "goal_score": pr.goal_verdict.get("score", 0) if pr.goal_verdict else None,
            "steps": len(pr.steps),
            "elapsed": round(elapsed, 1),
            "tokens": pr.usage.get("prompt_tokens", 0) + pr.usage.get("completion_tokens", 0),
        }
    except Exception as exc:
        elapsed = time.time() - t0
        print(f"\n  ERROR after {elapsed:.1f}s: {exc}")
        import traceback
        traceback.print_exc()
        return {
            "test": os.path.basename(pdf_path),
            "note": note,
            "status": "error",
            "error": str(exc)[:200],
            "elapsed": round(elapsed, 1),
        }


# ---------------------------------------------------------------------------
# Form tests
# ---------------------------------------------------------------------------

FORM_TESTS = [
    {"antenna_type": "yagi", "frequency_mhz": 28.4, "name": "3el 10m Yagi"},
    {"antenna_type": "moxon", "frequency_mhz": 14.175, "name": "20m Moxon"},
    {"antenna_type": "dipole", "frequency_mhz": 7.1, "name": "40m Dipole"},
    {"antenna_type": "vertical", "frequency_mhz": 14.175, "name": "20m Vertical"},
]


def test_form(spec: dict) -> dict:
    """Run the structured pipeline on form data."""
    from antenna_classifier.nec_pipeline import pipeline_from_form

    _banner(f"FORM: {spec['name']}")
    t0 = time.time()
    try:
        pr = pipeline_from_form(
            antenna_type=spec["antenna_type"],
            frequency_mhz=spec["frequency_mhz"],
            ground_type="free_space",
            description=f"Classic {spec['name']} design",
        )
        elapsed = time.time() - t0
        _print_pipeline_result(pr)
        print(f"\n  Elapsed: {elapsed:.1f}s")

        sim_steps = [s for s in pr.steps if s.name == "simulate"]
        sim_ok = any(s.status == "ok" for s in sim_steps)

        return {
            "test": spec["name"],
            "status": "ok",
            "nec2c_pass": sim_ok,
            "classified_type": pr.classified_type,
            "confidence": pr.confidence,
            "iterations": pr.iterations,
            "goal_score": pr.goal_verdict.get("score", 0) if pr.goal_verdict else None,
            "steps": len(pr.steps),
            "elapsed": round(elapsed, 1),
            "tokens": pr.usage.get("prompt_tokens", 0) + pr.usage.get("completion_tokens", 0),
        }
    except Exception as exc:
        elapsed = time.time() - t0
        print(f"\n  ERROR after {elapsed:.1f}s: {exc}")
        import traceback
        traceback.print_exc()
        return {
            "test": spec["name"],
            "status": "error",
            "error": str(exc)[:200],
            "elapsed": round(elapsed, 1),
        }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results: list[dict]) -> None:
    _banner("SUMMARY")
    ok = sum(1 for r in results if r["status"] == "ok")
    nec2c_pass = sum(1 for r in results if r.get("nec2c_pass"))
    fail = len(results) - ok
    print(f"  Tests: {len(results)}   OK: {ok}   FAIL: {fail}   nec2c pass: {nec2c_pass}")
    print()

    hdr = f"  {'Test':40s}  {'Time':>6s}  {'Status':>6s}  {'nec2c':>5s}  {'Type':>12s}  {'Conf':>5s}  {'Goals':>6s}  {'Iter':>4s}  {'Tok':>6s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in results:
        if r["status"] == "ok":
            gs = f"{r.get('goal_score', 0):.2f}" if r.get("goal_score") is not None else "n/a"
            n2c = "PASS" if r.get("nec2c_pass") else "FAIL"
            print(f"  {r['test']:40s}  {r['elapsed']:5.1f}s  {'OK':>6s}  {n2c:>5s}  "
                  f"{r.get('classified_type', '?'):>12s}  {r.get('confidence', 0):5.2f}  "
                  f"{gs:>6s}  {r.get('iterations', '?'):>4}  {r.get('tokens', 0):>6}")
        else:
            err = r.get("error", "")[:50]
            print(f"  {r['test']:40s}  {r['elapsed']:5.1f}s  {'ERROR':>6s}  {'':>5s}  "
                  f"{'':>12s}  {'':>5s}  {'':>6s}  {'':>4s}  {err}")


def main():
    parser = argparse.ArgumentParser(
        description="Test the structured 7-step NEC pipeline end-to-end.",
    )
    parser.add_argument("--pdf", type=str, help="Path to a PDF file")
    parser.add_argument("--pdf-type", type=str, default="", help="Antenna type hint for --pdf")
    parser.add_argument("--all-pdfs", action="store_true", help="Run all built-in PDF tests")
    parser.add_argument("--form", action="store_true", help="Run form-based tests")
    parser.add_argument("--all", action="store_true", help="Run all tests (form + all PDFs)")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set.")
        sys.exit(1)

    results: list[dict] = []

    if args.form or args.all:
        for spec in FORM_TESTS:
            results.append(test_form(spec))

    if args.all_pdfs or args.all:
        base = os.path.join(os.path.dirname(__file__), "..")
        for pt in PDF_TESTS:
            full = os.path.join(base, pt["path"])
            if os.path.exists(full):
                results.append(test_pdf(full, pt["antenna_type"], pt.get("note", "")))
            else:
                print(f"  SKIP (not found): {pt['path']}")

    if args.pdf:
        results.append(test_pdf(args.pdf, antenna_type=args.pdf_type))

    if results:
        print_summary(results)

        report_path = os.path.join(
            os.path.dirname(__file__), "..", "reports", "pipeline_test_results.json"
        )
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Report: {report_path}")


if __name__ == "__main__":
    main()
