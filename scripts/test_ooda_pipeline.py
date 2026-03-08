#!/usr/bin/env python3
"""Standalone test script for the full OODA NEC generation pipeline.

Exercises all three input sources:
  1. Form-based generation (with calculator-injected dimensions)
  2. URL-based generation (fetch a web page describing an antenna)
  3. PDF-based generation (from a local or downloaded PDF)

Requirements:
  - OPENAI_API_KEY env var set
  - Python >= 3.10
  - pip install beautifulsoup4 pdfplumber openai
  - NEC solver at localhost:8787 (optional — layers 3+ degrade gracefully)

Usage:
  python scripts/test_ooda_pipeline.py [--form] [--url] [--pdf PATH] [--all]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import time

# Ensure the src/ package is importable when run from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ---------------------------------------------------------------------------
# Pretty-printing helpers
# ---------------------------------------------------------------------------

def _banner(title: str) -> None:
    width = 72
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def _section(title: str) -> None:
    print(f"\n--- {title} ---")


def _print_result(result: dict) -> None:
    """Pretty-print the result dict from a generate_nec_from_* call."""
    _section("NEC content (first 40 lines)")
    nec = result.get("nec_content", "")
    for i, line in enumerate(nec.splitlines()[:40], 1):
        print(f"  {i:3d}| {line}")
    total_lines = len(nec.splitlines())
    if total_lines > 40:
        print(f"  ... ({total_lines - 40} more lines)")

    _section("Classification")
    print(f"  Type:       {result.get('classified_type', '?')}")
    print(f"  Confidence: {result.get('confidence', 0):.2f}")

    _section("Goal Verdict")
    gv = result.get("goal_verdict")
    if gv:
        print(f"  Score:    {gv.get('score', 0):.2f}")
        print(f"  Passed:   {gv.get('checks_passed', '?')}/{gv.get('checks_total', '?')}")
        for chk in gv.get("checks", [])[:8]:
            mark = "PASS" if chk.get("passed") else "FAIL"
            print(f"    [{mark}] {chk.get('name', '?')}: "
                  f"{chk.get('actual', '?')} (target: {chk.get('target', '?')})")
    else:
        print("  (no goal verdict — nec2c solver may be offline)")

    _section("Buildability")
    bld = result.get("buildability")
    if bld:
        print(f"  Score: {bld.get('score', 0):.0f}/100")
        dims = bld.get("dimensions", {})
        for k, v in dims.items():
            print(f"    {k}: {v:.0f}")
    else:
        print("  (no buildability — solver may be offline)")

    _section("Refinement Log")
    rlog = result.get("refinement_log", [])
    print(f"  Iterations: {len(rlog)}")
    for entry in rlog:
        layer = entry.get("layer", "?")
        passed = entry.get("passed", False)
        it = entry.get("iteration", "?")
        status = "PASS" if passed else "FAIL"
        extra = ""
        if "classified_type" in entry:
            extra += f" type={entry['classified_type']}"
        if "confidence" in entry:
            extra += f" conf={entry['confidence']:.2f}"
        if "goal_verdict" in entry and entry["goal_verdict"]:
            gsc = entry["goal_verdict"].get("score", 0) if isinstance(entry["goal_verdict"], dict) else 0
            extra += f" goal={gsc:.2f}"
        if "buildability_score" in entry and entry["buildability_score"] is not None:
            extra += f" build={entry['buildability_score']:.0f}"
        if "issue" in entry:
            extra += f" issue={entry['issue'][:80]}"
        print(f"  [{it}] {layer}: {status}{extra}")

    _section("Token Usage")
    usage = result.get("usage", {})
    print(f"  Prompt:     {usage.get('prompt_tokens', '?')}")
    print(f"  Completion: {usage.get('completion_tokens', '?')}")
    print(f"  Finish:     {usage.get('finish_reason', '?')}")


# ---------------------------------------------------------------------------
# Test: Form-based generation
# ---------------------------------------------------------------------------

FORM_TESTS = [
    {
        "name": "20m Moxon Rectangle",
        "antenna_type": "moxon",
        "frequency_mhz": 14.175,
        "ground_type": "free_space",
        "description": "Single-band Moxon rectangle for 20 metres.",
    },
    {
        "name": "3-element 10m Yagi",
        "antenna_type": "yagi",
        "frequency_mhz": 28.4,
        "ground_type": "free_space",
        "description": "3-element Yagi-Uda beam for 10 metres.",
    },
    {
        "name": "15m Hexbeam",
        "antenna_type": "hexbeam",
        "frequency_mhz": 21.2,
        "ground_type": "real",
        "description": "Broadband hexagonal beam for 15 metres.",
    },
]


def test_form_generation() -> list[dict]:
    """Run form-based generation tests with calculator injection."""
    from antenna_classifier.nec_generator import generate_nec_from_form

    results = []
    for test in FORM_TESTS:
        _banner(f"FORM TEST: {test['name']}")
        t0 = time.time()
        try:
            result = generate_nec_from_form(
                antenna_type=test["antenna_type"],
                frequency_mhz=test["frequency_mhz"],
                ground_type=test["ground_type"],
                description=test["description"],
            )
            elapsed = time.time() - t0
            _print_result(result)
            print(f"\n  Elapsed: {elapsed:.1f}s")
            results.append({
                "test": test["name"],
                "status": "ok",
                "type": result.get("classified_type"),
                "confidence": result.get("confidence"),
                "iterations": result.get("iterations"),
                "elapsed": round(elapsed, 1),
            })
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"\n  ERROR after {elapsed:.1f}s: {exc}")
            results.append({
                "test": test["name"],
                "status": "error",
                "error": str(exc),
                "elapsed": round(elapsed, 1),
            })
    return results


# ---------------------------------------------------------------------------
# Test: URL-based generation
# ---------------------------------------------------------------------------

URL_TESTS = [
    {
        "name": "G3TXQ Hexbeam (karinya.net)",
        "url": "http://www.karinya.net/g3txq/hexbeam/",
        "antenna_type": "hexbeam",
    },
    {
        "name": "ARRL Moxon Rectangle",
        "url": "https://www.arrl.org/moxon-rectangle",
        "antenna_type": "moxon",
    },
]


def test_url_generation() -> list[dict]:
    """Run URL-based generation tests (fetches live pages)."""
    from antenna_classifier.nec_generator import generate_nec_from_url

    results = []
    for test in URL_TESTS:
        _banner(f"URL TEST: {test['name']}")
        print(f"  URL: {test['url']}")
        t0 = time.time()
        try:
            result = generate_nec_from_url(
                test["url"],
                antenna_type=test.get("antenna_type", ""),
            )
            elapsed = time.time() - t0

            # Show extracted text preview
            url_text = result.get("url_text", "")
            _section("Extracted text (first 500 chars)")
            print(textwrap.indent(url_text[:500], "  "))

            _print_result(result)
            print(f"\n  Elapsed: {elapsed:.1f}s")
            results.append({
                "test": test["name"],
                "status": "ok",
                "type": result.get("classified_type"),
                "confidence": result.get("confidence"),
                "iterations": result.get("iterations"),
                "elapsed": round(elapsed, 1),
            })
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"\n  ERROR after {elapsed:.1f}s: {exc}")
            results.append({
                "test": test["name"],
                "status": "error",
                "error": str(exc),
                "elapsed": round(elapsed, 1),
            })
    return results


# ---------------------------------------------------------------------------
# Test: PDF-based generation
# ---------------------------------------------------------------------------

def test_pdf_generation(pdf_path: str, antenna_type: str = "") -> list[dict]:
    """Run PDF-based generation on a local file."""
    from antenna_classifier.nec_generator import generate_nec_from_pdf

    label = f"PDF TEST: {pdf_path}"
    if antenna_type:
        label += f" (type hint: {antenna_type})"
    _banner(label)
    results = []
    t0 = time.time()
    try:
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        result = generate_nec_from_pdf(
            pdf_bytes,
            extra_instructions="Build a NEC model from this antenna description.",
            antenna_type=antenna_type,
        )
        elapsed = time.time() - t0
        _print_result(result)
        print(f"\n  Elapsed: {elapsed:.1f}s")
        results.append({
            "test": os.path.basename(pdf_path),
            "status": "ok",
            "type": result.get("classified_type"),
            "confidence": result.get("confidence"),
            "iterations": result.get("iterations"),
            "elapsed": round(elapsed, 1),
        })
    except Exception as exc:
        elapsed = time.time() - t0
        print(f"\n  ERROR after {elapsed:.1f}s: {exc}")
        results.append({
            "test": os.path.basename(pdf_path),
            "status": "error",
            "error": str(exc),
            "elapsed": round(elapsed, 1),
        })
    return results


# ---------------------------------------------------------------------------
# Test: Calculator sanity check (no AI calls)
# ---------------------------------------------------------------------------

def test_calculators() -> None:
    """Quick sanity check that calculators produce valid dimensions."""
    from antenna_classifier.nec_calculators import calc_for_type, supported_types

    _banner("CALCULATOR SANITY CHECK (no AI)")
    types_to_test = [
        ("dipole", 14.175),
        ("yagi", 28.4),
        ("moxon", 14.175),
        ("hexbeam", 21.2),
        ("vertical", 7.15),
        ("magnetic_loop", 14.175),
        ("j_pole", 145.0),
    ]
    for atype, freq in types_to_test:
        calc = calc_for_type(atype, freq)
        if calc is None:
            print(f"  {atype:20s} @ {freq:8.3f} MHz — no calculator")
            continue
        # Verify dimensions are positive (values may be floats or lists)
        # Note: some values like radial_slope_deg can be zero legitimately
        def _positive(v):
            if isinstance(v, (list, tuple)):
                return all(x >= 0 for x in v)
            return v >= 0

        all_positive = all(_positive(v) for v in calc.dimensions.values())
        status = "OK" if all_positive else "WARN"

        def _fmt(v):
            if isinstance(v, (list, tuple)):
                return "[" + ",".join(f"{x:.4f}" for x in v) + "]"
            return f"{v:.4f}"

        dim_str = ", ".join(f"{k}={_fmt(v)}" for k, v in list(calc.dimensions.items())[:4])
        print(f"  [{status}] {atype:20s} @ {freq:8.3f} MHz — {dim_str}")

    print(f"\n  Supported types: {', '.join(sorted(supported_types()))}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test the OODA NEC generation pipeline end-to-end.",
    )
    parser.add_argument("--form", action="store_true", help="Run form-based tests")
    parser.add_argument("--url", action="store_true", help="Run URL-based tests")
    parser.add_argument("--pdf", type=str, help="Path to a PDF file to test")
    parser.add_argument("--pdf-type", type=str, default="", help="Antenna type hint for PDF test (e.g. moxon, yagi)")
    parser.add_argument("--calc", action="store_true", help="Run calculator sanity check only (no AI)")
    parser.add_argument("--all", action="store_true", help="Run form + URL tests")
    args = parser.parse_args()

    # Default: run calc check if nothing specified
    run_form = args.form or args.all
    run_url = args.url or args.all
    run_pdf = args.pdf is not None
    run_calc = args.calc or not (run_form or run_url or run_pdf)

    if (run_form or run_url or run_pdf) and not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set. AI tests require it.")
        print("  For calculator-only test, run: python scripts/test_ooda_pipeline.py --calc")
        sys.exit(1)

    all_results: list[dict] = []

    if run_calc:
        test_calculators()

    if run_form:
        all_results.extend(test_form_generation())

    if run_url:
        all_results.extend(test_url_generation())

    if run_pdf:
        all_results.extend(test_pdf_generation(args.pdf, antenna_type=args.pdf_type))

    # Summary
    if all_results:
        _banner("SUMMARY")
        ok = sum(1 for r in all_results if r["status"] == "ok")
        fail = len(all_results) - ok
        print(f"  Tests: {len(all_results)}   OK: {ok}   FAIL: {fail}\n")
        for r in all_results:
            status = r["status"].upper()
            extra = ""
            if r["status"] == "ok":
                extra = f"type={r.get('type')} conf={r.get('confidence', 0):.2f} iters={r.get('iterations')}"
            else:
                extra = r.get("error", "")[:60]
            print(f"  [{status:5s}] {r['test']:40s}  {r.get('elapsed', 0):6.1f}s  {extra}")

        # Write JSON report
        report_path = os.path.join(
            os.path.dirname(__file__), "..", "reports", "ooda_test_results.json"
        )
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n  Report written to: {report_path}")


if __name__ == "__main__":
    main()
