"""
CLI entry point for antenna-classifier.

Usage:
    antenna-classifier scan <directory>       Scan NEC files, validate, classify, report
    antenna-classifier check <file.nec>       Parse + validate + classify a single file
    antenna-classifier report <directory>      Generate a JSON/CSV catalog of all NEC files
    antenna-classifier fingerprint <path>      Show card-config fingerprint (file or dir)
    antenna-classifier similar <file> <dir>    Find NEC files with similar card structure
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from . import classifier, parser, validator
from .fingerprint import fingerprint as make_fingerprint, find_similar, similarity, build_archetype, classify_by_fingerprint


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="antenna-classifier",
        description="NEC file parser, validator, and antenna type classifier",
    )
    sub = ap.add_subparsers(dest="command")

    # --- scan ---
    p_scan = sub.add_parser("scan", help="Scan a directory of NEC files")
    p_scan.add_argument("directory", type=Path, help="Directory to scan (recursive)")
    p_scan.add_argument("--valid-only", action="store_true", help="Only show valid files")
    p_scan.add_argument("--type", dest="filter_type", help="Filter by antenna type")
    p_scan.add_argument("--min-confidence", type=float, default=0.0, help="Min classification confidence")

    # --- check ---
    p_check = sub.add_parser("check", help="Check a single NEC file")
    p_check.add_argument("file", type=Path, help="NEC file to check")

    # --- report ---
    p_report = sub.add_parser("report", help="Generate catalog report")
    p_report.add_argument("directory", type=Path, help="Directory to scan (recursive)")
    p_report.add_argument("--format", choices=["json", "csv"], default="json", help="Output format")
    p_report.add_argument("--output", "-o", type=Path, help="Output file (default: stdout)")

    # --- fingerprint ---
    p_fp = sub.add_parser("fingerprint", help="Show card-config fingerprint")
    p_fp.add_argument("path", type=Path, help="NEC file or directory")
    p_fp.add_argument("--json", dest="as_json", action="store_true", help="Output as JSON")

    # --- similar ---
    p_sim = sub.add_parser("similar", help="Find structurally similar NEC files")
    p_sim.add_argument("file", type=Path, help="Reference NEC file")
    p_sim.add_argument("directory", type=Path, help="Directory to search")
    p_sim.add_argument("-n", "--top", type=int, default=10, help="Number of results")
    p_sim.add_argument("--min-sim", type=float, default=0.5, help="Minimum similarity")

    # --- dashboard ---
    p_dash = sub.add_parser("dashboard", help="Launch interactive web dashboard")
    p_dash.add_argument("directory", type=Path, help="Directory of NEC files")
    p_dash.add_argument("--port", type=int, default=8501, help="Server port (default: 8501)")
    p_dash.add_argument("--host", default="0.0.0.0", help="Server host (default: 0.0.0.0)")
    p_dash.add_argument("--solver-url", default=None, help="NEC solver URL (default: http://localhost:8787)")

    args = ap.parse_args(argv)

    if args.command == "scan":
        return _cmd_scan(args)
    elif args.command == "check":
        return _cmd_check(args)
    elif args.command == "report":
        return _cmd_report(args)
    elif args.command == "fingerprint":
        return _cmd_fingerprint(args)
    elif args.command == "similar":
        return _cmd_similar(args)
    elif args.command == "dashboard":
        return _cmd_dashboard(args)
    else:
        ap.print_help()
        return 1


def _collect_nec_files(directory: Path) -> list[Path]:
    """Recursively find all .nec files."""
    return sorted(directory.rglob("*.nec"))


def _process_file(path: Path) -> dict:
    """Parse, validate, classify a single file. Returns a summary dict."""
    parsed = parser.parse_file(path)
    val = validator.validate(parsed)
    cls = classifier.classify(parsed)
    fp = make_fingerprint(parsed)
    return {
        "path": str(path),
        "filename": path.name,
        "valid": val.valid,
        "errors": len(val.errors),
        "warnings": len(val.warnings),
        "antenna_type": cls.antenna_type,
        "confidence": round(cls.confidence, 2),
        "subtypes": cls.subtypes,
        "evidence": cls.evidence,
        "frequency_mhz": cls.frequency_mhz,
        "band": cls.band,
        "element_count": cls.element_count,
        "ground_type": cls.ground_type,
        "wire_count": len(parsed.wire_cards),
        "card_count": len(parsed.cards),
        "fingerprint": fp.signature,
        "complexity": round(fp.complexity_score, 3),
        "feed_complexity": round(fp.feed_complexity, 3),
        "validation_issues": [
            {"severity": i.severity.value, "message": i.message, "line": i.line}
            for i in val.issues
        ],
    }


def _cmd_check(args: argparse.Namespace) -> int:
    path = args.file
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        return 1

    result = _process_file(path)
    parsed = parser.parse_file(path)

    # Header
    print(f"\n{'=' * 70}")
    print(f"  NEC File: {path.name}")
    print(f"{'=' * 70}")

    # Classification
    cls = classifier.classify(parsed)
    status = "VALID" if result["valid"] else "INVALID"
    print(f"  Status:     {status}")
    print(f"  Type:       {cls.label}")
    print(f"  Confidence: {result['confidence']:.0%}")
    if cls.frequency_mhz:
        print(f"  Frequency:  {cls.frequency_mhz} MHz ({cls.band or 'unknown band'})")
    print(f"  Elements:   {result['element_count']} wire group(s), {result['wire_count']} GW card(s)")
    print(f"  Ground:     {result['ground_type']}")

    # Evidence
    if cls.evidence:
        print(f"\n  Classification evidence:")
        for ev in cls.evidence:
            print(f"    - {ev}")

    # Validation
    val = validator.validate(parsed)
    if val.issues:
        print(f"\n  Validation ({len(val.errors)} error(s), {len(val.warnings)} warning(s)):")
        for issue in val.issues:
            icon = "X" if issue.severity.value == "error" else ("!" if issue.severity.value == "warning" else "i")
            loc = f" (line {issue.line})" if issue.line else ""
            print(f"    [{icon}] {issue.message}{loc}")

    # Fingerprint
    fp = make_fingerprint(parsed)
    print(f"\n  Fingerprint: {fp.signature}")
    print(f"  Complexity:  {fp.complexity_score:.3f}")
    print(f"  Feed:        {fp.feed_complexity:.3f}")

    print()
    return 0 if result["valid"] else 1


def _cmd_scan(args: argparse.Namespace) -> int:
    directory = args.directory
    if not directory.is_dir():
        print(f"Not a directory: {directory}", file=sys.stderr)
        return 1

    files = _collect_nec_files(directory)
    if not files:
        print(f"No .nec files found in {directory}")
        return 0

    total = len(files)
    valid_count = 0
    invalid_count = 0
    type_counts: dict[str, int] = {}

    print(f"\nScanning {total} NEC files in {directory}\n")
    print(f"{'Status':<8} {'Type':<20} {'Conf':>5} {'Freq':>10} {'Band':<8} {'File'}")
    print("-" * 90)

    for path in files:
        result = _process_file(path)

        if args.valid_only and not result["valid"]:
            continue
        if args.filter_type and result["antenna_type"] != args.filter_type:
            continue
        if result["confidence"] < args.min_confidence:
            continue

        status = "OK" if result["valid"] else "FAIL"
        freq_str = f"{result['frequency_mhz']:.1f}" if result["frequency_mhz"] else "-"
        band_str = result["band"] or "-"
        rel_path = path.relative_to(directory) if path.is_relative_to(directory) else path.name

        print(f"{status:<8} {result['antenna_type']:<20} {result['confidence']:>5.0%} {freq_str:>10} {band_str:<8} {rel_path}")

        if result["valid"]:
            valid_count += 1
        else:
            invalid_count += 1
        type_counts[result["antenna_type"]] = type_counts.get(result["antenna_type"], 0) + 1

    # Summary
    shown = valid_count + invalid_count
    print(f"\n{'=' * 90}")
    print(f"  Total: {total} files | Shown: {shown} | Valid: {valid_count} | Invalid: {invalid_count}")
    print(f"\n  Types found:")
    for atype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {atype:<24} {count}")
    print()

    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    directory = args.directory
    if not directory.is_dir():
        print(f"Not a directory: {directory}", file=sys.stderr)
        return 1

    files = _collect_nec_files(directory)
    if not files:
        print(f"No .nec files found in {directory}", file=sys.stderr)
        return 0

    records = []
    for path in files:
        result = _process_file(path)
        # Simplify for catalog output
        records.append({
            "filename": result["filename"],
            "path": result["path"],
            "valid": result["valid"],
            "antenna_type": result["antenna_type"],
            "confidence": result["confidence"],
            "fingerprint": result["fingerprint"],
            "complexity": result["complexity"],
            "frequency_mhz": result["frequency_mhz"],
            "band": result["band"],
            "element_count": result["element_count"],
            "wire_count": result["wire_count"],
            "ground_type": result["ground_type"],
            "errors": result["errors"],
            "warnings": result["warnings"],
            "evidence": "; ".join(result["evidence"]),
        })

    out = args.output.open("w") if args.output else sys.stdout

    if args.format == "json":
        json.dump(records, out, indent=2)
        out.write("\n")
    else:  # csv
        fieldnames = list(records[0].keys())
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    if args.output:
        out.close()
        print(f"Report written to {args.output} ({len(records)} files)", file=sys.stderr)

    return 0


def _cmd_fingerprint(args: argparse.Namespace) -> int:
    path = args.path
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = _collect_nec_files(path)
    else:
        print(f"Not found: {path}", file=sys.stderr)
        return 1

    if not files:
        print("No .nec files found")
        return 0

    results = []
    for f in files:
        parsed = parser.parse_file(f)
        fp = make_fingerprint(parsed)
        cls = classifier.classify(parsed)
        if args.as_json:
            entry = fp.to_dict()
            entry["file"] = str(f)
            entry["antenna_type"] = cls.antenna_type
            results.append(entry)
        else:
            rel = f.relative_to(path) if f != path and f.is_relative_to(path) else f.name
            print(f"{fp.signature:<50} {cls.antenna_type:<18} {rel}")

    if args.as_json:
        json.dump(results, sys.stdout, indent=2)
        print()

    return 0


def _cmd_similar(args: argparse.Namespace) -> int:
    ref_path = args.file
    search_dir = args.directory

    if not ref_path.is_file():
        print(f"File not found: {ref_path}", file=sys.stderr)
        return 1
    if not search_dir.is_dir():
        print(f"Not a directory: {search_dir}", file=sys.stderr)
        return 1

    # Build reference fingerprint
    ref_parsed = parser.parse_file(ref_path)
    ref_fp = make_fingerprint(ref_parsed)
    ref_cls = classifier.classify(ref_parsed)

    print(f"\nReference: {ref_path.name}")
    print(f"  Type: {ref_cls.antenna_type} ({ref_cls.confidence:.0%})")
    print(f"  Fingerprint: {ref_fp.signature}")
    print()

    # Build candidate fingerprints
    candidates = []
    for f in _collect_nec_files(search_dir):
        if f.resolve() == ref_path.resolve():
            continue
        parsed = parser.parse_file(f)
        fp = make_fingerprint(parsed)
        rel = f.relative_to(search_dir) if f.is_relative_to(search_dir) else f
        candidates.append((str(rel), fp))

    matches = find_similar(ref_fp, candidates, top_n=args.top, min_similarity=args.min_sim)

    if not matches:
        print("No similar files found.")
        return 0

    print(f"{'Similarity':>10}  {'Signature':<45} {'File'}")
    print("-" * 100)
    for label, sim, fp in matches:
        print(f"{sim:>10.1%}  {fp.signature:<45} {label}")

    print()
    return 0


def _cmd_dashboard(args) -> int:
    """Launch the interactive web dashboard."""
    from .dashboard import run_dashboard

    if not args.directory.is_dir():
        print(f"Error: {args.directory} is not a directory", file=sys.stderr)
        return 1
    run_dashboard(
        nec_dir=args.directory,
        port=args.port,
        host=args.host,
        solver_url=args.solver_url,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
