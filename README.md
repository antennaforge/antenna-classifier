# antenna-classifier

NEC file parser, validator, and antenna type classifier. Scans collections of NEC
antenna model files, validates that they parse correctly, and classifies each by
antenna type (yagi, dipole, vertical, loop, etc.).

## Purpose

Given a directory tree of `.nec` files (e.g. from the `antenna-agent-model/nec_files/`
collection), produce a curated catalog of **working design starters** — files that
parse cleanly and are tagged with their antenna type, frequency band, element count,
and ground configuration.

## Installation

```bash
pip install -e .
```

## Usage

### Check a single file

```bash
antenna-classifier check path/to/antenna.nec
```

### Scan a directory

```bash
antenna-classifier scan path/to/nec_files/
antenna-classifier scan path/to/nec_files/ --valid-only
antenna-classifier scan path/to/nec_files/ --type yagi --min-confidence 0.5
```

### Generate a catalog report

```bash
antenna-classifier report path/to/nec_files/ --format json -o catalog.json
antenna-classifier report path/to/nec_files/ --format csv -o catalog.csv
```

## Architecture

```
src/antenna_classifier/
├── __init__.py
├── cli.py            # Command-line interface (scan, check, report)
├── parser.py         # NEC file parser with SY variable resolution
├── validator.py      # Structural and domain-compliance validation
└── classifier.py     # Antenna type classification from geometry + metadata
```

### Parser (`parser.py`)

Two-pass parser adapted from the `antenna-agent-model` project:
1. **Pass 1**: Build symbol table from SY variable definition cards
2. **Pass 2**: Parse all cards with type-safe parameter handling using `CARD_SPECS`

Handles: comma/whitespace delimiters, inline comments, AWG wire gauge notation,
scientific notation, symbolic expressions.

### Validator (`validator.py`)

Checks for runnable antenna models:
- Geometry cards present (GW, GA, etc.)
- Excitation source (EX) defined
- Frequency (FR) specified
- Wire parameters valid (positive radius, non-zero length)
- Tag cross-references (EX/LD reference existing wire tags)
- Card ordering (geometry before control)

### Classifier (`classifier.py`)

Multi-strategy classification pipeline:
1. **Comment keywords** — CM/CE text searched for antenna type names (highest confidence)
2. **Filename keywords** — File/path names matched against known types
3. **Geometry heuristics** — Structural analysis of wire topology:
   - Helix: GH cards present
   - Patch: SP/SM surface patch cards
   - Loop/Quad: closed wire paths
   - Hexbeam: 6-fold wire symmetry
   - LPDA: monotonically-sized elements with TL feed
   - Vertical: primarily-vertical excited element with ground
   - Yagi: parallel horizontal elements along a boom axis
   - Phased array: multiple excitation sources
4. **Directory name fallback** — Infers type from parent folder

## Supported Antenna Types

yagi, dipole, vertical, loop, quad, hexbeam, lpda, phased_array, helix,
collinear, inverted_v, end_fed, j_pole, moxon, wire_array, patch, fractal,
magnetic_loop, bobtail_curtain, rhombic, beverage, discone, turnstile
