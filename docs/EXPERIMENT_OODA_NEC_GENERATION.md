# Experiment: OODA Loop NEC Generation (Reverse-Classification Claw)

**Date:** 2026-03-08
**Status:** IN-PROGRESS
**Repos:** antenna-classifier (oracle), antenna-claw (skill 5)

## Problem Statement

AI models (GPT-5.2) can extract antenna specifications from PDFs and free-text
descriptions, but consistently fail to translate those specs into correct NEC2
geometry. Typical failures:

- All wires stacked at origin (collapsed geometry)
- Wrong element count or layout for the antenna type
- Moxon bent-rectangle modeled as a simple Yagi
- Hexbeam 6-spoke star modeled as stacked dipoles
- Phased arrays without TL cards

The model understands *what* the antenna is, but not *how* to lay out the NEC
wire geometry for that class.

## Insight: Reverse Classification as Oracle

We already have a battle-tested antenna classifier (30 types, 236+ tests) and a
fingerprint engine (21-dim feature vectors) that can identify antenna types from
NEC geometry with high confidence. This is the **inverse** of what the LLM needs
to do.

If we treat classification as an oracle, we can:
1. Generate NEC from the LLM
2. Classify the generated NEC
3. Compare against the target type
4. Feed mismatches back to the LLM as structured error signals

This creates a closed-loop system — the LLM proposes, the classifier disposes.

## Design: OODA Loop Architecture

The refinement loop follows the OODA (Observe-Orient-Decide-Act) military
decision-making framework:

```
                    ┌──────────────────────────────────────────┐
                    │           ANTENNA CLAW (OODA)            │
                    │                                          │
    Source ──────►  │  ┌─────────┐    ┌─────────┐             │
   (PDF/URL/       │  │ OBSERVE │    │  ORIENT  │             │
    Form/Text)     │  │         │    │          │             │
                    │  │ Parse   │    │ Classify │             │
                    │  │ Extract │──►│ Finger-  │             │
                    │  │ Validate│    │ print    │             │
                    │  └─────────┘    │ Compare  │             │
                    │                 └────┬─────┘             │
                    │                      │                   │
                    │                 ┌────▼─────┐             │
                    │                 │  DECIDE  │             │
                    │                 │          │             │
                    │                 │ Match?   │──► YES: Done│
                    │                 │ Max iter?│             │
                    │                 │ Feedback │             │
                    │                 └────┬─────┘             │
                    │                      │ NO                │
                    │                 ┌────▼─────┐             │
                    │                 │   ACT    │             │
                    │                 │          │             │
                    │                 │ Re-prompt│             │
                    │                 │ LLM with │             │
                    │                 │ feedback │             │
                    │                 └────┬─────┘             │
                    │                      │                   │
                    │                      └───── loop ────────│
                    └──────────────────────────────────────────┘
```

### Phase Details

| Phase | Action | Tools |
|-------|--------|-------|
| **Observe** | Parse NEC output, validate structure (required cards, collapsed geometry, degenerate repetition). Extract text from source (PDF/URL/form). | `parser.parse_text()`, `validator.validate()`, `extract_pdf_text()`, `extract_url_text()` |
| **Orient** | Classify the generated NEC, fingerprint it, compare against target antenna type. Load per-type context references. | `classifier.classify()`, `fingerprint()`, `_load_type_context()` |
| **Decide** | Is classified_type == target_type with confidence ≥ 0.6? If yes, accept. If max iterations reached, return best. Otherwise, build structured feedback. | `_analyze_nec()` |
| **Act** | Append classifier feedback to conversation, re-prompt the LLM. Include mismatch details: expected type, got type, evidence, fingerprint delta. | `_generate_and_refine()` |

### Audit Trail

Every OODA iteration emits structured log entries:

```json
{
  "iteration": 2,
  "phase": "orient",
  "classified_type": "yagi",
  "target_type": "moxon",
  "confidence": 0.72,
  "fingerprint": "GW6:TAG6:EX1:LD:GN2",
  "evidence": ["6 GW wires in bent rectangle", "keyword: moxon"],
  "decision": "type_mismatch",
  "feedback_sent": true
}
```

Logs are:
- Emitted to Python `logging` (visible in Docker logs)
- Collected in `refinement_log` list (stored in Postgres metadata JSON)
- Viewable in the dashboard UI as an expandable audit trail

## Input Sources

| Source | Extraction Method | Status |
|--------|-------------------|--------|
| Form (type + freq + description) | Direct string | DONE |
| PDF upload | pdfplumber text + tables | DONE |
| URL (web page) | HTTP fetch + HTML→text (BeautifulSoup) | NEW |
| Free-text description | Direct string | DONE (via antenna-claw skill 2) |

### Web Extraction (New)

For URLs like `http://www.karinya.net/g3txq/hexbeam/`, the system:
1. Fetches the page via `urllib.request` (no JS rendering needed for most antenna sites)
2. Strips HTML to plain text + extracts tables via BeautifulSoup
3. Identifies dimensional data (element lengths, spacings, wire gauges)
4. Passes extracted text to the LLM with the same OODA loop

## Antenna-Claw Integration (Skill 5)

This becomes **Skill 5: NEC Claw** in the antenna-claw agent repo:

| # | Skill | Status |
|---|-------|--------|
| 1 | Blog content generation | DONE |
| 2 | NEC2 file generation (extraction) | DONE |
| 3 | Web search (references) | DONE |
| 4 | Comic/cartoon generation | IN-PROGRESS |
| 5 | **NEC Claw (OODA generation)** | **NEW** |

The skill combines:
- Web search (skill 3) for finding antenna reference pages
- Antenna extraction (skill 2) for parameter extraction
- Reverse-classification oracle from antenna-classifier
- OODA refinement loop for iterative correction

## Success Criteria

1. Moxon PDF → classified as `moxon` with confidence ≥ 0.6 (6 GW wires, bent rectangle)
2. Hexbeam URL → classified as `hexbeam` with confidence ≥ 0.6 (12 GW wires, 6 bands)
3. Average iterations to match: ≤ 2 (initial + 1 correction)
4. Audit trail shows clear decision path from input to accepted NEC

## Implementation Plan

1. **antenna-classifier**: `nec_generator.py` — OODA loop with `_analyze_nec()`, `_generate_and_refine()`, URL extraction, audit logging [IN-PROGRESS]
2. **antenna-classifier**: Dashboard — URL input field, audit trail display
3. **antenna-claw**: Skill 5 manifest + SKILL.md — documents the NEC Claw skill
4. **antenna-claw**: Orchestrator — `--mode nec-claw` with URL/PDF/text inputs

---

## Experiment 2: Calculator Seeding

**Date:** 2026-03-13
**Commit:** `6c3c960`

### Hypothesis

The OODA loop improves *structural correctness* (right number of wires, correct
card types, valid NEC syntax) but fails on *physics accuracy* — the LLM doesn't
know the actual dimensions that make a quad loop resonate at 21 MHz or what tip
gap a Moxon needs. If we inject physics-based calculator dimensions into the
prompt, the LLM has a concrete numerical starting point instead of inventing
geometry from scratch.

### Discovery: The Seeding Gap

Before this experiment, the three generation paths had very different levels of
calculator support:

| Path | Calculator injection | Notes/hints |
|------|---------------------|-------------|
| **Form** (`generate_nec_from_form`) | ✅ Full — summary + notes + nec_hints | Already wired since calculators were built |
| **PDF** (`generate_nec_from_pdf`) | ❌ **None** | Biggest gap — PDFs sent to LLM with zero physics |
| **URL** (`generate_nec_from_url`) | ⚠️ Partial — summary only, hardcoded 14.0 MHz | Ignored actual antenna frequency |

The PDF path — the most common real-world use case — had **zero** calculator
data. The LLM was flying blind on every PDF upload.

Additionally, the quad and hexbeam calculators produced correct dimension numbers
but included **zero NEC modelling hints**. The LLM got "driven side = 5.35m" but
not "each quad element is 4 GW cards forming a closed square loop" or "feed at
bottom centre — split the bottom wire into two halves".

### Changes Made

#### 1. Frequency guesser (`_guess_freq_mhz`)

New helper that extracts design frequency from free text:
- Explicit MHz values: `"21.1 MHz"` → 21.1, `"146 mhz"` → 146.0
- Ham-band references: `"20m band"` → 14.175, `"10-meter"` → 28.4
- Fallback: 14.175 MHz (20 m) if nothing found

Uses `_BAND_CENTRES` dict mapping band numbers to centre frequencies.

#### 2. PDF path calculator injection

`generate_nec_from_pdf()` now:
1. Guesses frequency from extracted PDF text
2. Calls `calc_for_type(antenna_type, freq)` for physics-based dimensions
3. Injects full prompt block: summary + bullet-point notes + NEC hints
4. Frames as "starting point" so LLM treats them as anchors, not suggestions

#### 3. URL path upgrade

`generate_nec_from_url()` upgraded:
- `calc_for_type(antenna_type, 14.0)` → `calc_for_type(antenna_type, _guess_freq_mhz(url_text, antenna_type))`
- Added full notes + NEC hints injection (was summary-only before)

#### 4. Quad NEC hints (7 hints)

```
• Each quad element is 4 GW cards forming a closed square loop
• Loops in Y-Z plane, spaced along X (boom axis)
• Driven element: 4 wires, each side {S}m, square from Y=-S/2 to Y=+S/2, Z=0 to Z=S
• Reflector: 4 wires, each side {S}m, at X=-{spacing}
• Use UNIQUE tag numbers per wire (e.g. reflector 1-4, driven 5-8)
• Feed at bottom centre of driven element — split bottom wire into two halves
  with excitation at the junction
• Wire endpoints MUST connect exactly to close each loop
```

#### 5. Hexbeam NEC hints (6 hints)

```
• 6 GW cards per band: 3 driven wire sections + 3 reflector sections
• Driven wire is a W-shape in X-Y plane; reflector below it by vertical_spacing
• Each element forms an inverted-V between adjacent spreader tips
• Driven: 6 wire sections around hexagonal frame (radius {R}m)
• Reflector: same pattern, offset Z=-{spacing}m
• Feed at centre of driven element
```

### Results: Before vs After Seeding

#### PDF Generation

| Test PDF | Without Seeding | With Seeding | Change |
|----------|----------------|--------------|--------|
| 9107030 quad (OCR) | 0.50 (3/6 goals) | **0.83 (5/6 goals)** | **+66%** |
| 3 Band Quad Loop | 0.67 (4/6), build 65 | 0.67 (4/6), build **73** | build +12% |
| DL2GMS Moxon 10-15m | 0.33 (2/6) | **0.50 (3/6)** | **+52%** |
| Ceecom Moxon 20m | 0.33 (2/6) | 0.33 (2/6) | same |
| POTA Performer vertical | 1.00 (4/4) | 0.75 (3/4) | −25% (see below) |

#### Form Generation

| Test | Without Seeding | With Seeding |
|------|----------------|--------------|
| 3-element 10m Yagi | 5/5 goals, 2 iterations | **6/6 goals, 1 iteration** |
| 20m Moxon | 0.33 (2/6) | 0.33 (2/6) |
| 15m Hexbeam | solver timeout | solver timeout, classification OK (0.80) |

### Lessons Learned

#### 1. The biggest win was filling a zero → something gap

The PDF path had *no* calculator injection at all. Going from zero physics
context to full calculator seeding produced the largest improvements. The OCR
quad jumped from 0.50 to 0.83 — the LLM finally knew what dimensions to use
instead of guessing.

**Takeaway:** Before adding sophistication, check whether the basics are even
wired up. The PDF path was the most-used generation path and had the least
support.

#### 2. Structural NEC hints matter more than raw dimensions

The quad calculator already produced correct driven-side and reflector-side
lengths. But without hints about "4 GW cards forming a closed loop" and "split
bottom wire for feed", the LLM would produce open-ended wires, single-wire
loops, or misplaced excitation points.

**Takeaway:** For complex antenna types, the LLM needs *construction recipes*,
not just measurements. "What shape to build" is harder for the LLM than "what
length to make it."

#### 3. More detail can cause over-engineering

The POTA vertical regressed from 1.00 to 0.75. The calculator hints mentioned
16 radials; the LLM built an overly complex ground-plane model that was
structurally correct but scored lower on simplicity goals. Without the hint, it
built a clean 4-radial model that scored perfectly.

**Takeaway:** Calculator hints should match the complexity level of the goals.
Simple antenna types may benefit from a "keep it minimal" note rather than
maximum structural detail.

#### 4. Moxon remains the hardest problem

Calculator seeding helped DL2GMS Moxon (0.33 → 0.50) but Moxon still doesn't
reliably pass. The calculator provides correct A/B/C/D dimensions and the Moxon
NEC context file provides a 6-GW reference example, yet the LLM still drifts
during geometry construction — particularly on tip-gap placement and the
U-shaped fold-back.

Potential next steps:
- **Template-based generation:** Generate the complete NEC deck in the calculator
  itself (not just dimensions), bypassing the LLM for geometry entirely
- **Geometry constraints in OODA:** Reject iterations where tip-gap or
  element-spacing violates calculator bounds by more than 10%
- **Fewer degrees of freedom:** Fix the wire topology and only let the LLM
  adjust lengths/positions

#### 5. Frequency guessing is good enough

`_guess_freq_mhz()` correctly extracted frequencies from all test PDFs:
- "21.1 MHz" → 21.1 (explicit)
- "27.5 MHz" → 27.5 (explicit)
- "40M" → 7.1 (band reference)
- "20m" → 14.175 (band reference)
- Fallback 14.175 when nothing found

The regex approach is simple and robust. No NLP or LLM call needed.

#### 6. JSON mode vs NEC-text mode: orthogonal to seeding

Earlier testing (Experiment 1b) showed JSON intermediate mode works well for
simple types (Yagi: first-try pass) but worse for complex types (quad: 0.20 vs
0.67 in NEC-text). Calculator seeding improves both modes because it addresses a
different problem — *what dimensions to use* vs *how to format the output*.

The two techniques are complementary:
- **NEC-text mode + calculator seeding** = best for complex types (quad, moxon)
- **JSON mode + calculator seeding** = best for simple types (yagi, vertical, dipole)

### Test Matrix Summary

```
                    No Seeding          With Seeding
                    NEC-text  JSON      NEC-text
                    --------  ----      --------
Yagi (form)         5/5       5/5       6/6 ★
POTA vertical       4/4       4/4       3/4 ↓
3 Band Quad (PDF)   4/6       1/5       4/6, build↑
9107030 quad (PDF)  3/6       1/5       5/6 ★★
DL2GMS Moxon (PDF)  2/6       —         3/6 ★
Ceecom Moxon (PDF)  2/6       —         2/6
Form Moxon          2/6       —         2/6
```

★ = notable improvement, ↓ = regression

---

## Related Work

- [antenna-claw/docs/AB_THINKING_EXPERIMENT.md](../../antenna-claw/docs/AB_THINKING_EXPERIMENT.md) — thinking vs non-thinking LLM comparison
- [antenna-claw/src/extract_antenna_params.py](../../antenna-claw/src/extract_antenna_params.py) — original extraction skill
- [antenna-claw/src/web_search.py](../../antenna-claw/src/web_search.py) — SearXNG web search
- Per-type NEC context docs: `src/antenna_classifier/nec_context/*.txt` (29 types)
