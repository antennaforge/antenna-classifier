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

## Related Work

- [antenna-claw/docs/AB_THINKING_EXPERIMENT.md](../../antenna-claw/docs/AB_THINKING_EXPERIMENT.md) — thinking vs non-thinking LLM comparison
- [antenna-claw/src/extract_antenna_params.py](../../antenna-claw/src/extract_antenna_params.py) — original extraction skill
- [antenna-claw/src/web_search.py](../../antenna-claw/src/web_search.py) — SearXNG web search
- Per-type NEC context docs: `src/antenna_classifier/nec_context/*.txt` (29 types)
