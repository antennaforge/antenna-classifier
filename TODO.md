# TODO - antenna-classifier

## Tasks

| Task | Domain | Priority | Status | Details |
|------|--------|----------|--------|---------|
| Add missing antenna types (quagi, delta_loop, v_beam, batwing, zigzag, wire_object) | classifier | high | done | v0.2.0. Added 6 types, expanded keyword map, improved hex matching. 2025-07-23 |
| Improve SY expression resolution | parser | high | done | v0.2.0. Inline comment stripping, math functions (cos/sin/tan/sqrt etc), case-insensitive vars, bare AWG, percent notation, tab-split fix. 2025-07-23 |
| Tag wire-grid objects as wire_object | classifier | high | done | v0.2.0. Added _classify_wire_object() for geometry-only NEC files. 2025-07-23 |
| Build card-config fingerprint engine | fingerprint | high | done | v0.2.0. New fingerprint.py module — Fingerprint dataclass, 21-dim feature vectors, cosine similarity, archetype profiles, CLI commands. 2025-07-23 |
| Add unit tests for parser | testing | high | not-started | Test SY resolution, card parsing, edge cases. 2026-03-06 |
| Add unit tests for validator | testing | high | not-started | Test each validation rule independently. 2026-03-06 |
| Add unit tests for classifier | testing | high | not-started | Test each heuristic with synthetic NEC content. 2026-03-06 |
| Add unit tests for fingerprint engine | testing | high | not-started | Test fingerprint generation, similarity, archetypes. 2025-07-23 |
| Improve geometry-based classification confidence | classifier | medium | not-started | Many files rely on comment/path keywords; pure geometry detection needs tuning. 2026-03-06 |
| Handle EZnec / MiniNec format variants | parser | medium | not-started | models_2/zz_EZnec and zz_MiniNec may have non-standard syntax. 2026-03-06 |
| Add NEC4-specific card support | parser | low | not-started | GD, IS, WG and other NEC4-only cards. 2026-03-06 |
| Export working-file set | cli | medium | not-started | Add `export` command to copy valid files into a clean directory tree organized by type. 2026-03-06 |
| Multiband detection | classifier | medium | not-started | Detect models with multiple FR entries or broadband characteristics. 2026-03-06 |
| Build archetype library from classified collection | fingerprint | medium | not-started | Use 978 classified files to build per-type archetype profiles for fingerprint-based classification. 2025-07-23 |
| Investigate 96 remaining invalid files | parser | medium | not-started | Some may be fixable with further parser improvements. 2025-07-23 |
| Docker NEC simulation integration | simulator | high | done | nec-solver Docker container (nec2c/nec2++/xnec2c), simulator.py client module, sanitize_nec auto-inserts GE/EN cards. 2025-07-24 |
| Interactive web dashboard | dashboard | high | done | FastAPI dashboard.py + Plotly.js frontend. Browse/filter/search files, view classification detail, run NEC simulations with SWR/impedance/pattern plots. CLI: `antenna-classifier dashboard <dir>`. 2025-07-24 |
