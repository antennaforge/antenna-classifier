# TODO - antenna-classifier

## Tasks

| Task | Domain | Priority | Status | Details |
|------|--------|----------|--------|---------|
| Add missing antenna types (quagi, delta_loop, v_beam, batwing, zigzag, wire_object) | classifier | high | done | v0.2.0. Added 6 types, expanded keyword map, improved hex matching. 2025-07-23 |
| Improve SY expression resolution | parser | high | done | v0.2.0. Inline comment stripping, math functions (cos/sin/tan/sqrt etc), case-insensitive vars, bare AWG, percent notation, tab-split fix. 2025-07-23 |
| Tag wire-grid objects as wire_object | classifier | high | done | v0.2.0. Added _classify_wire_object() for geometry-only NEC files. 2025-07-23 |
| Build card-config fingerprint engine | fingerprint | high | done | v0.2.0. New fingerprint.py module — Fingerprint dataclass, 21-dim feature vectors, cosine similarity, archetype profiles, CLI commands. 2025-07-23 |
| Add unit tests for parser | testing | high | done | 53 tests covering SY resolution, safe_eval, card parsing, preprocessors. 2026-03-06 |
| Add unit tests for validator | testing | high | done | 19 tests for validation rules (geometry, excitation, frequency, ground, wire params, tag refs). 2026-03-06 |
| Add unit tests for classifier | testing | high | done | 51 tests for comment/structural classification, edge cases, freq-to-band. 2026-03-06 |
| Add unit tests for fingerprint engine | testing | high | done | 35 tests for fingerprint generation, similarity, archetypes, complexity. 2026-03-06 |
| Improve geometry-based classification confidence | classifier | medium | not-started | Many files rely on comment/path keywords; pure geometry detection needs tuning. 2026-03-06 |
| Handle EZnec / MiniNec format variants | parser | medium | not-started | models_2/zz_EZnec and zz_MiniNec may have non-standard syntax. 2026-03-06 |
| Add NEC4-specific card support | parser | low | done | GD, IS, NX, UM, PL card specs added. 2026-03-06 |
| Export working-file set | cli | medium | done | `export` command copies/symlinks valid files into type-organized directory tree. 2026-03-06 |
| Multiband detection | classifier | medium | done | Detects multiple FR entries/sweeps, ClassificationResult.bands and is_multiband fields. 2026-03-06 |
| Build archetype library from classified collection | fingerprint | medium | not-started | Use 978 classified files to build per-type archetype profiles for fingerprint-based classification. 2025-07-23 |
| Investigate 96 remaining invalid files | parser | medium | not-started | Some may be fixable with further parser improvements. 2025-07-23 |
| Docker NEC simulation integration | simulator | high | done | nec-solver Docker container (nec2c/nec2++/xnec2c), simulator.py client module, sanitize_nec auto-inserts GE/EN cards. 2025-07-24 |
| NEC sanitizer comprehensive fixes | simulator | high | done | 98.5% success rate (1260/1279 files). Comma→space ordering fix, SQR/int()/cm builtins, AWG #0-#3, engineering notation, scientific notation protection, EX/LD segment clamping, Ctrl-Z stripping, LD type 6+ conversion, RP injection, NaN/inf sanitization. Remaining 19 failures: 10 nec2c limitations (NGF/patches/NEC-4), 9 compute timeouts. 2025-07-25 |
| Interactive web dashboard | dashboard | high | done | FastAPI dashboard.py + Plotly.js frontend. Browse/filter/search files, view classification detail, run NEC simulations with SWR/impedance/pattern plots. CLI: `antenna-classifier dashboard <dir>`. 2025-07-24 |
| 3D antenna visualization | visualizer | high | done | visualizer.py extracts geometry (GW/GA/GH/GS/GX/GM), viewer3d.js Three.js renderer with orbit controls, dashboard 3D View tab with feed point markers and ground plane. 16 unit tests. 2026-03-06 |
| SWR/impedance frequency sweep | simulator | high | done | _build_sweep_deck() modifies FR card for multi-freq analysis (±15% of design freq). simulate_sweep() + /api/sweep endpoint with adaptive points (21/11/7 based on wire count). Frontend auto-runs sweep+pattern in parallel when Simulation tab selected. 2026-03-07 |
| Fix radiation pattern rendering | dashboard | high | done | renderRadiationPattern() now detects azimuth (many phi, few theta) vs elevation (many theta, few phi) patterns and renders correctly as polar plots with shifted gain. 2026-03-07 |
| Remove comment-based classification | classifier | medium | not-started | Replace _classify_from_comments with pure geometry-based classification once 3D visualizer validates structural detection accuracy. 2026-03-06 |
