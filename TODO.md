# TODO - antenna-classifier

## Tasks

| Task | Domain | Priority | Status | Details |
|------|--------|----------|--------|---------|
| Add unit tests for parser | testing | high | not-started | Test SY resolution, card parsing, edge cases. 2026-03-06 |
| Add unit tests for validator | testing | high | not-started | Test each validation rule independently. 2026-03-06 |
| Add unit tests for classifier | testing | high | not-started | Test each heuristic with synthetic NEC content. 2026-03-06 |
| Improve geometry-based classification confidence | classifier | medium | not-started | Many files rely on comment/path keywords; pure geometry detection needs tuning. 2026-03-06 |
| Handle EZnec / MiniNec format variants | parser | medium | not-started | models_2/zz_EZnec and zz_MiniNec may have non-standard syntax. 2026-03-06 |
| Add NEC4-specific card support | parser | low | not-started | GD, IS, WG and other NEC4-only cards. 2026-03-06 |
| Export working-file set | cli | medium | not-started | Add `export` command to copy valid files into a clean directory tree organized by type. 2026-03-06 |
| Multiband detection | classifier | medium | not-started | Detect models with multiple FR entries or broadband characteristics. 2026-03-06 |
