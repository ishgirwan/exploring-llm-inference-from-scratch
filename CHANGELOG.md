# Changelog

What changed and when, newest first — including mistakes found and fixed, so
the history stays honest instead of quietly edited. One entry per meaningful
change; checkpoint tags (`v0.1`, `v0.2`, …) get their own entry.

Format: `YYYY-MM-DD — what changed, in one line.`

- 2026-06-12 — Chapter 9 §6 gains a where-AI-fits section; further-reading
  additions (Modal's FA4-inference post, nano-vLLM, FlexAttention); M30's
  contribution-target list widened.
- 2026-06-12 — Moved the nine chapter directories under `chapters/` and fixed
  every root↔chapter link. Old deep links into chapter files break; the
  chapter-to-chapter links survive unchanged.
- 2026-06-12 — Switched package management to uv: `uv.lock` committed, dev
  tools in a dependency group, CI runs `uv sync` + `uv run`.
- 2026-06-12 — Stage 0 scaffolding: Apache-2.0 license, Python skeleton
  (`pyproject.toml` + ruff, `common/` as a package, `tests/` with the
  GPU-marker skip), and CPU-only CI.
