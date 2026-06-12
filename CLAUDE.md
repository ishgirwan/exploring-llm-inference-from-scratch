# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A **public, first-person learning journey** about how LLM inference runs on GPUs —
kernels, memory, profiling, serving engines, and (Phase 5) writing competitive
custom kernels. It is **mostly prose**: ~40 Markdown files across nine
chapters plus `ROADMAP.md`, a short root `README.md`, `LEARNING_PATH.md`
(the stage-by-stage path that braids chapter reading into the labs — read-before /
build / in-lab questions / read-after / takeaway / carry-forward per stage), and
`FAILURES.md`/`CHANGELOG.md` stubs. The only code is the **Stage 0 skeleton**:
`pyproject.toml`, an empty `common/` package, `tests/` with a GPU-marker
auto-skip (`@pytest.mark.gpu` tests skip without CUDA), and a CPU-only CI
workflow. **Lint is `ruff check .`; tests are `pytest`** — both run inside
`.venv/` locally. The labs and benchmarks themselves (modules M0–M32) are
*planned* in `ROADMAP.md` and `LEARNING_PATH.md` but not yet built; the harness
internals (`bench.py`, `correctness.py`, `results_schema.py`) and
`BENCHMARKING.md` are deliberately left for the author to write as M0 learning
work — do not scaffold them unasked. "Working in this repo" still mostly means
**authoring or editing Markdown docs**.

`ROADMAP.md` is the entry point and master plan (the root `README.md` is a short
pointer into it). It
defines the M-module/phase model: **Chapters 1–8 are the reference library —
readable straight through, or pulled section-by-section just-in-time as
`LEARNING_PATH.md` braids them into the labs; Chapter 9 is Phase-5 prework, read
later, not before M0.** Each chapter is a
directory (`chapters/NN_topic/`) containing a `README.md` that indexes its
sections and numbered `NN_section.md` docs.

Hardware reality that shapes every doc: **there is no local NVIDIA GPU (remote-only).**
Nothing is run locally; GPU work happens on rented GPUs at M-module time.

## Authoring conventions (the core of this repo)

These are strict, span all docs, and are the main way to get work wrong. They are
not discoverable from any single file:

- **First-person learning-journey voice.** Every doc reads as the author's own
  journey of working something out ("I", "what confused me"), *never* as a marketed
  course or tutorial ("you will learn", "this course").
- **Self-contained, define every term inline.** A cold reader picks up any doc and
  is never left with an unexplained term or an un-bridged conceptual leap. Acronyms
  (HBM, SPMD, MMA, TMA, CTA, …) get a short inline gloss on first use, even when an
  earlier chapter also defined them.
- **Present truth, not doc-history.** State the bare present-tense fact. Never
  narrate edit history, who asked for a change, why it changed, or what it used to
  say. There is no "a tweet claims" / "reportedly" escape hatch — every line is the
  author's flat assertion, which raises the bar for accuracy (see below).
- **Open black boxes one level, then re-seal.** When a doc hits a sealed
  abstraction, open it one level (a diagram + a small truth table / worked example),
  then explicitly re-seal ("you don't need the full algebra here — the point is…").
- **Doc skeleton.** Each section doc: an intro that places it; a `Prerequisites:` /
  `Next:` line; numbered `## N.` sections; diagrams in ```text fences; a final
  `## N. What to carry forward` block (a ```text table mapping ideas → the M-modules
  that use them); and a bold **"The one sentence to keep:"** closer.

## Cross-references and the renumbering hazard

- **`§N` convention:** a **bare `§N`** (in prose or fences) means *the current
  doc's own* section N — never another doc's. To reference a sibling doc, use a
  **Markdown link** with descriptive text (`[Section 2](02_reading_a_real_kernel.md)`),
  or inside fences (where links don't render) a **descriptive name**, never a bare
  number. The chapter's file numbers and a doc's own section numbers overlap, so
  this distinction matters constantly.
- **Renumbering propagates.** Changing a section number, or an M-module number, or
  restructuring a doc, silently breaks references elsewhere. This has caused real
  bugs (an end-to-end restructure touched ~45 refs across 14 docs; an M-module
  renumber left stale `M24` refs in another chapter). **After any renumber, grep the
  whole repo** for the old numbers and any internal-count claims ("22 topics", "four
  phases") and fix every hit.
- **Anchor slugs (GitHub-flavored):** lowercase, punctuation stripped, spaces →
  hyphens, and an em-dash surrounded by spaces becomes a **double hyphen**
  (`…blackwell--b200…`). **Verify a cross-doc anchor by grepping the real heading**,
  not by deriving the slug from memory — arrow/symbol stripping is easy to get
  wrong. If a target is a file's first section, just link the file with no `#…`.

## Accuracy discipline

- **Verify technical claims before asserting them.** Because "present truth"
  forbids hedging, every GPU/compiler/hardware fact becomes a flat claim. Confirm
  product facts, instruction mnemonics, version-specific behavior, and numbers via
  web search before writing them; if a name/project can't be confirmed, omit it
  rather than define something unverifiable.
- **Never present un-run numbers as measured.** Quote external figures with
  attribution to their source; do not state a performance number as if the author
  measured it (there is no local GPU). Code shown in docs is *illustrative/
  representative* and labelled as such, not captured output.

## Git and environment

- Windows + PowerShell is the default shell. **PowerShell here-strings (`@'…'@`)
  only work in the PowerShell tool — in the Bash tool they are literal and corrupt
  the input.** For multi-line commit messages, write the message to a temp file and
  `git commit -F`, then delete it.
- End commit messages with the trailer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- `LF will be replaced by CRLF` warnings on `git add` are harmless (Windows line
  endings).
- `.claude/` and `.remember/` are gitignored session tooling — never commit them.
- Commit directly to `main` (solo repo, history is direct-to-main). Commit when
  asked; **push only when explicitly asked** ("push" / "push to remote"), since
  pushing is the outward, hard-to-reverse step.
