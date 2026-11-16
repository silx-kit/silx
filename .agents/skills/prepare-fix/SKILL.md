---
name: prepare-fix
description: Prepare the resolution of a silx GitHub issue number XXX - make sure the work happens on a branch/worktree named XXX_SHORT_DESCRIPTION, add a failing regression test, create the ISSUE_XXX.md work file (checklist + agent notes, later used to write the pull request text) and add a one-line changelog entry. Does not fix the bug itself. Use when asked to start working on, or get ready to fix, an issue.
---

# Prepare the resolution of issue XXX

Input: the issue number `XXX` of https://github.com/silx-kit/silx/issues.
Result: a correctly named branch, a regression test that currently fails, an `ISSUE_XXX.md` work file, and a changelog line. **The fix itself is out of scope**: do not modify the library code to make the test pass.

## 1. Understand the issue

Use the `issue` skill to read the issue and get the minimal reproducer and the suspected location. If it cannot be reproduced, stop and report it instead of inventing a test. If the issue is labelled `Difficulty: Easy`, stop: the AI policy (`doc/source/contribute/ai_policy.rst`) forbids AI on those.

## 2. Branch or worktree

The work must happen on a git branch or worktree named `XXX_SHORT_DESCRIPTION` (issue number, underscore, a few lowercase words separated by underscores, e.g. `633_fix_headers`).

1. Check the current branch (`git branch --show-current`) and worktrees (`git worktree list`).
2. If the current branch/worktree already starts with `XXX_`, keep it.
3. Otherwise, if a local branch or worktree for `XXX_*` already exists, switch to it (or use that worktree).
4. Otherwise create one from an up-to-date upstream `main` (`git fetch origin`, then branch from `origin/main`), either as a branch (`git switch -c XXX_short_description origin/main`) or as a sibling worktree (`git worktree add ../silx-XXX_short_description -b XXX_short_description origin/main`). Ask the user which one they prefer if it is not clear from the context.
5. Never discard or carry over uncommitted work silently: if the working tree is dirty, tell the user before switching. Do not push, and do not touch other branches.

All following steps are done on that branch/worktree.

## 3. Regression test

- Add the test to the existing test module next to the affected code (`src/silx/<package>/test/test_<module>.py`). Create a new module only if none fits; test directories are installed with `install_subdir`, so no `meson.build` change is needed.
- Write **pytest** tests (plain functions or classes, fixtures such as `tmp_path`; no `unittest`, no `suite()`), see `doc/source/contribute/write_tests.rst`. Build the data in the test (numpy arrays, `io.BytesIO`, `tmp_path`) rather than adding sample files; if a sample file is unavoidable, say so in `ISSUE_XXX.md` instead of committing a binary.
- Mark or skip tests needing optional dependencies, a display, OpenCL or OpenGL as the neighbouring tests do (see `src/silx/conftest.py`).
- Name the test after the issue, e.g. `test_bug_XXX`, and add a short docstring/comment mentioning `#XXX`.
- Derive it from the minimal reproducer of the `issue` skill: assert the **expected** behaviour, so the test fails today and passes once the bug is fixed.
- Run it with the `test` skill (`pytest path::test_name`) and confirm it **fails for the reason described in the issue** (not because of a typo or an import error). Run the surrounding module too, to make sure nothing else was broken.
- For a feature request, write the test that the requested feature must satisfy, and confirm it fails because the feature is missing.

## 4. `ISSUE_XXX.md`

Create `ISSUE_XXX.md` at the repository root. Do not commit or stage it: it is a scratch file, deleted once the issue is resolved.

Layout:

```markdown
# Issue #XXX: <title>

## Actions
- [ ] Issue read and reproduced
- [ ] Branch `XXX_short_description` in use
- [ ] Regression test `<path>::<test name>` added (fails before the fix)
- [ ] Changelog entry added
- [ ] Fix implemented
- [ ] Regression test passes
- [ ] Full test suite, `ruff check .` and `ruff format --check .` pass
- [ ] Pull request text written from this file
- [ ] This file deleted

## Summary
<2-3 lines: what is wrong, or what is requested>

## Reproducer
<the minimal snippet and its actual vs. expected output>

## Analysis
<suspected `path:line`, likely cause, confidence>

## Agents
< each agent should describe itself to know which model was used for each part of the work, one per line in a list>
- agent_name: LLM model used

## Notes
<comments left by whoever works on an item: date/agent name, the action concerned, what was done or found, dead ends. Append, do not rewrite others' notes.>
```

Add further checkbox items to the list when the analysis reveals more work (other formats affected, documentation to update, ...). Tick items as they are completed. Any agent working on the issue later reads this file first and records its progress in **Notes**.

## 5. Changelog

Add **one line** to `CHANGELOG.rst` (repository root), in the section under development (the topmost one, currently `3.2.0: ????/??/??`; compare with the version in `src/silx/_version.py`). If no such section exists, create it above the latest one with the same title/underline style.

- Entry format, as for the other lines: ``* `silx.module`: Description (`Issue #XXX <https://github.com/silx-kit/silx/issues/XXX>`_)``. A single bullet, no sub-bullets (use `PR #N` instead of `Issue #N` for a pull request without issue).
- Word it from the user's point of view ("Fixed ...", "Added ..."). Re-read it when the fix is finished and adjust it if the real fix differs from what was planned.

## 6. Hand-over

Report: the branch/worktree, the test (path and failure message), the `ISSUE_XXX.md` content summary and the changelog line. Do not attempt the fix unless asked.

## When the issue is resolved (for whoever does the fix)

- Tick the remaining items, with the `test` skill run the full suite, `ruff check .` and `ruff format --check .`.
- Write the pull request text from `ISSUE_XXX.md`, **as short as possible**: a one-line title, one to three lines of description (cause and fix), and `Closes #XXX`. State in the text that AI was used (AI policy); the PR must be submitted by a human. No checklist, no agent notes, no test log.
- Then delete `ISSUE_XXX.md`.
