# AGENTS.md

This file provides guidance to LLM-Agents when working with code in this repository.

## Project

FabIO reads/writes 2D X-ray detector images (~30 formats: EDF, CBF, TIFF, HDF5/Eiger/Lima, Mar345, Bruker, ...) exposing a unified API: `header` (dict) + `data` (numpy array). Python >= 3.11, built with meson-python + Cython.

## Commands

- Tests: use the `test` skill (`.agents/skills/test/SKILL.md`); `python run_tests.py` builds with meson then runs the suite.
- Run code against the sources (compiled, no install): use the `bootstrap` skill (`.agents/skills/bootstrap/SKILL.md`), e.g. `python bootstrap.py -c "import fabio; ..."`.
- Triage a GitHub issue (read-only, minimal reproducer): `issue` skill. Prepare its resolution (branch `XXX_short_description`, failing regression test, `ISSUE_XXX.md`, changelog line): `prepare-fix` skill.
- Build a wheel/sdist: `python -m build` (or `pip install .`)
- Changelog: `doc/source/Changelog.rst` (the root `CHANGELOG.rst` only points to it); one line per change, ending with the issue number, e.g. `(#612)`.
- Lint (as in CI): `ruff check src` — rules are an explicit allow-list in `pyproject.toml`; only add rules the codebase already complies with. Formatting with ruff is not enforced.

## Architecture

- **Meson is the source of truth for packaging.** Every Python file (including tests) must be listed in the `meson.build` of its directory, or it won't be installed/importable from the build.
- **Codec plugin model** (`src/fabio/`):
  - `fabioimage.py`: `FabioImage` base class (+ `FabioFrame`). Subclasses set `DESCRIPTION`, `DEFAULT_EXTENSIONS`, implement `read(fname, frame=None)` (and optionally `write`, `_readheader`, `_get_frame`/`nframes` for multi-frame formats). Transparent gz/bz2 handling goes through `fabioutils` file wrappers; `_need_a_seek_to_read` / `_need_a_real_file` class flags control it.
  - `fabioformats.py`: registry. `_default_codecs` lists `(module, class)` pairs — **order matters** for extension-based lookup. New formats must be added here. `templateimage.py` documents the full checklist for adding a format.
  - `openimage.py`: `fabio.open()` / `openimage()` / `openheader()` / `open_series()`. Format detection is primarily by **magic bytes** (`MAGIC_NUMBERS` + `_do_magic`), with extension fallback; HDF5 files are disambiguated (Eiger / Lima / Sparse / Lambda / generic hdf5) by inspecting attributes. `file.h5::/path` syntax selects a dataset.
  - `file_series.py`, `fabioutils.py`: filename numbering/series iteration helpers.
- **Native code** (`src/fabio/ext/`): Cython extensions for compression (CBF byte-offset, Mar345 PCK, AGI bitfield, CIF parsing, dense/sparse) with C sources in `ext/src`, `ext/include`. Pure-Python fallbacks live in `src/fabio/compression/`.
- **Apps** (`src/fabio/app/`): CLI entry points declared in `pyproject.toml` `[project.scripts]` (`fabio-convert`, `eiger2cbf`, `eiger2crysalis`, `densify_Bragg`, `hdf2neggia`, `fabio_viewer` (Qt)).
- `third_party/` is vendored code; `sandbox/` is scratch code excluded from releases and lint.
