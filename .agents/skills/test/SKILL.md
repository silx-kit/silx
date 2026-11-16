---
name: test
description: Build silx and run its test suite with pytest (all tests or a single module/function). Use when asked to run, check or debug tests.
---

# Run the silx tests

From the repository root:

- Full suite (first builds with meson, then tests that build):
  `python run_tests.py`
- Useful options: `--no-gui`, `--no-opencl`, `--no-opengl`, `--high-mem`, `--qt-binding=<PySide6|PyQt5|PyQt6>`, `-i` (test the installed package instead of building)
- A single module or test, on an installed/editable package:
  `pytest src/silx/io/test/test_url.py`
  `pytest src/silx/io/test/test_url.py::test_name`
- Lint and formatting (both enforced in CI):
  `ruff check .` and `ruff format --check .` (use `ruff format <file>` to fix)

## Notes

- Tests use **pytest** (new tests must not use `unittest`, see `doc/source/contribute/write_tests.rst`). There is no `suite()` registration.
- Tests live in a `test/` subfolder next to the code, in `test_<module>.py`. `test` directories are installed with `install_subdir`, so a new test file needs no `meson.build` change.
- An `ImportError` in the code under test often means a new `.py`/`.pyx` file is missing from the `py.install_sources` list of the `meson.build` of its package (`io`, `math`, `image`, `opencl`).
- GUI tests need a display (pytest-xvfb) and are skipped with `--no-gui`; OpenCL, OpenGL and high-memory tests are skipped with the matching options (see `src/silx/conftest.py`).
- Optional dependencies (pyopencl, PyOpenGL, Qt, h5pyd...) may be absent: tests must be skipped, not fail.
