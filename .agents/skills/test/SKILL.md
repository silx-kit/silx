---
name: test
description: Build FabIO and run its test suite (all tests or a single module/class/method). Use when asked to run, check or debug tests.
---

# Run the FabIO tests

From the repository root:

- Full suite (first builds with meson into `./build_py<XY>`, e.g. `build_py313` for Python 3.13, then tests that build):
  `python run_tests.py`
- A single module, class or method (dotted unittest names):
  `python run_tests.py fabio.test.codecs.test_edfimage`
  `python run_tests.py fabio.test.codecs.test_edfimage.TestEdfs.test_read`
- Test the installed package instead of building: `python run_tests.py --installed`
- Coverage: `python run_tests.py -c`; verbosity: `-v` / `-vv`
- Lint (as in CI): `ruff check src`

## Notes

- Tests use `unittest`; every module exposes a `suite()` function. A new test module must be added to `src/fabio/test/test_all.py` (or `src/fabio/test/codecs/__init__.py` for per-format tests) **and** to the matching `meson.build`, otherwise it is neither installed nor run.
- An `ImportError` in the tests often means a file is missing from the `meson.build` of its directory.
- Test images are downloaded on demand from `http://www.edna-site.org/pub/fabio/testimages` via `UtilsTest.getimage(...)` (`src/fabio/test/utilstest.py`, `utils/ExternalResources.py`). Set `FABIO_DATA` to a directory holding a local copy to avoid network access.
- `WITH_QT_TEST=0` skips the imports that depend on Qt.
