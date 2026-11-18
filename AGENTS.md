# AGENTS.md

This file provides guidance to LLM-Agent/Harnesses when working with code in this repository.

## Project

silx: Python toolkit for synchrotron data analysis (HDF5/SPEC/FabIO I/O, OpenCL processing, Qt plot widgets, `silx view` / `silx convert` / `silx compare` apps). Python >= 3.10, built with meson-python + Cython (`src/silx`).

Agent skills (`bootstrap`, `issue`, `prepare-fix`, `test`) live in `.agents/skills/`.

## AI policy

`doc/source/contribute/ai_policy.rst`: AI use must be disclosed in the PR; fully autonomous PR submission is forbidden; AI must not be used on issues labelled `Difficulty: Easy`.

## Commands

- Run code against the sources (builds with meson into `./build_py<XY>`, first call compiles Cython and is slow): `python bootstrap.py -c "import silx; ..."`, `python bootstrap.py silx view --help`.
- Editable install for development: `pip install meson-python ninja cython` then `pip install --no-build-isolation --editable .[dev]`.
- Tests use **pytest** (new tests must be pytest, not unittest; see `doc/source/contribute/write_tests.rst`):
  - `python run_tests.py` builds then runs the suite; options `--no-gui`, `--no-opencl`, `--no-opengl`, `--high-mem`, `--qt-binding=<PySide6|PyQt5|PyQt6>`, `-i` (test the installed package).
  - Single test on an installed/editable package: `pytest src/silx/io/test/test_url.py::test_name`.
  - Tests are placed in a `test/` subfolder next to the code, in `test_<module>.py`. GUI tests need a display (pytest-xvfb); OpenCL/GL/high-mem tests are skipped via options in `src/silx/conftest.py`.
- Lint/format (CI runs both): `ruff check .` and `ruff format --check .` (rules in `pyproject.toml`). Unlike elsewhere, formatting **is** enforced.
- Docs: `sphinx-build doc/source/ build/html`.
- Changelog: `CHANGELOG.rst` (root), one bullet per change in the section under development, in the form ``* `silx.module`: Description (`Issue #N <https://github.com/silx-kit/silx/issues/N>`_)``.

## Architecture

- Packages under `src/silx/`: `io` (HDF5 helpers, SPEC/fabio-as-h5 wrappers `spech5`/`fabioh5`/`fioh5`, `commonh5` base classes, `dictdump`, `nxdata`, `utils.open` with `file.h5::/path` URL syntax), `math` (colormap, fit, histogramnd, fft, marching cubes, median filter), `image` (sift, tomography, projection, backprojection), `opencl` (pyopencl processing, `codec/` for GPU bitshuffle-LZ4 and byte-offset decompression), `gui` (Qt widgets: `plot`, `plot3d`, `hdf5`, `data`, `fit`, `qt` compatibility layer), `app` (`view`, `convert`, `compare`; entry point `silx = silx.__main__:main`), `sx` (convenience `silx.sx` API), `utils`, `third_party`, `resources`.
- **Meson is the source of truth for packaging, and it is split by kind of package.** Subpackages containing Cython (`io`, `math`, `image`, `opencl`) have their own `meson.build` that lists **every Python file explicitly** via `py.install_sources` (new `.py`/`.pyx` files must be added there or are missing from the build and cause ImportError). Pure-Python subpackages (`app`, `gui`, `resources`, `sx`, `test`, `third_party`, `utils`) are installed wholesale with `install_subdir` listed in `src/silx/meson.build`; `test` directories of compiled packages are also installed with `install_subdir`.
- Native code is Cython (`*.pyx`, with `.pyi` stubs such as `math/combo.pyi`) plus C sources in `math/include` etc.; OpenCL kernels live in `resources/opencl`.
- GUI plot backends (matplotlib, OpenGL, optional pygfx) live in `gui/plot/backends`; Qt binding differences are hidden by `silx.gui.qt`, so import Qt from there, not directly.
- Optional dependencies (pyopencl, PyOpenGL, Qt, h5pyd, pygfx...) are guarded; tests and imports must degrade gracefully when they are missing.
