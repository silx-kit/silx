#!/usr/bin/env python3
# /*##########################################################################
#
# Copyright (c) 2015-2023 European Synchrotron Radiation Facility
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ###########################################################################*/

__authors__ = ["Jérôme Kieffer", "Thomas Vincent"]
__date__ = "07/11/2022"
__license__ = "MIT"

import sys
import os
import platform
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("silx.setup")

from setuptools import Extension
from setuptools.command.build_ext import build_ext

try:
    import numpy
except ImportError:
    raise ImportError(
        "To install this package, you must install numpy first\n"
        "(See https://pypi.org/project/numpy)"
    )


PROJECT = "silx"
if sys.version_info.major < 3:
    logger.error(PROJECT + " no longer supports Python2")

if "LANG" not in os.environ and sys.platform == "darwin":
    print(
        """WARNING: the LANG environment variable is not defined,
an utf-8 LANG is mandatory to use setup.py, you may face unexpected UnicodeError.
export LANG=en_US.utf-8
export LC_ALL=en_US.utf-8
"""
    )


# ############## #
# Compiler flags #
# ############## #


def parse_env_as_bool(key: str, default: Optional[bool] = None) -> Optional[bool]:
    """Parse `key` env. var. and convert its value to a boolean or None.

    If it cannot parse it or if None, `default` is returned.
    """
    content = os.environ.get(key, "")
    value = content.lower()
    if value in ["1", "true", "yes", "y"]:
        return True
    if value in ["0", "false", "no", "n"]:
        return False
    if value in ["none", ""]:
        return default
    msg = "Env variable '%s' contains '%s'. But a boolean or an empty \
        string was expected. Variable ignored."
    logger.warning(msg, key, content)
    return default


def get_use_openmp_from_env_var() -> bool:
    """Returns whether or not to build with OpenMP"""
    use_openmp = parse_env_as_bool("SILX_WITH_OPENMP", default=True)
    if use_openmp and platform.system() == "Darwin":
        logger.warning("OpenMP support ignored. Your platform does not support it.")
        return False
    return use_openmp


USE_OPENMP = get_use_openmp_from_env_var()
FORCE_CYTHON = parse_env_as_bool("SILX_FORCE_CYTHON", default=False)


class BuildExt(build_ext):
    """Handle extension compilation.

    Environment variables can custom the build of extensions, see the install documentation.

    If building with MSVC, compiler flags are converted from gcc flags.
    """

    COMPILE_ARGS_CONVERTER = {"-fopenmp": "/openmp"}

    LINK_ARGS_CONVERTER = {"-fopenmp": ""}

    description = "Build extensions"

    def patch_extension(self, ext: Extension):
        """Patch an extension according to requested Cython and OpenMP usage."""
        from Cython.Build import cythonize

        patched_exts = cythonize(
            [ext],
            compiler_directives={"embedsignature": True, "language_level": 3},
            force=FORCE_CYTHON,
        )
        ext.sources = patched_exts[0].sources

        # Remove OpenMP flags if OpenMP is disabled
        if not USE_OPENMP:
            ext.extra_compile_args = [
                f for f in ext.extra_compile_args if f != "-fopenmp"
            ]
            ext.extra_link_args = [f for f in ext.extra_link_args if f != "-fopenmp"]

        # Convert flags from gcc to MSVC if required
        if self.compiler.compiler_type == "msvc":
            extra_compile_args = [
                self.COMPILE_ARGS_CONVERTER.get(f, f) for f in ext.extra_compile_args
            ]
            # Avoid empty arg
            ext.extra_compile_args = [arg for arg in extra_compile_args if arg]

            extra_link_args = [
                self.LINK_ARGS_CONVERTER.get(f, f) for f in ext.extra_link_args
            ]
            # Avoid empty arg
            ext.extra_link_args = [arg for arg in extra_link_args if arg]

    def build_extensions(self):
        for ext in self.extensions:
            self.patch_extension(ext)
        build_ext.build_extensions(self)


# ##### #
# setup #
# ##### #


def get_project_configuration():
    """Returns project arguments for setup"""
    # Use installed numpy version as minimal required version
    # This is useful for wheels to advertise the numpy version they were built with
    numpy_requested_version = ">=%s" % numpy.version.version
    logger.info("Install requires: numpy %s", numpy_requested_version)

    install_requires = [
        # for most of the computation
        "numpy%s" % numpy_requested_version,
        # for version parsing
        "packaging",
        # for io support
        "h5py",
        "fabio>=0.9",
    ]
    if sys.version_info < (3, 9):
        install_requires.append("setuptools")  # For pkg_resources

    # extras requirements: target 'full' to install all dependencies at once
    full_requires = [
        # opencl
        "pyopencl",
        "Mako",
        # gui
        "qtconsole",
        "matplotlib>=3.1.0",
        "PyOpenGL",
        "python-dateutil",
        "PyQt5",
        # extra
        "hdf5plugin",
        "scipy",
        "Pillow",
        "bitshuffle",
    ]

    test_requires = ["pytest", "pytest-xvfb", "pytest-mock", "bitshuffle"]

    doc_requires = {
        "nbsphinx",
        "pandoc",
        "pillow",
        "pydata_sphinx_theme",
        "sphinx",
        "sphinx-autodoc-typehints",
        "sphinx-panels",
    }

    extras_require = {
        "full": full_requires,
        "doc": doc_requires,
        "test": test_requires,
    }

    # Here for packaging purpose only
    # Setting the SILX_FULL_INSTALL_REQUIRES environment variable
    # put all dependencies as install_requires
    if os.environ.get("SILX_FULL_INSTALL_REQUIRES") is not None:
        install_requires += full_requires

    # Set the SILX_INSTALL_REQUIRES_STRIP env. var. to a comma-separated
    # list of package names to remove them from install_requires
    install_requires_strip = os.environ.get("SILX_INSTALL_REQUIRES_STRIP")
    if install_requires_strip is not None:
        for package_name in install_requires_strip.split(","):
            install_requires.remove(package_name)

    def silx_io_specfile_define_macros():
        # Locale and platform management
        if sys.platform == "win32":
            return [("WIN32", None), ("SPECFILE_POSIX", None)]
        elif os.name.lower().startswith("posix"):
            # the best choice is to have _GNU_SOURCE defined
            # as a compilation flag because that allows the
            # use of strtod_l
            use_gnu_source = os.environ.get("SPECFILE_USE_GNU_SOURCE", "False")
            if use_gnu_source in ("True", "1"):  # 1 was the initially supported value
                return [("_GNU_SOURCE", 1)]
            return [("SPECFILE_POSIX", None)]
        else:
            return []

    ext_modules = [
        # silx.image
        Extension(
            name="silx.image.bilinear",
            sources=["src/silx/image/bilinear.pyx"],
            language="c",
        ),
        Extension(
            name="silx.image.marchingsquares._mergeimpl",
            sources=["src/silx/image/marchingsquares/_mergeimpl.pyx"],
            include_dirs=[
                numpy.get_include(),
                os.path.join(
                    os.path.dirname(__file__), "src", "silx", "utils", "include"
                ),
            ],
            language="c++",
            extra_link_args=["-fopenmp"],
            extra_compile_args=["-fopenmp"],
        ),
        Extension(
            name="silx.image.shapes",
            sources=["src/silx/image/shapes.pyx"],
            language="c",
        ),
        # silx.io
        Extension(
            name="silx.io.specfile",
            sources=[
                "src/silx/io/specfile/src/sfheader.c",
                "src/silx/io/specfile/src/sfinit.c",
                "src/silx/io/specfile/src/sflists.c",
                "src/silx/io/specfile/src/sfdata.c",
                "src/silx/io/specfile/src/sfindex.c",
                "src/silx/io/specfile/src/sflabel.c",
                "src/silx/io/specfile/src/sfmca.c",
                "src/silx/io/specfile/src/sftools.c",
                "src/silx/io/specfile/src/locale_management.c",
                "src/silx/io/specfile.pyx",
            ],
            define_macros=silx_io_specfile_define_macros(),
            include_dirs=["src/silx/io/specfile/include"],
            language="c",
        ),
        # silx.math
        Extension(
            name="silx.math._colormap",
            sources=["src/silx/math/_colormap.pyx"],
            language="c",
            include_dirs=[
                "src/silx/math/include",
                numpy.get_include(),
            ],
            extra_link_args=["-fopenmp"],
            extra_compile_args=["-fopenmp"],
        ),
        Extension(
            name="silx.math.chistogramnd",
            sources=[
                "src/silx/math/histogramnd/src/histogramnd_c.c",
                "src/silx/math/chistogramnd.pyx",
            ],
            include_dirs=[
                "src/silx/math/histogramnd/include",
                numpy.get_include(),
            ],
            language="c",
        ),
        Extension(
            name="silx.math.chistogramnd_lut",
            sources=["src/silx/math/chistogramnd_lut.pyx"],
            include_dirs=[
                "src/silx/math/histogramnd/include",
                numpy.get_include(),
            ],
            language="c",
        ),
        Extension(
            name="silx.math.combo",
            sources=["src/silx/math/combo.pyx"],
            include_dirs=["src/silx/math/include"],
            language="c",
        ),
        Extension(
            name="silx.math.interpolate",
            sources=["src/silx/math/interpolate.pyx"],
            language="c",
            include_dirs=[
                "src/silx/math/include",
                numpy.get_include(),
            ],
            extra_link_args=["-fopenmp"],
            extra_compile_args=["-fopenmp"],
        ),
        Extension(
            name="silx.math.marchingcubes",
            sources=[
                "src/silx/math/marchingcubes/mc_lut.cpp",
                "src/silx/math/marchingcubes.pyx",
            ],
            include_dirs=[
                "src/silx/math/marchingcubes",
                numpy.get_include(),
            ],
            language="c++",
        ),
        Extension(
            name="silx.math.medianfilter.medianfilter",
            sources=["src/silx/math/medianfilter/medianfilter.pyx"],
            include_dirs=[
                "src/silx/math/medianfilter/include",
                numpy.get_include(),
            ],
            language="c++",
            extra_link_args=["-fopenmp"],
            extra_compile_args=["-fopenmp"],
        ),
        # silx.math.fit
        Extension(
            name="silx.math.fit.filters",
            sources=[
                "src/silx/math/fit/filters/src/smoothnd.c",
                "src/silx/math/fit/filters/src/snip1d.c",
                "src/silx/math/fit/filters/src/snip2d.c",
                "src/silx/math/fit/filters/src/snip3d.c",
                "src/silx/math/fit/filters/src/strip.c",
                "src/silx/math/fit/filters.pyx",
            ],
            include_dirs=["src/silx/math/fit/filters/include"],
            language="c",
        ),
        Extension(
            name="silx.math.fit.functions",
            sources=[
                "src/silx/math/fit/functions/src/funs.c",
                "src/silx/math/fit/functions.pyx",
            ],
            include_dirs=["src/silx/math/fit/functions/include"],
            language="c",
        ),
        Extension(
            name="silx.math.fit.peaks",
            sources=[
                "src/silx/math/fit/peaks/src/peaks.c",
                "src/silx/math/fit/peaks.pyx",
            ],
            include_dirs=["src/silx/math/fit/peaks/include"],
            language="c",
        ),
    ]

    return dict(
        install_requires=install_requires,
        extras_require=extras_require,
        cmdclass=dict(build_ext=BuildExt),
        ext_modules=ext_modules,
    )


if __name__ == "__main__":
    from setuptools import setup

    setup(**get_project_configuration())
