# /*##########################################################################
#
# Copyright (c) 2015-2024 European Synchrotron Radiation Facility
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
"""This package provides test of the root modules"""

from __future__ import annotations

from collections.abc import Sequence
import importlib
import logging
import subprocess
import sys


try:
    import pytest  # noqa: F401
except ImportError:
    logging.getLogger(__name__).error(
        "pytest is required to run the tests, please install it."
    )
    raise


def run_tests(
    modules: str | Sequence[str] | None = ("silx",),
    verbosity: int = 0,
    args: Sequence[str] = (),
):
    """Run tests in a subprocess

    :param module: Name of the silx module to test
    :param verbosity: Requested level of verbosity
    :param args: List of extra arguments to pass to `pytest`
    """
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "--verbosity",
        str(verbosity),
    ]

    if args:
        cmd += list(args)

    if isinstance(modules, str):
        modules = (modules,)

    if modules:
        list_path = []
        for module in modules:
        # Retrieve folder for packages and file for modules
            imported_module = importlib.import_module(module)
            list_path.append(
                imported_module.__path__[0]
                if hasattr(imported_module, "__path__")
                else imported_module.__file__
                )
        cmd += list_path

    print("Running pytest with this command:")
    print(" ".join(f'"{i}"' if " " in i else i for i in cmd))
    return subprocess.run(cmd, check=False).returncode
