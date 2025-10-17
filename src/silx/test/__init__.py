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

import importlib
import os.path
import subprocess
import sys
import tempfile
from collections.abc import Sequence


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
            if getattr(imported_module, "__path__", None):
                module_path = imported_module.__path__[0]
            elif os.path.basename(imported_module.__file__) == "__init__.py":
                module_path = os.path.dirname(imported_module.__file__)
            else:
                module_path = imported_module.__file__

            list_path.append(module_path)
        cmd += list_path

    with tempfile.TemporaryDirectory(prefix="silx-") as workdir:
        print(f"Running pytest in `{workdir}` with this command:")
        print(" ".join(f'"{i}"' if " " in i else i for i in cmd))
        result = subprocess.run(cmd, check=False, cwd=workdir).returncode
    return result
