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
"""This package provides test of the root modules
"""

import importlib
import logging
import subprocess
import sys


try:
    import pytest
except ImportError:
    logging.getLogger(__name__).error(
        "pytest is required to run the tests, please install it."
    )
    raise


import silx


def run_tests(module: str = "silx", verbosity: int = 0, args=()):
    """Run tests in a subprocess

    :param module: Name of the silx module to test (default: 'silx')
    :param verbosity: Requested level of verbosity
    :param args: List of extra arguments to pass to `pytest`
    """
    # Retrieve folder for packages and file for modules
    imported_module = importlib.import_module(module)
    if hasattr(imported_module, "__path__"):
        tested_path = imported_module.__path__[0]
    else:
        tested_path = imported_module.__file__

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        tested_path,
        f"--rootdir={silx.__path__[0]}",
        "--verbosity",
        str(verbosity),
        # Handle warning as errors unless explicitly skipped
        "-Werror",
        "-Wignore:tostring() is deprecated. Use tobytes() instead.:DeprecationWarning",
        "-Wignore:Jupyter is migrating its paths to use standard platformdirs:DeprecationWarning",
    ] + list(args)

    return subprocess.run(cmd, check=False).returncode
