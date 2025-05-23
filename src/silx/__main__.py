#!/usr/bin/env python
# /*##########################################################################
#
# Copyright (c) 2017-2021 European Synchrotron Radiation Facility
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
"""This module describe silx applications which are available  through
the silx launcher.

Your environment should provide a command `silx`. You can reach help with
`silx --help`, and check the version with `silx --version`.
"""

__authors__ = ["V. Valls", "P. Knobel"]
__license__ = "MIT"
__date__ = "16/04/2025"


import logging

logging.basicConfig()

import multiprocessing
import sys
from silx.utils.launcher import Launcher
import silx._version


def main():
    """Main function of the launcher

    This function is referenced in the pyproject.toml file as single entry-point
    for all silx applications.

    :rtype: int
    :returns: The execution status
    """
    multiprocessing.freeze_support()

    launcher = Launcher(prog="silx", version=silx._version.version)
    launcher.add_command(
        "view",
        module_name="silx.app.view.main",
        description="Browse a data file with a GUI",
    )
    launcher.add_command(
        "convert",
        module_name="silx.app.convert",
        description="Convert and concatenate files into a HDF5 file",
    )
    launcher.add_command(
        "compare",
        module_name="silx.app.compare.main",
        description="Compare images with a GUI",
    )
    launcher.add_command(
        "test", module_name="silx.app.test_", description="Launch silx unittest"
    )
    status = launcher.execute(sys.argv)
    return status


if __name__ == "__main__":
    # executed when using python -m PROJECT_NAME
    status = main()
    sys.exit(status)
