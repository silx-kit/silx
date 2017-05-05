#!/usr/bin/env python
# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2004-2016 European Synchrotron Radiation Facility
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

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "04/04/2017"


import logging
logging.basicConfig()

import sys
from silx.utils.launcher import Launcher
import silx._version


def main():
    """Main function of the launcher

    :rtype: int
    :returns: The execution status
    """
    launcher = Launcher(prog="silx", version=silx._version.version)
    launcher.add_command("view",
                         module_name="silx.app.view",
                         description="Browse a data file with a GUI")
    status = launcher.execute(sys.argv)
    return status

status = main()
sys.exit(status)
