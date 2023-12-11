#!/usr/bin/env python
# /*##########################################################################
#
# Copyright (c) 2016-2021 European Synchrotron Radiation Facility
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
"""Example demonstrating the use of the widget CompareImages
"""

import sys
import logging
import numpy
import argparse
import os

import silx.io
from silx.gui import qt
import silx.test.utils
from silx.gui.plot.CompareImages import CompareImages

_logger = logging.getLogger(__name__)


def createTestData():
    data = numpy.arange(100 * 100)
    data = (data % 100) / 5.0
    data = numpy.sin(data)
    data1 = data.copy()
    data1.shape = 100, 100
    data2 = silx.test.utils.add_gaussian_noise(data, stdev=0.1)
    data2.shape = 100, 100
    return data1, data2


def createParser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        default=False,
        help="Set logging system in debug mode",
    )
    parser.add_argument(
        "--use-opengl-plot",
        dest="use_opengl_plot",
        action="store_true",
        default=False,
        help="Use OpenGL for plots (instead of matplotlib)",
    )
    return parser


if __name__ == "__main__":
    parser = createParser()
    options = parser.parse_args(sys.argv[1:])

    if options.debug:
        logging.root.setLevel(logging.DEBUG)

    if options.use_opengl_plot:
        backend = "gl"
    else:
        backend = "mpl"

    app = qt.QApplication([])

    data1, data2 = createTestData()
    window = CompareImages(backend=backend)
    window.setData(data1, data2)
    window.setVisible(True)
    app.exec()
