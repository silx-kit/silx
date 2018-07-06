#!/usr/bin/env python
# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2017 European Synchrotron Radiation Facility
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

import silx.io
from silx.gui import qt
import silx.test.utils
from silx.gui.plot.CompareImages import CompareImages

_logger = logging.getLogger(__name__)

try:
    import fabio
except ImportError:
    _logger.debug("Backtrace", exc_info=True)
    fabio = None

try:
    import PIL
except ImportError:
    _logger.debug("Backtrace", exc_info=True)
    PIL = None


def createTestData():
    data = numpy.arange(100 * 100)
    data = (data % 100) / 5.0
    data = numpy.sin(data)
    data1 = data.copy()
    data1.shape = 100, 100
    data2 = silx.test.utils.add_gaussian_noise(data, stdev=0.1)
    data2.shape = 100, 100
    return data1, data2


def loadImage(filename):
    try:
        return silx.io.get_data(filename)
    except Exception:
        _logger.debug("Error while loading image with silx.io", exc_info=True)

    if fabio is None and PIL is None:
        raise ImportError("fabio nor PIL are available")

    if fabio is not None:
        try:
            return fabio.open(filename).data
        except Exception:
            _logger.debug("Error while loading image with fabio", exc_info=True)

    if PIL is not None:
        try:
            return numpy.asarray(PIL.Image.open(filename))
        except Exception:
            _logger.debug("Error while loading image with PIL", exc_info=True)

    raise Exception("Impossible to load '%s' with the available image libraries" % filename)


file_description = """
Image data to compare (HDF5 file with path, EDF files, JPEG/PNG image files).
Data from HDF5 files can be accessed using dataset path and slicing as an URL: silx:../my_file.h5?path=/entry/data&slice=10
EDF file frames also can can be accessed using URL: fabio:../my_file.edf?slice=10
Using URL in command like usually have to be quoted: "URL".
"""


def createParser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'files',
        nargs=argparse.ZERO_OR_MORE,
        help=file_description)
    parser.add_argument(
        '--debug',
        dest="debug",
        action="store_true",
        default=False,
        help='Set logging system in debug mode')
    parser.add_argument(
        '--testdata',
        dest="testdata",
        action="store_true",
        default=False,
        help='Use synthetic images to test the application')
    parser.add_argument(
        '--use-opengl-plot',
        dest="use_opengl_plot",
        action="store_true",
        default=False,
        help='Use OpenGL for plots (instead of matplotlib)')
    return parser


if __name__ == "__main__":
    parser = createParser()
    options = parser.parse_args(sys.argv[1:])

    if options.debug:
        logging.root.setLevel(logging.DEBUG)

    if options.testdata:
        _logger.info("Generate test data")
        data1, data2 = createTestData()
    else:
        if len(options.files) != 2:
            raise Exception("Expected 2 images to compare them")
        data1 = loadImage(options.files[0])
        data2 = loadImage(options.files[1])

    if options.use_opengl_plot:
        backend = "gl"
    else:
        backend = "mpl"

    app = qt.QApplication([])
    window = CompareImages(backend=backend)
    window.setData(data1, data2)
    window.setVisible(True)
    app.exec_()
