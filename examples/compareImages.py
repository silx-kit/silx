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
        _logger.error("Error while loading image with silx.io", exc_info=True)

    if fabio is None and PIL is None:
        raise ImportError("fabio nor PIL are not available")

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


if __name__ == "__main__":
    if len(sys.argv) == 3:
        _logger.info("Load images from files")
        data1 = loadImage(sys.argv[1])
        data2 = loadImage(sys.argv[2])
    else:
        _logger.info("Generate test data")
        data1, data2 = createTestData()

    app = qt.QApplication([])
    window = CompareImages()
    window.setData(data1, data2)
    window.setVisible(True)
    app.exec_()
