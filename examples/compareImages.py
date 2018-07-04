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
from silx.gui import qt
import numpy
import silx.test.utils
from silx.gui.plot.CompareImages import CompareImages


def createTestData():
    data = numpy.arange(100 * 100)
    data = (data % 100) / 5.0
    data = numpy.sin(data)
    data1 = data.copy()
    data1.shape = 100, 100
    data2 = silx.test.utils.add_gaussian_noise(data, stdev=0.1)
    data2.shape = 100, 100
    return data1, data2


if __name__ == "__main__":
    if len(sys.argv) == 3:
        from PIL import Image
        data1 = numpy.asarray(Image.open(sys.argv[1]))
        data2 = numpy.asarray(Image.open(sys.argv[2]))
    else:
        data1, data2 = createTestData()

    app = qt.QApplication([])
    window = CompareImages()
    window.setData(data1, data2)
    window.setVisible(True)
    app.exec_()
