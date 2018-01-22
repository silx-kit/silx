# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017 European Synchrotron Radiation Facility
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
"""Test suite for :class:`ComplexImageView`"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "17/01/2018"


import unittest
import logging
import numpy

from silx.utils.testutils import ParametricTestCase
from silx.gui.plot import ComplexImageView

from .utils import PlotWidgetTestCase


logger = logging.getLogger(__name__)


class TestComplexImageView(PlotWidgetTestCase, ParametricTestCase):
    """Test suite of ComplexImageView widget"""

    def _createPlot(self):
        return ComplexImageView.ComplexImageView()

    def testPlot2DComplex(self):
        """Test API of ComplexImageView widget"""
        data = numpy.array(((0, 1j), (1, 1 + 1j)), dtype=numpy.complex)
        self.plot.setData(data)
        self.plot.setKeepDataAspectRatio(True)
        self.plot.getPlot().resetZoom()
        self.qWait(100)

        # Test colormap API
        colormap = self.plot.getColormap().copy()
        colormap.setName('magma')
        self.plot.setColormap(colormap)
        self.qWait(100)

        # Test all modes
        modes = self.plot.getSupportedVisualizationModes()
        for mode in modes:
            with self.subTest(mode=mode):
                self.plot.setVisualizationMode(mode)
                self.qWait(100)

        # Test origin and scale API
        self.plot.setScale((2, 1))
        self.qWait(100)
        self.plot.setOrigin((1, 1))
        self.qWait(100)

        # Test no data
        self.plot.setData(numpy.zeros((0, 0), dtype=numpy.complex))
        self.qWait(100)

        # Test float data
        self.plot.setData(numpy.arange(100, dtype=numpy.float).reshape(10, 10))
        self.qWait(100)


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(
        TestComplexImageView))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
