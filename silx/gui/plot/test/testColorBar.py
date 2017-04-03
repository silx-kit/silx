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
"""Basic tests for ColorBar featues"""

__authors__ = ["H. Payno"]
__license__ = "MIT"
__date__ = "03/04/2017"

import unittest
from silx.gui.plot.Colorbar import Gradation
import numpy

class TestFormats(unittest.TestCase):
    """Test that the format used to display the ticks labels are correct"""
    def testScientificDisplay(self):
        pass
        # TODO

class TestGradation(unittest.TestCase):
    """Test that interaction with the gradation is correct"""
    def setUp(self):
        self.colorMapLin1 = { 'name': 'gray', 'normalization': 'linear',
                    'autoscale': False, 'vmin': 0.0, 'vmax': 1.0 }

        self.colorMapLin2 = { 'name': 'viridis', 'normalization': 'linear',
                    'autoscale': False, 'vmin': -10, 'vmax': 0 }

        self.colorMapLog1 = { 'name': 'temperature', 'normalization': 'log',
                    'autoscale': False, 'vmin': 10.0, 'vmax': 1e8 }

        self.colorMapLog2 = { 'name': 'red', 'normalization': 'log',
                    'autoscale': False, 'vmin': 10, 'vmax': 12 }

        self.gradationWidget = Gradation(colormap=None, parent=None)

    def tearDown(self):
        self.gradationWidget.deleteLater()
        self.gradationWidget = None        

    def testRelativePositionLinear(self):
        self.gradationWidget.setColormap(self.colorMapLin1)
        self.assertTrue(
            self.gradationWidget.getValueFromRelativePosition(0.25) == 0.25)
        self.assertTrue(
            self.gradationWidget.getValueFromRelativePosition(0.5) == 0.5)
        self.assertTrue(
            self.gradationWidget.getValueFromRelativePosition(1.0) == 1.0)
        self.gradationWidget.setColormap(self.colorMapLin2)
        self.assertTrue(
            self.gradationWidget.getValueFromRelativePosition(0.25) == -7.5)
        self.assertTrue(
            self.gradationWidget.getValueFromRelativePosition(0.5) == -5.0)
        self.assertTrue(
            self.gradationWidget.getValueFromRelativePosition(1.0) == 0.0)

    def testRelativePositionLog(self):
        self.gradationWidget.setColormap(self.colorMapLog1)

        reVal = self.gradationWidget.getValueFromRelativePosition(0.25)
        thVal = self.getLogScaleValue(0.25, self.colorMapLog1['vmin'], self.colorMapLog1['vmax'] )
        self.assertTrue(thVal == reVal)

        thVal = self.gradationWidget.getValueFromRelativePosition(0.5)
        reVal = self.getLogScaleValue(0.5, self.colorMapLog1['vmin'], self.colorMapLog1['vmax'] )
        self.assertTrue(thVal == reVal)
        
        thVal = self.gradationWidget.getValueFromRelativePosition(1.0)
        reVal = self.getLogScaleValue(1.0, self.colorMapLog1['vmin'], self.colorMapLog1['vmax'] )
        self.assertTrue(thVal == reVal)
        
    def getLogScaleValue(self, relativeVal, vmin, vmax):
        assert(vmin > 0)
        assert(vmax > 0)
        assert((relativeVal >= 0.0) and (relativeVal <= 1.0))
        return vmin + numpy.exp((numpy.log10(vmax) - numpy.log10(vmin))*relativeVal)

class TestTickBar(unittest.TestCase):
    """Test that ticks displayed in the TickBar are correct"""
    def testLogNorm(self):
        pass

    def testLinearNorm(self):
        pass


def suite():
    test_suite = unittest.TestSuite()
    for ui in (TestFormats, TestGradation, TestTickBar):
        test_suite.addTest(
            unittest.defaultTestLoader.loadTestsFromTestCase(ui))

    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')