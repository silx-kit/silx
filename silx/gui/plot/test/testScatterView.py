# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2018 European Synchrotron Radiation Facility
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
"""Basic tests for ScatterView"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "06/03/2018"


import unittest

import numpy

from silx.gui.plot import ScatterView
from silx.gui.plot.test.utils import PlotWidgetTestCase


class TestScatterView(PlotWidgetTestCase):
    """Test of ScatterView widget"""

    def _createPlot(self):
        return ScatterView()

    def test(self):
        """Simple tests"""
        x = numpy.arange(100)
        y = numpy.arange(100)
        value = numpy.arange(100)
        self.plot.addScatter(x, y, value)
        self.qapp.processEvents()

        action = self.plot.getScatterToolBar().getXAxisLogarithmicAction()
        action.trigger()
        self.qapp.processEvents()

        maskAction = self.plot.getScatterToolBar().actions()[-1]
        maskAction.trigger()
        self.qapp.processEvents()


def suite():
    test_suite = unittest.TestSuite()
    loadTests = unittest.defaultTestLoader.loadTestsFromTestCase
    test_suite.addTest(loadTests(TestScatterView))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
