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

from silx.gui.plot.items import Axis, Scatter
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
        self.plot.setData(x, y, value)
        self.qapp.processEvents()

        data = self.plot.getData()
        self.assertEqual(len(data), 5)
        self.assertTrue(numpy.all(numpy.equal(x, data[0])))
        self.assertTrue(numpy.all(numpy.equal(y, data[1])))
        self.assertTrue(numpy.all(numpy.equal(value, data[2])))
        self.assertIsNone(data[3])  # xerror
        self.assertIsNone(data[4])  # yerror

        # Test access to scatter item
        self.assertIsInstance(self.plot.getScatterItem(), Scatter)

        # Test toolbar actions

        action = self.plot.getScatterToolBar().getXAxisLogarithmicAction()
        action.trigger()
        self.qapp.processEvents()

        maskAction = self.plot.getScatterToolBar().actions()[-1]
        maskAction.trigger()
        self.qapp.processEvents()

        # Test proxy API

        self.plot.resetZoom()
        self.qapp.processEvents()

        scale = self.plot.getXAxis().getScale()
        self.assertEqual(scale, Axis.LOGARITHMIC)

        scale = self.plot.getYAxis().getScale()
        self.assertEqual(scale, Axis.LINEAR)

        title = 'Test ScatterView'
        self.plot.setGraphTitle(title)
        self.assertEqual(self.plot.getGraphTitle(), title)

        self.qapp.processEvents()

        # Reset scatter data

        self.plot.setData(None, None, None)
        self.qapp.processEvents()

        data = self.plot.getData()
        self.assertEqual(len(data), 5)
        self.assertEqual(len(data[0]), 0)  # x
        self.assertEqual(len(data[1]), 0)  # y
        self.assertEqual(len(data[2]), 0)  # value
        self.assertIsNone(data[3])  # xerror
        self.assertIsNone(data[4])  # yerror


def suite():
    test_suite = unittest.TestSuite()
    loadTests = unittest.defaultTestLoader.loadTestsFromTestCase
    test_suite.addTest(loadTests(TestScatterView))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
