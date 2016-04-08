# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016 European Synchrotron Radiation Facility
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
"""Basic tests for Plot"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "02/03/2016"


import unittest

import numpy

from silx.gui.plot.Plot import Plot


class TestPlot(unittest.TestCase):
    """Basic tests of Plot without backend"""

    def testPlotTitleLabels(self):
        """Create a Plot and set the labels"""

        plot = Plot(backend='none')

        title, xlabel, ylabel = 'the title', 'x label', 'y label'
        plot.setGraphTitle(title)
        plot.setGraphXLabel(xlabel)
        plot.setGraphYLabel(ylabel)

        self.assertEqual(plot.getGraphTitle(), title)
        self.assertEqual(plot.getGraphXLabel(), xlabel)
        self.assertEqual(plot.getGraphYLabel(), ylabel)

    def testAddNoRemove(self):
        """add objects to the Plot"""

        plot = Plot(backend='none')
        plot.addCurve(x=(1, 2, 3), y=(3, 2, 1))
        plot.addImage(numpy.arange(100.).reshape(10, -1))
        plot.addItem(
            numpy.array((1., 10.)), numpy.array((10., 10.)), shape="rectangle")
        plot.addXMarker(10.)


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestPlot))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
