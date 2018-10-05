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
"""Basic tests for CurvesROIWidget"""

__authors__ = ["T. Vincent", "P. Knobel", "H. Payno"]
__license__ = "MIT"
__date__ = "16/11/2017"


import logging
import os.path
import unittest
from collections import OrderedDict
import numpy
from silx.gui import qt
from silx.test.utils import temp_dir
from silx.gui.utils.testutils import TestCaseQt
from silx.gui.plot import PlotWindow, CurvesROIWidget


_logger = logging.getLogger(__name__)


class TestCurvesROIWidget(TestCaseQt):
    """Basic test for CurvesROIWidget"""

    def setUp(self):
        super(TestCurvesROIWidget, self).setUp()
        self.plot = PlotWindow()
        self.plot.show()
        self.qWaitForWindowExposed(self.plot)

        self.widget = CurvesROIWidget.CurvesROIDockWidget(plot=self.plot, name='TEST')
        self.widget.show()
        self.qWaitForWindowExposed(self.widget)

    def tearDown(self):
        self.plot.setAttribute(qt.Qt.WA_DeleteOnClose)
        self.plot.close()
        del self.plot

        self.widget.setAttribute(qt.Qt.WA_DeleteOnClose)
        self.widget.close()
        del self.widget

        super(TestCurvesROIWidget, self).tearDown()

    def testEmptyPlot(self):
        """Empty plot, display ROI widget"""
        pass

    def testWithCurves(self):
        """Plot with curves: test all ROI widget buttons"""
        for offset in range(2):
            self.plot.addCurve(numpy.arange(1000),
                               offset + numpy.random.random(1000),
                               legend=str(offset))

        # Add two ROI
        self.mouseClick(self.widget.roiWidget.addButton, qt.Qt.LeftButton)
        self.mouseClick(self.widget.roiWidget.addButton, qt.Qt.LeftButton)

        # Change active curve
        self.plot.setActiveCurve(str(1))

        # Delete a ROI
        self.mouseClick(self.widget.roiWidget.delButton, qt.Qt.LeftButton)

        with temp_dir() as tmpDir:
            self.tmpFile = os.path.join(tmpDir, 'test.ini')

            # Save ROIs
            self.widget.roiWidget.save(self.tmpFile)
            self.assertTrue(os.path.isfile(self.tmpFile))

            # Reset ROIs
            self.mouseClick(self.widget.roiWidget.resetButton,
                            qt.Qt.LeftButton)

            # Load ROIs
            self.widget.roiWidget.load(self.tmpFile)

            del self.tmpFile

    def testMiddleMarker(self):
        """Test with middle marker enabled"""
        self.widget.roiWidget.setMiddleROIMarkerFlag(True)

        # Add a ROI
        self.mouseClick(self.widget.roiWidget.addButton, qt.Qt.LeftButton)

        xleftMarker = self.plot._getMarker(legend='ROI min').getXPosition()
        xMiddleMarker = self.plot._getMarker(legend='ROI middle').getXPosition()
        xRightMarker = self.plot._getMarker(legend='ROI max').getXPosition()
        self.assertAlmostEqual(xMiddleMarker,
                               xleftMarker + (xRightMarker - xleftMarker) / 2.)

    def testCalculation(self):
        x = numpy.arange(100.)
        y = numpy.arange(100.)

        # Add two curves
        self.plot.addCurve(x, y, legend="positive")
        self.plot.addCurve(-x, y, legend="negative")

        # Make sure there is an active curve and it is the positive one
        self.plot.setActiveCurve("positive")

        # Add two ROIs
        ddict = {}
        ddict["positive"] = {"from": 10, "to": 20, "type":"X"}
        ddict["negative"] = {"from": -20, "to": -10, "type":"X"}
        self.widget.roiWidget.setRois(ddict)

        # And calculate the expected output
        self.widget.calculateROIs()

        output = self.widget.roiWidget.getRois()
        self.assertEqual(output["positive"]["rawcounts"],
                         y[ddict["positive"]["from"]:ddict["positive"]["to"]+1].sum(),
                         "Calculation failed on positive X coordinates")

        # Set the curve with negative X coordinates as active
        self.plot.setActiveCurve("negative")

        # the ROIs should have been automatically updated
        output = self.widget.roiWidget.getRois()
        selection = numpy.nonzero((-x >= output["negative"]["from"]) & \
                                  (-x <= output["negative"]["to"]))[0]
        self.assertEqual(output["negative"]["rawcounts"],
                         y[selection].sum(), "Calculation failed on negative X coordinates")

    def testDeferedInit(self):
        x = numpy.arange(100.)
        y = numpy.arange(100.)
        self.plot.addCurve(x=x, y=y, legend="name", replace="True")
        roisDefs = OrderedDict([
            ["range1",
             OrderedDict([["from", 20], ["to", 200], ["type", "energy"]])],
            ["range2",
             OrderedDict([["from", 300], ["to", 500], ["type", "energy"]])]
        ])

        roiWidget = self.plot.getCurvesRoiDockWidget().roiWidget
        self.assertFalse(roiWidget._isInit)
        self.plot.getCurvesRoiDockWidget().setRois(roisDefs)
        self.assertTrue(len(roiWidget.getRois()) is len(roisDefs))
        self.plot.getCurvesRoiDockWidget().setVisible(True)
        self.assertTrue(len(roiWidget.getRois()) is len(roisDefs))


def suite():
    test_suite = unittest.TestSuite()
    for TestClass in (TestCurvesROIWidget,):
        test_suite.addTest(
            unittest.defaultTestLoader.loadTestsFromTestCase(TestClass))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
