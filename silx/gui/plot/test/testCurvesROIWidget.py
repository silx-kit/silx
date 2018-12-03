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

        self.widget = self.plot.getCurvesRoiDockWidget()

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

    def testWithCurves(self):
        """Plot with curves: test all ROI widget buttons"""
        for offset in range(2):
            self.plot.addCurve(numpy.arange(1000),
                               offset + numpy.random.random(1000),
                               legend=str(offset))

        # Add two ROI
        self.mouseClick(self.widget.roiWidget.addButton, qt.Qt.LeftButton)
        self.qWait(200)
        self.mouseClick(self.widget.roiWidget.addButton, qt.Qt.LeftButton)
        self.qWait(200)

        # Change active curve
        self.plot.setActiveCurve(str(1))

        # Delete a ROI
        self.mouseClick(self.widget.roiWidget.delButton, qt.Qt.LeftButton)
        self.qWait(200)

        with temp_dir() as tmpDir:
            self.tmpFile = os.path.join(tmpDir, 'test.ini')

            # Save ROIs
            self.widget.roiWidget.save(self.tmpFile)
            self.assertTrue(os.path.isfile(self.tmpFile))
            self.assertTrue(len(self.widget.getRois()) is 2)

            # Reset ROIs
            self.mouseClick(self.widget.roiWidget.resetButton,
                            qt.Qt.LeftButton)
            self.qWait(200)
            rois = self.widget.getRois()
            self.assertTrue(len(rois) is 1)
            print(rois)
            roiID = list(rois.keys())[0]
            self.assertTrue(rois[roiID].getName() == 'ICR')

            # Load ROIs
            self.widget.roiWidget.load(self.tmpFile)
            self.assertTrue(len(self.widget.getRois()) is 2)

            del self.tmpFile

    def testMiddleMarker(self):
        """Test with middle marker enabled"""
        self.widget.roiWidget.roiTable.setMiddleROIMarkerFlag(True)

        # Add a ROI
        self.mouseClick(self.widget.roiWidget.addButton, qt.Qt.LeftButton)

        for roiID in self.widget.roiWidget.roiTable._markersHandler._roiMarkerHandlers:
            handler = self.widget.roiWidget.roiTable._markersHandler._roiMarkerHandlers[roiID]
            assert handler.getMarker('min')
            xleftMarker = handler.getMarker('min').getXPosition()
            xMiddleMarker = handler.getMarker('middle').getXPosition()
            xRightMarker = handler.getMarker('max').getXPosition()
            thValue = xleftMarker + (xRightMarker - xleftMarker) / 2.
            self.assertAlmostEqual(xMiddleMarker, thValue)

    def testAreaCalculation(self):
        x = numpy.arange(100.)
        y = numpy.arange(100.)

        # Add two curves
        self.plot.addCurve(x, y, legend="positive")
        self.plot.addCurve(-x, y, legend="negative")

        # Make sure there is an active curve and it is the positive one
        self.plot.setActiveCurve("positive")

        # Add two ROIs
        roi_neg = CurvesROIWidget.ROI(name='negative', fromdata=-20,
                                      todata=-10, type_='X')
        roi_pos = CurvesROIWidget.ROI(name='positive', fromdata=10,
                                      todata=20, type_='X')

        self.widget.roiWidget.setRois((roi_pos, roi_neg))

        posCurve = self.plot.getCurve('positive')
        negCurve = self.plot.getCurve('negative')

        self.assertEqual(roi_pos.computeRawAndNetArea(posCurve),
                        (numpy.trapz(y=[10, 20], x=[10, 20]),
                        0.0))
        self.assertEqual(roi_pos.computeRawAndNetArea(negCurve),
                         (0.0, 0.0))
        self.assertEqual(roi_neg.computeRawAndNetArea(posCurve),
                         ((0.0), 0.0))
        self.assertEqual(roi_neg.computeRawAndNetArea(negCurve),
                         ((-150.0), 0.0))

    def testCountsCalculation(self):
        x = numpy.arange(100.)
        y = numpy.arange(100.)

        # Add two curves
        self.plot.addCurve(x, y, legend="positive")
        self.plot.addCurve(-x, y, legend="negative")

        # Make sure there is an active curve and it is the positive one
        self.plot.setActiveCurve("positive")

        # Add two ROIs
        roi_neg = CurvesROIWidget.ROI(name='negative', fromdata=-20,
                                      todata=-10, type_='X')
        roi_pos = CurvesROIWidget.ROI(name='positive', fromdata=10,
                                      todata=20, type_='X')

        self.widget.roiWidget.setRois((roi_pos, roi_neg))

        posCurve = self.plot.getCurve('positive')
        negCurve = self.plot.getCurve('negative')

        self.assertEqual(roi_pos.computeRawAndNetCounts(posCurve),
                         (y[10:21].sum(), 0.0))
        self.assertEqual(roi_pos.computeRawAndNetCounts(negCurve),
                         (0.0, 0.0))
        self.assertEqual(roi_neg.computeRawAndNetCounts(posCurve),
                         ((0.0), 0.0))
        self.assertEqual(roi_neg.computeRawAndNetCounts(negCurve),
                         (y[10:21].sum(), 0.0))

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
        self.plot.getCurvesRoiDockWidget().setRois(roisDefs)
        self.assertTrue(len(roiWidget.getRois()) is len(roisDefs))
        self.plot.getCurvesRoiDockWidget().setVisible(True)
        self.assertTrue(len(roiWidget.getRois()) is len(roisDefs))

    def testDictCompatibility(self):
        """Test that ROI api is valid with dict and not information is lost"""
        roiDict = {'from': 20, 'to': 200, 'type': 'energy', 'comment': 'no',
                   'name': 'myROI', 'calibration': [1, 2, 3]}
        roi = CurvesROIWidget.ROI._fromDict(roiDict)
        self.assertTrue(roi.toDict() == roiDict)

    def testShowAllROI(self):
        x = numpy.arange(100.)
        y = numpy.arange(100.)
        self.plot.addCurve(x=x, y=y, legend="name", replace="True")

        roisDefsDict = {
            "range1": {"from": 20, "to": 200,"type": "energy"},
            "range2": {"from": 300, "to": 500, "type": "energy"}
        }

        roisDefsObj = (
            CurvesROIWidget.ROI(name='range3', fromdata=20, todata=200,
                                type_='energy'),
            CurvesROIWidget.ROI(name='range4', fromdata=300, todata=500,
                                type_='energy')
        )
        self.widget.roiWidget.showAllMarkers(True)
        roiWidget = self.plot.getCurvesRoiDockWidget().roiWidget
        roiWidget.setRois(roisDefsDict)
        self.assertTrue(len(self.plot._getAllMarkers()) is 2*3)

        markersHandler = self.widget.roiWidget.roiTable._markersHandler
        roiWidget.showAllMarkers(True)
        ICRROI = markersHandler.getVisibleRois()
        self.assertTrue(len(ICRROI) is 2)

        roiWidget.showAllMarkers(False)
        ICRROI = markersHandler.getVisibleRois()
        self.assertTrue(len(ICRROI) is 1)

        roiWidget.setRois(roisDefsObj)
        self.qapp.processEvents()
        self.assertTrue(len(self.plot._getAllMarkers()) is 2*3)

        markersHandler = self.widget.roiWidget.roiTable._markersHandler
        roiWidget.showAllMarkers(True)
        ICRROI = markersHandler.getVisibleRois()
        self.assertTrue(len(ICRROI) is 2)

        roiWidget.showAllMarkers(False)
        ICRROI = markersHandler.getVisibleRois()
        self.assertTrue(len(ICRROI) is 1)

    def testRoiEdition(self):
        """Make sure if the ROI object is edited the ROITable will be updated
        """
        roi = CurvesROIWidget.ROI(name='linear', fromdata=0, todata=5)
        self.widget.roiWidget.setRois((roi, ))

        x = (0, 1, 1, 2, 2, 3)
        y = (1, 1, 2, 2, 1, 1)
        self.plot.addCurve(x=x, y=y, legend='linearCurve')
        self.plot.setActiveCurve(legend='linearCurve')
        self.widget.calculateROIs()

        roiTable = self.widget.roiWidget.roiTable
        indexesColumns = CurvesROIWidget.ROITable.COLUMNS_INDEX
        itemRawCounts = roiTable.item(0, indexesColumns['Raw Counts'])
        itemNetCounts = roiTable.item(0, indexesColumns['Net Counts'])

        self.assertTrue(itemRawCounts.text() == '8.0')
        self.assertTrue(itemNetCounts.text() == '2.0')

        itemRawArea = roiTable.item(0, indexesColumns['Raw Area'])
        itemNetArea = roiTable.item(0, indexesColumns['Net Area'])

        self.assertTrue(itemRawArea.text() == '4.0')
        self.assertTrue(itemNetArea.text() == '1.0')

        roi.setTo(2)
        itemRawArea = roiTable.item(0, indexesColumns['Raw Area'])
        self.assertTrue(itemRawArea.text() == '3.0')
        roi.setFrom(1)
        itemRawArea = roiTable.item(0, indexesColumns['Raw Area'])
        self.assertTrue(itemRawArea.text() == '2.0')

    def testRemoveActiveROI(self):
        roi = CurvesROIWidget.ROI(name='linear', fromdata=0, todata=5)
        self.widget.roiWidget.setRois((roi,))

        self.widget.roiWidget.roiTable.setActiveRoi(None)
        self.assertTrue(len(self.widget.roiWidget.roiTable.selectedItems()) is 0)
        self.widget.roiWidget.setRois((roi,))
        self.plot.setActiveCurve(legend='linearCurve')
        self.widget.calculateROIs()


def suite():
    test_suite = unittest.TestSuite()
    for TestClass in (TestCurvesROIWidget,):
        test_suite.addTest(
            unittest.defaultTestLoader.loadTestsFromTestCase(TestClass))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
