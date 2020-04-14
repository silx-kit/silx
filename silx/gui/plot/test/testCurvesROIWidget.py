# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2020 European Synchrotron Radiation Facility
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
from silx.gui.plot import items
from silx.gui.plot import Plot1D
from silx.test.utils import temp_dir
from silx.gui.utils.testutils import TestCaseQt, SignalListener
from silx.gui.plot import PlotWindow, CurvesROIWidget
from silx.gui.plot.CurvesROIWidget import ROITable
from silx.gui.utils.testutils import getQToolButtonFromAction
from silx.gui.plot.PlotInteraction import ItemsInteraction

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

    def testDummyAPI(self):
        """Simple test of the getRois and setRois API"""
        roi_neg = CurvesROIWidget.ROI(name='negative', fromdata=-20,
                                      todata=-10, type_='X')
        roi_pos = CurvesROIWidget.ROI(name='positive', fromdata=10,
                                      todata=20, type_='X')

        self.widget.roiWidget.setRois((roi_pos, roi_neg))

        rois_defs = self.widget.roiWidget.getRois()
        self.widget.roiWidget.setRois(rois=rois_defs)

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
            self.assertEqual(len(self.widget.getRois()), 2)

            # Reset ROIs
            self.mouseClick(self.widget.roiWidget.resetButton,
                            qt.Qt.LeftButton)
            self.qWait(200)
            rois = self.widget.getRois()
            self.assertEqual(len(rois), 1)
            roiID = list(rois.keys())[0]
            self.assertEqual(rois[roiID].getName(), 'ICR')

            # Load ROIs
            self.widget.roiWidget.load(self.tmpFile)
            self.assertEqual(len(self.widget.getRois()), 2)

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
        """Test result of area calculation"""
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
        """Test result of count calculation"""
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
        """Test behavior of the deferedInit"""
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
        self.assertEqual(len(roiWidget.getRois()), len(roisDefs))
        self.plot.getCurvesRoiDockWidget().setVisible(True)
        self.assertEqual(len(roiWidget.getRois()), len(roisDefs))

    def testDictCompatibility(self):
        """Test that ROI api is valid with dict and not information is lost"""
        roiDict = {'from': 20, 'to': 200, 'type': 'energy', 'comment': 'no',
                   'name': 'myROI', 'calibration': [1, 2, 3]}
        roi = CurvesROIWidget.ROI._fromDict(roiDict)
        self.assertEqual(roi.toDict(), roiDict)

    def testShowAllROI(self):
        """Test the show allROI action"""
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
        markers = [item for item in self.plot.getItems()
                   if isinstance(item, items.MarkerBase)]
        self.assertEqual(len(markers), 2*3)

        markersHandler = self.widget.roiWidget.roiTable._markersHandler
        roiWidget.showAllMarkers(True)
        ICRROI = markersHandler.getVisibleRois()
        self.assertEqual(len(ICRROI), 2)

        roiWidget.showAllMarkers(False)
        ICRROI = markersHandler.getVisibleRois()
        self.assertEqual(len(ICRROI), 1)

        roiWidget.setRois(roisDefsObj)
        self.qapp.processEvents()
        markers = [item for item in self.plot.getItems()
                   if isinstance(item, items.MarkerBase)]
        self.assertEqual(len(markers), 2*3)

        markersHandler = self.widget.roiWidget.roiTable._markersHandler
        roiWidget.showAllMarkers(True)
        ICRROI = markersHandler.getVisibleRois()
        self.assertEqual(len(ICRROI), 2)

        roiWidget.showAllMarkers(False)
        ICRROI = markersHandler.getVisibleRois()
        self.assertEqual(len(ICRROI), 1)

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
        """Test widget behavior when removing the active ROI"""
        roi = CurvesROIWidget.ROI(name='linear', fromdata=0, todata=5)
        self.widget.roiWidget.setRois((roi,))

        self.widget.roiWidget.roiTable.setActiveRoi(None)
        self.assertEqual(len(self.widget.roiWidget.roiTable.selectedItems()), 0)
        self.widget.roiWidget.setRois((roi,))
        self.plot.setActiveCurve(legend='linearCurve')
        self.widget.calculateROIs()

    def testEmitCurrentROI(self):
        """Test behavior of the CurvesROIWidget.sigROISignal"""
        roi = CurvesROIWidget.ROI(name='linear', fromdata=0, todata=5)
        self.widget.roiWidget.setRois((roi,))
        signalListener = SignalListener()
        self.widget.roiWidget.sigROISignal.connect(signalListener.partial())
        self.widget.show()
        self.qapp.processEvents()
        self.assertEqual(signalListener.callCount(), 0)
        self.assertIs(self.widget.roiWidget.roiTable.activeRoi, roi)
        roi.setFrom(0.0)
        self.qapp.processEvents()
        self.assertEqual(signalListener.callCount(), 0)
        roi.setFrom(0.3)
        self.qapp.processEvents()
        self.assertEqual(signalListener.callCount(), 1)


class TestRoiWidgetSignals(TestCaseQt):
    """Test Signals emitted by the RoiWidgetSignals"""

    def setUp(self):
        self.plot = Plot1D()
        x = range(20)
        y = range(20)
        self.plot.addCurve(x, y, legend='curve0')
        self.listener = SignalListener()
        self.curves_roi_widget = self.plot.getCurvesRoiWidget()
        self.curves_roi_widget.sigROISignal.connect(self.listener)
        assert self.curves_roi_widget.isVisible() is False
        assert self.listener.callCount() == 0
        self.plot.show()
        self.qWaitForWindowExposed(self.plot)

        toolButton = getQToolButtonFromAction(self.plot.getRoiAction())
        self.mouseClick(widget=toolButton, button=qt.Qt.LeftButton)

        self.curves_roi_widget.show()
        self.qWaitForWindowExposed(self.curves_roi_widget)

    def tearDown(self):
        self.plot = None

    def testSigROISignalAddRmRois(self):
        """Test SigROISignal when adding and removing ROIS"""
        self.assertEqual(self.listener.callCount(), 1)
        self.listener.clear()

        roi1 = CurvesROIWidget.ROI(name='linear', fromdata=0, todata=5)
        self.curves_roi_widget.roiTable.addRoi(roi1)
        self.assertEqual(self.listener.callCount(), 1)
        self.assertTrue(self.listener.arguments()[0][0]['current'] == 'linear')
        self.listener.clear()

        roi2 = CurvesROIWidget.ROI(name='linear2', fromdata=0, todata=5)
        self.curves_roi_widget.roiTable.addRoi(roi2)
        self.assertEqual(self.listener.callCount(), 1)
        self.assertTrue(self.listener.arguments()[0][0]['current'] == 'linear2')
        self.listener.clear()

        self.curves_roi_widget.roiTable.removeROI(roi2)
        self.assertEqual(self.listener.callCount(), 1)
        self.assertTrue(self.curves_roi_widget.roiTable.activeRoi == roi1)
        self.assertTrue(self.listener.arguments()[0][0]['current'] == 'linear')
        self.listener.clear()

        self.curves_roi_widget.roiTable.deleteActiveRoi()
        self.assertEqual(self.listener.callCount(), 1)
        self.assertTrue(self.curves_roi_widget.roiTable.activeRoi is None)
        self.assertTrue(self.listener.arguments()[0][0]['current'] is None)
        self.listener.clear()

        self.curves_roi_widget.roiTable.addRoi(roi1)
        self.assertEqual(self.listener.callCount(), 1)
        self.assertTrue(self.listener.arguments()[0][0]['current'] == 'linear')
        self.assertTrue(self.curves_roi_widget.roiTable.activeRoi == roi1)
        self.listener.clear()
        self.qapp.processEvents()

        self.curves_roi_widget.roiTable.removeROI(roi1)
        self.qapp.processEvents()
        self.assertEqual(self.listener.callCount(), 1)
        self.assertTrue(self.listener.arguments()[0][0]['current'] == 'ICR')
        self.listener.clear()

    def testSigROISignalModifyROI(self):
        """Test SigROISignal when modifying it"""
        self.curves_roi_widget.roiTable.setMiddleROIMarkerFlag(True)
        roi1 = CurvesROIWidget.ROI(name='linear', fromdata=2, todata=5)
        self.curves_roi_widget.roiTable.addRoi(roi1)
        self.curves_roi_widget.roiTable.setActiveRoi(roi1)

        # test modify the roi2 object
        self.listener.clear()
        roi1.setFrom(0.56)
        self.assertEqual(self.listener.callCount(), 1)
        self.listener.clear()
        roi1.setTo(2.56)
        self.assertEqual(self.listener.callCount(), 1)
        self.listener.clear()
        roi1.setName('linear2')
        self.assertEqual(self.listener.callCount(), 1)
        self.listener.clear()
        roi1.setType('new type')
        self.assertEqual(self.listener.callCount(), 1)

        # modify roi limits (from the gui)
        roi_marker_handler = self.curves_roi_widget.roiTable._markersHandler.getMarkerHandler(roi1.getID())
        for marker_type in ('min', 'max', 'middle'):
            with self.subTest(marker_type=marker_type):
                self.listener.clear()
                marker = roi_marker_handler.getMarker(marker_type)
                self.qapp.processEvents()
                items_interaction = ItemsInteraction(plot=self.plot)
                x_pix, y_pix = self.plot.dataToPixel(marker.getXPosition(), 1)
                items_interaction.beginDrag(x_pix, y_pix)
                self.qapp.processEvents()
                items_interaction.endDrag(x_pix+10, y_pix)
                self.qapp.processEvents()
                self.assertEqual(self.listener.callCount(), 1)

    def testSetActiveCurve(self):
        """Test sigRoiSignal when set an active curve"""
        roi1 = CurvesROIWidget.ROI(name='linear', fromdata=2, todata=5)
        self.curves_roi_widget.roiTable.addRoi(roi1)
        self.curves_roi_widget.roiTable.setActiveRoi(roi1)
        self.listener.clear()
        self.plot.setActiveCurve('curve0')
        self.assertEqual(self.listener.callCount(), 0)


def suite():
    test_suite = unittest.TestSuite()
    for TestClass in (TestCurvesROIWidget,):
        test_suite.addTest(
            unittest.defaultTestLoader.loadTestsFromTestCase(TestClass))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
