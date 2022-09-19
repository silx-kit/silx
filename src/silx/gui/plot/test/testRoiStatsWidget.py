# /*##########################################################################
#
# Copyright (c) 2016-2019 European Synchrotron Radiation Facility
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
"""Tests for ROIStatsWidget"""


from silx.gui.utils.testutils import TestCaseQt
from silx.gui import qt
from silx.gui.plot import PlotWindow
from silx.gui.plot.stats.stats import Stats
from silx.gui.plot.ROIStatsWidget import ROIStatsWidget
from silx.gui.plot.CurvesROIWidget import ROI
from silx.gui.plot.items.roi import RectangleROI, PolygonROI
from silx.gui.plot.StatsWidget import UpdateMode
import unittest
import numpy



class _TestRoiStatsBase(TestCaseQt):
    """Base class for several unittest relative to ROIStatsWidget"""
    def setUp(self):
        TestCaseQt.setUp(self)
        # define plot
        self.plot = PlotWindow()
        self.plot.addImage(numpy.arange(10000).reshape(100, 100),
                           legend='img1')
        self.img_item = self.plot.getImage('img1')
        self.plot.addCurve(x=numpy.linspace(0, 10, 56), y=numpy.arange(56),
                           legend='curve1')
        self.curve_item = self.plot.getCurve('curve1')
        self.plot.addHistogram(edges=numpy.linspace(0, 10, 56),
                               histogram=numpy.arange(56), legend='histo1')
        self.histogram_item = self.plot.getHistogram(legend='histo1')
        self.plot.addScatter(x=numpy.linspace(0, 10, 56),
                             y=numpy.linspace(0, 10, 56),
                             value=numpy.arange(56),
                             legend='scatter1')
        self.scatter_item = self.plot.getScatter(legend='scatter1')

        # stats widget
        self.statsWidget = ROIStatsWidget(plot=self.plot)

        # define stats
        stats = [
            ('sum', numpy.sum),
            ('mean', numpy.mean),
        ]
        self.statsWidget.setStats(stats=stats)

        # define rois
        self.roi1D = ROI(name='range1', fromdata=0, todata=4, type_='energy')
        self.rectangle_roi = RectangleROI()
        self.rectangle_roi.setGeometry(origin=(0, 0), size=(20, 20))
        self.rectangle_roi.setName('Initial ROI')
        self.polygon_roi = PolygonROI()
        points = numpy.array([[0, 5], [5, 0], [10, 5], [5, 10]])
        self.polygon_roi.setPoints(points)

    def statsTable(self):
        return self.statsWidget._statsROITable

    def tearDown(self):
        Stats._getContext.cache_clear()
        self.statsWidget.setAttribute(qt.Qt.WA_DeleteOnClose, True)
        self.statsWidget.close()
        self.statsWidget = None
        self.plot.setAttribute(qt.Qt.WA_DeleteOnClose, True)
        self.plot.close()
        self.plot = None
        TestCaseQt.tearDown(self)


class TestRoiStatsCouple(_TestRoiStatsBase):
    """
    Test different possible couple (roi, plotItem).
    Check that:
    
    * computation is correct if couple is valid
    * raise an error if couple is invalid
    """
    def testROICurve(self):
        """
        Test that the couple (ROI, curveItem) can be used for stats       
        """
        item = self.statsWidget.addItem(roi=self.roi1D,
                                        plotItem=self.curve_item)
        assert item is not None
        tableItems = self.statsTable()._itemToTableItems(item)
        self.assertEqual(tableItems['sum'].text(), '253')
        self.assertEqual(tableItems['mean'].text(), '11.0')

    def testRectangleImage(self):
        """
        Test that the couple (RectangleROI, imageItem) can be used for stats       
        """
        item = self.statsWidget.addItem(roi=self.rectangle_roi,
                                        plotItem=self.img_item)
        assert item is not None
        self.plot.addImage(numpy.ones(10000).reshape(100, 100),
                           legend='img1')
        self.qapp.processEvents()
        tableItems = self.statsTable()._itemToTableItems(item)
        self.assertEqual(tableItems['sum'].text(), str(float(21*21)))
        self.assertEqual(tableItems['mean'].text(), '1.0')

    def testPolygonImage(self):
        """
        Test that the couple (PolygonROI, imageItem) can be used for stats       
        """
        item = self.statsWidget.addItem(roi=self.polygon_roi,
                                        plotItem=self.img_item)
        assert item is not None
        tableItems = self.statsTable()._itemToTableItems(item)
        self.assertEqual(tableItems['sum'].text(), '22750')
        self.assertEqual(tableItems['mean'].text(), '455.0')

    def testROIImage(self):
        """
        Test that the couple (ROI, imageItem) is raising an error       
        """
        with self.assertRaises(TypeError):
            self.statsWidget.addItem(roi=self.roi1D,
                                     plotItem=self.img_item)

    def testRectangleCurve(self):
        """
        Test that the couple (rectangleROI, curveItem) is raising an error       
        """
        with self.assertRaises(TypeError):
            self.statsWidget.addItem(roi=self.rectangle_roi,
                                     plotItem=self.curve_item)

    def testROIHistogram(self):
        """
        Test that the couple (PolygonROI, imageItem) can be used for stats       
        """
        item = self.statsWidget.addItem(roi=self.roi1D,
                                        plotItem=self.histogram_item)
        assert item is not None
        tableItems = self.statsTable()._itemToTableItems(item)
        self.assertEqual(tableItems['sum'].text(), '253')
        self.assertEqual(tableItems['mean'].text(), '11.0')

    def testROIScatter(self):
        """
        Test that the couple (PolygonROI, imageItem) can be used for stats       
        """
        item = self.statsWidget.addItem(roi=self.roi1D,
                                        plotItem=self.scatter_item)
        assert item is not None
        tableItems = self.statsTable()._itemToTableItems(item)
        self.assertEqual(tableItems['sum'].text(), '253')
        self.assertEqual(tableItems['mean'].text(), '11.0')


class TestRoiStatsAddRemoveItem(_TestRoiStatsBase):
    """Test adding and removing (roi, plotItem) items"""
    def testAddRemoveItems(self):
        item1 = self.statsWidget.addItem(roi=self.roi1D,
                                         plotItem=self.scatter_item)
        self.assertTrue(item1 is not None)
        self.assertEqual(self.statsTable().rowCount(), 1)
        item2 = self.statsWidget.addItem(roi=self.roi1D,
                                         plotItem=self.histogram_item)
        self.assertTrue(item2 is not None)
        self.assertEqual(self.statsTable().rowCount(), 2)
        # try to add twice the same item
        item3 = self.statsWidget.addItem(roi=self.roi1D,
                                         plotItem=self.histogram_item)
        self.assertTrue(item3 is None)
        self.assertEqual(self.statsTable().rowCount(), 2)
        item4 = self.statsWidget.addItem(roi=self.roi1D,
                                         plotItem=self.curve_item)
        self.assertTrue(item4 is not None)
        self.assertEqual(self.statsTable().rowCount(), 3)

        self.statsWidget.removeItem(plotItem=item4._plot_item,
                                    roi=item4._roi)
        self.assertEqual(self.statsTable().rowCount(), 2)
        # try to remove twice the same item
        self.statsWidget.removeItem(plotItem=item4._plot_item,
                                    roi=item4._roi)
        self.assertEqual(self.statsTable().rowCount(), 2)
        self.statsWidget.removeItem(plotItem=item2._plot_item,
                                    roi=item2._roi)
        self.statsWidget.removeItem(plotItem=item1._plot_item,
                                    roi=item1._roi)
        self.assertEqual(self.statsTable().rowCount(), 0)


class TestRoiStatsRoiUpdate(_TestRoiStatsBase):
    """Test that the stats will be updated if the roi is updated"""
    def testChangeRoi(self):
        item = self.statsWidget.addItem(roi=self.rectangle_roi,
                                        plotItem=self.img_item)
        assert item is not None
        tableItems = self.statsTable()._itemToTableItems(item)
        self.assertEqual(tableItems['sum'].text(), '445410')
        self.assertEqual(tableItems['mean'].text(), '1010.0')

        # update roi
        self.rectangle_roi.setOrigin(position=(10, 10))
        self.assertNotEqual(tableItems['sum'].text(), '445410')
        self.assertNotEqual(tableItems['mean'].text(), '1010.0')

    def testUpdateModeScenario(self):
        """Test update according to a simple scenario"""
        self.statsWidget._setUpdateMode(UpdateMode.AUTO)
        item = self.statsWidget.addItem(roi=self.rectangle_roi,
                                        plotItem=self.img_item)

        assert item is not None
        tableItems = self.statsTable()._itemToTableItems(item)
        self.assertEqual(tableItems['sum'].text(), '445410')
        self.assertEqual(tableItems['mean'].text(), '1010.0')
        self.statsWidget._setUpdateMode(UpdateMode.MANUAL)
        self.rectangle_roi.setOrigin(position=(10, 10))
        self.qapp.processEvents()
        self.assertNotEqual(tableItems['sum'].text(), '445410')
        self.assertNotEqual(tableItems['mean'].text(), '1010.0')
        self.statsWidget._updateAllStats(is_request=True)
        self.assertNotEqual(tableItems['sum'].text(), '445410')
        self.assertNotEqual(tableItems['mean'].text(), '1010.0')


class TestRoiStatsPlotItemUpdate(_TestRoiStatsBase):
    """Test that the stats will be updated if the plot item is updated"""
    def testChangeImage(self):
        self.statsWidget._setUpdateMode(UpdateMode.AUTO)
        item = self.statsWidget.addItem(roi=self.rectangle_roi,
                                        plotItem=self.img_item)

        assert item is not None
        tableItems = self.statsTable()._itemToTableItems(item)
        self.assertEqual(tableItems['mean'].text(), '1010.0')

        # update plot
        self.plot.addImage(numpy.arange(100, 10100).reshape(100, 100),
                           legend='img1')
        self.assertNotEqual(tableItems['mean'].text(), '1059.5')

    def testUpdateModeScenario(self):
        """Test update according to a simple scenario"""
        self.statsWidget._setUpdateMode(UpdateMode.MANUAL)
        item = self.statsWidget.addItem(roi=self.rectangle_roi,
                                        plotItem=self.img_item)

        assert item is not None
        tableItems = self.statsTable()._itemToTableItems(item)
        self.assertEqual(tableItems['mean'].text(), '1010.0')
        self.plot.addImage(numpy.arange(100, 10100).reshape(100, 100),
                           legend='img1')
        self.assertEqual(tableItems['mean'].text(), '1010.0')
        self.statsWidget._updateAllStats(is_request=True)
        self.assertEqual(tableItems['mean'].text(), '1110.0')
