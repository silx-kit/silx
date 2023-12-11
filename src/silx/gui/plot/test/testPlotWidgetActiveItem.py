# /*##########################################################################
#
# Copyright (c) 2023 European Synchrotron Radiation Facility
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
"""Test PlotWidget active item"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "11/12/2023"


import numpy
import pytest

from silx.gui.utils.testutils import SignalListener

from silx.gui import qt
from silx.gui.plot.items.curve import CurveStyle

from .utils import PlotWidgetTestCase


class TestPlotActiveCurveImage(PlotWidgetTestCase):
    """Basic tests for active curve and image handling"""

    xData = numpy.arange(1000)
    yData = -500 + 100 * numpy.sin(xData)
    xData2 = xData + 1000
    yData2 = xData - 1000 + 200 * numpy.random.random(1000)

    def tearDown(self):
        self.plot.setActiveCurveHandling(False)
        super(TestPlotActiveCurveImage, self).tearDown()

    def testActiveCurveAndLabels(self):
        # Active curve handling off, no label change
        self.plot.setActiveCurveHandling(False)
        self.plot.getXAxis().setLabel("XLabel")
        self.plot.getYAxis().setLabel("YLabel")
        self.plot.addCurve((1, 2), (1, 2))
        self.assertEqual(self.plot.getXAxis().getLabel(), "XLabel")
        self.assertEqual(self.plot.getYAxis().getLabel(), "YLabel")

        self.plot.addCurve((1, 2), (2, 3), xlabel="x1", ylabel="y1")
        self.assertEqual(self.plot.getXAxis().getLabel(), "XLabel")
        self.assertEqual(self.plot.getYAxis().getLabel(), "YLabel")

        self.plot.clear()
        self.assertEqual(self.plot.getXAxis().getLabel(), "XLabel")
        self.assertEqual(self.plot.getYAxis().getLabel(), "YLabel")

        # Active curve handling on, label changes
        self.plot.setActiveCurveHandling(True)
        self.plot.getXAxis().setLabel("XLabel")
        self.plot.getYAxis().setLabel("YLabel")

        # labels changed as active curve
        self.plot.addCurve((1, 2), (1, 2), legend="1", xlabel="x1", ylabel="y1")
        self.plot.setActiveCurve("1")
        self.assertEqual(self.plot.getXAxis().getLabel(), "x1")
        self.assertEqual(self.plot.getYAxis().getLabel(), "y1")

        # labels not changed as not active curve
        self.plot.addCurve((1, 2), (2, 3), legend="2")
        self.assertEqual(self.plot.getXAxis().getLabel(), "x1")
        self.assertEqual(self.plot.getYAxis().getLabel(), "y1")

        # labels changed
        self.plot.setActiveCurve("2")
        self.assertEqual(self.plot.getXAxis().getLabel(), "XLabel")
        self.assertEqual(self.plot.getYAxis().getLabel(), "YLabel")

        self.plot.setActiveCurve("1")
        self.assertEqual(self.plot.getXAxis().getLabel(), "x1")
        self.assertEqual(self.plot.getYAxis().getLabel(), "y1")

        self.plot.clear()
        self.assertEqual(self.plot.getXAxis().getLabel(), "XLabel")
        self.assertEqual(self.plot.getYAxis().getLabel(), "YLabel")

    def testPlotActiveCurveSelectionMode(self):
        self.plot.clear()
        self.plot.setActiveCurveHandling(True)
        legend = "curve 1"
        self.plot.addCurve(self.xData, self.yData, legend=legend, color="green")

        # active curve should be None
        self.assertEqual(self.plot.getActiveCurve(just_legend=True), None)

        # active curve should be None when None is set as active curve
        self.plot.setActiveCurve(legend)
        current = self.plot.getActiveCurve(just_legend=True)
        self.assertEqual(current, legend)
        self.plot.setActiveCurve(None)
        current = self.plot.getActiveCurve(just_legend=True)
        self.assertEqual(current, None)

        # testing it automatically toggles if there is only one
        self.plot.setActiveCurveSelectionMode("legacy")
        current = self.plot.getActiveCurve(just_legend=True)
        self.assertEqual(current, legend)

        # active curve should not change when None set as active curve
        self.assertEqual(self.plot.getActiveCurveSelectionMode(), "legacy")
        self.plot.setActiveCurve(None)
        current = self.plot.getActiveCurve(just_legend=True)
        self.assertEqual(current, legend)

        # situation where no curve is active
        self.plot.clear()
        self.plot.setActiveCurveHandling(True)
        self.assertEqual(self.plot.getActiveCurveSelectionMode(), "atmostone")
        self.plot.addCurve(self.xData, self.yData, legend=legend, color="green")
        self.assertEqual(self.plot.getActiveCurve(just_legend=True), None)
        self.plot.addCurve(self.xData2, self.yData2, legend="curve 2", color="red")
        self.assertEqual(self.plot.getActiveCurve(just_legend=True), None)
        self.plot.setActiveCurveSelectionMode("legacy")
        self.assertEqual(self.plot.getActiveCurve(just_legend=True), None)

        # the first curve added should be active
        self.plot.clear()
        self.plot.addCurve(self.xData, self.yData, legend=legend, color="green")
        self.assertEqual(self.plot.getActiveCurve(just_legend=True), legend)
        self.plot.addCurve(self.xData2, self.yData2, legend="curve 2", color="red")
        self.assertEqual(self.plot.getActiveCurve(just_legend=True), legend)

    def testActiveCurveStyle(self):
        """Test change of active curve style"""
        self.plot.setActiveCurveHandling(True)
        self.plot.setActiveCurveStyle(color="black")
        style = self.plot.getActiveCurveStyle()
        self.assertEqual(style.getColor(), (0.0, 0.0, 0.0, 1.0))
        self.assertIsNone(style.getLineStyle())
        self.assertIsNone(style.getLineWidth())
        self.assertIsNone(style.getSymbol())
        self.assertIsNone(style.getSymbolSize())

        self.plot.addCurve(x=self.xData, y=self.yData, legend="curve1")
        curve = self.plot.getCurve("curve1")
        curve.setColor("blue")
        curve.setLineStyle("-")
        curve.setLineWidth(1)
        curve.setSymbol("o")
        curve.setSymbolSize(5)

        # Check default current style
        defaultStyle = curve.getCurrentStyle()
        self.assertEqual(
            defaultStyle,
            CurveStyle(
                color="blue", linestyle="-", linewidth=1, symbol="o", symbolsize=5
            ),
        )

        # Activate curve with highlight color=black
        self.plot.setActiveCurve("curve1")
        style = curve.getCurrentStyle()
        self.assertEqual(style.getColor(), (0.0, 0.0, 0.0, 1.0))
        self.assertEqual(style.getLineStyle(), "-")
        self.assertEqual(style.getLineWidth(), 1)
        self.assertEqual(style.getSymbol(), "o")
        self.assertEqual(style.getSymbolSize(), 5)

        # Change highlight to linewidth=2
        self.plot.setActiveCurveStyle(linewidth=2)
        style = curve.getCurrentStyle()
        self.assertEqual(style.getColor(), (0.0, 0.0, 1.0, 1.0))
        self.assertEqual(style.getLineStyle(), "-")
        self.assertEqual(style.getLineWidth(), 2)
        self.assertEqual(style.getSymbol(), "o")
        self.assertEqual(style.getSymbolSize(), 5)

        self.plot.setActiveCurve(None)
        self.assertEqual(curve.getCurrentStyle(), defaultStyle)

    def testActiveImageAndLabels(self):
        # Active image handling always on, no API for toggling it
        self.plot.getXAxis().setLabel("XLabel")
        self.plot.getYAxis().setLabel("YLabel")

        # labels changed as active curve
        self.plot.addImage(
            numpy.arange(100).reshape(10, 10), legend="1", xlabel="x1", ylabel="y1"
        )
        self.assertEqual(self.plot.getXAxis().getLabel(), "x1")
        self.assertEqual(self.plot.getYAxis().getLabel(), "y1")

        # labels not changed as not active curve
        self.plot.addImage(numpy.arange(100).reshape(10, 10), legend="2")
        self.assertEqual(self.plot.getXAxis().getLabel(), "x1")
        self.assertEqual(self.plot.getYAxis().getLabel(), "y1")

        # labels changed
        self.plot.setActiveImage("2")
        self.assertEqual(self.plot.getXAxis().getLabel(), "XLabel")
        self.assertEqual(self.plot.getYAxis().getLabel(), "YLabel")

        self.plot.setActiveImage("1")
        self.assertEqual(self.plot.getXAxis().getLabel(), "x1")
        self.assertEqual(self.plot.getYAxis().getLabel(), "y1")

        self.plot.clear()
        self.assertEqual(self.plot.getXAxis().getLabel(), "XLabel")
        self.assertEqual(self.plot.getYAxis().getLabel(), "YLabel")


class TestPlotWidgetSelection(PlotWidgetTestCase):
    """Test PlotWidget.selection and active items handling"""

    def _checkSelection(self, selection, current=None, selected=()):
        """Check current item and selected items."""
        self.assertIs(selection.getCurrentItem(), current)
        self.assertEqual(selection.getSelectedItems(), selected)

    def testSyncWithActiveItems(self):
        """Test update of PlotWidgetSelection according to active items"""
        listener = SignalListener()

        selection = self.plot.selection()
        selection.sigCurrentItemChanged.connect(listener)
        self._checkSelection(selection)

        # Active item is current
        self.plot.addImage(((0, 1), (2, 3)), legend="image")
        image = self.plot.getActiveImage()
        self.assertEqual(listener.callCount(), 1)
        self._checkSelection(selection, image, (image,))

        # No active = no current
        self.plot.setActiveImage(None)
        self.assertEqual(listener.callCount(), 2)
        self._checkSelection(selection)

        # Active item is current
        self.plot.setActiveImage("image")
        self.assertEqual(listener.callCount(), 3)
        self._checkSelection(selection, image, (image,))

        # Mosted recently "actived" item is current
        self.plot.addScatter((3, 2, 1), (0, 1, 2), (0, 1, 2), legend="scatter")
        scatter = self.plot.getActiveScatter()
        self.assertEqual(listener.callCount(), 4)
        self._checkSelection(selection, scatter, (scatter, image))

        # Previously mosted recently "actived" item is current
        self.plot.setActiveScatter(None)
        self.assertEqual(listener.callCount(), 5)
        self._checkSelection(selection, image, (image,))

        # Mosted recently "actived" item is current
        self.plot.setActiveScatter("scatter")
        self.assertEqual(listener.callCount(), 6)
        self._checkSelection(selection, scatter, (scatter, image))

        # No active = no current
        self.plot.setActiveImage(None)
        self.plot.setActiveScatter(None)
        self.assertEqual(listener.callCount(), 7)
        self._checkSelection(selection)

        # Mosted recently "actived" item is current
        self.plot.setActiveScatter("scatter")
        self.assertEqual(listener.callCount(), 8)
        self.plot.setActiveImage("image")
        self.assertEqual(listener.callCount(), 9)
        self._checkSelection(selection, image, (image, scatter))

        # Add a curve which is not active by default
        self.plot.addCurve((0, 1, 2), (0, 1, 2), legend="curve")
        curve = self.plot.getCurve("curve")
        self.assertEqual(listener.callCount(), 9)
        self._checkSelection(selection, image, (image, scatter))

        # Mosted recently "actived" item is current
        self.plot.setActiveCurve("curve")
        self.assertEqual(listener.callCount(), 10)
        self._checkSelection(selection, curve, (curve, image, scatter))

        # Add a curve which is not active by default
        self.plot.addCurve((0, 1, 2), (0, 1, 2), legend="curve2")
        curve2 = self.plot.getCurve("curve2")
        self.assertEqual(listener.callCount(), 10)
        self._checkSelection(selection, curve, (curve, image, scatter))

        # Mosted recently "actived" item is current, previous curve is removed
        self.plot.setActiveCurve("curve2")
        self.assertEqual(listener.callCount(), 11)
        self._checkSelection(selection, curve2, (curve2, image, scatter))

        # No items = no current
        self.plot.clear()
        self.assertEqual(listener.callCount(), 12)
        self._checkSelection(selection)

    def testPlotWidgetWithItems(self):
        """Test init of selection on a plot with items"""
        self.plot.addImage(((0, 1), (2, 3)), legend="image")
        self.plot.addScatter((3, 2, 1), (0, 1, 2), (0, 1, 2), legend="scatter")
        self.plot.addCurve((0, 1, 2), (0, 1, 2), legend="curve")
        self.plot.setActiveCurve("curve")

        selection = self.plot.selection()
        self.assertIsNotNone(selection.getCurrentItem())
        selected = selection.getSelectedItems()
        self.assertEqual(len(selected), 3)
        self.assertIn(self.plot.getActiveCurve(), selected)
        self.assertIn(self.plot.getActiveImage(), selected)
        self.assertIn(self.plot.getActiveScatter(), selected)

    def testSetCurrentItem(self):
        """Test setCurrentItem"""
        # Add items to the plot
        self.plot.addImage(((0, 1), (2, 3)), legend="image")
        image = self.plot.getActiveImage()
        self.plot.addScatter((3, 2, 1), (0, 1, 2), (0, 1, 2), legend="scatter")
        scatter = self.plot.getActiveScatter()
        self.plot.addCurve((0, 1, 2), (0, 1, 2), legend="curve")
        self.plot.setActiveCurve("curve")
        curve = self.plot.getActiveCurve()

        selection = self.plot.selection()
        self.assertIsNotNone(selection.getCurrentItem())
        self.assertEqual(len(selection.getSelectedItems()), 3)

        # Set current to None reset all active items
        selection.setCurrentItem(None)
        self._checkSelection(selection)
        self.assertIsNone(self.plot.getActiveCurve())
        self.assertIsNone(self.plot.getActiveImage())
        self.assertIsNone(self.plot.getActiveScatter())

        # Set current to an item makes it active
        selection.setCurrentItem(image)
        self._checkSelection(selection, image, (image,))
        self.assertIsNone(self.plot.getActiveCurve())
        self.assertIs(self.plot.getActiveImage(), image)
        self.assertIsNone(self.plot.getActiveScatter())

        # Set current to an item makes it active and keeps other active
        selection.setCurrentItem(curve)
        self._checkSelection(selection, curve, (curve, image))
        self.assertIs(self.plot.getActiveCurve(), curve)
        self.assertIs(self.plot.getActiveImage(), image)
        self.assertIsNone(self.plot.getActiveScatter())

        # Set current to an item makes it active and keeps other active
        selection.setCurrentItem(scatter)
        self._checkSelection(selection, scatter, (scatter, curve, image))
        self.assertIs(self.plot.getActiveCurve(), curve)
        self.assertIs(self.plot.getActiveImage(), image)
        self.assertIs(self.plot.getActiveScatter(), scatter)


@pytest.mark.usefixtures("use_opengl")
class TestPlotActiveCurveImage_Gl(TestPlotActiveCurveImage):
    backend = "gl"


@pytest.mark.usefixtures("use_opengl")
class TestPlotWidgetSelection_Gl(TestPlotWidgetSelection):
    backend = "gl"
