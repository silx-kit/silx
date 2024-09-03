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
from silx.gui.plot.items.curve import CurveStyle


@pytest.mark.parametrize("plotWidget", ("mpl", "gl"), indirect=True)
def testActiveCurveAndLabels(plotWidget):
    # Active curve handling off, no label change
    plotWidget.setActiveCurveHandling(False)
    plotWidget.getXAxis().setLabel("XLabel")
    plotWidget.getYAxis().setLabel("YLabel")
    plotWidget.addCurve((1, 2), (1, 2))
    assert plotWidget.getXAxis().getLabel() == "XLabel"
    assert plotWidget.getYAxis().getLabel() == "YLabel"

    plotWidget.addCurve((1, 2), (2, 3), xlabel="x1", ylabel="y1")
    assert plotWidget.getXAxis().getLabel() == "XLabel"
    assert plotWidget.getYAxis().getLabel() == "YLabel"

    plotWidget.clear()
    assert plotWidget.getXAxis().getLabel() == "XLabel"
    assert plotWidget.getYAxis().getLabel() == "YLabel"

    # Active curve handling on, label changes
    plotWidget.setActiveCurveHandling(True)
    plotWidget.getXAxis().setLabel("XLabel")
    plotWidget.getYAxis().setLabel("YLabel")

    # labels changed as active curve
    plotWidget.addCurve((1, 2), (1, 2), legend="1", xlabel="x1", ylabel="y1")
    plotWidget.setActiveCurve("1")
    assert plotWidget.getXAxis().getLabel() == "x1"
    assert plotWidget.getYAxis().getLabel() == "y1"

    # labels not changed as not active curve
    plotWidget.addCurve((1, 2), (2, 3), legend="2")
    assert plotWidget.getXAxis().getLabel() == "x1"
    assert plotWidget.getYAxis().getLabel() == "y1"

    # labels changed
    plotWidget.setActiveCurve("2")
    assert plotWidget.getXAxis().getLabel() == "XLabel"
    assert plotWidget.getYAxis().getLabel() == "YLabel"

    plotWidget.setActiveCurve("1")
    assert plotWidget.getXAxis().getLabel() == "x1"
    assert plotWidget.getYAxis().getLabel() == "y1"

    plotWidget.clear()
    assert plotWidget.getXAxis().getLabel() == "XLabel"
    assert plotWidget.getYAxis().getLabel() == "YLabel"

    plotWidget.setActiveCurveHandling(False)


@pytest.mark.parametrize("plotWidget", ("mpl", "gl"), indirect=True)
def testPlotActiveCurveSelectionMode(plotWidget):
    xData = numpy.arange(1000)
    yData = -500 + 100 * numpy.sin(xData)
    xData2 = xData + 1000
    yData2 = xData - 1000 + 200 * numpy.random.random(1000)

    plotWidget.clear()
    plotWidget.setActiveCurveHandling(True)
    legend = "curve 1"
    plotWidget.addCurve(xData, yData, legend=legend, color="green")

    # active curve should be None
    assert plotWidget.getActiveCurve(just_legend=True) is None

    # active curve should be None when None is set as active curve
    plotWidget.setActiveCurve(legend)
    current = plotWidget.getActiveCurve(just_legend=True)
    assert current == legend
    plotWidget.setActiveCurve(None)
    current = plotWidget.getActiveCurve(just_legend=True)
    assert current is None

    # testing it automatically toggles if there is only one
    plotWidget.setActiveCurveSelectionMode("legacy")
    current = plotWidget.getActiveCurve(just_legend=True)
    assert current == legend

    # active curve should not change when None set as active curve
    assert plotWidget.getActiveCurveSelectionMode() == "legacy"
    plotWidget.setActiveCurve(None)
    current = plotWidget.getActiveCurve(just_legend=True)
    assert current == legend

    # situation where no curve is active
    plotWidget.clear()
    plotWidget.setActiveCurveHandling(True)
    assert plotWidget.getActiveCurveSelectionMode() == "atmostone"
    plotWidget.addCurve(xData, yData, legend=legend, color="green")
    assert plotWidget.getActiveCurve(just_legend=True) is None
    plotWidget.addCurve(xData2, yData2, legend="curve 2", color="red")
    assert plotWidget.getActiveCurve(just_legend=True) is None
    plotWidget.setActiveCurveSelectionMode("legacy")
    assert plotWidget.getActiveCurve(just_legend=True) is None

    # the first curve added should be active
    plotWidget.clear()
    plotWidget.addCurve(xData, yData, legend=legend, color="green")
    assert plotWidget.getActiveCurve(just_legend=True) == legend
    plotWidget.addCurve(xData2, yData2, legend="curve 2", color="red")
    assert plotWidget.getActiveCurve(just_legend=True) == legend

    plotWidget.setActiveCurveHandling(False)


@pytest.mark.parametrize("plotWidget", ("mpl", "gl"), indirect=True)
def testActiveCurveStyle(plotWidget):
    """Test change of active curve style"""
    plotWidget.setActiveCurveHandling(True)
    plotWidget.setActiveCurveStyle(color="black")
    style = plotWidget.getActiveCurveStyle()
    assert style.getColor() == (0.0, 0.0, 0.0, 1.0)
    assert style.getLineStyle() is None
    assert style.getLineWidth() is None
    assert style.getSymbol() is None
    assert style.getSymbolSize() is None

    xData = numpy.arange(1000)
    yData = -500 + 100 * numpy.sin(xData)
    plotWidget.addCurve(x=xData, y=yData, legend="curve1")
    curve = plotWidget.getCurve("curve1")
    curve.setColor("blue")
    curve.setLineStyle("-")
    curve.setLineWidth(1)
    curve.setSymbol("o")
    curve.setSymbolSize(5)

    # Check default current style
    defaultStyle = curve.getCurrentStyle()
    assert defaultStyle == CurveStyle(
        color="blue", linestyle="-", linewidth=1, symbol="o", symbolsize=5
    )

    # Activate curve with highlight color=black
    plotWidget.setActiveCurve("curve1")
    style = curve.getCurrentStyle()
    assert style.getColor() == (0.0, 0.0, 0.0, 1.0)
    assert style.getLineStyle() == "-"
    assert style.getLineWidth() == 1
    assert style.getSymbol() == "o"
    assert style.getSymbolSize() == 5

    # Change highlight to linewidth=2
    plotWidget.setActiveCurveStyle(linewidth=2)
    style = curve.getCurrentStyle()
    assert style.getColor() == (0.0, 0.0, 1.0, 1.0)
    assert style.getLineStyle() == "-"
    assert style.getLineWidth() == 2
    assert style.getSymbol() == "o"
    assert style.getSymbolSize() == 5

    plotWidget.setActiveCurve(None)
    assert curve.getCurrentStyle() == defaultStyle

    plotWidget.setActiveCurveHandling(False)


@pytest.mark.parametrize("plotWidget", ("mpl", "gl"), indirect=True)
def testActiveImageAndLabels(plotWidget):
    # Active image handling always on, no API for toggling it
    plotWidget.getXAxis().setLabel("XLabel")
    plotWidget.getYAxis().setLabel("YLabel")

    # labels changed as active curve
    plotWidget.addImage(
        numpy.arange(100).reshape(10, 10), legend="1", xlabel="x1", ylabel="y1"
    )
    assert plotWidget.getXAxis().getLabel() == "x1"
    assert plotWidget.getYAxis().getLabel() == "y1"

    # labels not changed as not active curve
    plotWidget.addImage(numpy.arange(100).reshape(10, 10), legend="2")
    assert plotWidget.getXAxis().getLabel() == "x1"
    assert plotWidget.getYAxis().getLabel() == "y1"

    # labels changed
    plotWidget.setActiveImage("2")
    assert plotWidget.getXAxis().getLabel() == "XLabel"
    assert plotWidget.getYAxis().getLabel() == "YLabel"

    plotWidget.setActiveImage("1")
    assert plotWidget.getXAxis().getLabel() == "x1"
    assert plotWidget.getYAxis().getLabel() == "y1"

    plotWidget.clear()
    assert plotWidget.getXAxis().getLabel() == "XLabel"
    assert plotWidget.getYAxis().getLabel() == "YLabel"

    plotWidget.setActiveCurveHandling(False)


def _checkSelection(selection, current=None, selected=()):
    """Check current item and selected items."""
    assert selection.getCurrentItem() is current
    assert selection.getSelectedItems() == selected


@pytest.mark.parametrize("plotWidget", ("mpl", "gl"), indirect=True)
def testSelectionSyncWithActiveItems(plotWidget):
    """Test update of PlotWidgetSelection according to active items"""
    listener = SignalListener()

    selection = plotWidget.selection()
    selection.sigCurrentItemChanged.connect(listener)
    _checkSelection(selection)

    # Active item is current
    plotWidget.addImage(((0, 1), (2, 3)), legend="image")
    image = plotWidget.getActiveImage()
    assert listener.callCount() == 1
    _checkSelection(selection, image, (image,))

    # No active = no current
    plotWidget.setActiveImage(None)
    assert listener.callCount() == 2
    _checkSelection(selection)

    # Active item is current
    plotWidget.setActiveImage("image")
    assert listener.callCount() == 3
    _checkSelection(selection, image, (image,))

    # Mosted recently "actived" item is current
    plotWidget.addScatter((3, 2, 1), (0, 1, 2), (0, 1, 2), legend="scatter")
    scatter = plotWidget.getActiveScatter()
    assert listener.callCount() == 4
    _checkSelection(selection, scatter, (scatter, image))

    # Previously mosted recently "actived" item is current
    plotWidget.setActiveScatter(None)
    assert listener.callCount() == 5
    _checkSelection(selection, image, (image,))

    # Mosted recently "actived" item is current
    plotWidget.setActiveScatter("scatter")
    assert listener.callCount() == 6
    _checkSelection(selection, scatter, (scatter, image))

    # No active = no current
    plotWidget.setActiveImage(None)
    plotWidget.setActiveScatter(None)
    assert listener.callCount() == 7
    _checkSelection(selection)

    # Mosted recently "actived" item is current
    plotWidget.setActiveScatter("scatter")
    assert listener.callCount() == 8
    plotWidget.setActiveImage("image")
    assert listener.callCount() == 9
    _checkSelection(selection, image, (image, scatter))

    # Add a curve which is not active by default
    plotWidget.addCurve((0, 1, 2), (0, 1, 2), legend="curve")
    curve = plotWidget.getCurve("curve")
    assert listener.callCount() == 9
    _checkSelection(selection, image, (image, scatter))

    # Mosted recently "actived" item is current
    plotWidget.setActiveCurve("curve")
    assert listener.callCount() == 10
    _checkSelection(selection, curve, (curve, image, scatter))

    # Add a curve which is not active by default
    plotWidget.addCurve((0, 1, 2), (0, 1, 2), legend="curve2")
    curve2 = plotWidget.getCurve("curve2")
    assert listener.callCount() == 10
    _checkSelection(selection, curve, (curve, image, scatter))

    # Mosted recently "actived" item is current, previous curve is removed
    plotWidget.setActiveCurve("curve2")
    assert listener.callCount() == 11
    _checkSelection(selection, curve2, (curve2, image, scatter))

    # No items = no current
    plotWidget.clear()
    assert listener.callCount() == 12
    _checkSelection(selection)


@pytest.mark.parametrize("plotWidget", ("mpl", "gl"), indirect=True)
def testSelectionWithItems(plotWidget):
    """Test init of selection on a plot with items"""
    plotWidget.addImage(((0, 1), (2, 3)), legend="image")
    plotWidget.addScatter((3, 2, 1), (0, 1, 2), (0, 1, 2), legend="scatter")
    plotWidget.addCurve((0, 1, 2), (0, 1, 2), legend="curve")
    plotWidget.setActiveCurve("curve")

    selection = plotWidget.selection()
    assert selection.getCurrentItem() is not None
    selected = selection.getSelectedItems()
    assert len(selected) == 3
    assert plotWidget.getActiveCurve() in selected
    assert plotWidget.getActiveImage() in selected
    assert plotWidget.getActiveScatter() in selected


@pytest.mark.parametrize("plotWidget", ("mpl", "gl"), indirect=True)
def testSelectionSetCurrentItem(plotWidget):
    """Test setCurrentItem"""
    # Add items to the plot
    plotWidget.addImage(((0, 1), (2, 3)), legend="image")
    image = plotWidget.getActiveImage()
    plotWidget.addScatter((3, 2, 1), (0, 1, 2), (0, 1, 2), legend="scatter")
    scatter = plotWidget.getActiveScatter()
    plotWidget.addCurve((0, 1, 2), (0, 1, 2), legend="curve")
    plotWidget.setActiveCurve("curve")
    curve = plotWidget.getActiveCurve()

    selection = plotWidget.selection()
    assert selection.getCurrentItem() is not None
    assert len(selection.getSelectedItems()) == 3

    # Set current to None reset all active items
    selection.setCurrentItem(None)
    _checkSelection(selection)
    assert plotWidget.getActiveCurve() is None
    assert plotWidget.getActiveImage() is None
    assert plotWidget.getActiveScatter() is None

    # Set current to an item makes it active
    selection.setCurrentItem(image)
    _checkSelection(selection, image, (image,))
    assert plotWidget.getActiveCurve() is None
    assert plotWidget.getActiveImage() is image
    assert plotWidget.getActiveScatter() is None

    # Set current to an item makes it active and keeps other active
    selection.setCurrentItem(curve)
    _checkSelection(selection, curve, (curve, image))
    assert plotWidget.getActiveCurve() is curve
    assert plotWidget.getActiveImage() is image
    assert plotWidget.getActiveScatter() is None

    # Set current to an item makes it active and keeps other active
    selection.setCurrentItem(scatter)
    _checkSelection(selection, scatter, (scatter, curve, image))
    assert plotWidget.getActiveCurve() is curve
    assert plotWidget.getActiveImage() is image
    assert plotWidget.getActiveScatter() is scatter


def testSetActiveCurveWithInstance(plotWidget):
    """Test setting the active curve with a curve item instance"""
    plotWidget.addCurve((0, 1), (0, 1), legend="curve0")
    plotWidget.addCurve((0, 1), (1, 0), legend="curve1")
    curve0, curve1 = plotWidget.getItems()

    plotWidget.setActiveCurve(curve0)
    assert plotWidget.getActiveCurve() is curve0

    plotWidget.setActiveCurve(curve1)
    assert plotWidget.getActiveCurve() is curve1

    plotWidget.setActiveCurve(None)
    assert plotWidget.getActiveCurve() is None


def testSetActiveImageWithInstance(plotWidget):
    """Test setting the active image with an image item instance"""
    plotWidget.addImage(((0, 1), (2, 3)), legend="image")
    image = plotWidget.getItems()[0]

    plotWidget.setActiveImage(None)
    assert plotWidget.getActiveImage() is None

    plotWidget.setActiveImage(image)
    assert plotWidget.getActiveImage() is image


def testSetActiveScatterWithInstance(plotWidget):
    """Test setting the active scatter with a scatter item instance"""
    plotWidget.addScatter((0, 1), (0, 1), (0, 1), legend="scatter")
    scatter = plotWidget.getItems()[0]

    plotWidget.setActiveScatter(None)
    assert plotWidget.getActiveScatter() is None

    plotWidget.setActiveScatter(scatter)
    assert plotWidget.getActiveScatter() is scatter
