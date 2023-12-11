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
from silx.gui.plot import PlotWidget
from silx.gui.plot.items.curve import CurveStyle

from .utils import PlotWidgetTestCase


@pytest.mark.parametrize("backend", ("mpl", "gl"))
def testActiveCurveAndLabels(qWidgetFactory, backend, request):
    if backend == "gl":
        request.getfixturevalue("use_opengl")  # Skip test if OpenGL test disabled

    plot = qWidgetFactory(PlotWidget, backend=backend)

    # Active curve handling off, no label change
    plot.setActiveCurveHandling(False)
    plot.getXAxis().setLabel("XLabel")
    plot.getYAxis().setLabel("YLabel")
    plot.addCurve((1, 2), (1, 2))
    assert plot.getXAxis().getLabel() == "XLabel"
    assert plot.getYAxis().getLabel() == "YLabel"

    plot.addCurve((1, 2), (2, 3), xlabel="x1", ylabel="y1")
    assert plot.getXAxis().getLabel() == "XLabel"
    assert plot.getYAxis().getLabel() == "YLabel"

    plot.clear()
    assert plot.getXAxis().getLabel() == "XLabel"
    assert plot.getYAxis().getLabel() == "YLabel"

    # Active curve handling on, label changes
    plot.setActiveCurveHandling(True)
    plot.getXAxis().setLabel("XLabel")
    plot.getYAxis().setLabel("YLabel")

    # labels changed as active curve
    plot.addCurve((1, 2), (1, 2), legend="1", xlabel="x1", ylabel="y1")
    plot.setActiveCurve("1")
    assert plot.getXAxis().getLabel() == "x1"
    assert plot.getYAxis().getLabel() == "y1"

    # labels not changed as not active curve
    plot.addCurve((1, 2), (2, 3), legend="2")
    assert plot.getXAxis().getLabel() == "x1"
    assert plot.getYAxis().getLabel() == "y1"

    # labels changed
    plot.setActiveCurve("2")
    assert plot.getXAxis().getLabel() == "XLabel"
    assert plot.getYAxis().getLabel() == "YLabel"

    plot.setActiveCurve("1")
    assert plot.getXAxis().getLabel() == "x1"
    assert plot.getYAxis().getLabel() == "y1"

    plot.clear()
    assert plot.getXAxis().getLabel() == "XLabel"
    assert plot.getYAxis().getLabel() == "YLabel"

    plot.setActiveCurveHandling(False)


@pytest.mark.parametrize("backend", ("mpl", "gl"))
def testPlotActiveCurveSelectionMode(qWidgetFactory, backend, request):
    if backend == "gl":
        request.getfixturevalue("use_opengl")  # Skip test if OpenGL test disabled

    xData = numpy.arange(1000)
    yData = -500 + 100 * numpy.sin(xData)
    xData2 = xData + 1000
    yData2 = xData - 1000 + 200 * numpy.random.random(1000)

    plot = qWidgetFactory(PlotWidget)

    plot.clear()
    plot.setActiveCurveHandling(True)
    legend = "curve 1"
    plot.addCurve(xData, yData, legend=legend, color="green")

    # active curve should be None
    assert plot.getActiveCurve(just_legend=True) is None

    # active curve should be None when None is set as active curve
    plot.setActiveCurve(legend)
    current = plot.getActiveCurve(just_legend=True)
    assert current == legend
    plot.setActiveCurve(None)
    current = plot.getActiveCurve(just_legend=True)
    assert current is None

    # testing it automatically toggles if there is only one
    plot.setActiveCurveSelectionMode("legacy")
    current = plot.getActiveCurve(just_legend=True)
    assert current == legend

    # active curve should not change when None set as active curve
    assert plot.getActiveCurveSelectionMode() == "legacy"
    plot.setActiveCurve(None)
    current = plot.getActiveCurve(just_legend=True)
    assert current == legend

    # situation where no curve is active
    plot.clear()
    plot.setActiveCurveHandling(True)
    assert plot.getActiveCurveSelectionMode() == "atmostone"
    plot.addCurve(xData, yData, legend=legend, color="green")
    assert plot.getActiveCurve(just_legend=True) is None
    plot.addCurve(xData2, yData2, legend="curve 2", color="red")
    assert plot.getActiveCurve(just_legend=True) is None
    plot.setActiveCurveSelectionMode("legacy")
    assert plot.getActiveCurve(just_legend=True) is None

    # the first curve added should be active
    plot.clear()
    plot.addCurve(xData, yData, legend=legend, color="green")
    assert plot.getActiveCurve(just_legend=True) == legend
    plot.addCurve(xData2, yData2, legend="curve 2", color="red")
    assert plot.getActiveCurve(just_legend=True) == legend

    plot.setActiveCurveHandling(False)


@pytest.mark.parametrize("backend", ("mpl", "gl"))
def testActiveCurveStyle(qWidgetFactory, backend, request):
    """Test change of active curve style"""
    if backend == "gl":
        request.getfixturevalue("use_opengl")  # Skip test if OpenGL test disabled

    plot = qWidgetFactory(PlotWidget)

    plot.setActiveCurveHandling(True)
    plot.setActiveCurveStyle(color="black")
    style = plot.getActiveCurveStyle()
    assert style.getColor() == (0.0, 0.0, 0.0, 1.0)
    assert style.getLineStyle() is None
    assert style.getLineWidth() is None
    assert style.getSymbol() is None
    assert style.getSymbolSize() is None

    xData = numpy.arange(1000)
    yData = -500 + 100 * numpy.sin(xData)
    plot.addCurve(x=xData, y=yData, legend="curve1")
    curve = plot.getCurve("curve1")
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
    plot.setActiveCurve("curve1")
    style = curve.getCurrentStyle()
    assert style.getColor() == (0.0, 0.0, 0.0, 1.0)
    assert style.getLineStyle() == "-"
    assert style.getLineWidth() == 1
    assert style.getSymbol() == "o"
    assert style.getSymbolSize() == 5

    # Change highlight to linewidth=2
    plot.setActiveCurveStyle(linewidth=2)
    style = curve.getCurrentStyle()
    assert style.getColor() == (0.0, 0.0, 1.0, 1.0)
    assert style.getLineStyle() == "-"
    assert style.getLineWidth() == 2
    assert style.getSymbol() == "o"
    assert style.getSymbolSize() == 5

    plot.setActiveCurve(None)
    assert curve.getCurrentStyle() == defaultStyle

    plot.setActiveCurveHandling(False)


@pytest.mark.parametrize("backend", ("mpl", "gl"))
def testActiveImageAndLabels(qWidgetFactory, backend, request):
    if backend == "gl":
        request.getfixturevalue("use_opengl")  # Skip test if OpenGL test disabled

    plot = qWidgetFactory(PlotWidget)

    # Active image handling always on, no API for toggling it
    plot.getXAxis().setLabel("XLabel")
    plot.getYAxis().setLabel("YLabel")

    # labels changed as active curve
    plot.addImage(
        numpy.arange(100).reshape(10, 10), legend="1", xlabel="x1", ylabel="y1"
    )
    assert plot.getXAxis().getLabel() == "x1"
    assert plot.getYAxis().getLabel() == "y1"

    # labels not changed as not active curve
    plot.addImage(numpy.arange(100).reshape(10, 10), legend="2")
    assert plot.getXAxis().getLabel() == "x1"
    assert plot.getYAxis().getLabel() == "y1"

    # labels changed
    plot.setActiveImage("2")
    assert plot.getXAxis().getLabel() == "XLabel"
    assert plot.getYAxis().getLabel() == "YLabel"

    plot.setActiveImage("1")
    assert plot.getXAxis().getLabel() == "x1"
    assert plot.getYAxis().getLabel() == "y1"

    plot.clear()
    assert plot.getXAxis().getLabel() == "XLabel"
    assert plot.getYAxis().getLabel() == "YLabel"

    plot.setActiveCurveHandling(False)


def _checkSelection(selection, current=None, selected=()):
    """Check current item and selected items."""
    assert selection.getCurrentItem() is current
    assert selection.getSelectedItems() == selected


@pytest.mark.parametrize("backend", ("mpl", "gl"))
def testSyncWithActiveItems(qWidgetFactory, backend, request):
    """Test update of PlotWidgetSelection according to active items"""
    if backend == "gl":
        request.getfixturevalue("use_opengl")  # Skip test if OpenGL test disabled

    plot = qWidgetFactory(PlotWidget, backend=backend)

    listener = SignalListener()

    selection = plot.selection()
    selection.sigCurrentItemChanged.connect(listener)
    _checkSelection(selection)

    # Active item is current
    plot.addImage(((0, 1), (2, 3)), legend="image")
    image = plot.getActiveImage()
    assert listener.callCount() == 1
    _checkSelection(selection, image, (image,))

    # No active = no current
    plot.setActiveImage(None)
    assert listener.callCount() == 2
    _checkSelection(selection)

    # Active item is current
    plot.setActiveImage("image")
    assert listener.callCount() == 3
    _checkSelection(selection, image, (image,))

    # Mosted recently "actived" item is current
    plot.addScatter((3, 2, 1), (0, 1, 2), (0, 1, 2), legend="scatter")
    scatter = plot.getActiveScatter()
    assert listener.callCount() == 4
    _checkSelection(selection, scatter, (scatter, image))

    # Previously mosted recently "actived" item is current
    plot.setActiveScatter(None)
    assert listener.callCount() == 5
    _checkSelection(selection, image, (image,))

    # Mosted recently "actived" item is current
    plot.setActiveScatter("scatter")
    assert listener.callCount() == 6
    _checkSelection(selection, scatter, (scatter, image))

    # No active = no current
    plot.setActiveImage(None)
    plot.setActiveScatter(None)
    assert listener.callCount() == 7
    _checkSelection(selection)

    # Mosted recently "actived" item is current
    plot.setActiveScatter("scatter")
    assert listener.callCount() == 8
    plot.setActiveImage("image")
    assert listener.callCount() == 9
    _checkSelection(selection, image, (image, scatter))

    # Add a curve which is not active by default
    plot.addCurve((0, 1, 2), (0, 1, 2), legend="curve")
    curve = plot.getCurve("curve")
    assert listener.callCount() == 9
    _checkSelection(selection, image, (image, scatter))

    # Mosted recently "actived" item is current
    plot.setActiveCurve("curve")
    assert listener.callCount() == 10
    _checkSelection(selection, curve, (curve, image, scatter))

    # Add a curve which is not active by default
    plot.addCurve((0, 1, 2), (0, 1, 2), legend="curve2")
    curve2 = plot.getCurve("curve2")
    assert listener.callCount() == 10
    _checkSelection(selection, curve, (curve, image, scatter))

    # Mosted recently "actived" item is current, previous curve is removed
    plot.setActiveCurve("curve2")
    assert listener.callCount() == 11
    _checkSelection(selection, curve2, (curve2, image, scatter))

    # No items = no current
    plot.clear()
    assert listener.callCount() == 12
    _checkSelection(selection)


@pytest.mark.parametrize("backend", ("mpl", "gl"))
def testPlotWidgetWithItems(qWidgetFactory, backend, request):
    """Test init of selection on a plot with items"""
    if backend == "gl":
        request.getfixturevalue("use_opengl")  # Skip test if OpenGL test disabled

    plot = qWidgetFactory(PlotWidget, backend=backend)

    plot.addImage(((0, 1), (2, 3)), legend="image")
    plot.addScatter((3, 2, 1), (0, 1, 2), (0, 1, 2), legend="scatter")
    plot.addCurve((0, 1, 2), (0, 1, 2), legend="curve")
    plot.setActiveCurve("curve")

    selection = plot.selection()
    assert selection.getCurrentItem() is not None
    selected = selection.getSelectedItems()
    assert len(selected) == 3
    assert plot.getActiveCurve() in selected
    assert plot.getActiveImage() in selected
    assert plot.getActiveScatter() in selected


@pytest.mark.parametrize("backend", ("mpl", "gl"))
def testSetCurrentItem(qWidgetFactory, backend, request):
    """Test setCurrentItem"""
    if backend == "gl":
        request.getfixturevalue("use_opengl")  # Skip test if OpenGL test disabled

    plot = qWidgetFactory(PlotWidget, backend=backend)

    # Add items to the plot
    plot.addImage(((0, 1), (2, 3)), legend="image")
    image = plot.getActiveImage()
    plot.addScatter((3, 2, 1), (0, 1, 2), (0, 1, 2), legend="scatter")
    scatter = plot.getActiveScatter()
    plot.addCurve((0, 1, 2), (0, 1, 2), legend="curve")
    plot.setActiveCurve("curve")
    curve = plot.getActiveCurve()

    selection = plot.selection()
    assert selection.getCurrentItem() is not None
    assert len(selection.getSelectedItems()) == 3

    # Set current to None reset all active items
    selection.setCurrentItem(None)
    _checkSelection(selection)
    assert plot.getActiveCurve() is None
    assert plot.getActiveImage() is None
    assert plot.getActiveScatter() is None

    # Set current to an item makes it active
    selection.setCurrentItem(image)
    _checkSelection(selection, image, (image,))
    assert plot.getActiveCurve() is None
    assert plot.getActiveImage() is image
    assert plot.getActiveScatter() is None

    # Set current to an item makes it active and keeps other active
    selection.setCurrentItem(curve)
    _checkSelection(selection, curve, (curve, image))
    assert plot.getActiveCurve() is curve
    assert plot.getActiveImage() is image
    assert plot.getActiveScatter() is None

    # Set current to an item makes it active and keeps other active
    selection.setCurrentItem(scatter)
    _checkSelection(selection, scatter, (scatter, curve, image))
    assert plot.getActiveCurve() is curve
    assert plot.getActiveImage() is image
    assert plot.getActiveScatter() is scatter
