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
"""Tests of PlotWidget Axis items"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "15/06/2023"


from silx.gui.plot import PlotWidget


def testAxisIsVisible(qapp, qWidgetFactory):
    """Test Axis.isVisible method"""
    plotWidget = qWidgetFactory(PlotWidget)

    assert plotWidget.getXAxis().isVisible()
    assert plotWidget.getYAxis().isVisible()
    assert not plotWidget.getYAxis("right").isVisible()

    # Add curve on right axis
    plotWidget.addCurve((0, 1, 2), (1, 2, 3), yaxis="right")
    qapp.processEvents()

    assert plotWidget.getYAxis("right").isVisible()

    # hide curve on right axis
    curve = plotWidget.getItems()[0]
    curve.setVisible(False)
    qapp.processEvents()

    assert not plotWidget.getYAxis("right").isVisible()

    # show curve on right axis
    curve.setVisible(True)
    qapp.processEvents()

    assert plotWidget.getYAxis("right").isVisible()

    # Move curve to left axis
    curve.setYAxis("left")
    qapp.processEvents()

    assert not plotWidget.getYAxis("right").isVisible()


def testAxisSetScaleLogNoData(qapp, qWidgetFactory):
    """Test Axis.setScale('log') method with an empty plot

    Limits are reset only when negative
    """
    plotWidget = qWidgetFactory(PlotWidget)
    xaxis = plotWidget.getXAxis()
    yaxis = plotWidget.getYAxis()
    y2axis = plotWidget.getYAxis("right")

    xaxis.setLimits(-1.0, 1.0)
    yaxis.setLimits(2.0, 3.0)
    y2axis.setLimits(-2.0, -1.0)

    xaxis.setScale("log")
    qapp.processEvents()

    assert xaxis.getLimits() == (1.0, 100.0)
    assert yaxis.getLimits() == (2.0, 3.0)
    assert y2axis.getLimits() == (-2.0, -1.0)

    xaxis.setLimits(10.0, 20.0)

    yaxis.setScale("log")
    qapp.processEvents()

    assert xaxis.getLimits() == (10.0, 20.0)
    assert yaxis.getLimits() == (2.0, 3.0)  # Positive range is preserved
    assert y2axis.getLimits() == (1.0, 100.0)  # Negative min is reset


def testAxisSetScaleLogWithData(qapp, qWidgetFactory):
    """Test Axis.setScale('log') method with data

    Limits are reset only when negative and takes the data range into account
    """
    plotWidget = qWidgetFactory(PlotWidget)
    xaxis = plotWidget.getXAxis()
    yaxis = plotWidget.getYAxis()
    plotWidget.addCurve((-1, 1, 2, 3), (-1, 1, 2, 3))

    xaxis.setLimits(-1.0, 0.5)  # Limits contains no positive data
    yaxis.setLimits(-1.0, 2.0)  # Limits contains positive data

    xaxis.setScale("log")
    yaxis.setScale("log")
    qapp.processEvents()

    assert xaxis.getLimits() == (1.0, 3.0)  # Reset to positive data range
    assert yaxis.getLimits() == (1.0, 2.0)  # Keep max limit


def testAxisSetScaleLinear(qapp, qWidgetFactory):
    """Test Axis.setScale('linear') method: Limits are not changed"""
    plotWidget = qWidgetFactory(PlotWidget)
    xaxis = plotWidget.getXAxis()
    yaxis = plotWidget.getYAxis()
    y2axis = plotWidget.getYAxis("right")
    xaxis.setScale("log")
    yaxis.setScale("log")
    plotWidget.resetZoom()
    qapp.processEvents()

    xaxis.setLimits(10.0, 1000.0)
    yaxis.setLimits(20.0, 2000.0)
    y2axis.setLimits(30.0, 3000.0)

    xaxis.setScale("linear")
    qapp.processEvents()

    assert xaxis.getLimits() == (10.0, 1000.0)
    assert yaxis.getLimits() == (20.0, 2000.0)
    assert y2axis.getLimits() == (30.0, 3000.0)

    yaxis.setScale("linear")
    qapp.processEvents()

    assert xaxis.getLimits() == (10.0, 1000.0)
    assert yaxis.getLimits() == (20.0, 2000.0)
    assert y2axis.getLimits() == (30.0, 3000.0)
