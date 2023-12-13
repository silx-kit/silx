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
"""Test PlotWidget features related to data margins"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "11/05/2023"

import numpy
import pytest


def testDefaultDataMargins(plotWidget):
    """Test default PlotWidget data margins: No margins"""
    assert plotWidget.getDataMargins() == (0, 0, 0, 0)


def testResetZoomDataMarginsLinearAxes(qapp, plotWidget):
    """Test PlotWidget.setDataMargins effect on resetZoom with linear axis scales"""

    margins = 0.1, 0.2, 0.3, 0.4
    plotWidget.setDataMargins(*margins)

    plotWidget.resetZoom()
    qapp.processEvents()

    retrievedMargins = plotWidget.getDataMargins()
    assert retrievedMargins == margins

    dataRange = 100 - 1
    expectedXLimits = 1 - 0.1 * dataRange, 100 + 0.2 * dataRange
    expectedYLimits = 1 - 0.3 * dataRange, 100 + 0.4 * dataRange

    assert plotWidget.getXAxis().getLimits() == expectedXLimits
    assert plotWidget.getYAxis().getLimits() == expectedYLimits
    assert plotWidget.getYAxis(axis="right").getLimits() == expectedYLimits


def testResetZoomDataMarginsLogAxes(qapp, plotWidget):
    """Test PlotWidget.setDataMargins effect on resetZoom with log axis scales"""
    plotWidget.getXAxis().setScale("log")
    plotWidget.getYAxis().setScale("log")

    dataMargins = 0.1, 0.2, 0.3, 0.4
    plotWidget.setDataMargins(*dataMargins)

    plotWidget.resetZoom()
    qapp.processEvents()

    retrievedMargins = plotWidget.getDataMargins()
    assert retrievedMargins == dataMargins

    logMin, logMax = numpy.log10(1), numpy.log10(100)
    logRange = logMax - logMin
    expectedXLimits = pow(10.0, logMin - 0.1 * logRange), pow(
        10.0, logMax + 0.2 * logRange
    )
    expectedYLimits = pow(10.0, logMin - 0.3 * logRange), pow(
        10.0, logMax + 0.4 * logRange
    )

    assert plotWidget.getXAxis().getLimits() == expectedXLimits
    assert plotWidget.getYAxis().getLimits() == expectedYLimits
    assert plotWidget.getYAxis(axis="right").getLimits() == expectedYLimits


@pytest.mark.parametrize("margins", [False, True, (0, 0, 0, 0)])
def testSetLimitsNoDataMargins(plotWidget, margins):
    """Test PlotWidget.setLimits without data margins"""
    xlimits = 1, 2
    ylimits = 3, 4
    y2limits = 5, 6
    plotWidget.setLimits(*xlimits, *ylimits, *y2limits, margins=margins)

    assert plotWidget.getXAxis().getLimits() == xlimits
    assert plotWidget.getYAxis().getLimits() == ylimits
    assert plotWidget.getYAxis(axis="right").getLimits() == y2limits


@pytest.mark.parametrize(
    "margins,expectedLimits",
    [
        # margins=False: use limits as is
        (
            False,
            (1, 2, 3, 4, 5, 6),
        ),
        # margins=True: apply data margins
        (
            True,
            (1 - 0.1, 2 + 0.2, 3 - 0.3, 4 + 0.4, 5 - 0.3, 6 + 0.4),
        ),
        # margins=tuple: apply provided margins
        (
            (0.4, 0.3, 0.2, 0.1),
            (1 - 0.4, 2 + 0.3, 3 - 0.2, 4 + 0.1, 5 - 0.2, 6 + 0.1),
        ),
    ],
)
def testSetLimitsWithDataMargins(qapp, plotWidget, margins, expectedLimits):
    """Test PlotWidget.setLimits with data margins"""
    dataMargins = 0.1, 0.2, 0.3, 0.4
    limits = 1, 2, 3, 4, 5, 6

    plotWidget.setDataMargins(*dataMargins)
    plotWidget.setLimits(*limits, margins=margins)
    qapp.processEvents()

    retrievedLimits = (
        *plotWidget.getXAxis().getLimits(),
        *plotWidget.getYAxis().getLimits(),
        *plotWidget.getYAxis(axis="right").getLimits(),
    )
    assert retrievedLimits == expectedLimits
