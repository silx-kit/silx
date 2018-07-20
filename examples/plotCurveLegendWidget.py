# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2018 European Synchrotron Radiation Facility
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
"""This example illustrates the use of :class:`CurveLegendsWidget`.

:class:`CurveLegendsWidget` display curves style and legend currently visible
in a :class:`~silx.gui.plot.PlotWidget`
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "20/07/2018"

import numpy

from silx.gui import qt
from silx.gui.plot import Plot1D
from silx.gui.plot.tools.CurveLegendsWidget import CurveLegendsWidget
from silx.gui.widgets.BoxLayoutDockWidget import BoxLayoutDockWidget


# First create the QApplication
app = qt.QApplication([])

# Create a plot and add some curves
window = Plot1D()
window.setWindowTitle("CurveLegendWidgets demo")

x = numpy.linspace(-numpy.pi, numpy.pi, 100)
window.addCurve(x, 2. * numpy.random.random(100) - 1.,
                legend='random',
                symbol='s', linestyle='--',
                color='red')
window.addCurve(x, numpy.sin(x),
                legend='sin',
                symbol='o', linestyle=':',
                color='blue')
window.addCurve(x, numpy.cos(x),
                legend='cos',
                symbol='', linestyle='-',
                color='blue')


# Create a CurveLegendWidget associated to the plot
curveLegendsWidget = CurveLegendsWidget()
curveLegendsWidget.setPlotWidget(window)

# Add the CurveLegendsWidget as a dock widget to the plot
dock = BoxLayoutDockWidget()
dock.setWindowTitle('Curve legends')
dock.setWidget(curveLegendsWidget)
window.addDockWidget(qt.Qt.RightDockWidgetArea, dock)

# Show the plot and run the QApplication
window.setAttribute(qt.Qt.WA_DeleteOnClose)
window.show()

app.exec_()
