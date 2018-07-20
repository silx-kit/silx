"""This example illustrates the use of :class:`CurveLegendsWidget`.

:class:`CurveLegendsWidget` display curves style and legend currently visible
in a :class:`~silx.gui.plot.PlotWidget`
"""

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
