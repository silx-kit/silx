import numpy

from silx.gui import qt
from silx.gui.plot import Plot2D
from silx.gui.plot.LegendsWidget import LegendsWidget

app = qt.QApplication([])
window = qt.QMainWindow()
window.setWindowTitle("Plot Window")
window.resize(800, 600)

plot = Plot2D()
window.setCentralWidget(plot)

dock = qt.QDockWidget("Legend", window)
legends = LegendsWidget(plotWidget=plot)
dock.setWidget(legends)
window.addDockWidget(qt.Qt.RightDockWidgetArea, dock)

x = numpy.linspace(0, 10, 100)
c1 = plot.addCurve(x, numpy.sin(x), legend="Sin Curve", color="blue")
legends.addItem(plot.getCurve("Sin Curve"))
c2 = plot.addCurve(x, numpy.cos(x), legend="Cosine Curve", color="red")
legends.addItem(plot.getCurve("Cosine Curve"))
xScatter = numpy.random.rand(50) * 10
yScatter = numpy.random.rand(50) * 2 - 1
s1 = plot.addScatter(xScatter, yScatter, value=yScatter * 10, legend="Points")
legends.addItem(plot.getScatter("Points"))
s2 = plot.addScatter(xScatter - 1, yScatter - 1, value=yScatter * 10, legend="Points_2")
window.show()
app.exec_()
