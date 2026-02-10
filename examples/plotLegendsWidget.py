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
plot.addCurve(x, numpy.sin(x), legend="Sin Curve", color="blue")
plot.addCurve(x, numpy.cos(x), legend="Cosine Curve", color="red")
xScatter = numpy.random.rand(50) * 10
yScatter = numpy.random.rand(50) * 2 - 1
plot.addScatter(xScatter, yScatter, value=yScatter * 10, legend="Points")
window.show()
app.exec_()
