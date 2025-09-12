import numpy

from silx.gui import qt
from silx.gui.plot.items.curve import Curve as PlotCurve
from silx.gui.plot.items.scatter import Scatter as PlotScatter
from silx.gui.plot import Plot2D
from silx.gui.plot.LegendItem import LegendItemList

app = qt.QApplication([])
# Setup the plot window
plotWindow = qt.QMainWindow()
plotCentralWidget = qt.QWidget()
plotLayout = qt.QVBoxLayout(plotCentralWidget)
plotWindow.setCentralWidget(plotCentralWidget)
plotWindow.setWindowTitle("Plot Window")
plotWindow.resize(800, 600)

plot = Plot2D()
plotLayout.addWidget(plot)
infoWidget = LegendItemList(parent=plot)

# --- Curve Demonstration ---
xCurve = numpy.linspace(0, 10, 100)
yCurve = numpy.sin(xCurve)
curve1 = PlotCurve()
curve1.setData(xCurve, yCurve)
curve1.setName("Sin Curve")
plot.addItem(curve1)

# --- Scatter Demonstration ---
xScatter = numpy.random.rand(50) * 10
yScatter = numpy.random.rand(50) * 2 - 1
scatter1 = PlotScatter()
scatter1.setData(xScatter, yScatter, yScatter * 10)
scatter1.setName("Random Scatter")
scatter1.setSymbol("s")
scatter1.setSymbolSize(10)
plot.addItem(scatter1)

# Another curve for demonstration
y2 = numpy.cos(xCurve)
curve2 = PlotCurve()
curve2.setData(xCurve, y2)
curve2.setName("Cosine Curve")
plot.addItem(curve2)


plot.resetZoom()
plotWindow.show()
app.exec_()
