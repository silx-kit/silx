import numpy

from silx.gui import qt
from silx.gui.plot.items.curve import Curve as PlotCurve
from silx.gui.plot.items.scatter import Scatter as PlotScatter
from silx.gui.plot import Plot2D
from silx.gui.plot.LegendItem import LegendItemList

app = qt.QApplication([])
# Setup the plot window
plot_window = qt.QMainWindow()
plot_central_widget = qt.QWidget()
plot_layout = qt.QVBoxLayout(plot_central_widget)
plot_window.setCentralWidget(plot_central_widget)
plot_window.setWindowTitle("Plot Window")
plot_window.resize(800, 600)

plot = Plot2D()
plot_layout.addWidget(plot)
info_widget = LegendItemList()
info_widget.binding(plot)

# --- Curve Demonstration ---
x_curve = numpy.linspace(0, 10, 100)
y_curve = numpy.sin(x_curve)
curve1 = PlotCurve()
curve1.setData(x_curve, y_curve)
curve1.setName("Sin Curve")
plot.addItem(curve1)

# --- Scatter Demonstration ---
x_scatter = numpy.random.rand(50) * 10
y_scatter = numpy.random.rand(50) * 2 - 1
scatter1 = PlotScatter()
scatter1.setData(x_scatter, y_scatter, y_scatter * 10)
scatter1.setName("Random Scatter")
scatter1.setSymbol("s")
scatter1.setSymbolSize(10)
plot.addItem(scatter1)

# Another curve for demonstration
y2 = numpy.cos(x_curve)
curve2 = PlotCurve()
curve2.setData(x_curve, y2)
curve2.setName("Cosine Curve")
plot.addItem(curve2)

# Add a custom button to highlight a curve to demonstrate the signal handling
def highlight_curve1():
    curve1.setHighlighted(not curve1.isHighlighted())

highlight_button = qt.QPushButton("Toggle Highlight on Sinusoidal Curve")
plot_layout.addWidget(highlight_button)

plot_window.show()
info_widget.show()
app.exec_()
