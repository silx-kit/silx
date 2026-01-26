import numpy

from silx.gui import qt
from silx.gui.plot import Plot2D
from silx.gui.plot.LegendItem import LegendItemList


class FloatingLegend(LegendItemList):
    def __init__(self, plot):
        super().__init__(parent=plot, plotWidget=plot)
        # Style it to look like a floating panel
        self.setStyleSheet("""
            LegendItemList {
                background-color: rgba(255, 255, 255, 200);
                border: 1px solid #ACACAC;
                border-radius: 4px;
            }
        """)
        self.setAttribute(qt.Qt.WA_StyledBackground)

    def updatePosition(self):
        if self.parentWidget():
            margin = 10
            width = self.sizeHint().width() + 20
            width = min(width, 250)
            height = self.layout().sizeHint().height()
            x = self.parentWidget().width() - width - margin
            y = margin
            self.setGeometry(x, y, width, height)


app = qt.QApplication([])
plotWindow = qt.QMainWindow()
plotCentralWidget = qt.QWidget()
plotLayout = qt.QVBoxLayout(plotCentralWidget)
plotWindow.setCentralWidget(plotCentralWidget)
plotWindow.setWindowTitle("Plot Window")
plotWindow.resize(800, 600)

plot = Plot2D()
plotLayout.addWidget(plot)


legend = FloatingLegend(plot)
plot.resizeEvent = lambda event: (
    Plot2D.resizeEvent(plot, event),
    legend.updatePosition(),
)


# Curve 1
xCurve = numpy.linspace(0, 10, 100)
yCurve = numpy.sin(xCurve)
c1 = plot.addCurve(xCurve, yCurve, legend="Sin Curve", color="blue")

# Curve 2
y2 = numpy.cos(xCurve)
c2 = plot.addCurve(xCurve, y2, legend="Cosine Curve", color="red")

# Scatter

xScatter = numpy.random.rand(50) * 10
yScatter = numpy.random.rand(50) * 2 - 1
s1 = plot.addScatter(
    xScatter, yScatter, value=yScatter * 10, legend="Points", symbol="s"
)

plotWindow.show()
legend.updatePosition()
app.exec_()
