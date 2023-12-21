# /*##########################################################################
#
# Copyright (c) 2017-2021 European Synchrotron Radiation Facility
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

"""
This script compares the rendering of PlotWidget's matplotlib and OpenGL backends.
"""

from __future__ import annotations

__license__ = "MIT"

import numpy
import sys
import functools

from silx.gui import qt

from silx.gui.plot import PlotWidget
from silx.gui.plot import items
from silx.gui.plot.items.marker import Marker
from silx.gui.plot.utils.axis import SyncAxes


_DESCRIPTIONS = {}


class MyPlotWindow(qt.QMainWindow):
    """QMainWindow with selected tools"""

    def __init__(self, parent=None):
        super(MyPlotWindow, self).__init__(parent)

        # Create a PlotWidget
        self._plot1 = PlotWidget(parent=self, backend="mpl")
        self._plot1.setGraphTitle("matplotlib")
        self._plot2 = PlotWidget(parent=self, backend="opengl")
        self._plot2.setGraphTitle("opengl")

        self.constraintX = SyncAxes(
            [
                self._plot1.getXAxis(),
                self._plot2.getXAxis(),
            ]
        )
        self.constraintY = SyncAxes(
            [
                self._plot1.getYAxis(),
                self._plot2.getYAxis(),
            ]
        )

        plotWidget = qt.QWidget(self)
        plotLayout = qt.QHBoxLayout(plotWidget)
        plotLayout.addWidget(self._plot1)
        plotLayout.addWidget(self._plot2)
        plotLayout.setContentsMargins(0, 0, 0, 0)
        plotLayout.setContentsMargins(0, 0, 0, 0)

        options = self.createOptions(self)
        centralWidget = qt.QWidget(self)
        layout = qt.QHBoxLayout(centralWidget)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(options)
        layout.addWidget(plotWidget)

        self.setCentralWidget(centralWidget)

        self._state = {}

    def clear(self):
        self._state = {}

    def createOptions(self, parent):
        options = qt.QWidget(parent)
        layout = qt.QVBoxLayout(options)
        for id, description in _DESCRIPTIONS.items():
            label, _func = description
            button = qt.QPushButton(label, self)
            button.clicked.connect(functools.partial(self.showUseCase, id))
            layout.addWidget(button)
        layout.addStretch()
        return options

    def showUseCase(self, name: str):
        description = _DESCRIPTIONS.get(name)
        if description is None:
            raise ValueError(f"Unknown use case '{name}'")
        setupFunc = description[1]
        self.clear()
        for p in [self._plot1, self._plot2]:
            p.clear()
            setupFunc(self, p)
            p.resetZoom()

    def _register(name, label):
        def decorator(func):
            _DESCRIPTIONS[name] = (label, func)
            return func

        return decorator

    def _addLine(
        self,
        plot,
        lineWidth: float,
        lineStyle: str,
        color: str,
        gapColor: str | None,
        curve: bool,
    ):
        state = self._state.setdefault(plot, {})
        x = state.get("x", 0)
        y = state.get("y", 0)
        x += 10
        state["x"] = x
        state["y"] = y

        start = (x - 20, y + 0)
        stop = (x + 40, y + 100)

        def createShape():
            shape = items.Shape("polylines")
            shape.setPoints(numpy.array((start, stop)))
            shape.setLineWidth(lineWidth)
            shape.setLineStyle(lineStyle)
            shape.setColor(color)
            if gapColor is not None:
                shape.setLineGapColor(gapColor)
            return shape

        def createCurve():
            curve = items.Curve()
            array = numpy.array((start, stop)).T
            curve.setData(array[0], array[1])
            curve.setLineWidth(lineWidth)
            curve.setLineStyle(lineStyle)
            curve.setColor(color)
            curve.setSymbol("")
            if gapColor is not None:
                curve.setLineGapColor(gapColor)
            return curve

        if curve:
            plot.addItem(createCurve())
        else:
            plot.addItem(createShape())

    @_register("linewidth", "Line width")
    def _setupLineStyle(self, plot: PlotWidget):
        self._addLine(plot, 0.5, "-", "#0000FF", None, curve=False)
        self._addLine(plot, 1.0, "-", "#0000FF", None, curve=False)
        self._addLine(plot, 2.0, "-", "#0000FF", None, curve=False)
        self._addLine(plot, 4.0, "-", "#0000FF", None, curve=False)
        self._addLine(plot, 0.5, "-", "#00FFFF", None, curve=True)
        self._addLine(plot, 1.0, "-", "#00FFFF", None, curve=True)
        self._addLine(plot, 2.0, "-", "#00FFFF", None, curve=True)
        self._addLine(plot, 4.0, "-", "#00FFFF", None, curve=True)

    @_register("linestyle", "Line style")
    def _setupLineStyle(self, plot: PlotWidget):
        self._addLine(plot, 1.0, "--", "#0000FF", None, curve=False)
        self._addLine(plot, 1.0, "-.", "#0000FF", None, curve=False)
        self._addLine(plot, 1.0, ":", "#0000FF", None, curve=False)
        self._addLine(plot, 2.0, "--", "#00FFFF", None, curve=True)
        self._addLine(plot, 2.0, "-.", "#00FFFF", None, curve=True)
        self._addLine(plot, 2.0, ":", "#00FFFF", None, curve=True)

    @_register("gapcolor", "LineStyle Gap Color")
    def _setupLineStyleGapColor(self, plot):
        self._addLine(plot, 1.0, "-", "#FF00FF", "black", curve=False)
        self._addLine(plot, 1.0, "-.", "#FF00FF", "black", curve=False)
        self._addLine(plot, 1.0, "--", "#FF00FF", "black", curve=False)
        self._addLine(plot, 0.5, "--", "#FF00FF", "black", curve=False)
        self._addLine(plot, 1.5, "--", "#FF00FF", "black", curve=False)
        self._addLine(plot, 2.0, "--", "#FF00FF", "black", curve=False)
        plot.setGraphXLimits(0, 100)
        plot.setGraphYLimits(0, 100)

    @_register("curveshape", "Curve vs Shape")
    def _setupLineStyleCurveShape(self, plot):
        self._addLine(plot, 1.0, (0, (5, 5)), "#00FF00", None, curve=False)
        self._addLine(plot, 4.0, (0, (3, 3)), "#00FF00", None, curve=False)
        self._addLine(plot, 4.0, (0, (5, 5)), "#00FF00", None, curve=False)
        self._addLine(plot, 4.0, (0, (7, 7)), "#00FF00", None, curve=False)
        self._addLine(plot, 1.0, (0, (5, 5)), "#00FFFF", None, curve=True)
        self._addLine(plot, 4.0, (0, (3, 3)), "#00FFFF", None, curve=True)
        self._addLine(plot, 4.0, (0, (5, 5)), "#00FFFF", None, curve=True)
        self._addLine(plot, 4.0, (0, (7, 7)), "#00FFFF", None, curve=True)
        plot.setGraphXLimits(0, 100)
        plot.setGraphYLimits(0, 100)

    @_register("text", "Text")
    def _setupText(self, plot):
        plot.getDefaultColormap().setName("viridis")

        # Add an image to the plot
        x = numpy.outer(numpy.linspace(-10, 10, 200), numpy.linspace(-10, 5, 150))
        image = numpy.sin(x) / x
        plot.addImage(image)

        label = Marker()
        label.setPosition(40, 150)
        label.setText("No background")
        plot.addItem(label)

        label = Marker()
        label.setPosition(50, 50)
        label.setText("Foo bar\nmmmmmmmmmmmmmmmmmmmm")
        label.setBackgroundColor("#FFFFFF44")
        plot.addItem(label)

        label2 = Marker()
        label2.setPosition(70, 70)
        label2.setText("Foo bar")
        label2.setColor("red")
        label2.setBackgroundColor("#00000044")
        plot.addItem(label2)

        label3 = Marker()
        label3.setPosition(10, 70)
        label3.setText("Pioupiou")
        label3.setColor("yellow")
        label3.setBackgroundColor("#000000")
        plot.addItem(label3)

    @_register("marker", "Marker")
    def _setupMarker(self, plot):
        plot.getDefaultColormap().setName("viridis")

        # Add an image to the plot
        x = numpy.outer(numpy.linspace(-10, 10, 200), numpy.linspace(-10, 5, 150))
        image = numpy.sin(x) / x
        plot.addImage(image)

        label = Marker()
        label.setSymbol("o")
        label.setPosition(30, 30)
        label.setColor("white")
        plot.addItem(label)

        label = Marker()
        label.setSymbol(".")
        label.setPosition(50, 30)
        label.setColor("white")
        plot.addItem(label)

        label = Marker()
        label.setSymbol(",")
        label.setPosition(70, 30)
        label.setColor("white")
        plot.addItem(label)

        label = Marker()
        label.setSymbol("+")
        # label.setSymbolSize(100)
        label.setPosition(30, 50)
        label.setColor("white")
        plot.addItem(label)

        label = Marker()
        label.setSymbol("x")
        label.setPosition(50, 50)
        label.setColor("white")
        plot.addItem(label)

        label = Marker()
        label.setSymbol("d")
        label.setPosition(70, 50)
        label.setColor("white")
        plot.addItem(label)

        label = Marker()
        label.setSymbol("s")
        label.setPosition(30, 70)
        label.setColor("white")
        plot.addItem(label)

        label = Marker()
        label.setSymbol("|")
        label.setPosition(50, 70)
        label.setColor("white")
        plot.addItem(label)

        label = Marker()
        label.setSymbol("_")
        label.setPosition(70, 70)
        label.setColor("white")
        plot.addItem(label)

    @_register("arrows", "Arrows")
    def _setupArrows(self, plot):
        """Display few lines with markers."""
        plot.setDataMargins(0.1, 0.1, 0.1, 0.1)

        plot.addCurve(
            x=[-10, 0, 0, -10, -10], y=[90, 90, 10, 10, 90], legend="box1", color="gray"
        )
        plot.addCurve(
            x=[110, 100, 100, 110, 110],
            y=[90, 90, 10, 10, 90],
            legend="box2",
            color="gray",
        )
        plot.addCurve(
            y=[-10, 0, 0, -10, -10], x=[90, 90, 10, 10, 90], legend="box3", color="gray"
        )
        plot.addCurve(
            y=[110, 100, 100, 110, 110],
            x=[90, 90, 10, 10, 90],
            legend="box4",
            color="gray",
        )

        def addCompositeLine(
            source, destination, symbolSource, symbolDestination, legend, color
        ):
            line = numpy.array([source, destination]).T
            plot.addCurve(x=line[0, :], y=line[1, :], color=color, legend=legend)
            plot.addMarker(x=source[0], y=source[1], symbol=symbolSource, color=color)
            plot.addMarker(
                x=destination[0],
                y=destination[1],
                symbol=symbolDestination,
                color=color,
            )

        addCompositeLine([0, 50], [100, 50], "caretleft", "caretright", "l1", "red")
        addCompositeLine([0, 30], [100, 30], "tickup", "tickdown", "l2", "blue")
        addCompositeLine([0, 70], [100, 70], "|", "|", "l3", "black")

        addCompositeLine([50, 0], [50, 100], "caretdown", "caretup", "l4", "red")
        addCompositeLine([30, 0], [30, 100], "tickleft", "tickright", "l5", "blue")
        addCompositeLine([70, 0], [70, 100], "_", "_", "l6", "black")


def main():
    global app
    app = qt.QApplication([])

    # Create the ad hoc window containing a PlotWidget and associated tools
    window = MyPlotWindow()
    window.setAttribute(qt.Qt.WA_DeleteOnClose)
    window.show()
    if len(sys.argv) == 1:
        useCase = "linestyle"
    else:
        useCase = sys.argv[1]
    window.showUseCase(useCase)
    app.exec()


if __name__ == "__main__":
    main()
