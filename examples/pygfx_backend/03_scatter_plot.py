"""Scatter plot with pygfx backend.

Demonstrates: scatter points with colormap values, symbol sizes.
"""

import numpy
from silx.gui import qt
from silx.gui.plot import Plot1D


def main():
    app = qt.QApplication([])

    plot = Plot1D(backend="pygfx")
    plot.setWindowTitle("pygfx - Scatter Plot")
    plot.setGraphTitle("Random Scatter with Colormap")
    plot.setGraphXLabel("X")
    plot.setGraphYLabel("Y")
    plot.getDefaultColormap().setName("plasma")

    numpy.random.seed(42)
    n = 200
    x = numpy.random.randn(n)
    y = numpy.random.randn(n)
    value = numpy.sqrt(x**2 + y**2)  # distance from origin

    plot.addScatter(x, y, value, legend="distance", symbol="o")
    plot.setKeepDataAspectRatio(True)
    plot.resetZoom()
    plot.show()
    app.exec()


if __name__ == "__main__":
    main()
