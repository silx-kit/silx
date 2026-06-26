"""Logarithmic axes with pygfx backend.

Demonstrates: log scale on X and Y axes, grid.
"""

import numpy
from silx.gui import qt
from silx.gui.plot import Plot1D


def main():
    app = qt.QApplication([])

    plot = Plot1D(backend="pygfx")
    plot.setWindowTitle("pygfx - Log Axes")
    plot.setGraphTitle("Logarithmic Scale")
    plot.setGraphXLabel("Frequency (Hz)")
    plot.setGraphYLabel("Amplitude")

    x = numpy.logspace(0, 5, 200)

    # Power-law decay
    y1 = 1e6 * x**-1.5
    plot.addCurve(x, y1, legend="f^-1.5", color="blue", linewidth=2)

    # Exponential decay
    y2 = 1e4 * numpy.exp(-x / 1e4)
    plot.addCurve(x, y2, legend="exp decay", color="red", linewidth=2)

    plot.getXAxis().setScale("log")
    plot.getYAxis().setScale("log")
    plot.setGraphGrid("both")
    plot.setActiveCurveHandling(False)
    plot.resetZoom()
    plot.show()
    app.exec()


if __name__ == "__main__":
    main()
