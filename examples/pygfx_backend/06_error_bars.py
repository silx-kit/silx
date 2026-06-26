"""Error bars with pygfx backend.

Demonstrates: curves with x and y error bars.
"""

import numpy
from silx.gui import qt
from silx.gui.plot import Plot1D


def main():
    app = qt.QApplication([])

    plot = Plot1D(backend="pygfx")
    plot.setWindowTitle("pygfx - Error Bars")
    plot.setGraphTitle("Curves with Error Bars")
    plot.setGraphXLabel("X")
    plot.setGraphYLabel("Y")

    x = numpy.linspace(0, 10, 30)

    # Symmetric Y errors
    y1 = numpy.sin(x)
    yerr1 = 0.1 + 0.1 * numpy.abs(numpy.sin(x))
    plot.addCurve(
        x,
        y1,
        legend="sym Y error",
        color="blue",
        symbol="o",
        yerror=yerr1,
        linewidth=1.5,
    )

    # Asymmetric Y errors
    y2 = numpy.cos(x) + 3
    yerr_low = 0.2 * numpy.ones_like(x)
    yerr_high = 0.5 * numpy.abs(numpy.cos(x))
    plot.addCurve(
        x,
        y2,
        legend="asym Y error",
        color="red",
        symbol="s",
        yerror=numpy.array([yerr_low, yerr_high]),
        linewidth=1.5,
    )

    # X errors
    y3 = 0.5 * x - 1.5
    xerr = 0.3 * numpy.ones_like(x)
    plot.addCurve(
        x, y3, legend="X error", color="green", symbol="d", xerror=xerr, linewidth=1.5
    )

    plot.setActiveCurveHandling(False)
    plot.resetZoom()
    plot.show()
    app.exec()


if __name__ == "__main__":
    main()
