"""Basic curve plotting with pygfx backend.

Demonstrates: multiple curves, colors, line widths, symbols, fill, legend.
"""

import numpy
from silx.gui import qt
from silx.gui.plot import Plot1D


def main():
    app = qt.QApplication([])

    plot = Plot1D(backend="pygfx")
    plot.setWindowTitle("pygfx - Basic Curves")
    plot.setGraphTitle("Trigonometric Functions")
    plot.setGraphXLabel("X")
    plot.setGraphYLabel("Y")

    x = numpy.linspace(0, 4 * numpy.pi, 500)

    # Solid line
    plot.addCurve(x, numpy.sin(x), legend="sin(x)", color="blue", linewidth=2)
    # Dashed line with symbols
    plot.addCurve(
        x[::20],
        numpy.cos(x[::20]),
        legend="cos(x)",
        color="red",
        linewidth=1.5,
        linestyle="--",
        symbol="o",
    )
    # Filled curve
    plot.addCurve(
        x,
        0.5 * numpy.sin(2 * x),
        legend="0.5*sin(2x)",
        color="green",
        linewidth=1,
        fill=True,
    )

    plot.setActiveCurveHandling(False)
    plot.resetZoom()
    plot.show()
    app.exec()


if __name__ == "__main__":
    main()
