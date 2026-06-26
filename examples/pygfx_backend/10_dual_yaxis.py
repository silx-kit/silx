"""Dual Y-axis with pygfx backend.

Demonstrates: left and right Y axes with different scales.
"""

import numpy
from silx.gui import qt
from silx.gui.plot import Plot1D


def main():
    app = qt.QApplication([])

    plot = Plot1D(backend="pygfx")
    plot.setWindowTitle("pygfx - Dual Y Axis")
    plot.setGraphTitle("Temperature and Pressure")
    plot.setGraphXLabel("Time (s)")
    plot.getYAxis().setLabel("Temperature (K)")
    plot.getYAxis(axis="right").setLabel("Pressure (mbar)")

    x = numpy.linspace(0, 100, 300)

    # Left Y axis: temperature
    temp = 300 + 50 * numpy.sin(x / 10) + 5 * numpy.random.randn(len(x))
    plot.addCurve(x, temp, legend="Temperature", color="red", linewidth=2, yaxis="left")

    # Right Y axis: pressure
    pressure = 1e-6 + 5e-7 * numpy.cos(x / 15) + 1e-7 * numpy.random.randn(len(x))
    plot.addCurve(
        x, pressure, legend="Pressure", color="blue", linewidth=2, yaxis="right"
    )

    plot.setActiveCurveHandling(False)
    plot.resetZoom()
    plot.show()
    app.exec()


if __name__ == "__main__":
    main()
