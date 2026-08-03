"""Large dataset performance test with pygfx backend.

Demonstrates: rendering performance with large curve and image data.
"""

import numpy
from silx.gui import qt
from silx.gui.plot import Plot1D, Plot2D


def main():
    app = qt.QApplication([])

    # --- Large curve: 1M points ---
    plot1 = Plot1D(backend="pygfx")
    plot1.setWindowTitle("pygfx - 1M Points Curve")
    plot1.setGraphTitle("1,000,000 Points")

    n = 1_000_000
    x = numpy.linspace(0, 100, n)
    y = numpy.sin(x * 10) * numpy.exp(-x / 30) + 0.1 * numpy.random.randn(n)
    plot1.addCurve(x, y, legend="1M pts", color="blue", linewidth=1)
    plot1.resetZoom()
    plot1.resize(800, 400)
    plot1.show()

    # --- Large image: 2048x2048 ---
    plot2 = Plot2D(backend="pygfx")
    plot2.setWindowTitle("pygfx - 2048x2048 Image")
    plot2.setGraphTitle("2048 x 2048 Image")
    plot2.getDefaultColormap().setName("magma")

    size = 2048
    xx, yy = numpy.meshgrid(
        numpy.linspace(-10, 10, size), numpy.linspace(-10, 10, size)
    )
    image = numpy.sin(xx) * numpy.cos(yy) + 0.05 * numpy.random.random((size, size))
    plot2.addImage(image)
    plot2.setKeepDataAspectRatio(True)
    plot2.resetZoom()
    plot2.resize(600, 600)
    plot2.move(820, 0)
    plot2.show()

    app.exec()


if __name__ == "__main__":
    main()
