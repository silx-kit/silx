"""Image display with pygfx backend.

Demonstrates: 2D image with colormap, RGBA image, origin/scale, colorbar.
"""

import numpy
from silx.gui import qt
from silx.gui.plot import Plot2D


def main():
    app = qt.QApplication([])

    # --- Plot2D with colormap image ---
    plot = Plot2D(backend="pygfx")
    plot.setWindowTitle("pygfx - Image Display")
    plot.setGraphTitle("2D Gaussian + Noise")
    plot.setGraphXLabel("X")
    plot.setGraphYLabel("Y")

    # Generate a 2D Gaussian
    size = 256
    x = numpy.linspace(-3, 3, size)
    y = numpy.linspace(-3, 3, size)
    xx, yy = numpy.meshgrid(x, y)
    image = numpy.exp(-(xx**2 + yy**2)) + 0.1 * numpy.random.random((size, size))

    plot.getDefaultColormap().setName("viridis")
    plot.addImage(image, origin=(-3, -3), scale=(6 / size, 6 / size))
    plot.setKeepDataAspectRatio(True)
    plot.resetZoom()
    plot.show()

    # --- RGBA image window ---
    plot2 = Plot2D(backend="pygfx")
    plot2.setWindowTitle("pygfx - RGBA Image")
    plot2.setGraphTitle("RGBA Gradient")

    rgba = numpy.zeros((200, 300, 4), dtype=numpy.uint8)
    rgba[:, :, 0] = numpy.linspace(0, 255, 300)[numpy.newaxis, :]  # R gradient
    rgba[:, :, 1] = numpy.linspace(0, 255, 200)[:, numpy.newaxis]  # G gradient
    rgba[:, :, 2] = 128
    rgba[:, :, 3] = 255
    plot2.addImage(rgba)
    plot2.resetZoom()
    plot2.show()

    app.exec()


if __name__ == "__main__":
    main()
