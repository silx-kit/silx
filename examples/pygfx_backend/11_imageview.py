"""ImageView widget with pygfx backend.

Demonstrates: ImageView with side histograms, colormap, aspect ratio.
"""

import numpy
from silx.gui import qt
from silx.gui.plot.ImageView import ImageView


def main():
    app = qt.QApplication([])

    view = ImageView(backend="pygfx")
    view.setWindowTitle("pygfx - ImageView")
    view.setKeepDataAspectRatio(True)

    # Generate a multi-peak image
    size = 256
    x = numpy.linspace(-5, 5, size)
    y = numpy.linspace(-5, 5, size)
    xx, yy = numpy.meshgrid(x, y)

    image = (
        numpy.exp(-((xx - 1) ** 2 + (yy - 1) ** 2))
        + 0.7 * numpy.exp(-((xx + 2) ** 2 + (yy + 1) ** 2) / 0.5)
        + 0.3 * numpy.exp(-((xx - 2) ** 2 + (yy + 2) ** 2) / 2)
        + 0.05 * numpy.random.random((size, size))
    )

    view.setImage(image, origin=(-5, -5), scale=(10 / size, 10 / size))
    view.setColormap("viridis")
    view.show()
    app.exec()


if __name__ == "__main__":
    main()
