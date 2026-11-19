"""This script shows how to add a :class:`DynamicColormapAction` to a plot.

When the action is checked (toolbar button or W key), moving the mouse over
the image adjusts the colormap range to the data in a small region around the
cursor. The region is displayed as a dashed rectangle.

The image is made of 4 quadrants of noise with different amplitudes, and it
has a non-default origin and scale.
"""

import numpy

from silx.gui import qt
from silx.gui.plot import Plot2D
from silx.gui.plot.actions.image import DynamicColormapAction


def main():
    app = qt.QApplication([])

    image = numpy.zeros((1000, 1000), dtype=numpy.float32)
    image[:500, :500] = numpy.random.randn(500, 500)
    image[:500, 500:] = 10 * numpy.random.randn(500, 500)
    image[500:, :500] = 100 * numpy.random.randn(500, 500)
    image[500:, 500:] = 5 * numpy.random.randn(500, 500)

    plot = Plot2D()
    plot.addImage(image, origin=(-500.0, 200.0), scale=(0.5, 2.0))

    action = DynamicColormapAction(plot, parent=plot)
    action.setShortcut(qt.QKeySequence(qt.Qt.Key_W))
    plot.toolBar().addAction(action)
    plot.addAction(action)  # To enable the shortcut

    plot.show()
    app.exec()


if __name__ == "__main__":
    main()
