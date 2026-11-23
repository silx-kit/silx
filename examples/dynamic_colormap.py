"""This script shows how to adjust the colormap range to the data around the
mouse cursor.

This mode is provided by the colormap dialog ("Fixed scale on: Cursor").
In this example, it is enabled at startup and it can be toggled with the W key,
even when the colormap dialog is not displayed.

When enabled, moving the mouse over the image adjusts the colormap range to the
data in a small region around the cursor, displayed as a dashed rectangle.

The image is made of 4 quadrants of noise with different amplitudes, and it
has a non-default origin and scale.
"""

import numpy

from silx.gui import qt
from silx.gui.plot import Plot2D


def main():
    app = qt.QApplication([])

    image = numpy.zeros((1000, 1000), dtype=numpy.float32)
    image[:500, :500] = numpy.random.randn(500, 500)
    image[:500, 500:] = 10 * numpy.random.randn(500, 500)
    image[500:, :500] = 100 * numpy.random.randn(500, 500)
    image[500:, 500:] = 5 * numpy.random.randn(500, 500)

    plot = Plot2D()
    plot.addImage(image, origin=(-500.0, 200.0), scale=(0.5, 2.0))

    dialog = plot.getColormapAction().getColormapDialog()
    action = dialog.getScaleToCursorAction()
    action.setShortcut(qt.QKeySequence(qt.Qt.Key_W))
    plot.addAction(action)  # Enable the shortcut when the plot has focus
    action.setChecked(True)

    plot.show()
    app.exec()


if __name__ == "__main__":
    main()
