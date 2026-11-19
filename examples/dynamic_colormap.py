# /*##########################################################################
#
# Copyright (c) 2018-2021 European Synchrotron Radiation Facility
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ###########################################################################*/
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
