#!/usr/bin/env python
# /*##########################################################################
#
# Copyright (c) 2020 European Synchrotron Radiation Facility
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
"""Example illustrating the different profile tools.
"""

import numpy
import scipy.signal

from silx.gui import qt
from silx.gui.plot import Plot2D
from silx.gui.plot import ScatterView
from silx.gui.plot import StackView
from silx.gui.plot.tools.profile import toolbar


def createScatterData():
    nbPoints = 200
    nbX = int(numpy.sqrt(nbPoints))
    nbY = nbPoints // nbX + 1

    # Motor position
    yy = numpy.atleast_2d(numpy.ones(nbY)).T
    xx = numpy.atleast_2d(numpy.ones(nbX))

    positionX = numpy.linspace(10, 50, nbX) * yy
    positionX = positionX.reshape(nbX * nbY)
    positionX = positionX + numpy.random.rand(len(positionX)) - 0.5

    positionY = numpy.atleast_2d(numpy.linspace(20, 60, nbY)).T * xx
    positionY = positionY.reshape(nbX * nbY)
    positionY = positionY + numpy.random.rand(len(positionY)) - 0.5

    # Diodes position
    lut = scipy.signal.gaussian(max(nbX, nbY), std=8) * 10
    yy, xx = numpy.ogrid[:nbY, :nbX]
    signal = lut[yy] * lut[xx]
    diode1 = numpy.random.poisson(signal * 10)
    diode1 = diode1.reshape(nbX * nbY)
    return positionX, positionY, diode1


class Example(qt.QMainWindow):
    def __init__(self, parent=None):
        qt.QMainWindow.__init__(self, parent=parent)
        self._createPlot2D()
        self._createScatterView()
        self._createStackView()

        dataWidget = qt.QWidget(self)
        dataLayout = qt.QStackedLayout(dataWidget)
        dataLayout.addWidget(self.plot)
        dataLayout.addWidget(self.scatter)
        dataLayout.addWidget(self.stack)
        dataLayout.setCurrentWidget(self.plot)
        self.dataLayout = dataLayout

        clearButton = qt.QPushButton(self)
        clearButton.clicked.connect(self._clearData)
        clearButton.setText("Clear")

        imageButton = qt.QPushButton(self)
        imageButton.clicked.connect(self._updateImage)
        imageButton.setText("Intensity image")

        imageRgbButton = qt.QPushButton(self)
        imageRgbButton.clicked.connect(self._updateRgbImage)
        imageRgbButton.setText("RGB image")

        scatterButton = qt.QPushButton(self)
        scatterButton.clicked.connect(self._updateScatter)
        scatterButton.setText("Scatter")

        stackButton = qt.QPushButton(self)
        stackButton.clicked.connect(self._updateStack)
        stackButton.setText("Stack")

        options = qt.QWidget(self)
        layout = qt.QHBoxLayout(options)
        layout.addStretch()
        layout.addWidget(clearButton)
        layout.addWidget(imageButton)
        layout.addWidget(imageRgbButton)
        layout.addWidget(scatterButton)
        layout.addWidget(stackButton)
        layout.addStretch()

        widget = qt.QWidget(self)
        layout = qt.QVBoxLayout(widget)
        layout.addWidget(dataWidget)
        layout.addWidget(options)
        self.setCentralWidget(widget)

        self._updateImage()

    def _createPlot2D(self):
        plot = Plot2D(self)
        self.plot = plot

        toolBar = toolbar.ProfileToolBar(plot, plot)
        toolBar.setScheme("image")
        plot.addToolBar(toolBar)

        toolBar = plot.getProfileToolbar()
        toolBar.clear()

    def _createScatterView(self):
        plot = ScatterView(self)
        self.scatter = plot

        toolBar = toolbar.ProfileToolBar(plot, plot.getPlotWidget())
        toolBar.setScheme("scatter")
        plot.addToolBar(toolBar)

        toolBar = plot.getScatterProfileToolBar()
        toolBar.clear()

    def _createStackView(self):
        plot = StackView(self)
        self.stack = plot

        toolBar = toolbar.ProfileToolBar(plot, plot.getPlotWidget())
        toolBar.setScheme("imagestack")
        plot.addToolBar(toolBar)

        toolBar = plot.getProfileToolbar()
        toolBar.clear()

    def _clearData(self):
        image = self.plot.getActiveImage()
        if image is not None:
            self.plot.removeItem(image)
        self.scatter.setData(None, None, None)
        self.stack.clear()

    def _updateImage(self):
        x = numpy.outer(numpy.linspace(-10, 10, 200),
                        numpy.linspace(-5, 5, 150))
        image = numpy.sin(x) / x
        image = image * 10 + numpy.random.rand(*image.shape)

        self.plot.addImage(image)
        self.dataLayout.setCurrentWidget(self.plot)

    def _updateRgbImage(self):
        image = numpy.empty(shape=(200, 150, 3), dtype=numpy.uint8)
        x = numpy.outer(numpy.linspace(-10, 10, 200),
                        numpy.linspace(-5, 5, 150))
        r = numpy.sin(x) / x
        g = numpy.cos(x/10) * numpy.sin(x/10)
        b = x
        image[..., 0] = 100 + 200 * (r / r.max())
        image[..., 1] = 100 + 200 * (g / g.max())
        image[..., 2] = 100 + 200 * (b / b.max())
        image[...] = image + numpy.random.randint(0, 20, size=image.shape)

        self.plot.addImage(image)
        self.dataLayout.setCurrentWidget(self.plot)

    def _updateScatter(self):
        xx, yy, value = createScatterData()
        self.scatter.setData(xx, yy, value)
        self.dataLayout.setCurrentWidget(self.scatter)

    def _updateStack(self):
        a, b, c = numpy.meshgrid(numpy.linspace(-10, 10, 200),
                                 numpy.linspace(-10, 5, 150),
                                 numpy.linspace(-5, 10, 120),
                                 indexing="ij")
        raw = numpy.asarray(numpy.sin(a * b * c) / (a * b * c),
                            dtype='float32')
        raw = numpy.abs(raw)
        raw[numpy.isnan(raw)] = 0
        data = raw + numpy.random.poisson(raw * 10)
        self.stack.setStack(data)
        self.dataLayout.setCurrentWidget(self.stack)

def main():
    app = qt.QApplication([])
    widget = Example()
    widget.show()
    app.exec()

if __name__ == "__main__":
    main()
