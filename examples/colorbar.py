#!/usr/bin/env python
# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2017 European Synchrotron Radiation Facility
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
"""
Example to show the use of `ColorBarWidget` widget.
It can be associated to a plot.

In this exqmple the `ColorBarWidget` widget will display the colormap of the
active image.

To change the active image slick on the image you want to set active.
"""

__authors__ = ["H. Payno"]
__license__ = "MIT"
__date__ = "03/05/2017"


from silx.gui import qt
import numpy
from silx.gui.plot.Colormap import Colormap
from silx.gui.plot.ColorBar import ColorBarWidget
from silx.gui.plot.PlotWidget import PlotWidget

IMG_WIDTH = 100


class ColorBarShower(qt.QWidget):

    def __init__(self):
        qt.QWidget.__init__(self)

        self.setLayout(qt.QHBoxLayout())
        self.build_active_image_selector()
        self.layout().addWidget(self.image_selector)
        self.build_plot()
        self.layout().addWidget(self.plot)
        self.build_colorbar()
        self.layout().addWidget(self.colorbar)

        # connect radio button with the plot
        self.image_selector._qr1.toggled.connect(self.activateImageChanged)
        self.image_selector._qr2.toggled.connect(self.activateImageChanged)
        self.image_selector._qr3.toggled.connect(self.activateImageChanged)
        self.image_selector._qr4.toggled.connect(self.activateImageChanged)
        self.image_selector._qr5.toggled.connect(self.activateImageChanged)
        self.image_selector._qr6.toggled.connect(self.activateImageChanged)

    def build_active_image_selector(self):
        """Build the image selector widget"""
        self.image_selector = qt.QGroupBox()
        self.image_selector.setLayout(qt.QVBoxLayout())
        self.image_selector._qr1 = qt.QRadioButton('image1')
        self.image_selector._qr2 = qt.QRadioButton('image2')
        self.image_selector._qr3 = qt.QRadioButton('image3')
        self.image_selector._qr4 = qt.QRadioButton('image4')
        self.image_selector._qr5 = qt.QRadioButton('image5')
        self.image_selector._qr6 = qt.QRadioButton('image6')
        self.image_selector.layout().addWidget(self.image_selector._qr1)
        self.image_selector.layout().addWidget(self.image_selector._qr2)
        self.image_selector.layout().addWidget(self.image_selector._qr3)
        self.image_selector.layout().addWidget(self.image_selector._qr4)
        self.image_selector.layout().addWidget(self.image_selector._qr5)
        self.image_selector.layout().addWidget(self.image_selector._qr6)

    def activateImageChanged(self):
        if self.image_selector._qr1.isChecked():
            self.plot.setActiveImage('image1')
        if self.image_selector._qr2.isChecked():
            self.plot.setActiveImage('image2')
        if self.image_selector._qr3.isChecked():
            self.plot.setActiveImage('image3')
        if self.image_selector._qr4.isChecked():
            self.plot.setActiveImage('image4')
        if self.image_selector._qr5.isChecked():
            self.plot.setActiveImage('image5')
        if self.image_selector._qr6.isChecked():
            self.plot.setActiveImage('image6')

    def build_colorbar(self):
        self.colorbar = ColorBarWidget(parent=None)
        self.colorbar.setPlot(self.plot)

    def build_plot(self):
        image1 = numpy.exp(numpy.random.rand(IMG_WIDTH, IMG_WIDTH) * 10)
        image2 = numpy.linspace(-1000, 1000, IMG_WIDTH * IMG_WIDTH).reshape(IMG_WIDTH,
                                                                            IMG_WIDTH)
        image3 = numpy.linspace(-1, 1, IMG_WIDTH * IMG_WIDTH).reshape(IMG_WIDTH,
                                                                      IMG_WIDTH)
        image4 = numpy.linspace(-20, 50, IMG_WIDTH * IMG_WIDTH).reshape(IMG_WIDTH,
                                                                        IMG_WIDTH)
        image5 = image3
        image6 = image4

        # viridis colormap
        colormapViridis = Colormap(name='viridis',
                                   normalization='log',
                                   vmin=None,
                                   vmax=None)
        self.plot = PlotWidget()
        self.plot.addImage(data=image1,
                      origin=(0, 0),
                      replace=False,
                      legend='image1',
                      colormap=colormapViridis)
        self.plot.addImage(data=image2,
                      origin=(100, 0),
                      replace=False,
                      legend='image2',
                      colormap=colormapViridis)

        # red colormap
        colormapRed = Colormap(name='red',
                               normalization='linear',
                               vmin=None,
                               vmax=None)
        self.plot.addImage(data=image3,
                           origin=(0, 100),
                           replace=False,
                           legend='image3',
                           colormap=colormapRed)
        self.plot.addImage(data=image4,
                           origin=(100, 100),
                           replace=False,
                           legend='image4',
                           colormap=colormapRed)
        # gray colormap
        colormapGray = Colormap(name='gray',
                               normalization='linear',
                               vmin=1.0,
                               vmax=20.0)
        self.plot.addImage(data=image5,
                           origin=(0, 200),
                           replace=False,
                           legend='image5',
                           colormap=colormapGray)
        self.plot.addImage(data=image6,
                           origin=(100, 200),
                           replace=False,
                           legend='image6',
                           colormap=colormapGray)


if __name__ == '__main__':
    app = qt.QApplication([])
    widget = ColorBarShower()
    widget.show()
    app.exec_()
