# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2018 European Synchrotron Radiation Facility
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
"""A widget dedicated to compare 2 images.
"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "04/07/2018"


from silx.gui import qt
from silx.gui import plot
import logging
import numpy
from silx.gui.colors import Colormap


_logger = logging.getLogger(__name__)


class CompareImages(qt.QMainWindow):

    def __init__(self):
        qt.QMainWindow.__init__(self)
        self.setWindowTitle("Plot with synchronized axes")
        widget = qt.QWidget(self)
        self.setCentralWidget(widget)

        layout = qt.QGridLayout()
        widget.setLayout(layout)

        backend = "gl"

        self.__data1 = None
        self.__data2 = None
        self.__previousSeparatorPosition = None

        self.__plot2d = plot.Plot2D(parent=widget, backend=backend)
        self.__plot2d.setKeepDataAspectRatio(True)
        # self.__plot2d.setInteractiveMode('pan')
        self.__plot2d.sigPlotSignal.connect(self.__plotSlot)

        layout.addWidget(self.__plot2d)

        self.__plot2d.addXMarker(
            0,
            legend='separator',
            text='separator',
            draggable=True,
            color='blue',
            constraint=self.__separatorConstraint)
        self.__separator = self.__plot2d._getMarker('separator')

    def __plotSlot(self, event):
        """Handle events from the plot"""
        if event['event'] in ('markerMoving', 'markerMoved'):
            if event['label'] == 'separator':
                value = int(float(str(event['xdata'])))
                if self.__previousSeparatorPosition != value:
                    self.__separatorMoved(value)
                    self.__previousSeparatorPosition = value

    def __separatorConstraint(self, x, y):
        if self.__data1 is None:
            return 0, 0
        x = int(x)
        if x < 0:
            x = 0
        elif x > self.__data1.shape[1]:
            x = self.__data1.shape[1]
        return x, y

    def __separatorMoved(self, pos):
        if self.__data1 is None:
            return

        if pos <= 0:
            pos = 0
        elif pos >= self.__data1.shape[1]:
            pos = self.__data1.shape[1]
        data1 = self.__data1[:, 0:pos]
        data2 = self.__data2[:, pos:]
        self.__image1.setData(data1, copy=False)
        self.__image2.setData(data2, copy=False)
        self.__image2.setOrigin((pos, 0))

    def setData(self, image1, image2):
        self.__data1 = image1
        self.__data2 = image2
        self.__plot2d.addImage(self.__data1, legend="image1")
        self.__plot2d.addImage(self.__data2, legend="image2")
        self.__image1 = self.__plot2d.getImage("image1")
        self.__image2 = self.__plot2d.getImage("image2")

        # Set the separator into the middle
        middle = self.__data1.shape[1] // 2
        self.__separator.setPosition(middle, 0)
        self.__separatorMoved(middle)
        self.__previousSeparatorPosition = middle

        # Avoid to change the colormap range when the separator is moving
        # TODO: The colormap histogram will still be wrong
        if len(image1.shape) == 2:
            vmin = min(self.__data1.min(), self.__data2.min())
            vmax = max(self.__data1.max(), self.__data2.max())
            colormap = Colormap(vmin=vmin, vmax=vmax)
            self.__image1.setColormap(colormap)
            self.__image2.setColormap(colormap)
        else:
            # RGBA images
            pass
