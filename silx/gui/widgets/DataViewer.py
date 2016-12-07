# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016 European Synchrotron Radiation Facility
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
"""This module defines a widget designed to display data using to most adapted
view from available ones from silx.
"""
from __future__ import division

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "07/12/2016"

import numpy

try:
    import h5py
except ImportError:
    h5py = None


from silx.gui import qt
from silx.gui import plot
from silx.gui.widgets.ArrayTableWidget import ArrayTableWidget


class DataViewer(qt.QStackedWidget):
    """Widget to display any kind of data"""

    EMPTY_MODE = 0
    PLOT1D_MODE = 1
    PLOT2D_MODE = 2
    TEXT_MODE = 3
    ARRAY_MODE = 4

    displayModeChanged = qt.Signal(int)
    """Emitted when the display mode change"""

    dataChanged = qt.Signal()
    """Emitted when the data change"""

    def __init__(self, parent=None):
        """Constructor"""
        super(DataViewer, self).__init__(parent)

        self.__displayMode = None
        self.__data = None
        self.__plot1d = plot.Plot1D()
        self.__plot2d = plot.Plot2D()
        self.__array = ArrayTableWidget()
        self.__text = qt.QLabel()
        self.__text.setAlignment(qt.Qt.AlignCenter)
        self.__empty = qt.QLabel()

        self.__index1d = self.addWidget(self.__plot1d)
        self.__index2d = self.addWidget(self.__plot2d)
        self.__indexArray = self.addWidget(self.__array)
        self.__indexText = self.addWidget(self.__text)
        self.__indexEmpty = self.addWidget(self.__empty)

        self.displayNothing()

    def clear(self):
        self.setData(None)

    def __clearCurrentView(self):
        if self.__displayMode is None:
            pass
        elif self.__displayMode == self.EMPTY_MODE:
            pass
        elif self.__displayMode == self.TEXT_MODE:
            self.__text.setText("")
        elif self.__displayMode == self.PLOT1D_MODE:
            self.__plot1d.clear()
        elif self.__displayMode == self.PLOT2D_MODE:
            self.__plot2d.clear()
        elif self.__displayMode == self.ARRAY_MODE:
            self.__array.setArrayData(numpy.array([[]]))
        else:
            raise Exception("Unsupported mode")

    def __displayData(self, data):
        if data is None:
            return
        if self.__displayMode == self.EMPTY_MODE:
            pass
        elif self.__displayMode == self.TEXT_MODE:
            self.__text.setText(str(data))
        elif self.__displayMode == self.PLOT1D_MODE:
            self.__plot1d.addCurve(legend="data", x=range(len(data)), y=data)
        elif self.__displayMode == self.PLOT2D_MODE:
            self.__plot2d.addImage(legend="data", data=data)
        elif self.__displayMode == self.ARRAY_MODE:
            self.__array.setArrayData(data)
        else:
            raise Exception("Unsupported mode")

    def setDisplayMode(self, mode):
        if self.__displayMode == mode:
            return
        self.__clearCurrentView()
        self.__displayMode = mode
        self.__displayData(self.__data)
        if self.__displayMode == self.EMPTY_MODE:
            self.setCurrentIndex(self.__indexEmpty)
        elif self.__displayMode == self.TEXT_MODE:
            self.setCurrentIndex(self.__indexText)
        elif self.__displayMode == self.PLOT1D_MODE:
            self.setCurrentIndex(self.__index1d)
        elif self.__displayMode == self.PLOT2D_MODE:
            self.setCurrentIndex(self.__index2d)
        elif self.__displayMode == self.ARRAY_MODE:
            self.setCurrentIndex(self.__indexArray)
        else:
            raise Exception("Unsupported mode")
        self.displayModeChanged.emit(mode)

    def displayNothing(self):
        """Display no data"""
        self.setDisplayMode(self.EMPTY_MODE)

    def displayAsText(self):
        """Display a data using text"""
        self.setDisplayMode(self.TEXT_MODE)

    def displayAs1d(self):
        """Display a data using `silx.plot.Plot1D`"""
        self.setDisplayMode(self.PLOT1D_MODE)

    def displayAs2d(self):
        """Display a data using `silx.plot.Plot2D`"""
        self.setDisplayMode(self.PLOT2D_MODE)

    def displayAsArray(self):
        """Display the data using `silx.gui.widgets.ArrayTableWidget`"""
        self.setDisplayMode(self.ARRAY_MODE)

    def __updateView(self):
        """Display the data using the widget which fit the best"""
        data = self.__data

        isArray = isinstance(data, numpy.ndarray)
        isArray = isArray or (isinstance(data, h5py.Dataset) and data.shape != tuple())

        if data is None:
            self.__clearCurrentView()
            self.displayNothing()
        elif isArray:
            isAtomic = len(data.shape) == 0
            isNumeric = numpy.issubdtype(data.dtype, numpy.number)
            isCurve = len(data.shape) == 1
            isImage = len(data.shape) == 2
            if isAtomic:
                self.displayAsText()
            if isCurve and isNumeric:
                self.displayAs1d()
            elif isImage and isNumeric:
                self.displayAs2d()
            else:
                self.displayAsArray()
        else:
            self.displayAsText()

    def setData(self, data):
        self.__data = data
        self.dataChanged.emit()
        self.__updateView()
        self.__displayData(data)

    def data(self):
        return self.__data

    def getDisplayMode(self):
        return self.__displayMode
