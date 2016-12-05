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
__date__ = "05/12/2016"

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

    def __init__(self, parent=None):
        """Constructor"""
        super(DataViewer, self).__init__(parent)

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
        self.setCurrentIndex(self.__indexEmpty)

    def clear(self):
        self.__data = None
        self.__plot1d.clear()
        self.__plot2d.clear()
        self.__text.setText("")
        self.__array.setArrayData(numpy.array([[]]))

    def displayNothing(self):
        """Display no data"""
        self.setCurrentIndex(self.__indexEmpty)

    def displayAsString(self):
        """Display a data using text"""
        data = self.__data
        if isinstance(data, h5py.Dataset):
            data = data.value
        self.__text.setText(str(data))
        self.setCurrentIndex(self.__indexText)

    def displayAs1d(self):
        """Display a data using `silx.plot.Plot1D`"""
        self.__plot1d.clear()
        data = self.__data
        self.__plot1d.addCurve(legend="data", x=range(len(data)), y=data)
        self.setCurrentIndex(self.__index1d)

    def displayAs2d(self):
        """Display a data using `silx.plot.Plot2D`"""
        self.__plot2d.clear()
        self.__plot2d.addImage(legend="data", data=self.__data)
        self.setCurrentIndex(self.__index2d)

    def displayAsArray(self):
        """Display the data using `silx.gui.widgets.ArrayTableWidget`"""
        self.__array.setArrayData(self.__data)
        self.setCurrentIndex(self.__indexArray)

    def updateView(self):
        """Display the data using the widget which fit the best"""
        data = self.__data

        isArray = isinstance(data, numpy.ndarray)
        isArray = isArray or (isinstance(data, h5py.Dataset) and data.shape != tuple())

        if data is None:
            self.displayNothing()
        elif isArray:
            isAtomic = len(data.shape) == 0
            isNumeric = numpy.issubdtype(data.dtype, numpy.number)
            isCurve = len(data.shape) == 1
            isImage = len(data.shape) == 2
            if isAtomic:
                self.displayAsString()
            if isCurve and isNumeric:
                self.displayAs1d()
            elif isImage and isNumeric:
                self.displayAs2d()
            else:
                self.displayAsArray()
        else:
            self.displayAsString()

    def setData(self, data):
        self.__data = data
        self.updateView()

    def data(self):
        return self.__data
