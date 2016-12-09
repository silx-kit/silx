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
__date__ = "09/12/2016"

import numpy
import silx.gui.icons

try:
    import h5py
except ImportError:
    h5py = None


from silx.gui import qt
from silx.gui.widgets.DataViewer import DataViewer


class DataViewerSelector(qt.QWidget):
    """Widget to be able to select a custom view from the DataViewer"""

    def __init__(self, parent=None, dataViewer=None):
        """Constructor

        :param QWidget parent: The parent of the widget
        :param DataViewer dataViewer: The connected `DataViewer`
        """
        super(DataViewerSelector, self).__init__(parent)

        self.__buttons = []
        self.__dataViewer = None
        self.__group = qt.QButtonGroup(self)
        self.setLayout(qt.QHBoxLayout())
        self.layout().setMargin(0)

        iconSize = qt.QSize(16, 16)

        button = qt.QPushButton("1D")
        button.setIcon(silx.gui.icons.getQIcon("view-1d"))
        button.setIconSize(iconSize)
        button.setCheckable(True)
        button.clicked.connect(self.__displayAs1d)
        self.layout().addWidget(button)
        self.__group.addButton(button)
        self.__button1D = button

        button = qt.QPushButton("2D")
        button.setIcon(silx.gui.icons.getQIcon("view-2d"))
        button.setIconSize(iconSize)
        button.setCheckable(True)
        button.clicked.connect(self.__displayAs2d)
        self.layout().addWidget(button)
        self.__group.addButton(button)
        self.__button2D = button

        button = qt.QPushButton("Raw")
        button.setIcon(silx.gui.icons.getQIcon("view-raw"))
        button.setIconSize(iconSize)
        button.setCheckable(True)
        button.clicked.connect(self.__displayAsArray)
        self.layout().addWidget(button)
        self.__group.addButton(button)
        self.__buttonArray = button

        button = qt.QPushButton("Text")
        button.setIcon(silx.gui.icons.getQIcon("view-text"))
        button.setIconSize(iconSize)
        button.setCheckable(True)
        button.clicked.connect(self.__displayAsText)
        self.layout().addWidget(button)
        self.__group.addButton(button)
        self.__buttonText = button

        button = qt.QPushButton("Dummy")
        button.setCheckable(True)
        button.setVisible(False)
        self.layout().addWidget(button)
        self.__group.addButton(button)
        self.__buttonDummy = button

        self.layout().addStretch(1)

        self.__group.buttonClicked[qt.QAbstractButton].connect(self.__buttonClicked)

        if dataViewer is not None:
            self.setDataViewer(dataViewer)

    def setDataViewer(self, dataViewer):
        """Define the dataviewer connected to this status bar

        :param DataViewer dataViewer: The connected `DataViewer`
        """
        if self.__dataViewer is dataViewer:
            return
        if self.__dataViewer is not None:
            self.__dataViewer.dataChanged.disconnect(self.__dataChanged)
            self.__dataViewer.displayModeChanged.disconnect(self.__displayModeChanged)
        self.__dataViewer = dataViewer
        if self.__dataViewer is not None:
            self.__dataViewer.dataChanged.connect(self.__dataChanged)
            self.__dataViewer.displayModeChanged.connect(self.__displayModeChanged)
            self.__displayModeChanged(self.__dataViewer.getDisplayMode())
        self.__dataChanged()

    def setFlat(self, isFlat):
        self.__buttonText.setFlat(isFlat)
        self.__button1D.setFlat(isFlat)
        self.__button2D.setFlat(isFlat)
        self.__buttonArray.setFlat(isFlat)
        self.__buttonDummy.setFlat(isFlat)

    def __displayModeChanged(self, mode):
        """Called on display mode changed"""
        selectedButton = None
        if mode == DataViewer.TEXT_MODE:
            selectedButton = self.__buttonText
        elif mode == DataViewer.PLOT1D_MODE:
            selectedButton = self.__button1D
        elif mode == DataViewer.PLOT2D_MODE:
            selectedButton = self.__button2D
        elif mode == DataViewer.ARRAY_MODE:
            selectedButton = self.__buttonArray
        else:
            selectedButton = self.__buttonDummy

        selectedButton.setChecked(True)

    def __buttonClicked(self, button):
        pass

    def __displayAsText(self):
        """Display a data using text"""
        if self.__dataViewer is None:
            return
        self.__dataViewer.displayAsText()

    def __displayAs1d(self):
        """Display a data using `silx.plot.Plot1D`"""
        if self.__dataViewer is None:
            return
        self.__dataViewer.displayAs1d()

    def __displayAs2d(self):
        """Display a data using `silx.plot.Plot2D`"""
        if self.__dataViewer is None:
            return
        self.__dataViewer.displayAs2d()

    def __displayAsArray(self):
        """Display the data using `silx.gui.widgets.ArrayTableWidget`"""
        if self.__dataViewer is None:
            return
        self.__dataViewer.displayAsArray()

    def __dataChanged(self):
        """Called on data changed"""
        if self.__dataViewer is None:
            self.__button1D.setVisible(False)
            self.__button2D.setVisible(False)
            self.__buttonArray.setVisible(False)
            self.__buttonText.setVisible(False)
        elif isinstance(self.__dataViewer.data(), h5py.Group):
            self.__button1D.setVisible(False)
            self.__button2D.setVisible(False)
            self.__buttonArray.setVisible(False)
            self.__buttonText.setVisible(False)
        else:
            data = self.__dataViewer.data()
            if data is not None:
                isArray = isinstance(data, numpy.ndarray)
                isArray = isArray or (isinstance(data, h5py.Dataset) and data.shape != tuple())

                if isArray:
                    dimensionCount = len(data.shape)
                    isNumeric = numpy.issubdtype(data.dtype, numpy.number)
                else:
                    dimensionCount = 0
                    isNumeric = numpy.issubdtype(data.dtype, numpy.number)
            else:
                isArray = False
                isNumeric = False
                dimensionCount = None

            self.__button1D.setVisible(dimensionCount >= 1 and isNumeric)
            self.__button2D.setVisible(dimensionCount >= 2 and isNumeric)
            self.__buttonArray.setVisible(dimensionCount >= 2)
            self.__buttonText.setVisible(dimensionCount == 0)
