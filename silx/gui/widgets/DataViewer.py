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

try:
    import h5py
except ImportError:
    h5py = None


from silx.gui import qt
from silx.gui import plot
from silx.gui.widgets.ArrayTableWidget import ArrayTableWidget
from silx.gui.widgets.NumpyAxesSelector import NumpyAxesSelector


class DataView(object):
    """Holder for the data view.
    """

    def __init__(self, parent):
        """Constructor

        :param qt.QWidget parent: Parent of the hold widget
        """
        self.__parent = parent
        self.__widget = None

    def axiesNames(self):
        """Returns names of the expected axes of the view"""
        return []

    def getWidget(self):
        """Returns the widget hold in the view and displaying the data.

        :returns: qt.QWidget
        """
        if self.__widget is None:
            self.__widget = self.createWidget(self.__parent)
        return self.__widget

    def createWidget(self, parent):
        """Create the the widget displaying the data

        :param qt.QWidget parent: Parent of the widget
        :returns: qt.QWidget
        """
        raise NotImplementedError()

    def clear(self):
        """Clear the data from the view"""
        return None

    def setData(self, data):
        """Set the data displayed by the view

        :param data: Data to display
        :type data: numpy.ndarray or h5py.Dataset
        """
        return None


class _EmptyView(DataView):
    """Dummy view to display nothing"""

    def axiesNames(self):
        return []

    def createWidget(self, parent):
        return qt.QLabel(parent)


class _Plot1dView(DataView):
    """View displaying data using a 1d plot"""

    def __init__(self, parent):
        super(_Plot1dView, self).__init__(parent)
        self.__resetZoomNextTime = True

    def axiesNames(self):
        return ["y"]

    def createWidget(self, parent):
        return plot.Plot1D(parent=parent)

    def clear(self):
        self.getWidget().clear()
        self.__resetZoomNextTime = True

    def setData(self, data):
        self.getWidget().addCurve(legend="data",
                                  x=range(len(data)),
                                  y=data,
                                  resetzoom=self.__resetZoomNextTime)
        self.__resetZoomNextTime = False


class _Plot2dView(DataView):
    """View displaying data using a 2d plot"""

    def __init__(self, parent):
        super(_Plot2dView, self).__init__(parent)
        self.__resetZoomNextTime = True

    def axiesNames(self):
        return ["y", "x"]

    def createWidget(self, parent):
        widget = plot.Plot2D(parent=parent)
        widget.setKeepDataAspectRatio(True)
        widget.setGraphXLabel('X')
        widget.setGraphYLabel('Y')
        return widget

    def clear(self):
        self.getWidget().clear()
        self.__resetZoomNextTime = True

    def setData(self, data):
        self.getWidget().addImage(legend="data",
                                  data=data,
                                  resetzoom=self.__resetZoomNextTime)
        self.__resetZoomNextTime = False


class _ArrayView(DataView):
    """View displaying data using a 2d table"""

    def axiesNames(self):
        return ["col", "row"]

    def createWidget(self, parent):
        widget = ArrayTableWidget(parent)
        widget.displayAxesSelector(False)
        return widget

    def clear(self):
        self.getWidget().setArrayData(numpy.array([[]]))

    def setData(self, data):
        self.getWidget().setArrayData(data)


class _TextView(DataView):
    """View displaying data using text"""

    def axiesNames(self):
        return []

    def createWidget(self, parent):
        widget = qt.QLabel(parent)
        widget.setTextInteractionFlags(qt.Qt.TextSelectableByMouse)
        widget.setAlignment(qt.Qt.AlignCenter)
        return widget

    def clear(self):
        self.getWidget().setText("")

    def setData(self, data):
        if isinstance(data, h5py.Dataset):
            data = data[...]
        self.getWidget().setText(str(data))


class DataViewer(qt.QFrame):
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

        self.__stack = qt.QStackedWidget(self)
        self.__numpySelection = NumpyAxesSelector(self)
        self.__numpySelection.selectionChanged.connect(self.__numpySelectionChanged)

        self.setLayout(qt.QVBoxLayout(self))
        self.layout().addWidget(self.__stack, 1)

        group = qt.QGroupBox(self)
        group.setLayout(qt.QVBoxLayout())
        group.layout().addWidget(self.__numpySelection)
        group.setTitle("Axis selection")
        self.__axisSelection = group

        self.layout().addWidget(self.__axisSelection)

        self.__displayMode = None
        self.__data = None

        self.__views = {}
        self.__views[self.EMPTY_MODE] = _EmptyView(self.__stack)
        self.__views[self.PLOT1D_MODE] = _Plot1dView(self.__stack)
        self.__views[self.PLOT2D_MODE] = _Plot2dView(self.__stack)
        self.__views[self.TEXT_MODE] = _TextView(self.__stack)
        self.__views[self.ARRAY_MODE] = _ArrayView(self.__stack)

        # feed the stack widget
        self.__index = {}
        for modeId, view in self.__views.items():
            widget = view.getWidget()
            index = self.__stack.addWidget(widget)
            self.__index[modeId] = index

        self.displayNothing()

    def viewAxisExpected(self, displayMode):
        view = self.__views[displayMode]
        return len(view.axiesNames())

    def clear(self):
        self.setData(None)

    def __clearCurrentView(self):
        view = self.__views.get(self.__displayMode, None)
        if view is not None:
            view.clear()

    def __updateNumpySelectionAxis(self):
        self.__numpySelection.clear()
        view = self.__views[self.__displayMode]
        axisNames = view.axiesNames()
        if len(axisNames) > 0:
            previous = self.__numpySelection.blockSignals(True)
            self.__axisSelection.setVisible(True)
            self.__numpySelection.setAxisNames(axisNames)
            self.__numpySelection.setData(self.__data)
            self.__numpySelection.blockSignals(previous)
        else:
            self.__axisSelection.setVisible(False)

    def __updateDataInView(self):
        if self.__numpySelection.isVisible():
            data = self.__numpySelection.selectedData()
        else:
            data = self.__data

        view = self.__views[self.__displayMode]
        view.setData(data)

    def setDisplayMode(self, modeId):
        if self.__displayMode == modeId:
            return
        self.__clearCurrentView()
        self.__displayMode = modeId
        self.__updateNumpySelectionAxis()
        self.__updateDataInView()
        index = self.__index[modeId]
        self.__stack.setCurrentIndex(index)
        self.displayModeChanged.emit(modeId)

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
        self.__updateNumpySelectionAxis()
        self.__updateDataInView()

    def __numpySelectionChanged(self):
        self.__updateDataInView()

    def data(self):
        return self.__data

    def getDisplayMode(self):
        return self.__displayMode
