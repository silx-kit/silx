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
"""This module defines a widget designed to display data using to most adapted
view from available ones from silx.
"""
from __future__ import division

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "24/01/2017"

import numpy
import numbers
import logging

import silx.io
from silx.gui import qt
from silx.gui.data.NumpyAxesSelector import NumpyAxesSelector
from silx.gui.data.TextFormatter import TextFormatter
from silx.gui.hdf5 import H5Node

try:
    from silx.third_party import six
except ImportError:
    import six

_logger = logging.getLogger(__name__)


class DataInfo(object):
    """Store extracted information from a data"""

    def __init__(self, data):
        self.isArray = False
        self.interpretation = None
        self.isNumeric = False
        self.isRecord = False
        self.shape = tuple()
        self.dim = 0

        if data is None:
            return

        if isinstance(data, numpy.ndarray):
            self.isArray = True
        elif silx.io.is_dataset(data) and data.shape != tuple():
            self.isArray = True
        else:
            self.isArray = False

        if silx.io.is_dataset(data):
            self.interpretation = data.attrs.get("interpretation", None)
        else:
            self.interpretation = None

        if hasattr(data, "dtype"):
            self.isNumeric = numpy.issubdtype(data.dtype, numpy.number)
            self.isRecord = data.dtype.fields is not None
        else:
            self.isNumeric = isinstance(data, numbers.Number)
            self.isRecord = False

        if hasattr(data, "shape"):
            self.shape = data.shape
        else:
            self.shape = tuple()
        self.dim = len(self.shape)


class DataView(object):
    """Holder for the data view."""

    UNSUPPORTED = -1
    """Priority returned when the requested data can't be displayed by the
    view."""

    SUPPORTED_IF_NOTHING_BETTER = 0
    """Priority returned when the requested data can be displayed, but is
    really not convenient."""

    def __init__(self, parent, modeId):
        """Constructor

        :param qt.QWidget parent: Parent of the hold widget
        """
        self.__parent = parent
        self.__widget = None
        self.__modeId = modeId

    def modeId(self):
        """Returns the mode id"""
        return self.__modeId

    def normalizeData(self, data):
        """Returns a normalized data if the embbed a numpy or a dataset.
        Else returns the data."""
        if isinstance(data, H5Node):
            return data.h5py_object
        return data

    def axesNames(self):
        """Returns names of the expected axes of the view"""
        return []

    def customAxisNames(self):
        """Returns names of axes which can be custom by the user and provided
        to the view."""
        return []

    def setCustomAxisValue(self, name, value):
        """
        Set the value of a custom axis

        :param str name: Name of the custom axis
        :param int value: Value of the custom axis
        """
        pass

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

    def getDataPriority(self, data, info):
        """
        Returns the priority of using this view according to a data.

        - `UNSUPPORTED` means this view can't display this data
        - `SUPPORTED_IF_NOTHING_BETTER` means this view can display the data if there is no other choices
        - `1` means this view can display the data
        - `100` means this view should be used for this data
        - `1000` max value used by the views provided by silx
        - ...

        :param object data: The data to check
        :param DataInfo info: Pre-computed information on the data
        :rtype: int
        """
        return DataView.UNSUPPORTED

    def __lt__(self, other):
        return str(self) < str(other)


class _EmptyView(DataView):
    """Dummy view to display nothing"""

    def axesNames(self):
        return []

    def createWidget(self, parent):
        return qt.QLabel(parent)

    def getDataPriority(self, data, info):
        return DataView.UNSUPPORTED


class _Plot1dView(DataView):
    """View displaying data using a 1d plot"""

    def __init__(self, parent, modeId):
        super(_Plot1dView, self).__init__(parent, modeId)
        self.__resetZoomNextTime = True

    def axesNames(self):
        return ["y"]

    def createWidget(self, parent):
        from silx.gui import plot
        return plot.Plot1D(parent=parent)

    def clear(self):
        self.getWidget().clear()
        self.__resetZoomNextTime = True

    def setData(self, data):
        data = self.normalizeData(data)
        self.getWidget().addCurve(legend="data",
                                  x=range(len(data)),
                                  y=data,
                                  resetzoom=self.__resetZoomNextTime)
        self.__resetZoomNextTime = True

    def getDataPriority(self, data, info):
        if data is None or not info.isArray or not info.isNumeric:
            return DataView.UNSUPPORTED
        if info.dim < 1:
            return DataView.UNSUPPORTED
        if info.interpretation == "spectrum":
            return 1000
        if info.dim == 2 and info.shape[0] == 1:
            return 210
        if info.dim == 1:
            return 100
        else:
            return 10


class _Plot2dView(DataView):
    """View displaying data using a 2d plot"""

    def __init__(self, parent, modeId):
        super(_Plot2dView, self).__init__(parent, modeId)
        self.__resetZoomNextTime = True

    def axesNames(self):
        return ["y", "x"]

    def createWidget(self, parent):
        from silx.gui import plot
        widget = plot.Plot2D(parent=parent)
        widget.setKeepDataAspectRatio(True)
        widget.setGraphXLabel('X')
        widget.setGraphYLabel('Y')
        return widget

    def clear(self):
        self.getWidget().clear()
        self.__resetZoomNextTime = True

    def setData(self, data):
        data = self.normalizeData(data)
        self.getWidget().addImage(legend="data",
                                  data=data,
                                  resetzoom=self.__resetZoomNextTime)
        self.__resetZoomNextTime = False

    def getDataPriority(self, data, info):
        if data is None or not info.isArray or not info.isNumeric:
            return DataView.UNSUPPORTED
        if info.dim < 2:
            return DataView.UNSUPPORTED
        if info.interpretation == "image":
            return 1000
        if info.dim == 2:
            return 200
        else:
            return 190


class _Plot3dView(DataView):
    """View displaying data using a 3d plot"""

    def __init__(self, parent, modeId):
        super(_Plot3dView, self).__init__(parent, modeId)
        try:
            import silx.gui.plot3d
        except ImportError:
            _logger.warning("Plot3dView is not available")
            _logger.debug("Backtrace", exc_info=True)
            raise
        self.__resetZoomNextTime = True

    def axesNames(self):
        return ["z", "y", "x"]

    def createWidget(self, parent):
        from silx.gui.plot3d import ScalarFieldView
        from silx.gui.plot3d import SFViewParamTree

        plot = ScalarFieldView.ScalarFieldView(parent)
        plot.setAxesLabels(*reversed(self.axesNames()))
        plot.addIsosurface(
            lambda data: numpy.mean(data) + numpy.std(data), '#FF0000FF')

        # Create a parameter tree for the scalar field view
        options = SFViewParamTree.TreeView(plot)
        options.setSfView(plot)

        # Add the parameter tree to the main window in a dock widget
        dock = qt.QDockWidget()
        dock.setWidget(options)
        plot.addDockWidget(qt.Qt.RightDockWidgetArea, dock)

        return plot

    def clear(self):
        self.getWidget().setData(None)
        self.__resetZoomNextTime = True

    def setData(self, data):
        data = self.normalizeData(data)
        plot = self.getWidget()
        plot.setData(data)
        self.__resetZoomNextTime = False

    def getDataPriority(self, data, info):
        if data is None or not info.isArray or not info.isNumeric:
            return DataView.UNSUPPORTED
        if info.dim < 3:
            return DataView.UNSUPPORTED
        if info.dim == 3:
            return 100
        else:
            return 10


class _ArrayView(DataView):
    """View displaying data using a 2d table"""

    def axesNames(self):
        return ["col", "row"]

    def createWidget(self, parent):
        from silx.gui.data.ArrayTableWidget import ArrayTableWidget
        widget = ArrayTableWidget(parent)
        widget.displayAxesSelector(False)
        return widget

    def clear(self):
        self.getWidget().setArrayData(numpy.array([[]]))

    def setData(self, data):
        data = self.normalizeData(data)
        self.getWidget().setArrayData(data)

    def getDataPriority(self, data, info):
        if data is None or not info.isArray or info.isRecord:
            return DataView.UNSUPPORTED
        if info.dim < 2:
            return DataView.UNSUPPORTED
        if info.interpretation in ["scalar", "scaler"]:
            return 1000
        return 50


class _StackView(DataView):
    """View displaying data using a stack of images"""

    def __init__(self, parent, modeId):
        super(_StackView, self).__init__(parent, modeId)
        self.__resetZoomNextTime = True

    def axesNames(self):
        return ["depth", "y", "x"]

    def customAxisNames(self):
        return ["depth"]

    def setCustomAxisValue(self, name, value):
        if name == "depth":
            self.getWidget().setFrameNumber(value)
        else:
            raise Exception("Unsupported axis")

    def createWidget(self, parent):
        from silx.gui import plot
        widget = plot.StackView(parent=parent)
        widget.setKeepDataAspectRatio(True)
        widget.setLabels(self.axesNames())
        # hide default option panel
        widget.setOptionVisible(False)
        return widget

    def clear(self):
        self.getWidget().clear()
        self.__resetZoomNextTime = True

    def setData(self, data):
        data = self.normalizeData(data)
        self.getWidget().setStack(stack=data, reset=self.__resetZoomNextTime)
        self.__resetZoomNextTime = False

    def getDataPriority(self, data, info):
        if data is None or not info.isArray or not info.isNumeric:
            return DataView.UNSUPPORTED
        if info.dim < 3:
            return DataView.UNSUPPORTED
        if info.interpretation == "image":
            return 500
        return 90


class _RawView(DataView):
    """View displaying data using text"""

    def axesNames(self):
        return []

    def createWidget(self, parent):
        widget = qt.QTextEdit(parent)
        widget.setTextInteractionFlags(qt.Qt.TextSelectableByMouse)
        widget.setAlignment(qt.Qt.AlignLeft | qt.Qt.AlignTop)
        self.__formatter = TextFormatter(parent)
        return widget

    def clear(self):
        self.getWidget().setText("")

    def setData(self, data):
        data = self.normalizeData(data)
        if silx.io.is_dataset(data):
            data = data[()]
        text = self.__formatter.toString(data)
        self.getWidget().setText(text)

    def getDataPriority(self, data, info):
        if data is None:
            return DataView.UNSUPPORTED
        return DataView.SUPPORTED_IF_NOTHING_BETTER


class _RecordView(DataView):
    """View displaying data using text"""

    def axesNames(self):
        return ["data"]

    def createWidget(self, parent):
        from .RecordTableView import RecordTableView
        widget = RecordTableView(parent)
        widget.setWordWrap(False)
        return widget

    def clear(self):
        self.getWidget().setArrayData(None)

    def setData(self, data):
        data = self.normalizeData(data)
        widget = self.getWidget()
        widget.setArrayData(data)
        widget.resizeRowsToContents()
        widget.resizeColumnsToContents()

    def getDataPriority(self, data, info):
        if data is None or not info.isArray:
            return DataView.UNSUPPORTED
        if info.dim == 1:
            if info.interpretation in ["scalar", "scaler"]:
                return 1000
            if info.shape[0] == 1:
                return 110
            return 40
        elif info.isRecord:
            return 40
        return DataView.UNSUPPORTED


class _Hdf5View(DataView):
    """View displaying data using text"""

    def axesNames(self):
        return []

    def createWidget(self, parent):
        from .Hdf5TableModel import Hdf5TableModel
        widget = qt.QTableView()
        widget.setModel(Hdf5TableModel(widget))
        return widget

    def clear(self):
        self.getWidget().model().setObject(None)

    def setData(self, data):
        widget = self.getWidget()
        widget.model().setObject(data)
        header = widget.horizontalHeader()
        if qt.qVersion() < "5.0":
            setResizeMode = header.setResizeMode
        else:
            setResizeMode = header.setSectionResizeMode
        setResizeMode(0, qt.QHeaderView.Fixed)
        setResizeMode(1, qt.QHeaderView.Stretch)
        header.setStretchLastSection(True)

    def getDataPriority(self, data, info):
        widget = self.getWidget()
        if widget.model().isSupportedObject(data):
            return 1
        else:
            return DataView.UNSUPPORTED


class DataViewer(qt.QFrame):
    """Widget to display any kind of data

    .. image:: img/DataViewer.png

    The method :meth:`setData` allows to set any data to the widget. Mostly
    `numpy.array` and `h5py.Dataset` are supported with adapted views. Other
    data types are displayed using a text viewer.

    A default view is automatically selected when a data is set. The method
    :meth:`setDisplayMode` allows to change the view. To have a graphical tool
    to select the view, prefer using the widget :class:`DataViewerFrame`.

    The dimension of the input data and the expected dimension of the selected
    view can differ. For example you can display an image (2D) from 4D
    data. In this case a :class:`NumpyAxesSelector` is displayed to allow the
    user to select the axis mapping and the slicing of other axes.

    .. code-block:: python

        import numpy
        data = numpy.random.rand(500,500)
        viewer = DataViewer()
        viewer.setData(data)
        viewer.setVisible(True)
    """

    EMPTY_MODE = 0
    PLOT1D_MODE = 1
    PLOT2D_MODE = 2
    PLOT3D_MODE = 3
    RAW_MODE = 4
    ARRAY_MODE = 5
    STACK_MODE = 6
    RECORD_MODE = 7
    HDF5_MODE = 8

    displayModeChanged = qt.Signal(int)
    """Emitted when the display mode changes"""

    dataChanged = qt.Signal()
    """Emitted when the data changes"""

    currentAvailableViewsChanged = qt.Signal()
    """Emitted when the current available views (which support the current
    data) change"""

    def __init__(self, parent=None):
        """Constructor

        :param QWidget parent: The parent of the widget
        """
        super(DataViewer, self).__init__(parent)

        self.__stack = qt.QStackedWidget(self)
        self.__numpySelection = NumpyAxesSelector(self)
        self.__numpySelection.selectedAxisChanged.connect(self.__numpyAxisChanged)
        self.__numpySelection.selectionChanged.connect(self.__numpySelectionChanged)
        self.__numpySelection.customAxisChanged.connect(self.__numpyCustomAxisChanged)

        self.setLayout(qt.QVBoxLayout(self))
        self.layout().addWidget(self.__stack, 1)

        group = qt.QGroupBox(self)
        group.setLayout(qt.QVBoxLayout())
        group.layout().addWidget(self.__numpySelection)
        group.setTitle("Axis selection")
        self.__axisSelection = group

        self.layout().addWidget(self.__axisSelection)

        self.__currentAvailableViews = []
        self.__currentView = None
        self.__data = None
        self.__useAxisSelection = False

        views = [
            (_EmptyView, self.EMPTY_MODE),
            (_Hdf5View, self.HDF5_MODE),
            (_Plot1dView, self.PLOT1D_MODE),
            (_Plot2dView, self.PLOT2D_MODE),
            (_Plot3dView, self.PLOT3D_MODE),
            (_RawView, self.RAW_MODE),
            (_ArrayView, self.ARRAY_MODE),
            (_StackView, self.STACK_MODE),
            (_RecordView, self.RECORD_MODE),
        ]
        self.__views = {}
        for viewData in views:
            viewClass, modeId = viewData
            try:
                view = viewClass(self.__stack, modeId)
            except:
                continue
            self.__views[view.modeId()] = view

        # store stack index for each views
        self.__index = {}

        self.setDisplayMode(self.EMPTY_MODE)

    def clear(self):
        """Clear the widget"""
        self.setData(None)

    def normalizeData(self, data):
        """Returns a normalized data if the embbed a numpy or a dataset.
        Else returns the data."""
        if isinstance(data, H5Node):
            return data.h5py_object
        return data

    def __getStackIndex(self, view):
        """Get the stack index containing the view.

        :param DataView view: The view
        """
        if view not in self.__index:
            widget = view.getWidget()
            index = self.__stack.addWidget(widget)
            self.__index[view] = index
        else:
            index = self.__index[view]
        return index

    def __clearCurrentView(self):
        """Clear the current selected view"""
        view = self.__currentView
        if view is not None:
            view.clear()

    def __numpyCustomAxisChanged(self, name, value):
        view = self.__currentView
        if view is not None:
            view.setCustomAxisValue(name, value)

    def __updateNumpySelectionAxis(self):
        """
        Update the numpy-selector according to the needed axis names
        """
        previous = self.__numpySelection.blockSignals(True)
        self.__numpySelection.clear()
        axisNames = self.__currentView.axesNames()
        if len(axisNames) > 0:
            self.__useAxisSelection = True
            self.__numpySelection.setAxisNames(axisNames)
            self.__numpySelection.setCustomAxis(self.__currentView.customAxisNames())
            data = self.normalizeData(self.__data)
            self.__numpySelection.setData(data)
            if hasattr(data, "shape"):
                isVisible = not (len(axisNames) == 1 and len(data.shape) == 1)
            else:
                isVisible = True
            self.__axisSelection.setVisible(isVisible)
        else:
            self.__useAxisSelection = False
            self.__axisSelection.setVisible(False)
        self.__numpySelection.blockSignals(previous)

    def __updateDataInView(self):
        """
        Update the views using the current data
        """
        if self.__useAxisSelection:
            self.__displayedData = self.__numpySelection.selectedData()
        else:
            self.__displayedData = self.__data

        qt.QTimer.singleShot(10, self.__setDataInView)

    def __setDataInView(self):
        self.__currentView.setData(self.__displayedData)

    def __setDisplayedView(self, view):
        """Set the displayed view.

        Change the displayed view according to the view itself.

        :param DataView view: The DataView to use to display the data
        """
        if self.__currentView is view:
            return
        self.__clearCurrentView()
        self.__currentView = view
        self.__updateNumpySelectionAxis()
        self.__updateDataInView()
        stackIndex = self.__getStackIndex(self.__currentView)
        self.__stack.setCurrentIndex(stackIndex)
        self.displayModeChanged.emit(view.modeId())

    def setDisplayMode(self, modeId):
        """Set the displayed view using display mode.

        Change the displayed view according to the requested mode.

        :param int modeId: Display mode, one of

            - `EMPTY_MODE`: display nothing
            - `PLOT1D_MODE`: display the data as a curve
            - `PLOT2D_MODE`: display the data as an image
            - `TEXT_MODE`: display the data as a text
            - `ARRAY_MODE`: display the data as a table
        """
        try:
            view = self.__views[modeId]
        except KeyError:
            raise ValueError("Display mode %s is unknown" % modeId)
        self.__setDisplayedView(view)

    def __updateView(self):
        """Display the data using the widget which fit the best"""
        data = self.__data

        # sort available views according to priority
        info = DataInfo(data)
        priorities = [v.getDataPriority(data, info) for v in self.__views.values()]
        views = zip(priorities, self.__views.values())
        views = filter(lambda t: t[0] >= DataView.SUPPORTED_IF_NOTHING_BETTER, views)
        views = sorted(views, reverse=True)

        # store available views
        if len(views) == 0:
            self.__setCurrentAvailableViews([])
            available = []
        else:
            if views[0][0] != 0:
                # remove 0-priority, if other are available
                views = list(filter(lambda t: t[0] != DataView.SUPPORTED_IF_NOTHING_BETTER, views))
            available = [v[1] for v in views]
            self.__setCurrentAvailableViews(available)

        # display the view with the most priority (the default view)
        view = self.getDefaultViewFromAvailableViews(data, available)
        self.__clearCurrentView()
        self.__setDisplayedView(view)

    def getDefaultViewFromAvailableViews(self, data, available):
        """Returns the default view which will be used according to available
        views.

        :param object data: data which will be displayed
        :param list[view] available: List of available views, from highest
            priority to lowest.
        :rtype: DataView
        """
        if len(available) > 0:
            # returns the view with the highest priority
            view = available[0]
        else:
            # else returns the empty view
            view = self.__views[DataViewer.EMPTY_MODE]
        return view

    def __setCurrentAvailableViews(self, availableViews):
        """Set the current available viewa

        :param List[DataView] availableViews: Current available viewa
        """
        self.__currentAvailableViews = availableViews
        self.currentAvailableViewsChanged.emit()

    def currentAvailableViews(self):
        """Returns the list of available views for the current data

        :rtype: List[DataView]
        """
        return self.__currentAvailableViews

    def availableViews(self):
        """Returns the list of registered views

        :rtype: List[DataView]
        """
        return self.__views.values()

    def setData(self, data):
        """Set the data to view.

        It mostly can be a h5py.Dataset or a numpy.ndarray. Other kind of
        objects will be displayed as text rendering.

        :param numpy.ndarray data: The data.
        """
        self.__data = data
        self.__displayedData = None
        self.__updateView()
        self.__updateNumpySelectionAxis()
        self.__updateDataInView()
        self.dataChanged.emit()

    def __numpyAxisChanged(self):
        """
        Called when axis selection of the numpy-selector changed
        """
        self.__clearCurrentView()

    def __numpySelectionChanged(self):
        """
        Called when data selection of the numpy-selector changed
        """
        self.__updateDataInView()

    def data(self):
        """Returns the data"""
        return self.__data

    def displayMode(self):
        """Returns the current display mode"""
        return self.__currentView.modeId()
