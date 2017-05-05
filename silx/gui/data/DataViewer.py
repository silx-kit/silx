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

from silx.gui.data import DataViews
from silx.gui.data.DataViews import _normalizeData
import logging
from silx.gui import qt
from silx.gui.data.NumpyAxesSelector import NumpyAxesSelector

try:
    from silx.third_party import six
except ImportError:
    import six

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "10/04/2017"


_logger = logging.getLogger(__name__)


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
    PLOT1D_MODE = 10
    PLOT2D_MODE = 20
    PLOT3D_MODE = 30
    RAW_MODE = 40
    RAW_ARRAY_MODE = 41
    RAW_RECORD_MODE = 42
    RAW_SCALAR_MODE = 43
    STACK_MODE = 50
    HDF5_MODE = 60

    displayedViewChanged = qt.Signal(object)
    """Emitted when the displayed view changes"""

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
        self.__userSelectedView = None

        self.__views = []
        self.__index = {}
        """store stack index for each views"""

        self._initializeViews()

    def _initializeViews(self):
        """Inisialize the available views"""
        views = self.createDefaultViews(self.__stack)
        self.__views = list(views)
        self.setDisplayMode(self.EMPTY_MODE)

    def createDefaultViews(self, parent=None):
        """Create and returns available views which can be displayed by default
        by the data viewer. It is called internally by the widget. It can be
        overwriten to provide a different set of viewers.

        :param QWidget parent: QWidget parent of the views
        :rtype: list[silx.gui.data.DataViews.DataView]
        """
        viewClasses = [
            DataViews._EmptyView,
            DataViews._Hdf5View,
            DataViews._NXdataView,
            DataViews._Plot1dView,
            DataViews._Plot2dView,
            DataViews._Plot3dView,
            DataViews._RawView,
            DataViews._StackView,
        ]
        views = []
        for viewClass in viewClasses:
            try:
                view = viewClass(parent)
                views.append(view)
            except Exception:
                _logger.warning("%s instantiation failed. View is ignored" % viewClass.__name__)
                _logger.debug("Backtrace", exc_info=True)

        return views

    def clear(self):
        """Clear the widget"""
        self.setData(None)

    def normalizeData(self, data):
        """Returns a normalized data if the embed a numpy or a dataset.
        Else returns the data."""
        return _normalizeData(data)

    def __getStackIndex(self, view):
        """Get the stack index containing the view.

        :param silx.gui.data.DataViews.DataView view: The view
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
        info = DataViews.DataInfo(self.__data)
        axisNames = self.__currentView.axesNames(self.__data, info)
        if info.isArray and self.__data is not None and len(axisNames) > 0:
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

    def setDisplayedView(self, view):
        """Set the displayed view.

        Change the displayed view according to the view itself.

        :param silx.gui.data.DataViews.DataView view: The DataView to use to display the data
        """
        self.__userSelectedView = view
        self._setDisplayedView(view)

    def _setDisplayedView(self, view):
        """Internal set of the displayed view.

        Change the displayed view according to the view itself.

        :param silx.gui.data.DataViews.DataView view: The DataView to use to display the data
        """
        if self.__currentView is view:
            return
        self.__clearCurrentView()
        self.__currentView = view
        self.__updateNumpySelectionAxis()
        self.__updateDataInView()
        stackIndex = self.__getStackIndex(self.__currentView)
        if self.__currentView is not None:
            self.__currentView.select()
        self.__stack.setCurrentIndex(stackIndex)
        self.displayedViewChanged.emit(view)

    def getViewFromModeId(self, modeId):
        """Returns the first available view which have the requested modeId.

        :param int modeId: Requested mode id
        :rtype: silx.gui.data.DataViews.DataView
        """
        for view in self.__views:
            if view.modeId() == modeId:
                return view
        return view

    def setDisplayMode(self, modeId):
        """Set the displayed view using display mode.

        Change the displayed view according to the requested mode.

        :param int modeId: Display mode, one of

            - `EMPTY_MODE`: display nothing
            - `PLOT1D_MODE`: display the data as a curve
            - `PLOT2D_MODE`: display the data as an image
            - `PLOT3D_MODE`: display the data as an isosurface
            - `RAW_MODE`: display the data as a table
            - `STACK_MODE`: display the data as a stack of images
            - `HDF5_MODE`: display the data as a table
        """
        try:
            view = self.getViewFromModeId(modeId)
        except KeyError:
            raise ValueError("Display mode %s is unknown" % modeId)
        self._setDisplayedView(view)

    def displayedView(self):
        """Returns the current displayed view.

        :rtype: silx.gui.data.DataViews.DataView
        """
        return self.__currentView

    def addView(self, view):
        """Allow to add a view to the dataview.

        If the current data support this view, it will be displayed.

        :param DataView view: A dataview
        """
        self.__views.append(view)
        # TODO It can be skipped if the view do not support the data
        self.__updateAvailableViews()

    def removeView(self, view):
        """Allow to remove a view which was available from the dataview.

        If the view was displayed, the widget will be updated.

        :param DataView view: A dataview
        """
        self.__views.remove(view)
        self.__stack.removeWidget(view.getWidget())
        # invalidate the full index. It will be updated as expected
        self.__index = {}

        if self.__userSelectedView is view:
            self.__userSelectedView = None

        if view is self.__currentView:
            self.__updateView()
        else:
            # TODO It can be skipped if the view is not part of the
            # available views
            self.__updateAvailableViews()

    def __updateAvailableViews(self):
        """
        Update available views from the current data.
        """
        data = self.__data
        # sort available views according to priority
        info = DataViews.DataInfo(data)
        priorities = [v.getDataPriority(data, info) for v in self.__views]
        views = zip(priorities, self.__views)
        views = filter(lambda t: t[0] > DataViews.DataView.UNSUPPORTED, views)
        views = sorted(views, reverse=True)

        # store available views
        if len(views) == 0:
            self.__setCurrentAvailableViews([])
            available = []
        else:
            available = [v[1] for v in views]
            self.__setCurrentAvailableViews(available)

    def __updateView(self):
        """Display the data using the widget which fit the best"""
        data = self.__data

        # update available views for this data
        self.__updateAvailableViews()
        available = self.__currentAvailableViews

        # display the view with the most priority (the default view)
        view = self.getDefaultViewFromAvailableViews(data, available)
        self.__clearCurrentView()
        try:
            self._setDisplayedView(view)
        except Exception as e:
            # in case there is a problem to read the data, try to use a safe
            # view
            view = self.getSafeViewFromAvailableViews(data, available)
            self._setDisplayedView(view)
            raise e

    def getSafeViewFromAvailableViews(self, data, available):
        """Returns a view which is sure to display something without failing
        on rendering.

        :param object data: data which will be displayed
        :param list[view] available: List of available views, from highest
            priority to lowest.
        :rtype: DataView
        """
        hdf5View = self.getViewFromModeId(DataViewer.HDF5_MODE)
        if hdf5View in available:
            return hdf5View
        return self.getViewFromModeId(DataViewer.EMPTY_MODE)

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
            if self.__userSelectedView in available:
                return self.__userSelectedView
            self.__userSelectedView = None
            view = available[0]
        else:
            # else returns the empty view
            view = self.getViewFromModeId(DataViewer.EMPTY_MODE)
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
        return self.__views

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
