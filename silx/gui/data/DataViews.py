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
"""This module defines a views used by :class:`silx.gui.data.DataViewer`.
"""

import logging
import numbers
import numpy

import silx.io
from silx.gui import qt, icons
from silx.gui.data.TextFormatter import TextFormatter
from silx.io import nxdata
from silx.gui.hdf5 import H5Node
from silx.io.nxdata import NXdata

__authors__ = ["V. Valls", "P. Knobel"]
__license__ = "MIT"
__date__ = "07/04/2017"

_logger = logging.getLogger(__name__)


# DataViewer modes
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


def _normalizeData(data):
    """Returns a normalized data.

    If the data embed a numpy data or a dataset it is returned.
    Else returns the input data."""
    if isinstance(data, H5Node):
        return data.h5py_object
    return data


def _normalizeComplex(data):
    """Returns a normalized complex data.

    If the data is a numpy data with complex, returns the
    absolute value.
    Else returns the input data."""
    if hasattr(data, "dtype"):
        isComplex = numpy.issubdtype(data.dtype, numpy.complex)
    else:
        isComplex = isinstance(data, numbers.Complex)
    if isComplex:
        data = numpy.absolute(data)
    return data


class DataInfo(object):
    """Store extracted information from a data"""

    def __init__(self, data):
        data = self.normalizeData(data)
        self.isArray = False
        self.interpretation = None
        self.isNumeric = False
        self.isComplex = False
        self.isRecord = False
        self.isNXdata = False
        self.shape = tuple()
        self.dim = 0

        if data is None:
            return

        if silx.io.is_group(data) and nxdata.is_valid_nxdata(data):
            self.isNXdata = True
            nxd = nxdata.NXdata(data)

        if isinstance(data, numpy.ndarray):
            self.isArray = True
        elif silx.io.is_dataset(data) and data.shape != tuple():
            self.isArray = True
        else:
            self.isArray = False

        if silx.io.is_dataset(data):
            self.interpretation = data.attrs.get("interpretation", None)
        elif self.isNXdata:
            self.interpretation = nxd.interpretation
        else:
            self.interpretation = None

        if hasattr(data, "dtype"):
            self.isNumeric = numpy.issubdtype(data.dtype, numpy.number)
            self.isRecord = data.dtype.fields is not None
            self.isComplex = numpy.issubdtype(data.dtype, numpy.complex)
        elif self.isNXdata:
            self.isNumeric = numpy.issubdtype(nxd.signal.dtype,
                                              numpy.number)
            self.isComplex = numpy.issubdtype(nxd.signal.dtype, numpy.complex)
        else:
            self.isNumeric = isinstance(data, numbers.Number)
            self.isComplex = isinstance(data, numbers.Complex)
            self.isRecord = False

        if hasattr(data, "shape"):
            self.shape = data.shape
        elif self.isNXdata:
            self.shape = nxd.signal.shape
        else:
            self.shape = tuple()
        self.dim = len(self.shape)

    def normalizeData(self, data):
        """Returns a normalized data if the embed a numpy or a dataset.
        Else returns the data."""
        return _normalizeData(data)


class DataView(object):
    """Holder for the data view."""

    UNSUPPORTED = -1
    """Priority returned when the requested data can't be displayed by the
    view."""

    def __init__(self, parent, modeId=None, icon=None, label=None):
        """Constructor

        :param qt.QWidget parent: Parent of the hold widget
        """
        self.__parent = parent
        self.__widget = None
        self.__modeId = modeId
        if label is None:
            label = self.__class__.__name__
        self.__label = label
        if icon is None:
            icon = qt.QIcon()
        self.__icon = icon

    def icon(self):
        """Returns the default icon"""
        return self.__icon

    def label(self):
        """Returns the default label"""
        return self.__label

    def modeId(self):
        """Returns the mode id"""
        return self.__modeId

    def normalizeData(self, data):
        """Returns a normalized data if the embed a numpy or a dataset.
        Else returns the data."""
        return _normalizeData(data)

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

    def isWidgetInitialized(self):
        """Returns true if the widget is already initialized.
        """
        return self.__widget is not None

    def select(self):
        """Called when the view is selected to display the data.
        """
        return

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

    def axesNames(self, data, info):
        """Returns names of the expected axes of the view, according to the
        input data.

        :param data: Data to display
        :type data: numpy.ndarray or h5py.Dataset
        :param DataInfo info: Pre-computed information on the data
        :rtype: list[str]
        """
        return []

    def getDataPriority(self, data, info):
        """
        Returns the priority of using this view according to a data.

        - `UNSUPPORTED` means this view can't display this data
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


class CompositeDataView(DataView):
    """Data view which can display a data using different view according to
    the kind of the data."""

    def __init__(self, parent, modeId=None, icon=None, label=None):
        """Constructor

        :param qt.QWidget parent: Parent of the hold widget
        """
        super(CompositeDataView, self).__init__(parent, modeId, icon, label)
        self.__views = {}
        self.__currentView = None

    def addView(self, dataView):
        """Add a new dataview to the available list."""
        self.__views[dataView] = None

    def getBestView(self, data, info):
        """Returns the best view according to priorities."""
        info = DataInfo(data)
        views = [(v.getDataPriority(data, info), v) for v in self.__views.keys()]
        views = filter(lambda t: t[0] > DataView.UNSUPPORTED, views)
        views = sorted(views, reverse=True)

        if len(views) == 0:
            return None
        elif views[0][0] == DataView.UNSUPPORTED:
            return None
        else:
            return views[0][1]

    def customAxisNames(self):
        if self.__currentView is None:
            return
        return self.__currentView.customAxisNames()

    def setCustomAxisValue(self, name, value):
        if self.__currentView is None:
            return
        self.__currentView.setCustomAxisValue(name, value)

    def __updateDisplayedView(self):
        widget = self.getWidget()
        if self.__currentView is None:
            return

        # load the widget if it is not yet done
        index = self.__views[self.__currentView]
        if index is None:
            w = self.__currentView.getWidget()
            index = widget.addWidget(w)
            self.__views[self.__currentView] = index
        if widget.currentIndex() != index:
            widget.setCurrentIndex(index)
            self.__currentView.select()

    def select(self):
        self.__updateDisplayedView()
        if self.__currentView is not None:
            self.__currentView.select()

    def createWidget(self, parent):
        return qt.QStackedWidget()

    def clear(self):
        for v in self.__views.keys():
            v.clear()

    def setData(self, data):
        if self.__currentView is None:
            return
        self.__updateDisplayedView()
        self.__currentView.setData(data)

    def axesNames(self, data, info):
        view = self.getBestView(data, info)
        self.__currentView = view
        return view.axesNames(data, info)

    def getDataPriority(self, data, info):
        view = self.getBestView(data, info)
        self.__currentView = view
        if view is None:
            return DataView.UNSUPPORTED
        else:
            return view.getDataPriority(data, info)


class _EmptyView(DataView):
    """Dummy view to display nothing"""

    def __init__(self, parent):
        DataView.__init__(self, parent, modeId=EMPTY_MODE)

    def axesNames(self, data, info):
        return []

    def createWidget(self, parent):
        return qt.QLabel(parent)

    def getDataPriority(self, data, info):
        return DataView.UNSUPPORTED


class _Plot1dView(DataView):
    """View displaying data using a 1d plot"""

    def __init__(self, parent):
        super(_Plot1dView, self).__init__(
            parent=parent,
            modeId=PLOT1D_MODE,
            label="Curve",
            icon=icons.getQIcon("view-1d"))
        self.__resetZoomNextTime = True

    def createWidget(self, parent):
        from silx.gui import plot
        return plot.Plot1D(parent=parent)

    def clear(self):
        self.getWidget().clear()
        self.__resetZoomNextTime = True

    def normalizeData(self, data):
        data = DataView.normalizeData(self, data)
        data = _normalizeComplex(data)
        return data

    def setData(self, data):
        data = self.normalizeData(data)
        self.getWidget().addCurve(legend="data",
                                  x=range(len(data)),
                                  y=data,
                                  resetzoom=self.__resetZoomNextTime)
        self.__resetZoomNextTime = True

    def axesNames(self, data, info):
        return ["y"]

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

    def __init__(self, parent):
        super(_Plot2dView, self).__init__(
            parent=parent,
            modeId=PLOT2D_MODE,
            label="Image",
            icon=icons.getQIcon("view-2d"))
        self.__resetZoomNextTime = True

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

    def normalizeData(self, data):
        data = DataView.normalizeData(self, data)
        data = _normalizeComplex(data)
        return data

    def setData(self, data):
        data = self.normalizeData(data)
        self.getWidget().addImage(legend="data",
                                  data=data,
                                  resetzoom=self.__resetZoomNextTime)
        self.__resetZoomNextTime = False

    def axesNames(self, data, info):
        return ["y", "x"]

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

    def __init__(self, parent):
        super(_Plot3dView, self).__init__(
            parent=parent,
            modeId=PLOT3D_MODE,
            label="Cube",
            icon=icons.getQIcon("view-3d"))
        try:
            import silx.gui.plot3d  #noqa
        except ImportError:
            _logger.warning("Plot3dView is not available")
            _logger.debug("Backtrace", exc_info=True)
            raise
        self.__resetZoomNextTime = True

    def createWidget(self, parent):
        from silx.gui.plot3d import ScalarFieldView
        from silx.gui.plot3d import SFViewParamTree

        plot = ScalarFieldView.ScalarFieldView(parent)
        plot.setAxesLabels(*reversed(self.axesNames(None, None)))
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

    def normalizeData(self, data):
        data = DataView.normalizeData(self, data)
        data = _normalizeComplex(data)
        return data

    def setData(self, data):
        data = self.normalizeData(data)
        plot = self.getWidget()
        plot.setData(data)
        self.__resetZoomNextTime = False

    def axesNames(self, data, info):
        return ["z", "y", "x"]

    def getDataPriority(self, data, info):
        if data is None or not info.isArray or not info.isNumeric:
            return DataView.UNSUPPORTED
        if info.dim < 3:
            return DataView.UNSUPPORTED
        if min(data.shape) < 2:
            return DataView.UNSUPPORTED
        if info.dim == 3:
            return 100
        else:
            return 10


class _ArrayView(DataView):
    """View displaying data using a 2d table"""

    def __init__(self, parent):
        DataView.__init__(self, parent, modeId=RAW_ARRAY_MODE)

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

    def axesNames(self, data, info):
        return ["col", "row"]

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

    def __init__(self, parent):
        super(_StackView, self).__init__(
            parent=parent,
            modeId=STACK_MODE,
            label="Image stack",
            icon=icons.getQIcon("view-2d-stack"))
        self.__resetZoomNextTime = True

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
        widget.setLabels(self.axesNames(None, None))
        # hide default option panel
        widget.setOptionVisible(False)
        return widget

    def clear(self):
        self.getWidget().clear()
        self.__resetZoomNextTime = True

    def normalizeData(self, data):
        data = DataView.normalizeData(self, data)
        data = _normalizeComplex(data)
        return data

    def setData(self, data):
        data = self.normalizeData(data)
        self.getWidget().setStack(stack=data, reset=self.__resetZoomNextTime)
        self.__resetZoomNextTime = False

    def axesNames(self, data, info):
        return ["depth", "y", "x"]

    def getDataPriority(self, data, info):
        if data is None or not info.isArray or not info.isNumeric:
            return DataView.UNSUPPORTED
        if info.dim < 3:
            return DataView.UNSUPPORTED
        if info.interpretation == "image":
            return 500
        return 90


class _ScalarView(DataView):
    """View displaying data using text"""

    def __init__(self, parent):
        DataView.__init__(self, parent, modeId=RAW_SCALAR_MODE)

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

    def axesNames(self, data, info):
        return []

    def getDataPriority(self, data, info):
        data = self.normalizeData(data)
        if data is None:
            return DataView.UNSUPPORTED
        if silx.io.is_group(data):
            return DataView.UNSUPPORTED
        return 2


class _RecordView(DataView):
    """View displaying data using text"""

    def __init__(self, parent):
        DataView.__init__(self, parent, modeId=RAW_RECORD_MODE)

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

    def axesNames(self, data, info):
        return ["data"]

    def getDataPriority(self, data, info):
        if info.isRecord:
            return 40
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

    def __init__(self, parent):
        super(_Hdf5View, self).__init__(
            parent=parent,
            modeId=HDF5_MODE,
            label="HDF5",
            icon=icons.getQIcon("view-hdf5"))

    def createWidget(self, parent):
        from .Hdf5TableView import Hdf5TableView
        widget = Hdf5TableView(parent)
        return widget

    def clear(self):
        widget = self.getWidget()
        widget.setData(None)

    def setData(self, data):
        widget = self.getWidget()
        widget.setData(data)

    def axesNames(self, data, info):
        return []

    def getDataPriority(self, data, info):
        widget = self.getWidget()
        if widget.isSupportedData(data):
            return 1
        else:
            return DataView.UNSUPPORTED


class _RawView(CompositeDataView):
    """View displaying data as raw data.

    This implementation use a 2d-array view, or a record array view, or a
    raw text output.
    """

    def __init__(self, parent):
        super(_RawView, self).__init__(
            parent=parent,
            modeId=RAW_MODE,
            label="Raw",
            icon=icons.getQIcon("view-raw"))
        self.addView(_ScalarView(parent))
        self.addView(_ArrayView(parent))
        self.addView(_RecordView(parent))


class _NXdataScalarView(DataView):
    """DataView using a table view for displaying NXdata scalars:
    0-D signal or n-D signal with *@interpretation=scalar*"""
    def __init__(self, parent):
        DataView.__init__(self, parent)

    def createWidget(self, parent):
        from silx.gui.data.ArrayTableWidget import ArrayTableWidget
        widget = ArrayTableWidget(parent)
        # widget.displayAxesSelector(False)
        return widget

    def axesNames(self, data, info):
        return ["col", "row"]

    def clear(self):
        self.getWidget().setArrayData(numpy.array([[]]),
                                      labels=True)

    def setData(self, data):
        data = self.normalizeData(data)
        signal = NXdata(data).signal
        self.getWidget().setArrayData(signal,
                                      labels=True)

    def getDataPriority(self, data, info):
        data = self.normalizeData(data)
        if info.isNXdata:
            nxd = NXdata(data)
            if nxd.signal_is_0d or nxd.interpretation in ["scalar", "scaler"]:
                return 100
        return DataView.UNSUPPORTED


class _NXdataCurveView(DataView):
    """DataView using a Plot1D for displaying NXdata curves:
    1-D signal or n-D signal with *@interpretation=spectrum*.

    It also handles basic scatter plots:
    a 1-D signal with one axis whose values are not monotonically increasing.
    """
    def __init__(self, parent):
        DataView.__init__(self, parent)

    def createWidget(self, parent):
        from silx.gui.data.NXdataWidgets import ArrayCurvePlot
        widget = ArrayCurvePlot(parent)
        return widget

    def axesNames(self, data, info):
        # disabled (used by default axis selector widget in Hdf5Viewer)
        return []

    def clear(self):
        self.getWidget().clear()

    def setData(self, data):
        data = self.normalizeData(data)
        nxd = NXdata(data)
        signal_name = data.attrs["signal"]
        group_name = data.name
        if nxd.axes_names[-1] is not None:
            x_errors = nxd.get_axis_errors(nxd.axes_names[-1])
        else:
            x_errors = None

        self.getWidget().setCurveData(nxd.signal, nxd.axes[-1],
                                      yerror=nxd.errors, xerror=x_errors,
                                      ylabel=signal_name, xlabel=nxd.axes_names[-1],
                                      title="NXdata group " + group_name)

    def getDataPriority(self, data, info):
        data = self.normalizeData(data)
        if info.isNXdata:
            nxd = NXdata(data)
            if nxd.is_x_y_value_scatter or nxd.is_unsupported_scatter:
                return DataView.UNSUPPORTED
            if nxd.signal_is_1d and \
                    not nxd.interpretation in ["scalar", "scaler"]:
                return 100
            if nxd.interpretation == "spectrum":
                return 100
        return DataView.UNSUPPORTED


class _NXdataXYVScatterView(DataView):
    """DataView using a Plot1D for displaying NXdata 3D scatters as
    a scatter of coloured points (1-D signal with 2 axes)"""
    def __init__(self, parent):
        DataView.__init__(self, parent)

    def createWidget(self, parent):
        from silx.gui.data.NXdataWidgets import ArrayCurvePlot
        widget = ArrayCurvePlot(parent)
        return widget

    def axesNames(self, data, info):
        # disabled (used by default axis selector widget in Hdf5Viewer)
        return []

    def clear(self):
        self.getWidget().clear()

    def setData(self, data):
        data = self.normalizeData(data)
        nxd = NXdata(data)
        signal_name = data.attrs["signal"]
        # signal_errors = nx.errors   # not supported
        group_name = data.name
        x_axis, y_axis = nxd.axes[-2:]

        x_label, y_label = nxd.axes_names[-2:]
        if x_label is not None:
            x_errors = nxd.get_axis_errors(x_label)
        else:
            x_errors = None

        if y_label is not None:
            y_errors = nxd.get_axis_errors(y_label)
        else:
            y_errors = None

        self.getWidget().setCurveData(y_axis, x_axis, values=nxd.signal,
                                      yerror=y_errors, xerror=x_errors,
                                      ylabel=signal_name, xlabel=x_label,
                                      title="NXdata group " + group_name)

    def getDataPriority(self, data, info):
        data = self.normalizeData(data)
        if info.isNXdata:
            if NXdata(data).is_x_y_value_scatter:
                return 100
        return DataView.UNSUPPORTED


class _NXdataImageView(DataView):
    """DataView using a Plot2D for displaying NXdata images:
    2-D signal or n-D signals with *@interpretation=spectrum*."""
    def __init__(self, parent):
        DataView.__init__(self, parent)

    def createWidget(self, parent):
        from silx.gui.data.NXdataWidgets import ArrayImagePlot
        widget = ArrayImagePlot(parent)
        return widget

    def axesNames(self, data, info):
        return []

    def clear(self):
        self.getWidget().clear()

    def setData(self, data):
        data = self.normalizeData(data)
        nxd = NXdata(data)
        signal_name = data.attrs["signal"]
        group_name = data.name
        y_axis, x_axis = nxd.axes[-2:]
        y_label, x_label = nxd.axes_names[-2:]

        self.getWidget().setImageData(
                     nxd.signal, x_axis=x_axis, y_axis=y_axis,
                     signal_name=signal_name, xlabel=x_label, ylabel=y_label,
                     title="NXdata group %s: %s" % (group_name, signal_name))

    def getDataPriority(self, data, info):
        data = self.normalizeData(data)
        if info.isNXdata:
            nxd = NXdata(data)
            if nxd.signal_is_2d:
                if nxd.interpretation not in ["scalar", "spectrum", "scaler"]:
                    return 100
            if nxd.interpretation == "image":
                return 100
        return DataView.UNSUPPORTED


class _NXdataStackView(DataView):
    def __init__(self, parent):
        DataView.__init__(self, parent)

    def createWidget(self, parent):
        from silx.gui.data.NXdataWidgets import ArrayStackPlot
        widget = ArrayStackPlot(parent)
        return widget

    def axesNames(self, data, info):
        return []

    def clear(self):
        self.getWidget().clear()

    def setData(self, data):
        data = self.normalizeData(data)
        nxd = NXdata(data)
        signal_name = data.attrs["signal"]
        group_name = data.name
        z_axis, y_axis, x_axis = nxd.axes[-3:]
        z_label, y_label, x_label = nxd.axes_names[-3:]

        self.getWidget().setStackData(
                     nxd.signal, x_axis=x_axis, y_axis=y_axis, z_axis=z_axis,
                     signal_name=signal_name,
                     xlabel=x_label, ylabel=y_label, zlabel=z_label,
                     title="NXdata group %s: %s" % (group_name, signal_name))

    def getDataPriority(self, data, info):
        data = self.normalizeData(data)
        if info.isNXdata:
            nxd = NXdata(data)
            if nxd.signal_ndim >= 3:
                if nxd.interpretation not in ["scalar", "scaler",
                                              "spectrum", "image"]:
                    return 100
        return DataView.UNSUPPORTED


class _NXdataView(CompositeDataView):
    """Composite view displaying NXdata groups using the most adequate
    widget depending on the dimensionality."""
    def __init__(self, parent):
        super(_NXdataView, self).__init__(
            parent=parent,
            label="NXdata",
            icon=icons.getQIcon("view-nexus"))

        self.addView(_NXdataScalarView(parent))
        self.addView(_NXdataCurveView(parent))
        self.addView(_NXdataXYVScatterView(parent))
        self.addView(_NXdataImageView(parent))
        self.addView(_NXdataStackView(parent))
