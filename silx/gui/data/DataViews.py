# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2018 European Synchrotron Radiation Facility
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

from collections import OrderedDict
import logging
import numbers
import numpy

import silx.io
from silx.gui import qt, icons
from silx.gui.data.TextFormatter import TextFormatter
from silx.io import nxdata
from silx.gui.hdf5 import H5Node
from silx.io.nxdata import get_attr_as_unicode
from silx.gui.colors import Colormap
from silx.gui.dialog.ColormapDialog import ColormapDialog

__authors__ = ["V. Valls", "P. Knobel"]
__license__ = "MIT"
__date__ = "23/05/2018"

_logger = logging.getLogger(__name__)


# DataViewer modes
EMPTY_MODE = 0
PLOT1D_MODE = 10
IMAGE_MODE = 20
PLOT2D_MODE = 21
COMPLEX_IMAGE_MODE = 22
PLOT3D_MODE = 30
RAW_MODE = 40
RAW_ARRAY_MODE = 41
RAW_RECORD_MODE = 42
RAW_SCALAR_MODE = 43
RAW_HEXA_MODE = 44
STACK_MODE = 50
HDF5_MODE = 60
NXDATA_MODE = 70
NXDATA_INVALID_MODE = 71
NXDATA_SCALAR_MODE = 72
NXDATA_CURVE_MODE = 73
NXDATA_XYVSCATTER_MODE = 74
NXDATA_IMAGE_MODE = 75
NXDATA_STACK_MODE = 76


def _normalizeData(data):
    """Returns a normalized data.

    If the data embed a numpy data or a dataset it is returned.
    Else returns the input data."""
    if isinstance(data, H5Node):
        if data.is_broken:
            return None
        return data.h5py_object
    return data


def _normalizeComplex(data):
    """Returns a normalized complex data.

    If the data is a numpy data with complex, returns the
    absolute value.
    Else returns the input data."""
    if hasattr(data, "dtype"):
        isComplex = numpy.issubdtype(data.dtype, numpy.complexfloating)
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
        self.isVoid = False
        self.isComplex = False
        self.isBoolean = False
        self.isRecord = False
        self.hasNXdata = False
        self.isInvalidNXdata = False
        self.shape = tuple()
        self.dim = 0
        self.size = 0

        if data is None:
            return

        if silx.io.is_group(data):
            nxd = nxdata.get_default(data)
            nx_class = get_attr_as_unicode(data, "NX_class")
            if nxd is not None:
                self.hasNXdata = True
                # can we plot it?
                is_scalar = nxd.signal_is_0d or nxd.interpretation in ["scalar", "scaler"]
                if not (is_scalar or nxd.is_curve or nxd.is_x_y_value_scatter or
                        nxd.is_image or nxd.is_stack):
                    # invalid: cannot be plotted by any widget
                    self.isInvalidNXdata = True
            elif nx_class == "NXdata":
                # group claiming to be NXdata could not be parsed
                self.isInvalidNXdata = True
            elif nx_class == "NXentry" and "default" in data.attrs:
                # entry claiming to have a default NXdata could not be parsed
                self.isInvalidNXdata = True
            elif nx_class == "NXroot" or silx.io.is_file(data):
                # root claiming to have a default entry
                if "default" in data.attrs:
                    def_entry = data.attrs["default"]
                    if def_entry in data and "default" in data[def_entry].attrs:
                        # and entry claims to have default NXdata
                        self.isInvalidNXdata = True

        if isinstance(data, numpy.ndarray):
            self.isArray = True
        elif silx.io.is_dataset(data) and data.shape != tuple():
            self.isArray = True
        else:
            self.isArray = False

        if silx.io.is_dataset(data):
            if "interpretation" in data.attrs:
                self.interpretation = get_attr_as_unicode(data, "interpretation")
            else:
                self.interpretation = None
        elif self.hasNXdata:
            self.interpretation = nxd.interpretation
        else:
            self.interpretation = None

        if hasattr(data, "dtype"):
            if numpy.issubdtype(data.dtype, numpy.void):
                # That's a real opaque type, else it is a structured type
                self.isVoid = data.dtype.fields is None
            self.isNumeric = numpy.issubdtype(data.dtype, numpy.number)
            self.isRecord = data.dtype.fields is not None
            self.isComplex = numpy.issubdtype(data.dtype, numpy.complexfloating)
            self.isBoolean = numpy.issubdtype(data.dtype, numpy.bool_)
        elif self.hasNXdata:
            self.isNumeric = numpy.issubdtype(nxd.signal.dtype,
                                              numpy.number)
            self.isComplex = numpy.issubdtype(nxd.signal.dtype, numpy.complexfloating)
            self.isBoolean = numpy.issubdtype(nxd.signal.dtype, numpy.bool_)
        else:
            self.isNumeric = isinstance(data, numbers.Number)
            self.isComplex = isinstance(data, numbers.Complex)
            self.isBoolean = isinstance(data, bool)
            self.isRecord = False

        if hasattr(data, "shape"):
            self.shape = data.shape
        elif self.hasNXdata:
            self.shape = nxd.signal.shape
        else:
            self.shape = tuple()
        if self.shape is not None:
            self.dim = len(self.shape)

        if hasattr(data, "shape") and data.shape is None:
            # This test is expected to avoid to fall done on the h5py issue
            # https://github.com/h5py/h5py/issues/1044
            self.size = 0
        elif hasattr(data, "size"):
            self.size = int(data.size)
        else:
            self.size = 1

    def normalizeData(self, data):
        """Returns a normalized data if the embed a numpy or a dataset.
        Else returns the data."""
        return _normalizeData(data)


class DataViewHooks(object):
    """A set of hooks defined to custom the behaviour of the data views."""

    def getColormap(self, view):
        """Returns a colormap for this view."""
        return None

    def getColormapDialog(self, view):
        """Returns a color dialog for this view."""
        return None


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
        self.__hooks = None

    def getHooks(self):
        """Returns the data viewer hooks used by this view.

        :rtype: DataViewHooks
        """
        return self.__hooks

    def setHooks(self, hooks):
        """Set the data view hooks to use with this view.

        :param DataViewHooks hooks: The data view hooks to use
        """
        self.__hooks = hooks

    def defaultColormap(self):
        """Returns a default colormap.

        :rtype: Colormap
        """
        colormap = None
        if self.__hooks is not None:
            colormap = self.__hooks.getColormap(self)
        if colormap is None:
            colormap = Colormap(name="viridis")
        return colormap

    def defaultColorDialog(self):
        """Returns a default color dialog.

        :rtype: ColormapDialog
        """
        dialog = None
        if self.__hooks is not None:
            dialog = self.__hooks.getColormapDialog(self)
        if dialog is None:
            dialog = ColormapDialog()
            dialog.setModal(False)
        return dialog

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
        input data. A none value will disable the default axes selectior.

        :param data: Data to display
        :type data: numpy.ndarray or h5py.Dataset
        :param DataInfo info: Pre-computed information on the data
        :rtype: list[str] or None
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
        self.__views = OrderedDict()
        self.__currentView = None

    def setHooks(self, hooks):
        """Set the data context to use with this view.

        :param DataViewHooks hooks: The data view hooks to use
        """
        super(CompositeDataView, self).setHooks(hooks)
        if hooks is not None:
            for v in self.__views:
                v.setHooks(hooks)

    def addView(self, dataView):
        """Add a new dataview to the available list."""
        hooks = self.getHooks()
        if hooks is not None:
            dataView.setHooks(hooks)
        self.__views[dataView] = None

    def availableViews(self):
        """Returns the list of registered views

        :rtype: List[DataView]
        """
        return list(self.__views.keys())

    def getBestView(self, data, info):
        """Returns the best view according to priorities."""
        views = [(v.getDataPriority(data, info), v) for v in self.__views.keys()]
        views = filter(lambda t: t[0] > DataView.UNSUPPORTED, views)
        views = sorted(views, key=lambda t: t[0], reverse=True)

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

    def replaceView(self, modeId, newView):
        """Replace a data view with a custom view.
        Return True in case of success, False in case of failure.

        .. note::

            This method must be called just after instantiation, before
            the viewer is used.

        :param int modeId: Unique mode ID identifying the DataView to
            be replaced.
        :param DataViews.DataView newView: New data view
        :return: True if replacement was successful, else False
        """
        oldView = None
        for view in self.__views:
            if view.modeId() == modeId:
                oldView = view
                break
            elif isinstance(view, CompositeDataView):
                # recurse
                hooks = self.getHooks()
                if hooks is not None:
                    newView.setHooks(hooks)
                if view.replaceView(modeId, newView):
                    return True
        if oldView is None:
            return False

        # replace oldView with new view in dict
        self.__views = OrderedDict(
                (newView, None) if view is oldView else (view, idx) for
                view, idx in self.__views.items())
        return True


class _EmptyView(DataView):
    """Dummy view to display nothing"""

    def __init__(self, parent):
        DataView.__init__(self, parent, modeId=EMPTY_MODE)

    def axesNames(self, data, info):
        return None

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
        if info.size <= 0:
            return DataView.UNSUPPORTED
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
        widget.setDefaultColormap(self.defaultColormap())
        widget.getColormapAction().setColorDialog(self.defaultColorDialog())
        widget.getIntensityHistogramAction().setVisible(True)
        widget.setKeepDataAspectRatio(True)
        widget.getXAxis().setLabel('X')
        widget.getYAxis().setLabel('Y')
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
        if info.size <= 0:
            return DataView.UNSUPPORTED
        if (data is None or
                not info.isArray or
                not (info.isNumeric or info.isBoolean)):
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

        def computeIsolevel(data):
            data = data[numpy.isfinite(data)]
            if len(data) == 0:
                return 0
            else:
                return numpy.mean(data) + numpy.std(data)

        plot.addIsosurface(computeIsolevel, '#FF0000FF')

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
        if info.size <= 0:
            return DataView.UNSUPPORTED
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


class _ComplexImageView(DataView):
    """View displaying data using a ComplexImageView"""

    def __init__(self, parent):
        super(_ComplexImageView, self).__init__(
            parent=parent,
            modeId=COMPLEX_IMAGE_MODE,
            label="Complex Image",
            icon=icons.getQIcon("view-2d"))

    def createWidget(self, parent):
        from silx.gui.plot.ComplexImageView import ComplexImageView
        widget = ComplexImageView(parent=parent)
        widget.setColormap(self.defaultColormap(), mode=ComplexImageView.Mode.ABSOLUTE)
        widget.setColormap(self.defaultColormap(), mode=ComplexImageView.Mode.SQUARE_AMPLITUDE)
        widget.setColormap(self.defaultColormap(), mode=ComplexImageView.Mode.REAL)
        widget.setColormap(self.defaultColormap(), mode=ComplexImageView.Mode.IMAGINARY)
        widget.getPlot().getColormapAction().setColorDialog(self.defaultColorDialog())
        widget.getPlot().getIntensityHistogramAction().setVisible(True)
        widget.getPlot().setKeepDataAspectRatio(True)
        widget.getXAxis().setLabel('X')
        widget.getYAxis().setLabel('Y')
        return widget

    def clear(self):
        self.getWidget().setData(None)

    def normalizeData(self, data):
        data = DataView.normalizeData(self, data)
        return data

    def setData(self, data):
        data = self.normalizeData(data)
        self.getWidget().setData(data)

    def axesNames(self, data, info):
        return ["y", "x"]

    def getDataPriority(self, data, info):
        if info.size <= 0:
            return DataView.UNSUPPORTED
        if data is None or not info.isArray or not info.isComplex:
            return DataView.UNSUPPORTED
        if info.dim < 2:
            return DataView.UNSUPPORTED
        if info.interpretation == "image":
            return 1000
        if info.dim == 2:
            return 200
        else:
            return 190


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
        if info.size <= 0:
            return DataView.UNSUPPORTED
        if data is None or not info.isArray or info.isRecord:
            return DataView.UNSUPPORTED
        if info.dim < 2:
            return DataView.UNSUPPORTED
        if info.interpretation in ["scalar", "scaler"]:
            return 1000
        return 500


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
        widget.setColormap(self.defaultColormap())
        widget.getPlot().getColormapAction().setColorDialog(self.defaultColorDialog())
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
        # Override the colormap, while setStack overwrite it
        self.getWidget().setColormap(self.defaultColormap())
        self.__resetZoomNextTime = False

    def axesNames(self, data, info):
        return ["depth", "y", "x"]

    def getDataPriority(self, data, info):
        if info.size <= 0:
            return DataView.UNSUPPORTED
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
        d = self.normalizeData(data)
        if silx.io.is_dataset(d):
            d = d[()]
        dtype = None
        if data is not None:
            if hasattr(data, "dtype"):
                dtype = data.dtype
        text = self.__formatter.toString(d, dtype)
        self.getWidget().setText(text)

    def axesNames(self, data, info):
        return []

    def getDataPriority(self, data, info):
        if info.size <= 0:
            return DataView.UNSUPPORTED
        data = self.normalizeData(data)
        if info.shape is None:
            return DataView.UNSUPPORTED
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
        if len(data) < 100:
            widget.resizeRowsToContents()
            widget.resizeColumnsToContents()

    def axesNames(self, data, info):
        return ["data"]

    def getDataPriority(self, data, info):
        if info.size <= 0:
            return DataView.UNSUPPORTED
        if info.isRecord:
            return 40
        if data is None or not info.isArray:
            return DataView.UNSUPPORTED
        if info.dim == 1:
            if info.interpretation in ["scalar", "scaler"]:
                return 1000
            if info.shape[0] == 1:
                return 510
            return 500
        elif info.isRecord:
            return 40
        return DataView.UNSUPPORTED


class _HexaView(DataView):
    """View displaying data using text"""

    def __init__(self, parent):
        DataView.__init__(self, parent, modeId=RAW_HEXA_MODE)

    def createWidget(self, parent):
        from .HexaTableView import HexaTableView
        widget = HexaTableView(parent)
        return widget

    def clear(self):
        self.getWidget().setArrayData(None)

    def setData(self, data):
        data = self.normalizeData(data)
        widget = self.getWidget()
        widget.setArrayData(data)

    def axesNames(self, data, info):
        return []

    def getDataPriority(self, data, info):
        if info.size <= 0:
            return DataView.UNSUPPORTED
        if info.isVoid:
            return 2000
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
        return None

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
        self.addView(_HexaView(parent))
        self.addView(_ScalarView(parent))
        self.addView(_ArrayView(parent))
        self.addView(_RecordView(parent))


class _ImageView(CompositeDataView):
    """View displaying data as 2D image

    It choose between Plot2D and ComplexImageView widgets
    """

    def __init__(self, parent):
        super(_ImageView, self).__init__(
            parent=parent,
            modeId=IMAGE_MODE,
            label="Image",
            icon=icons.getQIcon("view-2d"))
        self.addView(_ComplexImageView(parent))
        self.addView(_Plot2dView(parent))


class _InvalidNXdataView(DataView):
    """DataView showing a simple label with an error message
    to inform that a group with @NX_class=NXdata cannot be
    interpreted by any NXDataview."""
    def __init__(self, parent):
        DataView.__init__(self, parent,
                          modeId=NXDATA_INVALID_MODE)
        self._msg = ""

    def createWidget(self, parent):
        widget = qt.QLabel(parent)
        widget.setWordWrap(True)
        widget.setStyleSheet("QLabel { color : red; }")
        return widget

    def axesNames(self, data, info):
        return []

    def clear(self):
        self.getWidget().setText("")

    def setData(self, data):
        self.getWidget().setText(self._msg)

    def getDataPriority(self, data, info):
        data = self.normalizeData(data)

        if not info.isInvalidNXdata:
            return DataView.UNSUPPORTED

        if info.hasNXdata:
            self._msg = "NXdata seems valid, but cannot be displayed "
            self._msg += "by any existing plot widget."
        else:
            nx_class = get_attr_as_unicode(data, "NX_class")
            if nx_class == "NXdata":
                # invalid: could not even be parsed by NXdata
                self._msg = "Group has @NX_class = NXdata, but could not be interpreted"
                self._msg += " as valid NXdata."
            elif nx_class == "NXentry":
                self._msg = "NXentry group provides a @default attribute,"
                default_nxdata_name = data.attrs["default"]
                if default_nxdata_name not in data:
                    self._msg += " but no corresponding NXdata group exists."
                elif get_attr_as_unicode(data[default_nxdata_name], "NX_class") != "NXdata":
                    self._msg += " but the corresponding item is not a "
                    self._msg += "NXdata group."
                else:
                    self._msg += " but the corresponding NXdata seems to be"
                    self._msg += " malformed."
            elif nx_class == "NXroot" or silx.io.is_file(data):
                default_entry = data[data.attrs["default"]]
                default_nxdata_name = default_entry.attrs["default"]
                self._msg = "NXroot group provides a @default attribute "
                self._msg += "pointing to a NXentry which defines its own "
                self._msg += "@default attribute, "
                if default_nxdata_name not in default_entry:
                    self._msg += " but no corresponding NXdata group exists."
                elif get_attr_as_unicode(default_entry[default_nxdata_name],
                                         "NX_class") != "NXdata":
                    self._msg += " but the corresponding item is not a "
                    self._msg += "NXdata group."
                else:
                    self._msg += " but the corresponding NXdata seems to be"
                    self._msg += " malformed."
        return 100


class _NXdataScalarView(DataView):
    """DataView using a table view for displaying NXdata scalars:
    0-D signal or n-D signal with *@interpretation=scalar*"""
    def __init__(self, parent):
        DataView.__init__(self, parent,
                          modeId=NXDATA_SCALAR_MODE)

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
        # data could be a NXdata or an NXentry
        nxd = nxdata.get_default(data, validate=False)
        signal = nxd.signal
        self.getWidget().setArrayData(signal,
                                      labels=True)

    def getDataPriority(self, data, info):
        data = self.normalizeData(data)

        if info.hasNXdata and not info.isInvalidNXdata:
            nxd = nxdata.get_default(data, validate=False)
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
        DataView.__init__(self, parent,
                          modeId=NXDATA_CURVE_MODE)

    def createWidget(self, parent):
        from silx.gui.data.NXdataWidgets import ArrayCurvePlot
        widget = ArrayCurvePlot(parent)
        return widget

    def axesNames(self, data, info):
        # disabled (used by default axis selector widget in Hdf5Viewer)
        return None

    def clear(self):
        self.getWidget().clear()

    def setData(self, data):
        data = self.normalizeData(data)
        nxd = nxdata.get_default(data, validate=False)
        signals_names = [nxd.signal_name] + nxd.auxiliary_signals_names
        if nxd.axes_dataset_names[-1] is not None:
            x_errors = nxd.get_axis_errors(nxd.axes_dataset_names[-1])
        else:
            x_errors = None

        # this fix is necessary until the next release of PyMca (5.2.3 or 5.3.0)
        # see https://github.com/vasole/pymca/issues/144 and https://github.com/vasole/pymca/pull/145
        if not hasattr(self.getWidget(), "setCurvesData") and \
                hasattr(self.getWidget(), "setCurveData"):
            _logger.warning("Using deprecated ArrayCurvePlot API, "
                            "without support of auxiliary signals")
            self.getWidget().setCurveData(nxd.signal, nxd.axes[-1],
                                          yerror=nxd.errors, xerror=x_errors,
                                          ylabel=nxd.signal_name, xlabel=nxd.axes_names[-1],
                                          title=nxd.title or nxd.signal_name)
            return

        self.getWidget().setCurvesData([nxd.signal] + nxd.auxiliary_signals, nxd.axes[-1],
                                       yerror=nxd.errors, xerror=x_errors,
                                       ylabels=signals_names, xlabel=nxd.axes_names[-1],
                                       title=nxd.title or signals_names[0])

    def getDataPriority(self, data, info):
        data = self.normalizeData(data)
        if info.hasNXdata and not info.isInvalidNXdata:
            if nxdata.get_default(data, validate=False).is_curve:
                return 100
        return DataView.UNSUPPORTED


class _NXdataXYVScatterView(DataView):
    """DataView using a Plot1D for displaying NXdata 3D scatters as
    a scatter of coloured points (1-D signal with 2 axes)"""
    def __init__(self, parent):
        DataView.__init__(self, parent,
                          modeId=NXDATA_XYVSCATTER_MODE)

    def createWidget(self, parent):
        from silx.gui.data.NXdataWidgets import XYVScatterPlot
        widget = XYVScatterPlot(parent)
        return widget

    def axesNames(self, data, info):
        # disabled (used by default axis selector widget in Hdf5Viewer)
        return None

    def clear(self):
        self.getWidget().clear()

    def setData(self, data):
        data = self.normalizeData(data)
        nxd = nxdata.get_default(data, validate=False)

        x_axis, y_axis = nxd.axes[-2:]
        if x_axis is None:
            x_axis = numpy.arange(nxd.signal.size)
        if y_axis is None:
            y_axis = numpy.arange(nxd.signal.size)

        x_label, y_label = nxd.axes_names[-2:]
        if x_label is not None:
            x_errors = nxd.get_axis_errors(x_label)
        else:
            x_errors = None

        if y_label is not None:
            y_errors = nxd.get_axis_errors(y_label)
        else:
            y_errors = None

        self.getWidget().setScattersData(y_axis, x_axis, values=[nxd.signal] + nxd.auxiliary_signals,
                                         yerror=y_errors, xerror=x_errors,
                                         ylabel=y_label, xlabel=x_label,
                                         title=nxd.title,
                                         scatter_titles=[nxd.signal_name] + nxd.auxiliary_signals_names)

    def getDataPriority(self, data, info):
        data = self.normalizeData(data)
        if info.hasNXdata and not info.isInvalidNXdata:
            if nxdata.get_default(data, validate=False).is_x_y_value_scatter:
                return 100

        return DataView.UNSUPPORTED


class _NXdataImageView(DataView):
    """DataView using a Plot2D for displaying NXdata images:
    2-D signal or n-D signals with *@interpretation=spectrum*."""
    def __init__(self, parent):
        DataView.__init__(self, parent,
                          modeId=NXDATA_IMAGE_MODE)

    def createWidget(self, parent):
        from silx.gui.data.NXdataWidgets import ArrayImagePlot
        widget = ArrayImagePlot(parent)
        widget.getPlot().setDefaultColormap(self.defaultColormap())
        widget.getPlot().getColormapAction().setColorDialog(self.defaultColorDialog())
        return widget

    def axesNames(self, data, info):
        # disabled (used by default axis selector widget in Hdf5Viewer)
        return None

    def clear(self):
        self.getWidget().clear()

    def setData(self, data):
        data = self.normalizeData(data)
        nxd = nxdata.get_default(data, validate=False)
        isRgba = nxd.interpretation == "rgba-image"

        # last two axes are Y & X
        img_slicing = slice(-2, None) if not isRgba else slice(-3, -1)
        y_axis, x_axis = nxd.axes[img_slicing]
        y_label, x_label = nxd.axes_names[img_slicing]

        self.getWidget().setImageData(
            [nxd.signal] + nxd.auxiliary_signals,
            x_axis=x_axis, y_axis=y_axis,
            signals_names=[nxd.signal_name] + nxd.auxiliary_signals_names,
            xlabel=x_label, ylabel=y_label,
            title=nxd.title, isRgba=isRgba)

    def getDataPriority(self, data, info):
        data = self.normalizeData(data)

        if info.hasNXdata and not info.isInvalidNXdata:
            if nxdata.get_default(data, validate=False).is_image:
                return 100

        return DataView.UNSUPPORTED


class _NXdataStackView(DataView):
    def __init__(self, parent):
        DataView.__init__(self, parent,
                          modeId=NXDATA_STACK_MODE)

    def createWidget(self, parent):
        from silx.gui.data.NXdataWidgets import ArrayStackPlot
        widget = ArrayStackPlot(parent)
        widget.getStackView().setColormap(self.defaultColormap())
        widget.getStackView().getPlot().getColormapAction().setColorDialog(self.defaultColorDialog())
        return widget

    def axesNames(self, data, info):
        # disabled (used by default axis selector widget in Hdf5Viewer)
        return None

    def clear(self):
        self.getWidget().clear()

    def setData(self, data):
        data = self.normalizeData(data)
        nxd = nxdata.get_default(data, validate=False)
        signal_name = nxd.signal_name
        z_axis, y_axis, x_axis = nxd.axes[-3:]
        z_label, y_label, x_label = nxd.axes_names[-3:]
        title = nxd.title or signal_name

        widget = self.getWidget()
        widget.setStackData(
                     nxd.signal, x_axis=x_axis, y_axis=y_axis, z_axis=z_axis,
                     signal_name=signal_name,
                     xlabel=x_label, ylabel=y_label, zlabel=z_label,
                     title=title)
        # Override the colormap, while setStack overwrite it
        widget.getStackView().setColormap(self.defaultColormap())

    def getDataPriority(self, data, info):
        data = self.normalizeData(data)
        if info.hasNXdata and not info.isInvalidNXdata:
            if nxdata.get_default(data, validate=False).is_stack:
                return 100

        return DataView.UNSUPPORTED


class _NXdataView(CompositeDataView):
    """Composite view displaying NXdata groups using the most adequate
    widget depending on the dimensionality."""
    def __init__(self, parent):
        super(_NXdataView, self).__init__(
            parent=parent,
            label="NXdata",
            modeId=NXDATA_MODE,
            icon=icons.getQIcon("view-nexus"))

        self.addView(_InvalidNXdataView(parent))
        self.addView(_NXdataScalarView(parent))
        self.addView(_NXdataCurveView(parent))
        self.addView(_NXdataXYVScatterView(parent))
        self.addView(_NXdataImageView(parent))
        self.addView(_NXdataStackView(parent))
