# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017 European Synchrotron Radiation Facility
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
"""This module defines a view for a NXdata HDF5 group, to be used in
:class:`silx.gui.data.DataViewer.DataViewer`
"""
import logging
import numpy

# scatter plot handling
import matplotlib.cm
import matplotlib.colors
import silx.gui.plot.MPLColormap

from silx.io import nxdata
from silx.gui import qt, icons
from silx.gui.data.DataViews import DataView, CompositeDataView
from silx.gui.data.ArrayTableWidget import ArrayTableWidget
from silx.gui.plot import Plot1D, Plot2D, StackView
from silx.gui.data.NumpyAxesSelector import NumpyAxesSelector

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "22/02/2017"


_logger = logging.getLogger(__name__)


def _applyColormap(values,
                   colormap=silx.gui.plot.MPLColormap.viridis,
                   minVal=None,
                   maxVal=None, ):
    """Compute RGBA array of shape (n, 4) from a 1D array of length n.

    :param values: 1D array of values
    :param colormap: colormap to be used
    """
    values_array = numpy.array(values)
    if minVal is None:
        minVal = values_array.min()
    if maxVal is None:
        maxVal = values_array.max()
    sm = matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(vmin=minVal, vmax=maxVal),
            cmap=colormap)
    colors = sm.to_rgba(values)
    return colors


class NXdataScalarView(DataView):
    """DataView using a table view for displaying NXdata scalars:
    0-D signal or n-D signal with *@interpretation=scalar*"""
    def __init__(self, parent):
        DataView.__init__(self, parent)

    def createWidget(self, parent):
        widget = ArrayTableWidget(parent)
        #widget.displayAxesSelector(False)
        return widget

    def axesNames(self, data, info):
        return ["col", "row"]

    def clear(self):
        self.getWidget().setArrayData(numpy.array([[]]),
                                      labels=True)

    def setData(self, data):
        data = self.normalizeData(data)
        signal = nxdata.get_signal(data)
        self.getWidget().setArrayData(signal,
                                      labels=True)

    def getDataPriority(self, data, info):
        data = self.normalizeData(data)
        if info.isNXdata and (
                    nxdata.signal_is_0D(data) or
                    nxdata.get_interpretation(data) in ["scalar", "scaler"]):
            return 100
        return DataView.UNSUPPORTED


class ArrayCurvePlot(qt.QWidget):
    """
    Widget for plotting a curve from a multi-dimensional signal array
    and a 1D axis array.

    The signal array can have an arbitrary number of dimensions, the only
    limitation being that the last dimension must have the same length as
    the axis array.

    The widget provides sliders to select indices on the first (n - 1)
    dimensions of the signal array, and buttons to add/replace selected
    curves to the plot.

    This widget also handles simple 2D or 3D scatter plots (third dimension
    displayed as colour of points).
    """
    def __init__(self, parent=None):
        """

        :param parent: Parent QWidget
        """
        super(ArrayCurvePlot, self).__init__(parent)

        self.__signal = None
        self.__signal_name = None
        self.__signal_errors = None
        self.__axis = None
        self.__axis_name = None
        self.__axis_errors = None
        self.__values = None

        self.__first_curve_added = False

        self._plot = Plot1D(self)
        self._selector = NumpyAxesSelector(self)
        self._selector.setNamedAxesSelectorVisibility(False)
        self.__selector_is_connected = False

        layout = qt.QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plot,  0, 0)
        layout.addWidget(self._selector, 1, 0)

        self.setLayout(layout)

    def setCurveData(self, y, x=None, values=None,
                     yerror=None, xerror=None,
                     ylabel=None, xlabel=None, title=None):
        """

        :param y: dataset to be represented by the y (vertical) axis.
            For a scatter, this must be a 1D array and x and values must be
            1-D arrays of the same size.
            In other cases, it can be a n-D array whose last dimension must
            have the same length as x (and values must be None)
        :param x: 1-D dataset used as the curve's x values. If provided,
            its lengths must be equal to the length of the last dimension of
            ``y`` (and equal to the length of ``value``, for a scatter plot).
        :param values: Values, to be provided for a x-y-value scatter plot.
            This will be used to compute the color map and assign colors
            to the points.
        :param yerror: 1-D dataset of errors for y, or None
        :param xerror: 1-D dataset of errors for x, or None
        :param ylabel: Label for Y axis
        :param xlabel: Label for X axis
        :param title: Graph title
        """
        self.__signal = y
        self.__signal_name = ylabel
        self.__signal_errors = yerror
        self.__axis = x
        self.__axis_name = xlabel
        self.__axis_errors = xerror
        self.__values = values

        if self.__selector_is_connected:
            self._selector.selectionChanged.disconnect(self._replaceCurve)
            self.__selector_is_connected = False
        self._selector.setData(y)
        self._selector.setAxisNames([ylabel or "Y"])

        self._plot.setGraphTitle(title or "")
        self._plot.setGraphXLabel(self.__axis_name or "X")
        self._plot.setGraphYLabel(self.__signal_name or "Y")
        self._replaceCurve()

        if not self.__selector_is_connected:
            self._selector.selectionChanged.connect(self._replaceCurve)
            self.__selector_is_connected = True

    def _addCurve(self, replace=False):
        y = self._selector.selectedData()
        x = self.__axis
        if x is None:
            x = numpy.arange(len(y))
        elif numpy.isscalar(x) or len(x) == 1:
            # constant axis
            x = x * numpy.ones_like(y)
        elif len(x) == 2 and len(y) != 2:
            # linear calibration a*x + b
            x = x[0] * numpy.arange(len(y)) + x[1]
        legend = self.__signal_name + "["
        for sl in self._selector.selection():
            if sl == slice(None):
                legend += ":, "
            else:
                legend += str(sl) + ", "
        legend = legend[:-2] + "]"
        if self.__signal_errors is not None:
            y_errors = self.__signal_errors[self._selector.selection()]
        else:
            y_errors = None
        # values: x-y-v scatter
        if self.__values is not None:
            rgbacolors = _applyColormap(self.__values)
            self._plot.addCurve(x, y, color=rgbacolors,
                                legend=legend,
                                xlabel=self.__axis_name,
                                ylabel=self.__signal_name,
                                xerror=self.__axis_errors,
                                yerror=y_errors,
                                resetzoom=True, replace=replace,
                                symbol="o", linestyle="")

        # x monotonically increasing: curve
        elif numpy.all(numpy.diff(x) > 0):
            self._plot.addCurve(x, y, legend=legend,
                                xlabel=self.__axis_name,
                                ylabel=self.__signal_name,
                                xerror=self.__axis_errors,
                                yerror=y_errors,
                                resetzoom=True, replace=replace)

        # scatter
        else:
            self._plot.addCurve(x, y, legend=legend,
                                xlabel=self.__axis_name,
                                ylabel=self.__signal_name,
                                xerror=self.__axis_errors,
                                yerror=y_errors,
                                resetzoom=True, replace=replace,
                                symbol="o", linestyle="")

    def _replaceCurve(self):
        self._addCurve(replace=True)

    def clear(self):
        self._plot.clear()

    def _addAllCurves(self):
        self.clear()
        curve_len = self.__signal.shape[-1]
        all_curves = numpy.reshape(self.__signal,
                                   (-1, curve_len))
        x = self.__axis if self.__axis is not None else numpy.arange(curve_len)
        for i in range(all_curves.shape[0]):
            y = all_curves[i]

            if len(self.__signal.shape) > 1:
                nD_index = numpy.unravel_index(i, self.__signal.shape[:-1])
            else:
                nD_index = tuple()

            legend = self.__signal_name + "["
            for j in nD_index:
                legend += "%d, " % j
                legend += ":]"
            if self.__signal_errors is not None:
                y_errors = self.__signal_errors[nD_index + (slice(None), )]
            else:
                y_errors = None
            self._plot.addCurve(x, y, legend=legend,
                                xlabel=self.__axis_name,
                                ylabel=self.__signal_name,
                                xerror=self.__axis_errors,
                                yerror=y_errors,
                                resetzoom=True)


class NXdataCurveView(DataView):
    """DataView using a Plot1D for displaying NXdata curves:
    1-D signal or n-D signal with *@interpretation=spectrum*.

    It also handles basic scatter plots:
    a 1-D signal with one axis whose values are not monotonically increasing.
    """
    def __init__(self, parent):
        DataView.__init__(self, parent)

    def createWidget(self, parent):
        widget = ArrayCurvePlot(parent)
        return widget

    def axesNames(self, data, info):
        # disabled (used by default axis selector widget in Hdf5Viewer)
        return []

    def clear(self):
        self.getWidget().clear()

    def setData(self, data):
        data = self.normalizeData(data)
        signal = nxdata.get_signal(data)
        signal_name = data.attrs["signal"]
        signal_errors = nxdata.get_signal_errors(data)
        group_name = data.name
        x_axis = nxdata.get_axes(data)[-1]
        x_label = nxdata.get_axes_names(data)[-1]
        if x_label is not None:
            x_errors = nxdata.get_axis_errors(
                    data,
                    x_label)
        else:
            x_errors = None

        self.getWidget().setCurveData(signal, x_axis,
                                      yerror=signal_errors, xerror=x_errors,
                                      ylabel=signal_name, xlabel=x_label,
                                      title="NXdata group " + group_name)

    def getDataPriority(self, data, info):
        data = self.normalizeData(data)
        if info.isNXdata:
            if nxdata.is_x_y_value_scatter(data) or nxdata.is_unsupported_scatter(data):
                return DataView.UNSUPPORTED
            if nxdata.signal_is_1D(data) and \
                    not nxdata.get_interpretation(data) in ["scalar", "scaler"]:
                return 100
            if nxdata.get_interpretation(data) == "spectrum":
                return 100
        return DataView.UNSUPPORTED


class NXdataXYVScatterView(DataView):
    """DataView using a Plot1D for displaying NXdata 3D scatters as
    a scatter of coloured points (1-D signal with 2 axes)"""
    def __init__(self, parent):
        DataView.__init__(self, parent)

    def createWidget(self, parent):
        widget = ArrayCurvePlot(parent)
        return widget

    def axesNames(self, data, info):
        # disabled (used by default axis selector widget in Hdf5Viewer)
        return []

    def clear(self):
        self.getWidget().clear()

    def setData(self, data):
        data = self.normalizeData(data)
        signal = nxdata.get_signal(data)
        signal_name = data.attrs["signal"]
        # signal_errors = nxdata.get_signal_errors(data)   # not supported
        group_name = data.name
        x_axis, y_axis = nxdata.get_axes(data)[-2:]

        x_label, y_label = nxdata.get_axes_names(data)[-2:]
        if x_label is not None:
            x_errors = nxdata.get_axis_errors(
                    data,
                    x_label)
        else:
            x_errors = None

        if y_label is not None:
            y_errors = nxdata.get_axis_errors(
                    data,
                    y_label)
        else:
            y_errors = None

        self.getWidget().setCurveData(y_axis, x_axis, values=signal,
                                      yerror=y_errors, xerror=x_errors,
                                      ylabel=signal_name, xlabel=x_label,
                                      title="NXdata group " + group_name)

    def getDataPriority(self, data, info):
        data = self.normalizeData(data)
        if info.isNXdata:
            if nxdata.is_x_y_value_scatter(data):
                return 100
        return DataView.UNSUPPORTED


class ArrayImagePlot(qt.QWidget):
    """
    Widget for plotting an image from a multi-dimensional signal array
    and two 1D axes array.

    The signal array can have an arbitrary number of dimensions, the only
    limitation being that the last two dimensions must have the same length as
    the axes arrays.

    Sliders are provided to select indices on the first (n - 2) dimensions of
    the signal array, and the plot is updated to show the image corresponding
    to the selection.

    If one or both of the axes does not have regularly spaced values, the
    the image is plotted as a coloured scatter plot.
    """
    def __init__(self, parent=None):
        """

        :param parent: Parent QWidget
        """
        super(ArrayImagePlot, self).__init__(parent)

        self.__signal = None
        self.__signal_name = None
        self.__x_axis = None
        self.__x_axis_name = None
        self.__y_axis = None
        self.__y_axis_name = None

        self._plot = Plot2D(self)
        self._plot.setDefaultColormap(    # FIXME
                {"name": "viridis",
                 "normalization": "linear",
                 "autoscale": True})
        self._legend = qt.QLabel(self)
        self._selector = NumpyAxesSelector(self)
        self._selector.setNamedAxesSelectorVisibility(False)
        self.__selector_is_connected = False

        layout = qt.QVBoxLayout()
        layout.addWidget(self._plot)
        layout.addWidget(self._legend)
        layout.addWidget(self._selector)

        self.setLayout(layout)

    def setImageData(self, signal,
                     x_axis=None, y_axis=None,
                     signal_name=None,
                     xlabel=None, ylabel=None,
                     title=None):
        """

        :param signal: n-D dataset, whose last 2 dimensions are used as the
            image's values.
        :param x_axis: 1-D dataset used as the image's x coordinates. If
            provided, its lengths must be equal to the length of the last
            dimension of ``signal``.
        :param y_axis: 1-D dataset used as the image's y. If provided,
            its lengths must be equal to the length of the 2nd to last
            dimension of ``signal``.
        :param signal_name: Label used in the legend
        :param xlabel: Label for X axis
        :param ylabel: Label for Y axis
        :param title: Graph title
        """
        if self.__selector_is_connected:
            self._selector.selectionChanged.disconnect(self._updateImage)
            self.__selector_is_connected = False

        self.__signal = signal
        self.__signal_name = signal_name or ""
        self.__x_axis = x_axis
        self.__x_axis_name = xlabel
        self.__y_axis = y_axis
        self.__y_axis_name = ylabel

        self._selector.setData(signal)
        self._selector.setAxisNames([ylabel or "Y", xlabel or "X"])

        self._plot.setGraphTitle(title or "")
        self._plot.setGraphXLabel(self.__x_axis_name or "X")
        self._plot.setGraphYLabel(self.__y_axis_name or "Y")

        self._updateImage()

        if not self.__selector_is_connected:
            self._selector.selectionChanged.connect(self._updateImage)
            self.__selector_is_connected = True

    @staticmethod
    def _image_axes_are_regular(x_axis, y_axis):
        """Return True if both x_axis and y_axis are regularly spaced arrays

        :param x_axis: 1D numpy array
        :param y_axis: 1D numpy array
        """
        delta_x = x_axis[1:] - x_axis[:-1]
        if not numpy.isclose(delta_x, delta_x[0]).all():
            return False
        delta_y = y_axis[1:] - y_axis[:-1]
        if not numpy.isclose(delta_y, delta_y[0]).all():
            return False
        return True

    @staticmethod
    def _get_origin_scale(axis):
        """Assuming axis is a regularly spaced 1D array,
        return a tuple (origin, scale) where:
            - origin = axis[0]
            - scale = axis[1] - axis[0]
        :param axis: 1D numpy array
        :return: Tuple (axis[0], axis[1] - axis[0])
        """
        return axis[0], axis[1] - axis[0]

    def _updateImage(self):
        img = self._selector.selectedData()
        x_axis = self.__x_axis
        y_axis = self.__y_axis

        is_regular_image = False
        if x_axis is None and y_axis is None:
            is_regular_image = True
            origin = (0, 0)
            scale = (1., 1.)
        else:
            if x_axis is None:
                # no calibration
                x_axis = numpy.arange(img.shape[-1])
            elif numpy.isscalar(x_axis) or len(x_axis) == 1:
                # constant axis
                x_axis = x_axis * numpy.ones((img.shape[-1], ))
            elif len(x_axis) == 2:
                # linear calibration
                x_axis = x_axis[0] * numpy.arange(img.shape[-1]) + x_axis[1]

            if y_axis is None:
                y_axis = numpy.arange(img.shape[-2])
            elif numpy.isscalar(y_axis) or len(y_axis) == 1:
                y_axis = y_axis * numpy.ones((img.shape[-2], ))
            elif len(y_axis) == 2:
                y_axis = y_axis[0] * numpy.arange(img.shape[-2]) + y_axis[1]

            if self._image_axes_are_regular(x_axis, y_axis):
                is_regular_image = True
                xorigin, xscale = self._get_origin_scale(x_axis)
                yorigin, yscale = self._get_origin_scale(y_axis)
                origin = (xorigin, yorigin)
                scale = (xscale, yscale)

        legend = self.__signal_name + "["
        for sl in self._selector.selection():
            if sl == slice(None):
                legend += ":, "
            else:
                legend += str(sl) + ", "
        legend = legend[:-2] + "]"
        self._legend.setText("Displayed data: " + legend)

        if is_regular_image:
            # single regular image
            self._plot.addImage(img, legend=legend,
                                xlabel=self.__x_axis_name,
                                ylabel=self.__y_axis_name,
                                origin=origin, scale=scale)
        else:
            # FIXME: use addScatter
            scatterx, scattery = numpy.meshgrid(x_axis, y_axis)
            rgbacolor = _applyColormap(numpy.ravel(img))
            self._plot.addCurve(
                    numpy.ravel(scatterx), numpy.ravel(scattery),
                    color=rgbacolor, symbol="o", linestyle="")

    def clear(self):
        self._plot.clear()


class NXdataImageView(DataView):
    """DataView using a Plot2D for displaying NXdata images:
    2-D signal or n-D signals with *@interpretation=spectrum*."""
    def __init__(self, parent):
        DataView.__init__(self, parent)

    def createWidget(self, parent):
        widget = ArrayImagePlot(parent)
        return widget

    def axesNames(self, data, info):
        return []

    def clear(self):
        self.getWidget().clear()

    def setData(self, data):
        data = self.normalizeData(data)
        signal = nxdata.get_signal(data)
        signal_name = data.attrs["signal"]
        group_name = data.name
        y_axis, x_axis = nxdata.get_axes(data)[-2:]
        y_label, x_label = nxdata.get_axes_names(data)[-2:]

        self.getWidget().setImageData(
                     signal, x_axis=x_axis, y_axis=y_axis,
                     signal_name=signal_name, xlabel=x_label, ylabel=y_label,
                     title="NXdata group %s: %s" % (group_name, signal_name))

    def getDataPriority(self, data, info):
        data = self.normalizeData(data)
        if info.isNXdata:
            if nxdata.signal_is_2D(data):
                if nxdata.get_interpretation(data) not in ["scalar",
                                                           "spectrum",
                                                           "scaler"]:
                    return 100
            if nxdata.get_interpretation(data) == "image":
                return 100
        return DataView.UNSUPPORTED


class ArrayStackPlot(qt.QWidget):
    """
    Widget for plotting a n-D array (n >= 3) as a stack of images.
    Three axis arrays can be provided to calibrate the axes.

    The signal array can have an arbitrary number of dimensions, the only
    limitation being that the last 3 dimensions must have the same length as
    the axes arrays.

    Sliders are provided to select indices on the first (n - 3) dimensions of
    the signal array, and the plot is updated to load the stack corresponding
    to the selection.
    """
    def __init__(self, parent=None):
        """

        :param parent: Parent QWidget
        """
        super(ArrayStackPlot, self).__init__(parent)

        self.__signal = None
        self.__signal_name = None
        # the Z, Y, X axes apply to the last three dimensions of the signal
        # (in that order)
        self.__z_axis = None
        self.__z_axis_name = None
        self.__y_axis = None
        self.__y_axis_name = None
        self.__x_axis = None
        self.__x_axis_name = None

        self._stack_view = StackView(self)
        self._legend = qt.QLabel(self)
        self._selector = NumpyAxesSelector(self)
        self._selector.setNamedAxesSelectorVisibility(False)
        self.__selector_is_connected = False

        layout = qt.QVBoxLayout()
        layout.addWidget(self._stack_view)
        layout.addWidget(self._legend)
        layout.addWidget(self._selector)

        self.setLayout(layout)

    def setStackData(self, signal,
                     x_axis=None, y_axis=None, z_axis=None,
                     signal_name=None,
                     xlabel=None, ylabel=None, zlabel=None,
                     title=None):
        """

        :param signal: n-D dataset, whose last 3 dimensions are used as the
            3D stack values.
        :param x_axis: 1-D dataset used as the image's x coordinates. If
            provided, its lengths must be equal to the length of the last
            dimension of ``signal``.
        :param y_axis: 1-D dataset used as the image's y. If provided,
            its lengths must be equal to the length of the 2nd to last
            dimension of ``signal``.
        :param z_axis: 1-D dataset used as the image's z. If provided,
            its lengths must be equal to the length of the 3rd to last
            dimension of ``signal``.
        :param signal_name: Label used in the legend
        :param xlabel: Label for X axis
        :param ylabel: Label for Y axis
        :param zlabel: Label for Z axis
        :param title: Graph title
        """
        if self.__selector_is_connected:
            self._selector.selectionChanged.disconnect(self._updateStack)
            self.__selector_is_connected = False

        self.__signal = signal
        self.__signal_name = signal_name or ""
        self.__x_axis = x_axis
        self.__x_axis_name = xlabel
        self.__y_axis = y_axis
        self.__y_axis_name = ylabel
        self.__z_axis = z_axis
        self.__z_axis_name = zlabel

        self._selector.setData(signal)
        self._selector.setAxisNames([ylabel or "Y", xlabel or "X", zlabel or "Z"])

        self._stack_view.setGraphTitle(title or "")
        # by default, the z axis is the image position (dimension not plotted)
        self._stack_view.setGraphXLabel(self.__x_axis_name or "X")
        self._stack_view.setGraphYLabel(self.__y_axis_name or "Y")

        self._updateStack()

        if not self.__selector_is_connected:
            self._selector.selectionChanged.connect(self._updateStack)
            self.__selector_is_connected = True

    @staticmethod
    def _get_origin_scale(axis):
        """Assuming axis is a regularly spaced 1D array,
        return a tuple (origin, scale) where:
            - origin = axis[0]
            - scale = (axis[n-1] - axis[0]) / (n -1)
        :param axis: 1D numpy array
        :return: Tuple (axis[0], (axis[-1] - axis[0]) / (len(axis) - 1))
        """
        return axis[0], (axis[-1] - axis[0]) / (len(axis) - 1)

    def _updateStack(self):
        """Update displayed stack according to the current axes selector
        data."""
        stk = self._selector.selectedData()
        x_axis = self.__x_axis
        y_axis = self.__y_axis
        z_axis = self.__z_axis

        if x_axis is None:
            xorigin, xscale = 0., 1.
            x_axis = numpy.arange(stk.shape[-1])
        else:
            xorigin, xscale = self._get_origin_scale(x_axis)
        if y_axis is None:
            yorigin, yscale = 0., 1.
            y_axis = numpy.arange(stk.shape[-2])
        else:
            yorigin, yscale = self._get_origin_scale(y_axis)
        if z_axis is None:
            zorigin, zscale = 0., 1.
            z_axis = numpy.arange(stk.shape[-3])
        else:
            zorigin, zscale = self._get_origin_scale(z_axis)

        origin = (xorigin, yorigin)  # TODO
        scale = (xscale, yscale)

        legend = self.__signal_name + "["
        for sl in self._selector.selection():
            if sl == slice(None):
                legend += ":, "
            else:
                legend += str(sl) + ", "
        legend = legend[:-2] + "]"
        self._legend.setText("Displayed data: " + legend)

        self._stack_view.setStack(stk, ) # legend=legend,
                            # origin=origin, scale=scale)  # TODO
        self._stack_view.setLabels(
                labels=[self.__z_axis_name,
                        self.__y_axis_name,
                        self.__x_axis_name])

    def clear(self):
        self._stack_view.clear()


class NXdataStackView(DataView):
    def __init__(self, parent):
        DataView.__init__(self, parent)

    def createWidget(self, parent):
        widget = ArrayStackPlot(parent)
        return widget

    def axesNames(self, data, info):
        return []

    def clear(self):
        self.getWidget().clear()

    def setData(self, data):
        data = self.normalizeData(data)
        signal = nxdata.get_signal(data)
        signal_name = data.attrs["signal"]
        group_name = data.name
        z_axis, y_axis, x_axis = nxdata.get_axes(data)[-3:]
        z_label, y_label, x_label = nxdata.get_axes_names(data)[-3:]

        self.getWidget().setStackData(
                     signal, x_axis=x_axis, y_axis=y_axis, z_axis=z_axis,
                     signal_name=signal_name,
                     xlabel=x_label, ylabel=y_label, zlabel=z_label,
                     title="NXdata group %s: %s" % (group_name, signal_name))

    def getDataPriority(self, data, info):
        data = self.normalizeData(data)
        if info.isNXdata:
            if nxdata.get_signal_ndim(data) >= 3:
                if nxdata.get_interpretation(data) not in ["scalar", "scaler",
                                                           "spectrum",
                                                           "image"]:  # handled by NXdataImageView
                    return 100
        return DataView.UNSUPPORTED


class NXdataView(CompositeDataView):
    """Composite view displaying NXdata groups using the most adequate
    widget depending on the dimensionality."""
    def __init__(self, parent):
        super(NXdataView, self).__init__(
            parent=parent,
            label="NXdata",
            icon=icons.getQIcon("view-nexus"))

        self.addView(NXdataScalarView(parent))
        self.addView(NXdataCurveView(parent))
        self.addView(NXdataXYVScatterView(parent))
        self.addView(NXdataImageView(parent))
        self.addView(NXdataStackView(parent))
