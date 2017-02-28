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
from silx.gui.plot.MPLColormap import viridis

from silx.io import nxdata
from silx.gui import qt, icons
from silx.gui.data.DataViews import DataView, CompositeDataView
from silx.gui.data.ArrayTableWidget import ArrayTableWidget
from silx.gui.plot import Plot1D, Plot2D
from silx.gui.data.NumpyAxesSelector import NumpyAxesSelector

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "22/02/2017"


_logger = logging.getLogger(__name__)


class NXdataScalarView(DataView):
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
        if info.isNXdata and nxdata.signal_is_0D(data):
            return 100
        return DataView.UNSUPPORTED


class ArrayCurvePlot(qt.QWidget):
    """
    Widget for plotting a curve from a multi-dimensional signal array
    and a 1D axes array.

    The signal array can have an arbitrary number of dimensions, the only
    limitation being that the last dimension must have the same length as
    the axis array.
    """
    def __init__(self, parent=None):
        """

        :param parent: Parent QWidget
        """
        super(ArrayCurvePlot, self).__init__(parent)

        self.__signal = None
        self.__signal_name = None
        self.__axis = None
        self.__axis_name = None

        self._plot = Plot1D(self)
        self._selector = NumpyAxesSelector(self)
        self._addButton = qt.QPushButton("Add curve", self)
        self._addButton.clicked.connect(self._addCurve)
        self._replaceButton = qt.QPushButton("Replace curves", self)
        self._replaceButton.clicked.connect(self._replaceCurve)
        self._clearButton = qt.QPushButton("Clear plot", self)
        self._clearButton.clicked.connect(self.clear)

        layout = qt.QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plot,  0, 0, 1, 3)
        layout.addWidget(self._selector, 1, 0, 1, 3)
        layout.addWidget(self._addButton, 2, 0)
        layout.addWidget(self._replaceButton, 2, 1)
        layout.addWidget(self._clearButton, 2, 2)

        self.setLayout(layout)

    def setCurveData(self, signal, axis=None,
                     ylabel=None, xlabel=None,
                     title=None):
        """

        :param signal: n-D dataset, whose last dimension is used as the
            curve's y values.
        :param axis: 1-D dataset used as the curve's x values. If provided,
            its lengths must be equal to the length of the last dimension of
            ``signal``.
        :param ylabel: Label for Y axis
        :param xlabel: Label for X axis
        :param title: Graph title
        """
        self.__signal = signal
        self.__signal_name = ylabel
        self.__axis = axis
        self.__axis_name = xlabel

        self._selector.setData(signal)
        self._selector.setAxisNames([ylabel or "Y"])

        self._plot.setGraphTitle(title or "")
        self._plot.setGraphXLabel(self.__axis_name or "X")
        self._plot.setGraphYLabel(self.__signal_name or "Y")
        self._addCurve()

    def _addCurve(self, replace=False):
        y = self._selector.selectedData()
        x = self.__axis
        if x is None:
            x = numpy.arange(len(y))
        legend = self.__signal_name + "["
        for sl in self._selector.selection():
            if sl == slice(None):
                legend += ":, "
            else:
                legend += str(sl) + ", "
        legend = legend[:-2] + "]"
        self._plot.addCurve(x, y, legend=legend,
                            xlabel=self.__axis_name,
                            ylabel=self.__signal_name,
                            resetzoom=True, replace=replace)

    def _replaceCurve(self):
        self._addCurve(replace=True)

    def clear(self):
        self._plot.clear()


class NXdataCurveView(DataView):
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
        group_name = data.name
        x_axis = nxdata.get_axes(data)[-1]
        x_label = nxdata.get_axes_names(data)[-1]

        self.getWidget().setCurveData(signal, x_axis,
                                      ylabel=signal_name, xlabel=x_label,
                                      title="NXdata group " + group_name)

    def getDataPriority(self, data, info):
        data = self.normalizeData(data)
        if info.isNXdata and nxdata.signal_is_1D(data):
            return 100
        return DataView.UNSUPPORTED


def image_axes_are_regular(x_axis, y_axis):
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


def get_origin_scale(axis):
    """Assuming axis is a regularly spaced 1D array,
    return a tuple (origin, scale) where:
        - origin = axis[0]
        - scale = axis[1] - axis[0]
    :param axis: 1D numpy array
    :return: Tuple (axis[0], axis[1] - axis[0])
    """
    return axis[0], axis[1] - axis[0]


# fixme: use addScatter when available
def _applyColormap(values,
                   colormap=viridis,
                   minVal=None,
                   maxVal=None,):
    """Compute RGBA array of shape (n, 4) from a 1D array of length n.

    :param values: 1D array of values
    :param colormap: colormap to be used
    """
    if minVal is None:
        minVal = values.min()
    if maxVal is None:
        maxVal = values.max()

    sm = matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(vmin=minVal, vmax=maxVal),
            cmap=colormap)
    colors = sm.to_rgba(values)

    return colors


class NXdata2dView(DataView):
    def __init__(self, parent):
        DataView.__init__(self, parent)

    def createWidget(self, parent):
        from silx.gui.plot import Plot2D
        widget = Plot2D(parent)
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

        self.getWidget().setGraphTitle("NXdata group " + group_name)
        self.getWidget().setGraphXLabel(x_label if x_label is not None else "X")
        self.getWidget().setGraphYLabel(y_label if y_label is not None else "Y")

        if len(signal.shape) == 2:
            is_regular_image = False
            if x_axis is None and y_axis is None:
                is_regular_image = True
                origin = (0, 0)
                scale = (1., 1.)
            elif image_axes_are_regular(x_axis, y_axis):
                is_regular_image = True
                xorigin, xscale = get_origin_scale(x_axis)
                yorigin, yscale = get_origin_scale(y_axis)
                origin = (xorigin, yorigin)
                scale = (xscale, yscale)
            if is_regular_image:
                # single regular image
                legend = "NXdata image " + group_name
                self.getWidget().addImage(signal, legend=legend,
                                          xlabel=x_label, ylabel=signal_name,
                                          origin=origin, scale=scale)
            else:
                # TODO: use addScatter
                scatterx, scattery = numpy.meshgrid(x_axis, y_axis)
                rgbacolor = _applyColormap(numpy.ravel(signal))
                numpy.ravel(signal)
                self.getWidget().addCurve(
                        numpy.ravel(scatterx), numpy.ravel(scattery),
                        color=rgbacolor, symbol="o", linestyle="")

        # TODO:
        # else:
        #     # array of curves (spectra)
        #     assert x_axis.shape == signal.shape[-1:]
        #     _recursive_addcurve(plot=self.getWidget(),
        #                         x=x_axis, signal=signal,
        #                         xlabel=x_label, ylabel=signal_name,
        #                         legend_prefix="NXdata spectrum ")


    def getDataPriority(self, data, info):
        data = self.normalizeData(data)
        if info.isNXdata and nxdata.signal_is_2D(data):
            return 100
        return DataView.UNSUPPORTED


class NXdataView(CompositeDataView):
    """Composite view displaying NXdata groups using the most adequate
    widget depending on the dimensionality."""
    def __init__(self, parent):
        super(NXdataView, self).__init__(
            parent=parent,
            # modeId=DataViewer.NXDATA_MODE,
            label="NXdata",
            icon=icons.getQIcon("view-hdf5"))

        self.addView(NXdataScalarView(parent))
        self.addView(NXdataCurveView(parent))
        self.addView(NXdata2dView(parent))
