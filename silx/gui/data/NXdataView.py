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
from silx.gui import icons
from silx.gui.data.DataViews import DataView, CompositeDataView

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "22/02/2017"


_logger = logging.getLogger(__name__)


def _recursive_addcurve(plot, x, signal, xlabel=None, ylabel=None,
                        legend_prefix="NXdata spectrum ",
                        index=tuple()):
    """Plot all curves in an array with an arbitrary number of dimensions (n).
    The last dimension of signal represents the sample number and must have
    the same length as the 1-D array x.
    All previous dimensions of the signal array represent the (n-1) dimensional
    index of the curves.
    The legend is the legend_prefix concatenated with the index (e.g.
    "NXdata spectrum (511, 1023)")

    Example::

        import numpy
        x = numpy.arange(10)

        signal = numpy.zeros((5, 10, 10))
        for i in range(signal.shape[0]):
            for j in range(signal.shape[1]):
                signal[i, j] = i*10 + x*j

        from silx.gui import qt
        from silx.gui.plot import Plot1D
        app = qt.QApplication([])
        plot = Plot1D()

        _recursive_addcurve(plot, x, signal,
                            legend_prefix="a*10 + b*x with (a, b)=")

        plot.show()
        app.exec_()

    """
    if len(index) == len(signal.shape) - 1:
        y = signal[index]
        plot.addCurve(x=x, y=y, xlabel=xlabel, ylabel=ylabel,
                      legend=legend_prefix + str(index))
    else:
        for i in range(signal.shape[len(index)]):
            _recursive_addcurve(plot, x, signal, xlabel, ylabel,
                                legend_prefix,
                                index + (i, ))


class NXdata1dView(DataView):
    def __init__(self, parent):
        DataView.__init__(self, parent)

    def createWidget(self, parent):
        from silx.gui.plot import Plot1D
        widget = Plot1D(parent)
        return widget

    def axesNames(self, data, info):
        # used by axis selector widget in Hdf5Viewer
        return []

    def clear(self):
        self.getWidget().clear()

    def setData(self, data):
        signal = nxdata.get_signal(data)
        signal_name = data.attrs["signal"]
        signal_len = signal.shape[-1]
        group_name = data.name
        x_axis = nxdata.get_axes(data)[-1]
        x_label = nxdata.get_axes_names(data)[-1]

        if x_axis is None:
            x_axis = range(signal_len)

        if len(signal.shape) == 1:
            # single curve
            assert x_axis.shape == signal.shape
            legend = "NXdata curve " + group_name
            self.getWidget().addCurve(legend=legend,
                                      x=x_axis, y=signal,
                                      xlabel=x_label, ylabel=signal_name)
        else:
            # array of curves (spectra)
            assert x_axis.shape == signal.shape[-1:]
            _recursive_addcurve(plot=self.getWidget(),
                                x=x_axis, signal=signal,
                                xlabel=x_label, ylabel=signal_name,
                                legend_prefix="NXdata spectrum ")

        self.getWidget().setGraphTitle("NXdata group " + group_name)

    def getDataPriority(self, data, info):
        if info.isNXdata and nxdata.signal_is_1D(data):
            return 100
        return DataView.UNSUPPORTED


def image_axes_are_regular(x_axis, y_axis):
    """Return True if both x_axis and y_axis are regularly spaced arrays

    :param x_axis: 1D numpy array
    :param y_axis: 1D numpy array
    """
    delta_x = x_axis[1:] - x_axis[:-1]
    bool_x = delta_x == delta_x[0]
    if False in bool_x:
        return False

    delta_y = y_axis[1:] - y_axis[:-1]
    bool_y = delta_y == delta_x[0]
    if False in bool_y:
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
        signal = nxdata.get_signal(data)
        signal_name = data.attrs["signal"]
        group_name = data.name
        y_axis, x_axis = nxdata.get_axes(data)[-2:]
        y_label, x_label = nxdata.get_axes_names(data)[-2:]

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

        self.getWidget().setGraphTitle("NXdata group " + group_name)

    def getDataPriority(self, data, info):
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

        self.addView(NXdata1dView(parent))
        self.addView(NXdata2dView(parent))
