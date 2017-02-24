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
from silx.io import nxdata

from silx.gui import icons
from .DataViewer import DataView, CompositeDataView

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

    def getDataPriority(self, data, info):
        if info.isNXdata and nxdata.signal_is_1D(data):
            return 100
        return DataView.UNSUPPORTED


class NXdataView(CompositeDataView):
    """Composite view displaying NXdata groups using the most adequate
    widget depending on the dimensionality."""
    # TODO: normalizeData,
    def __init__(self, parent):
        super(NXdataView, self).__init__(
            parent=parent,
            # modeId=DataViewer.NXDATA_MODE,
            label="NXdata",
            icon=icons.getQIcon("view-hdf5"))

        # TODO self.addView(_NXdata1dView(parent)

    # def createWidget(self, parent):
    #     return qt.QStackedWidget()
    #
    # def clear(self):
    #     # TODO
    #     pass
    #
    # def setData(self, data):
    #     """
    #
    #     :param data: ``h5py.Group`` with an attribute ``NX_class = NXdata``
    #         following the NeXus specification for NXdata.
    #     """
    #     pass  # TODO

    def axesNames(self, data, info):
        return []

    def getDataPriority(self, data, info):
        if nxdata.is_valid(data):
            return 100
        else:
            return DataView.UNSUPPORTED
