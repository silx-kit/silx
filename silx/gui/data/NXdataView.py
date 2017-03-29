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
import numpy

from silx.io.nxdata import NXdata
from silx.gui import icons
from silx.gui.data.DataViews import DataView, CompositeDataView

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "22/02/2017"


class NXdataScalarView(DataView):
    """DataView using a table view for displaying NXdata scalars:
    0-D signal or n-D signal with *@interpretation=scalar*"""
    def __init__(self, parent):
        DataView.__init__(self, parent)

    def createWidget(self, parent):
        from silx.gui.data.ArrayTableWidget import ArrayTableWidget
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
        signal = NXdata(data).signal
        self.getWidget().setArrayData(signal,
                                      labels=True)

    def getDataPriority(self, data, info):
        data = self.normalizeData(data)
        if info.isNXdata:
            nxd = NXdata(data)
            if nxd.signal_is_0D or nxd.interpretation in ["scalar", "scaler"]:
                return 100
        return DataView.UNSUPPORTED


class NXdataCurveView(DataView):
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
            if nxd.signal_is_1D and \
                    not nxd.interpretation in ["scalar", "scaler"]:
                return 100
            if nxd.interpretation == "spectrum":
                return 100
        return DataView.UNSUPPORTED


class NXdataXYVScatterView(DataView):
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


class NXdataImageView(DataView):
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
            if nxd.signal_is_2D:
                if nxd.interpretation not in ["scalar", "spectrum", "scaler"]:
                    return 100
            if nxd.interpretation == "image":
                return 100
        return DataView.UNSUPPORTED


class NXdataStackView(DataView):
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
