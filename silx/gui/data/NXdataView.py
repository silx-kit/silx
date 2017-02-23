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
import silx.io

from silx.gui import icons
from silx.gui import qt
from .DataViewer import DataView, DataViewer, CompositeDataView

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "22/02/2017"


_logger = logging.getLogger(__name__)


def is_NXdata(group):   # noqa
    """Check if a h5py group is a **valid** NX_data group.

    Log warnings to troubleshoot malformed NXdata groups.
    See specification http://download.nexusformat.org/sphinx/classes/base_classes/NXdata.html

    :param group: h5py-like group
    :raise TypeError: if group is not a h5py group, a spech5 group,
        or a fabioh5 group
    """
    if not silx.io.is_group(group):
        raise TypeError("group must be a h5py-like group")
    if group.attrs.get("NX_class") != "NXdata":
        return False
    if "signal" not in group.attrs:
        _logger.warning("NXdata group does not define a signal attr.")
        return False

    signal_name = group.attrs["signal"]
    if signal_name not in group or not silx.io.is_dataset(group[signal_name]):
        _logger.warning(
                "Cannot find signal dataset '%s' in NXdata group" % signal_name)
        return False

    signal_ndim = len(group[signal_name].shape)

    if "axes" in group.attrs:
        axes = group.attrs.get("axes")
        if isinstance(axes, str):
            axes = [axes]

        if signal_ndim != len(axes):
            # FIXME: an axis dataset can apply to more than 1 dimension (check len(@AXISNAME_indices))
            _logger.warning("%d axes defined for %dD signal" %
                            (len(axes), signal_ndim))
            return False

        for axis in axes:
            if axis == ".":
                continue
            if axis not in group or not silx.io.is_dataset(group[axis]):
                _logger.warning("Could not find axis dataset '%d'" % axis +
                                " in NXdata group")
                return False
            if len(group[axis].shape) != 1:
                _logger.warning("Axis %s is not a 1D dataset" % axis)
                return False
    return True


def NXdata_signal(group):
    """Return the signal dataset in a NXdata group.

    :param group: h5py-like group following the NeXus *NXdata* specification.
    :return: Dataset whose name is specified in the *signal* attribute
        of *group*.
    :rtype: Dataset
    """
    if not is_NXdata(group):
        raise TypeError("group is not a valid NXdata class")

    return group[group.attrs["signal"]]


def NXdata_axes(group):
    """Return the axes datasets in a NXdata group.

    :param group: h5py-like Group following the NeXus *NXdata* specification.
    :return: List of datasets whose names are specified in the *axes*
        attribute of *group*. The output list has as many elements as there
        are dimensions in the signal. If a dimension has no dimension scale,
        `None` is inserted in the list in its position.
    :rtype: list[Dataset or None]
    """
    if not is_NXdata(group):
        raise TypeError("group is not a valid NXdata class")

    signal_ndims = len(NXdata_signal(group).shape)
    axes = group.attrs.get("axes")

    if axes is None:
        return [None for _i in range(signal_ndims)]

    # single string means that there is a single axis
    if isinstance(axes, str):
        return [group[axes]]

    # axes is a list of string
    return [group[axis] for axis in axes]
    # FIXME: axes not necessarily in the right order, check @AXISNAME_indices


class NXdata1dView(DataView):   # noqa
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
        signal = NXdata_signal(data)
        x_axis = NXdata_axis(data)
        data = self.normalizeData(data)
        self.getWidget().addCurve(legend="data",
                                  x=range(len(data)),
                                  y=data,
                                  resetzoom=self.__resetZoomNextTime)
        self.__resetZoomNextTime = True



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
        if is_NXdata(data):
            return 100
        else:
            return DataView.UNSUPPORTED
