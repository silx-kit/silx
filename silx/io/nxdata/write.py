# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017-2018 European Synchrotron Radiation Facility
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

import os
import logging

import h5py
import numpy
import six

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "17/04/2018"


_logger = logging.getLogger(__name__)


def _str_to_utf8(text):
    return numpy.array(text, dtype=h5py.special_dtype(vlen=six.text_type))


def save_NXdata(filename, signal, axes=None,
                signal_name="data", axes_names=None,
                signal_long_name=None, axes_long_names=None,
                signal_errors=None, axes_errors=None,
                title=None, interpretation=None,
                nxentry_name="entry", nxdata_name=None):
    """Write data to an NXdata group.

    .. note::

        No consistency checks are made regarding the dimensionality of the
        signal and number of axes. The user is responsible for providing
        meaningful data, that can be interpreted by visualization software.

    :param str filename: Path to output file. If the file does not
        exists, it is created.
    :param numpy.ndarray signal: Signal array.
    :param List[numpy.ndarray] axes: List of axes arrays.
    :param str signal_name: Name of signal dataset, in output file
    :param List[str] axes_names: List of dataset names for axes, in
        output file
    :param str signal_long_name: *@long_name* attribute for signal, or None.
    :param  axes_long_names: None, or list of long names
        for axes
    :type axes_long_names: List[str, None]
    :param numpy.ndarray signal_errors: Array of errors associated with the
        signal
    :param axes_errors: List of arrays of errors
        associated with each axis
    :type axes_errors: List[numpy.ndarray, None]
    :param str title: Graph title (saved as a "title" dataset) or None.
    :param str interpretation: *@interpretation* attribute ("spectrum",
        "image", "rgba-image" or None). This is only needed in cases of
        ambiguous dimensionality, e.g. a 3D array which represents a RGBA
        image rather than a stack.
    :param str nxentry_name: Name of group in which the NXdata group
        is created. By default, "/entry" is used.

        .. note::

            The Nexus format specification requires for NXdata groups
            be part of a NXentry group.
            The specified group should have attribute *@NX_class=NXentry*, in
            order for the created file to be nexus compliant.
    :param str nxdata_name: Name of NXdata group. If omitted (None), the
        function creates a new group using the first available name ("data0",
        or "data1"...).
        Overwriting an existing group (or dataset) is not supported, you must
        delete it yourself prior to calling this function if this is what you
        want.
    :return: True if save was successful, else False.
    """
    if h5py is None:
        raise ImportError("h5py could not be imported, but is required by "
                          "save_NXdata function")

    if axes_names is not None:
        assert axes is not None, "Axes names defined, but missing axes arrays"
        assert len(axes) == len(axes_names), \
            "Mismatch between number of axes and axes_names"

    if axes is not None and axes_names is None:
        axes_names = []
        for i, axis in enumerate(axes):
            axes_names.append("dim%d" % i if axis is not None else ".")
    if axes is None:
        axes = []

    # Open file in
    if os.path.exists(filename):
        errmsg = "Cannot write/append to existing path %s"
        if not os.path.isfile(filename):
            errmsg += " (not a file)"
            _logger.error(errmsg, filename)
            return False
        if not os.access(filename, os.W_OK):
            errmsg += " (no permission to write)"
            _logger.error(errmsg, filename)
            return False
        mode = "r+"
    else:
        mode = "w-"

    with h5py.File(filename, mode=mode) as h5f:
        # get or create entry
        if nxentry_name is not None:
            entry = h5f.require_group(nxentry_name)
            if "default" not in h5f.attrs:
                # set this entry as default
                h5f.attrs["default"] = _str_to_utf8(nxentry_name)
            if "NX_class" not in entry.attrs:
                entry.attrs["NX_class"] = u"NXentry"
        else:
            # write NXdata into the root of the file (invalid nexus!)
            entry = h5f

        # Create NXdata group
        if nxdata_name is not None:
            if nxdata_name in entry:
                _logger.error("Cannot assign an NXdata group to an existing"
                              " group or dataset")
                return False
        else:
            # no name specified, take one that is available
            nxdata_name = "data0"
            i = 1
            while nxdata_name in entry:
                _logger.info("%s item already exists in NXentry group," +
                             " trying %s", nxdata_name, "data%d" % i)
                nxdata_name = "data%d" % i
                i += 1

        data_group = entry.create_group(nxdata_name)
        data_group.attrs["NX_class"] = u"NXdata"
        data_group.attrs["signal"] = _str_to_utf8(signal_name)
        if axes:
            data_group.attrs["axes"] = _str_to_utf8(axes_names)
        if title:
            # not in NXdata spec, but implemented by nexpy
            data_group["title"] = title
            # better way imho
            data_group.attrs["title"] = _str_to_utf8(title)

        signal_dataset = data_group.create_dataset(signal_name,
                                                   data=signal)
        if signal_long_name:
            signal_dataset.attrs["long_name"] = _str_to_utf8(signal_long_name)
        if interpretation:
            signal_dataset.attrs["interpretation"] = _str_to_utf8(interpretation)

        for i, axis_array in enumerate(axes):
            if axis_array is None:
                assert axes_names[i] in [".", None], \
                    "Axis name defined for dim %d but no axis array" % i
                continue
            axis_dataset = data_group.create_dataset(axes_names[i],
                                                     data=axis_array)
            if axes_long_names is not None:
                axis_dataset.attrs["long_name"] = _str_to_utf8(axes_long_names[i])

        if signal_errors is not None:
            data_group.create_dataset("errors",
                                      data=signal_errors)

        if axes_errors is not None:
            assert isinstance(axes_errors, (list, tuple)), \
                "axes_errors must be a list or a tuple of ndarray or None"
            assert len(axes_errors) == len(axes_names), \
                "Mismatch between number of axes_errors and axes_names"
            for i, axis_errors in enumerate(axes_errors):
                if axis_errors is not None:
                    dsname = axes_names[i] + "_errors"
                    data_group.create_dataset(dsname,
                                              data=axis_errors)
        if "default" not in entry.attrs:
            # set this NXdata as default
            entry.attrs["default"] = nxdata_name

    return True
