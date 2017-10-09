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
"""This module provides a collection of functions to work with h5py-like
groups following the NeXus *NXdata* specification.

See http://download.nexusformat.org/sphinx/classes/base_classes/NXdata.html

"""
import logging
import numpy
from .utils import is_dataset, is_group
from silx.third_party import six

_logger = logging.getLogger(__name__)


_INTERPDIM = {"scalar": 0,
              "spectrum": 1,
              "image": 2,
              # "rgba-image": 3, "hsla-image": 3, "cmyk-image": 3, # TODO
              "vertex": 1}  # 3D scatter: 1D signal + 3 axes (x, y, z) of same legth
"""Number of signal dimensions associated to each possible @interpretation
attribute.
"""


def _nxdata_warning(msg):
    """Log a warning message prefixed with
    *"NXdata warning: "*

    :param str msg: Warning message
    """
    _logger.warning("NXdata warning: " + msg)


def get_attr_as_string(item, attr_name, default=None):
    """Return item.attrs[attr_name]. If it is a byte-string or an array of
    byte-strings, return it as a default python string.

    For Python 3, this involves a coercion from bytes into unicode.
    For Python 2, there is nothing special to do, as strings are bytes.

    :param item: Group or dataset
    :param attr_name: Attribute name
    :return: item.attrs[attr_name]
    """
    attr = item.attrs.get(attr_name, default)
    if six.PY2:
        return attr
    if six.PY3:
        if hasattr(attr, "decode"):
            # byte-string
            return attr.decode("utf-8")
        elif isinstance(attr, numpy.ndarray) and hasattr(attr[0], "decode"):
            # array of byte-strings
            return [element.decode("utf-8") for element in attr]
        else:
            # attr is not a byte-string
            return attr


def is_valid_nxdata(group):   # noqa
    """Check if a h5py group is a **valid** NX_data group.

    If the group does not have attribute *@NX_class=NXdata*, this function
    simply returns *False*.

    Else, warning messages are logged to troubleshoot malformed NXdata groups
    prior to returning *False*.

    :param group: h5py-like group
    :return: True if this NXdata group is valid.
    :raise: TypeError if group is not a h5py group, a spech5 group,
        or a fabioh5 group
    """
    if not is_group(group):
        raise TypeError("group must be a h5py-like group")
    if get_attr_as_string(group, "NX_class") != "NXdata":
        return False
    if "signal" not in group.attrs:
        _logger.warning("NXdata group does not define a signal attr.")
        return False

    signal_name = get_attr_as_string(group, "signal")
    if signal_name not in group or not is_dataset(group[signal_name]):
        _logger.warning(
            "Cannot find signal dataset '%s' in NXdata group" % signal_name)
        return False

    ndim = len(group[signal_name].shape)

    if "axes" in group.attrs:
        axes_names = get_attr_as_string(group, "axes")
        if isinstance(axes_names, str):
            axes_names = [axes_names]

        if 1 < ndim < len(axes_names):
            # ndim = 1 and several axes could be a scatter
            _nxdata_warning(
                "More @axes defined than there are " +
                "signal dimensions: " +
                "%d axes, %d dimensions." % (len(axes_names), ndim))
            return False

        # case of less axes than dimensions: number of axes must match
        # dimensionality defined by @interpretation
        if ndim > len(axes_names):
            interpretation = get_attr_as_string(group[signal_name], "interpretation")
            if interpretation is None:
                interpretation = get_attr_as_string(group, "interpretation")
            if interpretation is None:
                _nxdata_warning("No @interpretation and not enough" +
                                " @axes defined.")
                return False

            if interpretation not in _INTERPDIM:
                _nxdata_warning("Unrecognized @interpretation=" + interpretation +
                                " for data with wrong number of defined @axes.")
                return False

            if len(axes_names) != _INTERPDIM[interpretation]:
                _nxdata_warning(
                    "%d-D signal with @interpretation=%s " % (ndim, interpretation) +
                    "must define %d or %d axes." % (ndim, _INTERPDIM[interpretation]))
                return False

        # Test consistency of @uncertainties
        uncertainties_names = get_attr_as_string(group, "uncertainties")
        if uncertainties_names is None:
            uncertainties_names = get_attr_as_string(group[signal_name], "uncertainties")
        if isinstance(uncertainties_names, str):
            uncertainties_names = [uncertainties_names]
        if uncertainties_names is not None:
            if len(uncertainties_names) != len(axes_names):
                _nxdata_warning("@uncertainties does not define the same " +
                                "number of fields than @axes")
                return False

        # Test individual axes
        is_scatter = True   # true if all axes have the same size as the signal
        signal_size = 1
        for dim in group[signal_name].shape:
            signal_size *= dim
        polynomial_axes_names = []
        for i, axis_name in enumerate(axes_names):
            if axis_name == ".":
                continue
            if axis_name not in group or not is_dataset(group[axis_name]):
                _nxdata_warning("Could not find axis dataset '%s'" % axis_name)
                return False

            axis_size = 1
            for dim in group[axis_name].shape:
                axis_size *= dim

            if len(group[axis_name].shape) != 1:
                # too me, it makes only sense to have a n-D axis if it's total
                # size is exactly the signal's size (weird n-d scatter)
                if axis_size != signal_size:
                    _nxdata_warning("Axis %s is not a 1D dataset" % axis_name +
                                    " and its shape does not match the signal's shape")
                    return False
                axis_len = axis_size
            else:
                # for a  1-d axis,
                fg_idx = group[axis_name].attrs.get("first_good", 0)
                lg_idx = group[axis_name].attrs.get("last_good", len(group[axis_name]) - 1)
                axis_len = lg_idx + 1 - fg_idx

            if axis_len != signal_size:
                if axis_len not in group[signal_name].shape + (1, 2):
                    _nxdata_warning(
                        "Axis %s number of elements does not " % axis_name +
                        "correspond to the length of any signal dimension,"
                        " it does not appear to be a constant or a linear calibration," +
                        " and this does not seem to be a scatter plot.")
                    return False
                elif axis_len in (1, 2):
                    polynomial_axes_names.append(axis_name)
                is_scatter = False
            else:
                if not is_scatter:
                    _nxdata_warning(
                        "Axis %s number of elements is equal " % axis_name +
                        "to the length of the signal, but this does not seem" +
                        " to be a scatter (other axes have different sizes)")
                    return False

            # Test individual uncertainties
            errors_name = axis_name + "_errors"
            if errors_name not in group and uncertainties_names is not None:
                errors_name = uncertainties_names[i]
                if errors_name in group and axis_name not in polynomial_axes_names:
                    if group[errors_name].shape != group[axis_name].shape:
                        _nxdata_warning(
                            "Errors '%s' does not have the same " % errors_name +
                            "dimensions as axis '%s'." % axis_name)
                        return False

    # test dimensions of errors associated with signal
    if "errors" in group and is_dataset(group["errors"]):
        if group["errors"].shape != group[signal_name].shape:
            _nxdata_warning("Dataset containing standard deviations must " +
                            "have the same dimensions as the signal.")
            return False
    return True


class NXdata(object):
    """

    :param group: h5py-like group following the NeXus *NXdata* specification.
    """
    def __init__(self, group):
        if not is_valid_nxdata(group):
            raise TypeError("group is not a valid NXdata class")
        super(NXdata, self).__init__()

        self._is_scatter = None
        self._axes = None

        self.group = group
        """h5py-like group object compliant with NeXus NXdata specification.
        """

        self.signal = self.group[get_attr_as_string(self.group, "signal")]
        """Signal dataset in this NXdata group.
        """

        # ndim will be available in very recent h5py versions only
        self.signal_ndim = getattr(self.signal, "ndim",
                                   len(self.signal.shape))

        self.signal_is_0d = self.signal_ndim == 0
        self.signal_is_1d = self.signal_ndim == 1
        self.signal_is_2d = self.signal_ndim == 2
        self.signal_is_3d = self.signal_ndim == 3

        self.axes_names = []
        """List of axes names in a NXdata group.

        This attribute is similar to :attr:`axes_dataset_names` except that
        if an axis dataset has a "@long_name" attribute, it will be used
        instead of the dataset name.
        """
        # check if axis dataset defines @long_name
        for i, dsname in enumerate(self.axes_dataset_names):
            if dsname is not None and "long_name" in self.group[dsname].attrs:
                self.axes_names.append(get_attr_as_string(self.group[dsname], "long_name"))
            else:
                self.axes_names.append(dsname)

        # excludes scatters
        self.signal_is_1d = self.signal_is_1d and len(self.axes) <= 1  # excludes n-D scatters

    @property
    def interpretation(self):
        """*@interpretation* attribute associated with the *signal*
        dataset of the NXdata group. ``None`` if no interpretation
        attribute is present.

        The *interpretation* attribute provides information about the last
        dimensions of the signal. The allowed values are:

             - *"scalar"*: 0-D data to be plotted
             - *"spectrum"*: 1-D data to be plotted
             - *"image"*: 2-D data to be plotted
             - *"vertex"*: 3-D data to be plotted

        For example, a 3-D signal with interpretation *"spectrum"* should be
        considered to be a 2-D array of 1-D data. A 3-D signal with
        interpretation *"image"* should be interpreted as a 1-D array (a list)
        of 2-D images. An n-D array with interpretation *"image"* should be
        interpreted as an (n-2)-D array of images.

        A warning message is logged if the returned interpretation is not one
        of the allowed values, but no error is raised and the unknown
        interpretation is returned anyway.
        """
        allowed_interpretations = [None, "scalar", "spectrum", "image",
                                   # "rgba-image", "hsla-image", "cmyk-image"  # TODO
                                   "vertex"]

        interpretation = get_attr_as_string(self.signal, "interpretation")
        if interpretation is None:
            interpretation = get_attr_as_string(self.group, "interpretation")

        if interpretation not in allowed_interpretations:
            _logger.warning("Interpretation %s is not valid." % interpretation +
                            " Valid values: " + ", ".join(allowed_interpretations))
        return interpretation

    @property
    def axes(self):
        """List of the axes datasets.

        The list typically has as many elements as there are dimensions in the
        signal dataset, the exception being scatter plots which typically
        use a 1D signal and several 1D axes of the same size.

        If an axis dataset applies to several dimensions of the signal, it
        will be repeated in the list.

        If a dimension of the signal has no dimension scale (i.e. there is a
        "." in that position in the *@axes* array), `None` is inserted in the
        output list in its position.

        .. note::

            In theory, the *@axes* attribute defines as many entries as there
            are dimensions in the signal. In such a case, there is no ambiguity.
            If this is not the case, this implementation relies on the existence
            of an *@interpretation* (*spectrum* or *image*) attribute in the
            *signal* dataset.

        .. note::

            If an axis dataset defines attributes @first_good or @last_good,
            the output will be a numpy array resulting from slicing that
            axis to keep only the good index range: axis[first_good:last_good + 1]

        :rtype: list[Dataset or 1D array or None]
        """
        if self._axes is not None:
            # use cache
            return self._axes
        ndims = len(self.signal.shape)
        axes_names = get_attr_as_string(self.group, "axes")
        interpretation = self.interpretation

        if axes_names is None:
            self._axes = [None for _i in range(ndims)]
            return self._axes

        if isinstance(axes_names, str):
            axes_names = [axes_names]

        if len(axes_names) == ndims:
            # axes is a list of strings, one axis per dim is explicitly defined
            axes = [None] * ndims
            for i, axis_n in enumerate(axes_names):
                if axis_n != ".":
                    axes[i] = self.group[axis_n]
        elif interpretation is not None:
            # case of @interpretation attribute defined: we expect 1, 2 or 3 axes
            # corresponding to the 1, 2, or 3 last dimensions of the signal
            assert len(axes_names) == _INTERPDIM[interpretation]
            axes = [None] * (ndims - _INTERPDIM[interpretation])
            for axis_n in axes_names:
                if axis_n != ".":
                    axes.append(self.group[axis_n])
                else:
                    axes.append(None)
        else:   # scatter
            axes = []
            for axis_n in axes_names:
                if axis_n != ".":
                    axes.append(self.group[axis_n])
                else:
                    axes.append(None)
        # keep only good range of axis data
        for i, axis in enumerate(axes):
            if axis is None:
                continue
            if "first_good" not in axis.attrs and "last_good" not in axis.attrs:
                continue
            fg_idx = axis.attrs.get("first_good", 0)
            lg_idx = axis.attrs.get("last_good", len(axis) - 1)
            axes[i] = axis[fg_idx:lg_idx + 1]

        self._axes = axes
        return self._axes

    @property
    def axes_dataset_names(self):
        """
        If an axis dataset applies to several dimensions of the signal, its
        name will be repeated in the list.

        If a dimension of the signal has no dimension scale (i.e. there is a
        "." in that position in the *@axes* array), `None` is inserted in the
        output list in its position.
        """
        axes_dataset_names = get_attr_as_string(self.group, "axes")
        if axes_dataset_names is None:
            axes_dataset_names = get_attr_as_string(self.group, "axes")

        ndims = len(self.signal.shape)
        if axes_dataset_names is None:
            return [None] * ndims

        if isinstance(axes_dataset_names, str):
            axes_dataset_names = [axes_dataset_names]

        for i, axis_name in enumerate(axes_dataset_names):
            if hasattr(axis_name, "decode"):
                axis_name = axis_name.decode()
            if axis_name == ".":
                axes_dataset_names[i] = None

        if len(axes_dataset_names) != ndims:
            if self.is_scatter and ndims == 1:
                return list(axes_dataset_names)
            # @axes may only define 1 or 2 axes if @interpretation=spectrum/image.
            # Use the existing names for the last few dims, and prepend with Nones.
            assert len(axes_dataset_names) == _INTERPDIM[self.interpretation]
            all_dimensions_names = [None] * (ndims - _INTERPDIM[self.interpretation])
            for axis_name in axes_dataset_names:
                all_dimensions_names.append(axis_name)
            return all_dimensions_names

        return list(axes_dataset_names)

    def get_axis_errors(self, axis_name):
        """Return errors (uncertainties) associated with an axis.

        If the axis has attributes @first_good or @last_good, the output
        is trimmed accordingly (a numpy array will be returned rather than a
        dataset).

        :param str axis_name: Name of axis dataset. This dataset **must exist**.
        :return: Dataset with axis errors, or None
        :raise: KeyError if this group does not contain a dataset named axis_name
        """
        # ensure axis_name is decoded, before comparing it with decoded attributes
        if hasattr(axis_name, "decode"):
            axis_name = axis_name.decode("utf-8")
        if axis_name not in self.group:
            # tolerate axis_name given as @long_name
            for item in self.group:
                long_name = get_attr_as_string(self.group[item], "long_name")
                if long_name is not None and long_name == axis_name:
                    axis_name = item
                    break

        if axis_name not in self.group:
            raise KeyError("group does not contain a dataset named '%s'" % axis_name)

        len_axis = len(self.group[axis_name])

        fg_idx = self.group[axis_name].attrs.get("first_good", 0)
        lg_idx = self.group[axis_name].attrs.get("last_good", len_axis - 1)

        # case of axisname_errors dataset present
        errors_name = axis_name + "_errors"
        if errors_name in self.group and is_dataset(self.group[errors_name]):
            if fg_idx != 0 or lg_idx != (len_axis - 1):
                return self.group[errors_name][fg_idx:lg_idx + 1]
            else:
                return self.group[errors_name]
        # case of uncertainties dataset name provided in @uncertainties
        uncertainties_names = get_attr_as_string(self.group, "uncertainties")
        if uncertainties_names is None:
            uncertainties_names = get_attr_as_string(self.signal, "uncertainties")
        if isinstance(uncertainties_names, str):
            uncertainties_names = [uncertainties_names]
        if uncertainties_names is not None:
            # take the uncertainty with the same index as the axis in @axes
            axes_ds_names = get_attr_as_string(self.group, "axes")
            if axes_ds_names is None:
                axes_ds_names = get_attr_as_string(self.signal, "axes")
            if isinstance(axes_ds_names, str):
                axes_ds_names = [axes_ds_names]
            elif isinstance(axes_ds_names, numpy.ndarray):
                # transform numpy.ndarray into list
                axes_ds_names = list(axes_ds_names)
            assert isinstance(axes_ds_names, list)
            if hasattr(axes_ds_names[0], "decode"):
                axes_ds_names = [ax_name.decode("utf-8") for ax_name in axes_ds_names]
            if axis_name not in axes_ds_names:
                raise KeyError("group attr @axes does not mention a dataset " +
                               "named '%s'" % axis_name)
            errors = self.group[uncertainties_names[list(axes_ds_names).index(axis_name)]]
            if fg_idx == 0 and lg_idx == (len_axis - 1):
                return errors      # dataset
            else:
                return errors[fg_idx:lg_idx + 1]    # numpy array
        return None

    @property
    def errors(self):
        """Return errors (uncertainties) associated with the signal values.

        :return: Dataset with errors, or None
        """
        if "errors" not in self.group:
            return None
        return self.group["errors"]

    @property
    def is_scatter(self):
        """True if the signal is 1D and all the axes have the
        same size as the signal."""
        if self._is_scatter is not None:
            return self._is_scatter
        if not self.signal_is_1d:
            self._is_scatter = False
        else:
            self._is_scatter = True
            sigsize = 1
            for dim in self.signal.shape:
                sigsize *= dim
            for axis in self.axes:
                if axis is None:
                    continue
                axis_size = 1
                for dim in axis.shape:
                    axis_size *= dim
                self._is_scatter = self._is_scatter and (axis_size == sigsize)
        return self._is_scatter

    @property
    def is_x_y_value_scatter(self):
        """True if this is a scatter with a signal and two axes."""
        return self.is_scatter and len(self.axes) == 2

    # we currently have no widget capable of plotting 4D data
    @property
    def is_unsupported_scatter(self):
        """True if this is a scatter with a signal and more than 2 axes."""
        return self.is_scatter and len(self.axes) > 2
