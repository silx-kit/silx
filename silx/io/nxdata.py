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
"""This module provides a collection of functions to work with h5py-like
groups following the NeXus *NXdata* specification.

See http://download.nexusformat.org/sphinx/classes/base_classes/NXdata.html

"""
import logging
import os
import os.path
import numpy
from .utils import is_dataset, is_group, is_file
from silx.third_party import six

try:
    import h5py
except ImportError:
    h5py = None

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "12/02/2018"

_logger = logging.getLogger(__name__)


_INTERPDIM = {"scalar": 0,
              "spectrum": 1,
              "image": 2,
              "rgba-image": 3,  # "hsla-image": 3, "cmyk-image": 3, # TODO
              "vertex": 1}  # 3D scatter: 1D signal + 3 axes (x, y, z) of same legth
"""Number of signal dimensions associated to each possible @interpretation
attribute.
"""


def _nxdata_warning(msg, group_name=""):
    """Log a warning message prefixed with
    *"NXdata warning: "*

    :param str msg: Warning message
    :param str group_name: Name of NXdata group this warning relates to
    """
    warning_prefix = "NXdata warning"
    if group_name:
        warning_prefix += " (group %s): " % group_name
    else:
        warning_prefix += ": "
    _logger.warning(warning_prefix + msg)


def get_attr_as_string(item, attr_name, default=None):
    """Return item.attrs[attr_name]. If it is a byte-string or an array of
    byte-strings, return it as a default python string.

    For Python 3, this involves a coercion from bytes into unicode.
    For Python 2, there is nothing special to do, as strings are bytes.

    :param item: Group or dataset
    :param attr_name: Attribute name
    :param default: Value to be returned if attribute is not found.
    :return: item.attrs[attr_name]
    """
    attr = item.attrs.get(attr_name, default)
    if six.PY2:
        if isinstance(attr, six.text_type):
            # unicode
            return attr.encode("utf-8")
        else:
            return attr
    if six.PY3:
        if hasattr(attr, "decode"):
            # byte-string
            return attr.decode("utf-8")
        elif isinstance(attr, numpy.ndarray) and not attr.shape and\
                hasattr(attr[()], "decode"):
            # byte string as ndarray scalar
            return attr[()].decode("utf-8")
        elif isinstance(attr, numpy.ndarray) and len(attr.shape) and\
                hasattr(attr[0], "decode"):
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
    :raise TypeError: if group is not a h5py group, a spech5 group,
        or a fabioh5 group
    """
    if not is_group(group):
        raise TypeError("group must be a h5py-like group")
    if get_attr_as_string(group, "NX_class") != "NXdata":
        return False
    if "signal" not in group.attrs:
        _logger.info("NXdata group %s does not define a signal attr. "
                     "Testing legacy specification.", group.name)
        signal_name = None
        for key in group:
            if "signal" in group[key].attrs:
                signal_name = key
                signal_attr = group[key].attrs["signal"]
                if signal_attr in [1, b"1", u"1"]:
                    # This is the main (default) signal
                    break
        if signal_name is None:
            _nxdata_warning("No @signal attribute on the NXdata group, "
                            "and no dataset with a @signal=1 attr found",
                            group.name)
            return False
    else:
        signal_name = get_attr_as_string(group, "signal")

    if signal_name not in group or not is_dataset(group[signal_name]):
        _nxdata_warning(
            "Cannot find signal dataset '%s'" % signal_name,
            group.name)
        return False

    auxiliary_signals_names = get_attr_as_string(group, "auxiliary_signals",
                                                 default=[])
    if isinstance(auxiliary_signals_names, (six.text_type, six.binary_type)):
        auxiliary_signals_names = [auxiliary_signals_names]
    for asn in auxiliary_signals_names:
        if asn not in group or not is_dataset(group[asn]):
            _nxdata_warning(
                "Cannot find auxiliary signal dataset '%s'" % asn,
                group.name)
            return False
        if group[signal_name].shape != group[asn].shape:
            _nxdata_warning("Auxiliary signal dataset '%s' does not" % asn +
                            " have the same shape as the main signal.",
                            group.name)
            return False

    ndim = len(group[signal_name].shape)

    if "axes" in group.attrs:
        axes_names = get_attr_as_string(group, "axes")
        if isinstance(axes_names, (six.text_type, six.binary_type)):
            axes_names = [axes_names]

        if 1 < ndim < len(axes_names):
            # ndim = 1 with several axes could be a scatter
            _nxdata_warning(
                "More @axes defined than there are " +
                "signal dimensions: " +
                "%d axes, %d dimensions." % (len(axes_names), ndim),
                group.name)
            return False

        # case of less axes than dimensions: number of axes must match
        # dimensionality defined by @interpretation
        if ndim > len(axes_names):
            interpretation = get_attr_as_string(group[signal_name], "interpretation")
            if interpretation is None:
                interpretation = get_attr_as_string(group, "interpretation")
            if interpretation is None:
                _nxdata_warning("No @interpretation and not enough" +
                                " @axes defined.", group.name)
                return False

            if interpretation not in _INTERPDIM:
                _nxdata_warning("Unrecognized @interpretation=" + interpretation +
                                " for data with wrong number of defined @axes.",
                                group.name)
                return False
            if interpretation == "rgba-image":
                if ndim != 3 or group[signal_name].shape[-1] not in [3, 4]:
                    _nxdata_warning(
                        "Inconsistent RGBA Image. Expected 3 dimensions with " +
                        "last one of length 3 or 4. Got ndim=%d " % ndim +
                        "with last dimension of length %d." % group[signal_name].shape[-1],
                        group.name)
                    return False
                if len(axes_names) != 2:
                    _nxdata_warning(
                        "Inconsistent number of axes for RGBA Image. Expected "
                        "3, but got %d." % ndim, group.name)
                    return False

            elif len(axes_names) != _INTERPDIM[interpretation]:
                _nxdata_warning(
                    "%d-D signal with @interpretation=%s " % (ndim, interpretation) +
                    "must define %d or %d axes." % (ndim, _INTERPDIM[interpretation]),
                    group.name)
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
                                "number of fields than @axes", group.name)
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
                _nxdata_warning("Could not find axis dataset '%s'" % axis_name,
                                group.name)
                return False

            axis_size = 1
            for dim in group[axis_name].shape:
                axis_size *= dim

            if len(group[axis_name].shape) != 1:
                # too me, it makes only sense to have a n-D axis if it's total
                # size is exactly the signal's size (weird n-d scatter)
                if axis_size != signal_size:
                    _nxdata_warning("Axis %s is not a 1D dataset" % axis_name +
                                    " and its shape does not match the signal's shape",
                                    group.name)
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
                        " and this does not seem to be a scatter plot.", group.name)
                    return False
                elif axis_len in (1, 2):
                    polynomial_axes_names.append(axis_name)
                is_scatter = False
            else:
                if not is_scatter:
                    _nxdata_warning(
                        "Axis %s number of elements is equal " % axis_name +
                        "to the length of the signal, but this does not seem" +
                        " to be a scatter (other axes have different sizes)",
                        group.name)
                    return False

            # Test individual uncertainties
            errors_name = axis_name + "_errors"
            if errors_name not in group and uncertainties_names is not None:
                errors_name = uncertainties_names[i]
                if errors_name in group and axis_name not in polynomial_axes_names:
                    if group[errors_name].shape != group[axis_name].shape:
                        _nxdata_warning(
                            "Errors '%s' does not have the same " % errors_name +
                            "dimensions as axis '%s'." % axis_name, group.name)
                        return False

    # test dimensions of errors associated with signal
    if "errors" in group and is_dataset(group["errors"]):
        if group["errors"].shape != group[signal_name].shape:
            _nxdata_warning("Dataset containing standard deviations must " +
                            "have the same dimensions as the signal.",
                            group.name)
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

        self.signal = self.group[self.signal_dataset_name]
        """Main signal dataset in this NXdata group.

        In case more than one signal is present in this group,
        the other ones can be found in :attr:`auxiliary_signals`.
        """

        self.signal_name = get_attr_as_string(self.signal, "long_name")
        """Signal long name, as specified in the @long_name attribute of the
        signal dataset. If not specified, the dataset name is used."""
        if self.signal_name is None:
            self.signal_name = self.signal_dataset_name

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
    def signal_dataset_name(self):
        """Name of the main signal dataset."""
        signal_dataset_name = get_attr_as_string(self.group, "signal")
        if signal_dataset_name is None:
            # find a dataset with @signal == 1
            for dsname in self.group:
                signal_attr = self.group[dsname].attrs.get("signal")
                if signal_attr in [1, b"1", u"1"]:
                    # This is the main (default) signal
                    signal_dataset_name = dsname
                    break
        assert signal_dataset_name is not None
        return signal_dataset_name

    @property
    def auxiliary_signals_dataset_names(self):
        """Sorted list of names of the auxiliary signals datasets.

        These are the names provided by the *@auxiliary_signals* attribute
        on the NXdata group.

        In case the NXdata group does not specify a *@signal* attribute
        but has a dataset with an attribute *@signal=1*,
        we look for datasets with attributes *@signal=2, @signal=3...*
        (deprecated NXdata specification)."""
        signal_dataset_name = get_attr_as_string(self.group, "signal")
        if signal_dataset_name is not None:
            auxiliary_signals_names = get_attr_as_string(self.group, "auxiliary_signals")
            if auxiliary_signals_names is not None:
                if not isinstance(auxiliary_signals_names,
                                  (tuple, list, numpy.ndarray)):
                    # tolerate a single string, but coerce into a list
                    return [auxiliary_signals_names]
                return list(auxiliary_signals_names)
            return []

        # try old spec, @signal=1 (2, 3...) on dataset
        numbered_names = []
        for dsname in self.group:
            if dsname == self.signal_dataset_name:
                # main signal, not auxiliary
                continue
            ds = self.group[dsname]
            signal_attr = ds.attrs.get("signal")
            if signal_attr is not None and not is_dataset(ds):
                _logger.warning("Item %s with @signal=%s is not a dataset (%s)",
                                dsname, signal_attr, type(ds))
                continue
            if signal_attr is not None:
                try:
                    signal_number = int(signal_attr)
                except (ValueError, TypeError):
                    _logger.warning("Could not parse attr @signal=%s on "
                                    "dataset %s as an int",
                                    signal_attr, dsname)
                    continue
                numbered_names.append((signal_number, dsname))
        return [a[1] for a in sorted(numbered_names)]

    @property
    def auxiliary_signals_names(self):
        """List of names of the auxiliary signals.

        Similar to :attr:`auxiliary_signals_dataset_names`, but the @long_name
        is used when this attribute is present, instead of the dataset name.
        """
        signal_names = []
        for asdn in self.auxiliary_signals_dataset_names:
            if "long_name" in self.group[asdn].attrs:
                signal_names.append(self.group[asdn].attrs["long_name"])
            else:
                signal_names.append(asdn)
        return signal_names

    @property
    def auxiliary_signals(self):
        """List of all auxiliary signal datasets."""
        return [self.group[dsname] for dsname in self.auxiliary_signals_dataset_names]

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
                                   "rgba-image",  # "hsla-image", "cmyk-image"
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
        signal dataset, the exception being scatter plots which use a 1D
        signal and multiple 1D axes of the same size.

        If an axis dataset applies to several dimensions of the signal, it
        will be repeated in the list.

        If a dimension of the signal has no dimension scale, `None` is
        inserted in its position in the list.

        .. note::

            The *@axes* attribute should define as many entries as there
            are dimensions in the signal, to avoid  any ambiguity.
            If this is not the case, this implementation relies on the existence
            of an *@interpretation* (*spectrum* or *image*) attribute in the
            *signal* dataset.

        .. note::

            If an axis dataset defines attributes @first_good or @last_good,
            the output will be a numpy array resulting from slicing that
            axis (*axis[first_good:last_good + 1]*).

        :rtype: List[Dataset or 1D array or None]
        """
        if self._axes is not None:
            # use cache
            return self._axes
        axes = []
        for axis_name in self.axes_dataset_names:
            if axis_name is None:
                axes.append(None)
            else:
                axes.append(self.group[axis_name])

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
        """List of axes dataset names.

        If an axis dataset applies to several dimensions of the signal, its
        name will be repeated in the list.

        If a dimension of the signal has no dimension scale (i.e. there is a
        "." in that position in the *@axes* array), `None` is inserted in the
        output list in its position.
        """
        numbered_names = []     # used in case of @axis=0 (old spec)
        axes_dataset_names = get_attr_as_string(self.group, "axes")
        if axes_dataset_names is None:
            # try @axes on signal dataset (older NXdata specification)
            axes_dataset_names = get_attr_as_string(self.signal, "axes")
            if axes_dataset_names is not None:
                # we expect a comma separated string
                if hasattr(axes_dataset_names, "split"):
                    axes_dataset_names = axes_dataset_names.split(":")
            else:
                # try @axis on the individual datasets (oldest NXdata specification)
                for dsname in self.group:
                    if not is_dataset(self.group[dsname]):
                        continue
                    axis_attr = self.group[dsname].attrs.get("axis")
                    if axis_attr is not None:
                        try:
                            axis_num = int(axis_attr)
                        except (ValueError, TypeError):
                            _logger.warning("Could not interpret attr @axis as"
                                            "int on dataset %s", dsname)
                            continue
                        numbered_names.append((axis_num, dsname))

        ndims = len(self.signal.shape)
        if axes_dataset_names is None:
            if numbered_names:
                axes_dataset_names = []
                numbers = [a[0] for a in numbered_names]
                names = [a[1] for a in numbered_names]
                for i in range(ndims):
                    if i in numbers:
                        axes_dataset_names.append(names[numbers.index(i)])
                    else:
                        axes_dataset_names.append(None)
                return axes_dataset_names
            else:
                return [None] * ndims

        if isinstance(axes_dataset_names, (six.text_type, six.binary_type)):
            axes_dataset_names = [axes_dataset_names]

        for i, axis_name in enumerate(axes_dataset_names):
            if hasattr(axis_name, "decode"):
                axis_name = axis_name.decode()
            if axis_name == ".":
                axes_dataset_names[i] = None

        if len(axes_dataset_names) != ndims:
            if self.is_scatter and ndims == 1:
                # case of a 1D signal with arbitrary number of axes
                return list(axes_dataset_names)
            if self.interpretation != "rgba-image":
                # @axes may only define 1 or 2 axes if @interpretation=spectrum/image.
                # Use the existing names for the last few dims, and prepend with Nones.
                assert len(axes_dataset_names) == _INTERPDIM[self.interpretation]
                all_dimensions_names = [None] * (ndims - _INTERPDIM[self.interpretation])
                for axis_name in axes_dataset_names:
                    all_dimensions_names.append(axis_name)
            else:
                # 2 axes applying to the first two dimensions.
                # The 3rd signal dimension is expected to contain 3(4) RGB(A) values.
                assert len(axes_dataset_names) == 2
                all_dimensions_names = [axn for axn in axes_dataset_names]
                all_dimensions_names.append(None)
            return all_dimensions_names

        return list(axes_dataset_names)

    @property
    def title(self):
        """Plot title. If not found, returns an empty string.

        This attribute does not appear in the NXdata specification, but it is
        implemented in *nexpy* as a dataset named "title" inside the NXdata
        group. This dataset is expected to contain text.

        Because the *nexpy* approach could cause a conflict if the signal
        dataset or an axis dataset happened to be called "title", we also
        support providing the title as an attribute of the NXdata group.
        """
        title = self.group.get("title")
        data_dataset_names = [self.signal_name] + self.axes_dataset_names
        if (title is not None and is_dataset(title) and
                "title" not in data_dataset_names):
            return str(title[()])

        title = self.group.attrs.get("title")
        if title is None:
            return ""
        return str(title)

    def get_axis_errors(self, axis_name):
        """Return errors (uncertainties) associated with an axis.

        If the axis has attributes @first_good or @last_good, the output
        is trimmed accordingly (a numpy array will be returned rather than a
        dataset).

        :param str axis_name: Name of axis dataset. This dataset **must exist**.
        :return: Dataset with axis errors, or None
        :raise KeyError: if this group does not contain a dataset named axis_name
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

    @property
    def is_curve(self):
        """This property is True if the signal is 1D or :attr:`interpretation` is
        *"spectrum"*, and there is at most one axis with a consistent length.
        """
        if self.signal_is_0d or self.interpretation not in [None, "spectrum"]:
            return False
        # the axis, if any, must be of the same length as the last dimension
        # of the signal, or of length 2 (a + b *x scale)
        if self.axes[-1] is not None and len(self.axes[-1]) not in [
                self.signal.shape[-1], 2]:
            return False
        if self.interpretation is None:
            # We no longer test whether x values are monotonic
            # (in the past, in that case, we used to consider it a scatter)
            return self.signal_is_1d
        # everything looks good
        return True

    @property
    def is_image(self):
        """True if the signal is 2D, or 3D with last dimension of length 3 or 4
        and interpretation *rgba-image*, or >2D with interpretation *image*.
        The axes (if any) length must also be consistent with the signal shape.
        """
        if self.interpretation in ["scalar", "spectrum", "scaler"]:
            return False
        if self.signal_is_0d or self.signal_is_1d:
            return False
        if not self.signal_is_2d and \
                        self.interpretation not in ["image", "rgba-image"]:
            return False
        if self.signal_is_3d and self.interpretation == "rgba-image":
            if self.signal.shape[-1] not in [3, 4]:
                return False
            img_axes = self.axes[0:2]
            img_shape = self.signal.shape[0:2]
        else:
            img_axes = self.axes[-2:]
            img_shape = self.signal.shape[-2:]
        for i, axis in enumerate(img_axes):
            if axis is not None and len(axis) not in [img_shape[i], 2]:
                return False

        return True

    @property
    def is_stack(self):
        """True in the signal is at least 3D and interpretation is not
        "scalar", "spectrum", "image" or "rgba-image".
        The axes length must also be consistent with the last 3 dimensions
        of the signal.
        """
        if self.signal_ndim < 3 or self.interpretation in [
                "scalar", "scaler", "spectrum", "image", "rgba-image"]:
            return False
        stack_shape = self.signal.shape[-3:]
        for i, axis in enumerate(self.axes[-3:]):
            if axis is not None and len(axis) not in [stack_shape[i], 2]:
                return False
        return True


def is_NXentry_with_default_NXdata(group):
    """Return True if group is a valid NXentry defining a valid default
    NXdata."""
    if not is_group(group):
        return False

    if get_attr_as_string(group, "NX_class") != "NXentry":
        return False

    default_nxdata_name = group.attrs.get("default")
    if default_nxdata_name is None or default_nxdata_name not in group:
        return False

    default_nxdata_group = group.get(default_nxdata_name)

    if not is_group(default_nxdata_group):
        return False

    return is_valid_nxdata(default_nxdata_group)


def is_NXroot_with_default_NXdata(group):
    """Return True if group is a valid NXroot defining a default NXentry
    defining a valid default NXdata."""
    if not is_group(group):
        return False

    # A NXroot is supposed to be at the root of a data file, and @NX_class
    # is therefore optional. We accept groups that are not located at the root
    # if they have @NX_class=NXroot (use case: several nexus files archived
    # in a single HDF5 file)
    if get_attr_as_string(group, "NX_class") != "NXroot" and not is_file(group):
        return False

    default_nxentry_name = group.attrs.get("default")
    if default_nxentry_name is None or default_nxentry_name not in group:
        return False

    default_nxentry_group = group.get(default_nxentry_name)
    return is_NXentry_with_default_NXdata(default_nxentry_group)


def get_default(group):
    """Return a :class:`NXdata` object corresponding to the default NXdata group
    in the group specified as parameter.

    This function can find the NXdata if the group is already a NXdata, or
    if it is a NXentry defining a default NXdata, or if it is a NXroot
    defining such a default valid NXentry.

    Return None if no valid NXdata could be found.

    :param group: h5py-like group following the Nexus specification
        (NXdata, NXentry or NXroot).
    :return: :class:`NXdata` object or None
    :raise TypeError: if group is not a h5py-like group
    """
    if not is_group(group):
        raise TypeError("Provided parameter is not a h5py-like group")

    if is_NXroot_with_default_NXdata(group):
        default_entry = group[group.attrs["default"]]
        default_data = default_entry[default_entry.attrs["default"]]
    elif is_NXentry_with_default_NXdata(group):
        default_data = group[group.attrs["default"]]
    elif is_valid_nxdata(group):
        default_data = group
    else:
        return None

    return NXdata(default_data)


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
