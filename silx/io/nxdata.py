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

from .utils import is_dataset, is_group

_logger = logging.getLogger(__name__)


# TODO: tests (need to create a valid NXdata sample)


INTERPDIM = {"scalar": 0,
             "spectrum": 1,
             "image": 2,
             # "rgba-image": 3, "hsla-image": 3, "cmyk-image": 3, # TODO
             "vertex": 3,}
"""Number of data dimensions associated to each possible @interpretation
attribute.
"""


def NXdata_warning(msg):
    """Log a warning message prefixed with
    *"NXdata warning: "*

    :param str msg: Warning message
    """
    _logger.warning("NXdata warning: " + msg)


def is_valid(group):   # noqa
    """Check if a h5py group is a **valid** NX_data group.

    Warning messages are logged to troubleshoot malformed NXdata groups.

    :param group: h5py-like group
    :raise TypeError: if group is not a h5py group, a spech5 group,
        or a fabioh5 group
    """
    if not is_group(group):
        raise TypeError("group must be a h5py-like group")
    if group.attrs.get("NX_class") != "NXdata":
        return False
    if "signal" not in group.attrs:
        _logger.warning("NXdata group does not define a signal attr.")
        return False

    signal_name = group.attrs["signal"]
    if signal_name not in group or not is_dataset(group[signal_name]):
        _logger.warning(
                "Cannot find signal dataset '%s' in NXdata group" % signal_name)
        return False

    ndim = len(group[signal_name].shape)

    if "axes" in group.attrs:
        axes_names = group.attrs.get("axes")
        if isinstance(axes_names, str):
            axes_names = [axes_names]

        if ndim < len(axes_names):
            NXdata_warning(
                    "More @axes defined than there are " +
                    "signal dimensions: " +
                    "%d axes, %d dimensions." % (len(axes_names), ndim))
            return False

        # case of less axes than dimensions: number of axes must match
        # dimensionality defined by @interpretation
        if ndim > len(axes_names):
            interpretation = group[signal_name].attrs.get("interpretation", None)
            if interpretation is None:
                interpretation = group.attrs.get("interpretation", None)
            if interpretation is None:
                NXdata_warning("No @interpretation and wrong" +
                               " number of @axes defined.")
                return False

            if interpretation not in INTERPDIM:
                NXdata_warning("Unrecognized @interpretation=" + interpretation +
                               " for data with wrong number of defined @axes.")
                return False

            if len(axes_names) != INTERPDIM[interpretation]:
                NXdata_warning(
                        "%d-D signal with @interpretation=%s " % (ndim, interpretation) +
                        "must define %d or %d axes." % (ndim, INTERPDIM[interpretation]))
                return False

        # Test consistency of @uncertainties
        uncertainties_names = group.attrs.get("uncertainties")
        if uncertainties_names is None:
            uncertainties_names = group[signal_name].attrs.get("uncertainties")
        if isinstance(uncertainties_names, str):
            uncertainties_names = [uncertainties_names]
        if uncertainties_names is not None:
            if len(uncertainties_names) != len(axes_names):
                NXdata_warning("@uncertainties does not define the same " +
                               "number of fields than @axes")
                return False

        # Test individual axes
        for i, axis_name in enumerate(axes_names):
            if axis_name == ".":
                continue
            if axis_name not in group or not is_dataset(group[axis_name]):
                NXdata_warning("Could not find axis dataset '%s'" % axis_name)
                return False
            if len(group[axis_name].shape) != 1:
                # FIXME: is this a valid constraint?
                NXdata_warning("Axis %s is not a 1D dataset" % axis_name)
                return False

            if "first_good" in group[axis_name].attrs or "last_good" in group[axis_name].attrs:
                fg_idx = group[axis_name].attrs.get("first_good", 0)
                lg_idx = group[axis_name].attrs.get("last_good", len(group[axis_name]) - 1)
                axis_len = lg_idx + 1 - fg_idx
            else:
                axis_len = len(group[axis_name])
            if axis_len not in group[signal_name].shape:
                # TODO: test the len() vs the specific dimension this axes applies to
                NXdata_warning(
                        "Axis %s number of elements does not " % axis_name +
                        "correspond to the length of any signal dimension.")
                return False
            # Test individual uncertainties
            errors_name = axis_name + "_errors"
            if errors_name not in group and uncertainties_names is not None:
                errors_name = uncertainties_names[i]
                if errors_name in group:
                    if group[errors_name].shape != group[axis_name]:
                        NXdata_warning(
                            "Errors '%s' does not have the same " % errors_name +
                            "shape as axis '%s'." % axis_name)
                    return False
    return True


def validate_NXdata(f):
    """Wrapper for functions taking a NXdata group as the first argument
    or as a keyword argument named *group*.
    The NXdata group is validated prior to applying the wrapped function,
    and a *TypeError* is raised if it is not valid."""
    def wrapper(*args, **kwargs):
        group = kwargs.get("group", None)
        if group is None:
            group = args[0]
        if not is_valid(group):
            raise TypeError("group is not a valid NXdata class")
        return f(*args, **kwargs)
    return wrapper


@validate_NXdata
def get_signal(group):
    """Return the signal dataset in a NXdata group.

    :param group: h5py-like group following the NeXus *NXdata* specification.
    :return: Dataset whose name is specified in the *signal* attribute
        of *group*.
    :rtype: Dataset
    """
    return group[group.attrs["signal"]]


@validate_NXdata
def get_interpretation(group):
    """Return the *@interpretation* attribute associated with the *signal*
    dataset of an NXdata group. ``None`` is returned if no interpretation
    attribute is found.

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

    :param group: h5py-like Group following the NeXus *NXdata* specification.
    :return: Interpretation attribute associated with the *signal* dataset
        or with the NXdata group itself.
    :rtype: str or None
    """
    allowed_interpretations = [None, "scalar", "spectrum", "image",
                               # "rgba-image", "hsla-image", "cmyk-image"  # TODO
                               "vertex"]

    interpretation = get_signal(group).attrs.get("interpretation", None)
    if interpretation is None:
        interpretation = group.attrs.get("interpretation", None)

    if interpretation not in allowed_interpretations:
        _logger.warning("Interpretation %s is not valid." % interpretation +
                        " Valid values: " + ", ".join(allowed_interpretations))
    return interpretation


@validate_NXdata
def get_axes(group):
    """Return a list of the axes datasets in a NXdata group.

    The output list has as many elements as there are dimensions in the
    signal dataset.

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

    :param group: h5py-like Group following the NeXus *NXdata* specification.
    :return: List of datasets whose names are specified in the *axes*
        attribute of *group*, sorted in the order in which they should be
        applied to the corresponding dimension of the signal dataset.
    :rtype: list[Dataset or 1D array or None]
    """
    ndims = len(get_signal(group).shape)
    axes_names = group.attrs.get("axes")

    if axes_names is None:
        return [None for _i in range(ndims)]

    if isinstance(axes_names, str):
        axes_names = [axes_names]

    if len(axes_names) == ndims:
        # axes is a list of strings, one axis per dim is explicitly defined
        axes = [None] * ndims
        for i, axis_n in enumerate(axes_names):
            if axis_n != ".":
                try:
                    axes[i] = group[axis_n]
                except KeyError:
                    raise KeyError(
                            "No dataset matching axis name " + axis_n)
    else:
        # case of @interpretation attribute defined: we expect 1, 2 or 3 axes
        # corresponding to the 1, 2, or 3 last dimensions of the signal
        interpretation = get_interpretation(group)
        assert len(axes_names) == INTERPDIM[interpretation]
        axes = [None] * (ndims - INTERPDIM[interpretation])
        for axis_n in axes_names:
            if axis_n != ".":
                axes.append(group[axis_n])
            else:
                axes.append(None)
    # keep only good range of axis data
    for i, axis in enumerate(axes):
        if axis is None:
            continue
        if "first_good" not in axis.attrs and "last_good" not in axis.attrs:
            continue
        fg_idx = axis.attrs.get("first_good") or 0
        lg_idx = axis.attrs.get("last_good") or (len(axis) - 1)
        axes[i] = axis[fg_idx:lg_idx + 1]
    return axes


@validate_NXdata
def get_axes_dataset_names(group):
    """Return a list of axes datasest names in a NXdata group.

    If an axis dataset applies to several dimensions of the signal, its
    name will be repeated in the list.

    If a dimension of the signal has no dimension scale (i.e. there is a
    "." in that position in the *@axes* array), `None` is inserted in the
    output list in its position.
    """
    axes_dataset_names = group.attrs.get("axes")
    if axes_dataset_names is None:
       axes_dataset_names = get_signal(group).attrs.get("axes")

    ndims = len(get_signal(group).shape)
    if axes_dataset_names is None:
        return [None] * ndims

    if isinstance(axes_dataset_names, str):
        axes_dataset_names = [axes_dataset_names]

    if len(axes_dataset_names) != ndims:
        # @axes may only define 1 or 2 axes if @interpretation=spectrum/image.
        # Use the existing names for the last few dims, and prepend with Nones.
        interpretation = get_interpretation(group)
        assert len(axes_dataset_names) == INTERPDIM[interpretation]
        all_dimensions_names = [None] * (ndims - INTERPDIM[interpretation])
        for axis_name in axes_dataset_names:
            if axis_name == ".":
                all_dimensions_names.append(None)
            else:
                all_dimensions_names.append(axis_name)
        return all_dimensions_names

    for i, axis_name in enumerate(axes_dataset_names):
        if axis_name == ".":
            axes_dataset_names[i] = None
    return axes_dataset_names


@validate_NXdata
def get_axes_names(group):
    """Return a list of axes names in a NXdata group.

    This method is similar to :meth:`get_axes_dataset_names` except that
    if an axis dataset has a "@long_name" attribute, it will be returned
    instead of the dataset name.

    :param group: h5py-like Group following the NeXus *NXdata* specification.
    :rtype: list[str or None]
    """
    axes_dataset_names = get_axes_dataset_names(group)

    axes_names = []
    # check if axis dataset defines @long_name
    for i, dsname in enumerate(axes_dataset_names):
        if dsname is not None and "long_name" in group[dsname].attrs:
            axes_names.append(group[dsname].attrs["long_name"])
        else:
            axes_names.append(dsname)
    return axes_names


@validate_NXdata
def get_axis_errors(group, axis_name):
    """

    :param Group group: h5py-like group complying with the NeXus
        *NXdata* specification.
    :param str axis_name: Name of axis dataset. This dataset **must exist**.
    :return: Dataset with axis errors, or none
    :raise: KeyError if group does not contain a dataset named axis_name
    """
    if axis_name not in group:
        # tolerate axis_name given as @long_name
        for item in group:
            long_name = group[item].attrs.get("long_name")
            if long_name is not None and long_name == axis_name:
                axis_name = item
                break

    if axis_name not in group:
        raise KeyError("group does not contain a dataset named '%s'" % axis_name)

    # case of axisname_errors dataset present
    errors_name = axis_name + "_errors"
    if errors_name in group and is_dataset(group[errors_name]):
        return group[errors_name]
    # case of uncertainties dataset name provided in @uncertainties
    uncertainties_names = group.attrs.get("uncertainties")
    if uncertainties_names is None:
        uncertainties_names = get_signal(group).attrs.get("uncertainties")
    if isinstance(uncertainties_names, str):
        uncertainties_names = [uncertainties_names]
    if uncertainties_names is not None:
        # take the uncertainty with the same index as the axis in @axes
        axes_ds_names = group.attrs.get("axes")
        if axes_ds_names is None:
            axes_ds_names = get_signal(group).attrs.get("axes")
        if isinstance(axes_ds_names, str):
            axes_ds_names = [axes_ds_names]
        if axis_name not in axes_ds_names:
            raise KeyError("group attr @axes does not mention a dataset " +
                           "named '%s'" % axis_name)
        return uncertainties_names[axes_ds_names.index(axis_name)]
    return None


def signal_is_0D(group):
    """Return True if NXdata signal dataset is 0-D or if
    *@interpretation="scalar"*

    :param group: h5py-like Group following the NeXus *NXdata* specification.
    :return: Boolean
    """
    return len(get_signal(group).shape) == 0


def signal_is_1D(group):
    """Return True if NXdata signal dataset is 1-D

    :param group: h5py-like Group following the NeXus *NXdata* specification.
    :return: Boolean
    """
    return len(get_signal(group).shape) == 1


def signal_is_2D(group):
    """Return True if NXdata signal dataset is 2-D

    :param group: h5py-like Group following the NeXus *NXdata* specification.
    :return: Boolean
    """
    return len(get_signal(group).shape) == 2


def signal_is_3D(group):
    """Return True if NXdata signal dataset is 3-D

    :param group: h5py-like Group following the NeXus *NXdata* specification.
    :return: Boolean
    """
    return len(get_signal(group).shape) == 3


@validate_NXdata
def signal_is_scalar(group):
    """Return True if  *@interpretation="scalar"*

    :param group: h5py-like Group following the NeXus *NXdata* specification.
    :return: Boolean
    """
    # misspelled word "scaler" is used on NeXus website
    return get_interpretation(group) in ["scalar", "scaler"]


@validate_NXdata
def signal_is_spectrum(group):
    """Return True if *@interpretation="spectrum"*

    :param group: h5py-like Group following the NeXus *NXdata* specification.
    :return: Boolean
    """
    return get_interpretation(group) == "spectrum"


@validate_NXdata
def signal_is_spectrum(group):
    """Return True if *@interpretation="spectrum"*

    :param group: h5py-like Group following the NeXus *NXdata* specification.
    :return: Boolean
    """
    return get_interpretation(group) == "image"


@validate_NXdata
def signal_is_vertex(group):
    """Return True if *@interpretation="vertex"*

    :param group: h5py-like Group following the NeXus *NXdata* specification.
    :return: Boolean
    """
    return get_interpretation(group) == "vertex"
