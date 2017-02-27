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
        axes = group.attrs.get("axes")
        if isinstance(axes, str):
            axes = [axes]

        if ndim < len(axes):
            NXdata_warning(
                    "More @axes defined than there are " +
                    "signal dimensions: " +
                    "%d axes, %d dimensions." % (len(axes), ndim))
            return False

        # case of less axes than dimensions: number of axes must match
        # dimensionality defined by @interpretation
        if ndim > len(axes):
            interpretation = get_interpretation(group)
            if interpretation is None:
                NXdata_warning("No @interpretation and wrong" +
                               " number of @axes defined.")
                return False

            if interpretation not in INTERPDIM:
                NXdata_warning("Unrecognized @interpretation=" + interpretation +
                               " for data with wrong number of defined @axes.")
                return False

            if len(axes) != INTERPDIM[interpretation]:
                NXdata_warning(
                        "%d-D signal with @interpretation=%s " % (ndim, interpretation) +
                        "must define %d or %d axes." % (ndim, INTERPDIM[interpretation]))
                return False

        # Axes dataset must exist and be 1D (?)
        for axis in axes:
            if axis == ".":
                continue
            if axis not in group or not is_dataset(group[axis]):
                NXdata_warning("Could not find axis dataset '%s'" % axis)
                return False
            if len(group[axis].shape) != 1:
                # FIXME: is this a valid constraint?
                NXdata_warning("Axis %s is not a 1D dataset" % axis)
                return False

            if len(group[axis]) not in group[signal_name].shape:
                # TODO: test the len() vs the specific dimension this axes applies to
                NXdata_warning(
                        "Axis %s number of elements does not " % axis +
                        "correspond to the length of any signal dimension.")
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
        return f(group)
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

    :param group: h5py-like Group following the NeXus *NXdata* specification.
    :return: List of datasets whose names are specified in the *axes*
        attribute of *group*, sorted in the order in which they should be
        applied to the corresponding dimension of the signal dataset.
    :rtype: list[Dataset or None]
    """
    ndims = len(get_signal(group).shape)
    axes_names = group.attrs.get("axes")

    if axes_names is None:
        return [None for _i in range(ndims)]

    if isinstance(axes_names, str):
        axes_names = [axes_names]

    if len(axes_names) == ndims:
        # axes is a list of strings, one axis per dim is explicitely defined
        axes = [None] * ndims
        for i, axis_n in enumerate(axes_names):
            if axis_n != ".":
                try:
                    axes[i] = group[axis_n]
                except KeyError:
                    raise KeyError(
                            "No dataset matching axis name " + axis_n)
        return axes

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
    return axes


@validate_NXdata
def get_axes_names(group):
    """Return a list of axes names in a NXdata group.

    If an axis dataset applies to several dimensions of the signal, it
    will be repeated in the list.

    If a dimension of the signal has no dimension scale (i.e. there is a
    "." in that position in the *@axes* array), `None` is inserted in the
    output list in its position.


    :param group: h5py-like Group following the NeXus *NXdata* specification.
    :rtype: list[str or None]
    """
    axes_names = group.attrs.get("axes")
    if isinstance(axes_names, str):
         axes_names = [axes_names]

    ndims = len(get_signal(group).shape)
    if axes_names is None:
        axes_names = [None] * ndims
    if len(axes_names) == ndims:
        return axes_names

    interpretation = get_interpretation(group)
    assert len(axes_names) == INTERPDIM[interpretation]
    all_dimensions_names = [None] * (ndims - INTERPDIM[interpretation])
    for axis_name in axes_names:
        if axis_name == ".":
            all_dimensions_names.append(None)
        else:
            all_dimensions_names.append(axis_name)
    return all_dimensions_names


@validate_NXdata
def signal_is_0D(group):
    """Return True if NXdata signal dataset is 0-D or if
    *@interpretation="scalar"*

    :param group: h5py-like Group following the NeXus *NXdata* specification.
    :return: Boolean
    """
    ndim = len(get_signal(group).shape)

    if ndim == 0:
        return True
    if get_interpretation(group) == "scalar":
        return True
    return False


@validate_NXdata
def signal_is_1D(group):
    """Return True if NXdata signal dataset is 1-D and *@interpretation*
    is not *"scalar"*,  or if the dataset has more than 1 dimension
    and has *@interpretation="spectrum"*

    :param group: h5py-like Group following the NeXus *NXdata* specification.
    :return: Boolean
    """
    ndim = len(get_signal(group).shape)

    if ndim == 1 and get_interpretation(group) != "scalar":
        return True
    if ndim > 1 and get_interpretation(group) == "spectrum":
        return True
    return False


@validate_NXdata
def signal_is_2D(group):
    """Return True if NXdata signal dataset is 2-D and *@interpretation*
    is not *"spectrum"* and not *"scalar"*, or if the dataset has more than
    2 dimensions and has *@interpretation="image"*

    :param group: h5py-like Group following the NeXus *NXdata* specification.
    :return: Boolean
    """
    ndim = len(get_signal(group).shape)

    if ndim == 2 and get_interpretation(group) not in ["spectrum", "scalar"]:
        return True
    if ndim > 2 and get_interpretation(group) == "image":
        return True
    return False


@validate_NXdata
def signal_is_3D(group):
    """Return True if NXdata signal dataset is 3-D and *@interpretation*
    is not one of *["spectrum", "scalar", "image"]*, or if the dataset has
    more than 3 dimensions and has *@interpretation="vertex"*.

    :param group: h5py-like Group following the NeXus *NXdata* specification.
    :return: Boolean
    """
    ndim = len(get_signal(group).shape)
    interp = get_interpretation(group)

    if ndim == 3 and interp not in ["spectrum", "scalar", "image"]:
        return True
    if ndim > 3 and get_interpretation(group) == "vertex":
        return True
    return False
