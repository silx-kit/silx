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
"""NXdata validation module.
"""
import logging

from silx.io.utils import is_dataset, is_group, is_file
from silx.third_party import six
from ._utils import get_attr_as_unicode, _INTERPDIM

try:
    import h5py
except ImportError:
    h5py = None

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "12/02/2018"

_logger = logging.getLogger(__name__)


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


def _get_signal_name(group):
    """Return the name of the (main) signal in a NXdata group.
    Return None if this info is missing (invalid NXdata).

    """
    signal_name = get_attr_as_unicode(group, "signal", default=None)
    if signal_name is None:
        _logger.info("NXdata group %s does not define a signal attr. "
                     "Testing legacy specification.", group.name)
        for key in group:
            if "signal" in group[key].attrs:
                signal_name = key
                signal_attr = group[key].attrs["signal"]
                if signal_attr in [1, b"1", u"1"]:
                    # This is the main (default) signal
                    break
    return signal_name


def _get_auxiliary_signals_names(group):
    """Return list of auxiliary signals names"""
    auxiliary_signals_names = get_attr_as_unicode(group, "auxiliary_signals",
                                                  default=[])
    if isinstance(auxiliary_signals_names, (six.text_type, six.binary_type)):
        auxiliary_signals_names = [auxiliary_signals_names]
    return auxiliary_signals_names


def _validate_auxiliary_signals(group, signal_name, auxiliary_signals_names):
    """Check data dimensionality and size. Return False if invalid."""
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
    return True


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
    if get_attr_as_unicode(group, "NX_class") != "NXdata":
        return False

    signal_name = _get_signal_name(group)
    if signal_name is None:
        _nxdata_warning("No @signal attribute on the NXdata group, "
                        "and no dataset with a @signal=1 attr found",
                        group.name)
        return False

    if signal_name not in group or not is_dataset(group[signal_name]):
        _nxdata_warning(
            "Cannot find signal dataset '%s'" % signal_name,
            group.name)
        return False

    auxiliary_signals_names = _get_auxiliary_signals_names(group)
    _validate_auxiliary_signals(group, signal_name, auxiliary_signals_names)

    ndim = len(group[signal_name].shape)

    if "axes" in group.attrs:
        axes_names = get_attr_as_unicode(group, "axes")
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
            interpretation = get_attr_as_unicode(group[signal_name], "interpretation")
            if interpretation is None:
                interpretation = get_attr_as_unicode(group, "interpretation")
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
        uncertainties_names = get_attr_as_unicode(group, "uncertainties")
        if uncertainties_names is None:
            uncertainties_names = get_attr_as_unicode(group[signal_name], "uncertainties")
        if isinstance(uncertainties_names, six.text_type):
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


def is_NXentry_with_default_NXdata(group, validate=True):
    """Return True if group is a valid NXentry defining a valid default
    NXdata.

    :param group: h5py-like object.
    :param bool validate: Set this to False if you are sure that the target group
        is valid NXdata (i.e. :func:`silx.io.nxdata.is_valid_nxdata(target_group)`
        returns True). Parameter provided for optimisation purposes."""
    if not is_group(group):
        return False

    if get_attr_as_unicode(group, "NX_class") != "NXentry":
        return False

    default_nxdata_name = group.attrs.get("default")
    if default_nxdata_name is None or default_nxdata_name not in group:
        return False

    default_nxdata_group = group.get(default_nxdata_name)

    if not is_group(default_nxdata_group):
        return False

    if not validate:
        return True
    else:
        return is_valid_nxdata(default_nxdata_group)


def is_NXroot_with_default_NXdata(group, validate=True):
    """Return True if group is a valid NXroot defining a default NXentry
    defining a valid default NXdata.

    :param group: h5py-like object.
    :param bool validate: Set this to False if you are sure that the target group
        is valid NXdata (i.e. :func:`silx.io.nxdata.is_valid_nxdata(target_group)`
        returns True). Parameter provided for optimisation purposes.
    """
    if not is_group(group):
        return False

    # A NXroot is supposed to be at the root of a data file, and @NX_class
    # is therefore optional. We accept groups that are not located at the root
    # if they have @NX_class=NXroot (use case: several nexus files archived
    # in a single HDF5 file)
    if get_attr_as_unicode(group, "NX_class") != "NXroot" and not is_file(group):
        return False

    default_nxentry_name = group.attrs.get("default")
    if default_nxentry_name is None or default_nxentry_name not in group:
        return False

    default_nxentry_group = group.get(default_nxentry_name)
    return is_NXentry_with_default_NXdata(default_nxentry_group,
                                          validate=validate)
