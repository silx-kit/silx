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
"""Utility functions used by NXdata validation and parsing."""

import copy
import logging
import numpy

from silx.io import is_dataset
from silx.utils.deprecation import deprecated
from silx.third_party import six


__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "17/04/2018"


_INTERPDIM = {"scalar": 0,
              "spectrum": 1,
              "image": 2,
              "rgba-image": 3,  # "hsla-image": 3, "cmyk-image": 3, # TODO
              "vertex": 1}  # 3D scatter: 1D signal + 3 axes (x, y, z) of same legth
"""Number of signal dimensions associated to each possible @interpretation
attribute.
"""


_logger = logging.getLogger(__name__)


@deprecated(since_version="0.8.0", replacement="get_attr_as_unicode")
def get_attr_as_string(*args, **kwargs):
    return get_attr_as_unicode(*args, **kwargs)


def get_attr_as_unicode(item, attr_name, default=None):
    """Return item.attrs[attr_name] as unicode or as a
    list of unicode.

    Numpy arrays of strings or bytes returned by h5py are converted to
    lists of unicode.

    :param item: Group or dataset
    :param attr_name: Attribute name
    :param default: Value to be returned if attribute is not found.
    :return: item.attrs[attr_name]
    """
    attr = item.attrs.get(attr_name, default)

    if isinstance(attr, six.binary_type):
        # byte-string
        return attr.decode("utf-8")
    elif isinstance(attr, numpy.ndarray) and not attr.shape:
        if isinstance(attr[()], six.binary_type):
            # byte string as ndarray scalar
            return attr[()].decode("utf-8")
        else:
            # other scalar, possibly unicode
            return attr[()]
    elif isinstance(attr, numpy.ndarray) and len(attr.shape):
        if hasattr(attr[0], "decode"):
            # array of byte-strings
            return [element.decode("utf-8") for element in attr]
        else:
            # other array, most likely unicode objects
            return [element for element in attr]
    else:
        return copy.deepcopy(attr)


def _get_uncertainties_names(group, signal_name):
    # Test consistency of @uncertainties
    uncertainties_names = get_attr_as_unicode(group, "uncertainties")
    if uncertainties_names is None:
        uncertainties_names = get_attr_as_unicode(group[signal_name], "uncertainties")
    if isinstance(uncertainties_names, six.text_type):
        uncertainties_names = [uncertainties_names]
    return uncertainties_names


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


def _are_auxiliary_signals_valid(group, signal_name, auxiliary_signals_names):
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


def _has_valid_number_of_axes(group, signal_name, num_axes):
    ndims = len(group[signal_name].shape)
    if 1 < ndims < num_axes:
        # ndim = 1 with several axes could be a scatter
        _nxdata_warning(
            "More @axes defined than there are " +
            "signal dimensions: " +
            "%d axes, %d dimensions." % (num_axes, ndims),
            group.name)
        return False

    # case of less axes than dimensions: number of axes must match
    # dimensionality defined by @interpretation
    if ndims > num_axes:
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
            if ndims != 3 or group[signal_name].shape[-1] not in [3, 4]:
                _nxdata_warning(
                    "Inconsistent RGBA Image. Expected 3 dimensions with " +
                    "last one of length 3 or 4. Got ndim=%d " % ndims +
                    "with last dimension of length %d." % group[signal_name].shape[-1],
                    group.name)
                return False
            if num_axes != 2:
                _nxdata_warning(
                    "Inconsistent number of axes for RGBA Image. Expected "
                    "3, but got %d." % ndims, group.name)
                return False

        elif num_axes != _INTERPDIM[interpretation]:
            _nxdata_warning(
                "%d-D signal with @interpretation=%s " % (ndims, interpretation) +
                "must define %d or %d axes." % (ndims, _INTERPDIM[interpretation]),
                group.name)
            return False
    return True
