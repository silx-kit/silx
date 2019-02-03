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
import six

from silx.io import is_dataset
from silx.utils.deprecation import deprecated


__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "17/04/2018"


nxdata_logger = logging.getLogger("silx.io.nxdata")


INTERPDIM = {"scalar": 0,
             "spectrum": 1,
             "image": 2,
             "rgba-image": 3,  # "hsla-image": 3, "cmyk-image": 3, # TODO
             "vertex": 1}  # 3D scatter: 1D signal + 3 axes (x, y, z) of same legth
"""Number of signal dimensions associated to each possible @interpretation
attribute.
"""


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


def get_uncertainties_names(group, signal_name):
    # Test consistency of @uncertainties
    uncertainties_names = get_attr_as_unicode(group, "uncertainties")
    if uncertainties_names is None:
        uncertainties_names = get_attr_as_unicode(group[signal_name], "uncertainties")
    if isinstance(uncertainties_names, six.text_type):
        uncertainties_names = [uncertainties_names]
    return uncertainties_names


def get_signal_name(group):
    """Return the name of the (main) signal in a NXdata group.
    Return None if this info is missing (invalid NXdata).

    """
    signal_name = get_attr_as_unicode(group, "signal", default=None)
    if signal_name is None:
        nxdata_logger.info("NXdata group %s does not define a signal attr. "
                           "Testing legacy specification.", group.name)
        for key in group:
            if "signal" in group[key].attrs:
                signal_name = key
                signal_attr = group[key].attrs["signal"]
                if signal_attr in [1, b"1", u"1"]:
                    # This is the main (default) signal
                    break
    return signal_name


def get_auxiliary_signals_names(group):
    """Return list of auxiliary signals names"""
    auxiliary_signals_names = get_attr_as_unicode(group, "auxiliary_signals",
                                                  default=[])
    if isinstance(auxiliary_signals_names, (six.text_type, six.binary_type)):
        auxiliary_signals_names = [auxiliary_signals_names]
    return auxiliary_signals_names


def validate_auxiliary_signals(group, signal_name, auxiliary_signals_names):
    """Check data dimensionality and size. Return False if invalid."""
    issues = []
    for asn in auxiliary_signals_names:
        if asn not in group or not is_dataset(group[asn]):
            issues.append(
                "Cannot find auxiliary signal dataset '%s'" % asn)
        elif group[signal_name].shape != group[asn].shape:
            issues.append("Auxiliary signal dataset '%s' does not" % asn +
                           " have the same shape as the main signal.")
    return issues


def validate_number_of_axes(group, signal_name, num_axes):
    issues = []
    ndims = len(group[signal_name].shape)
    if 1 < ndims < num_axes:
        # ndim = 1 with several axes could be a scatter
        issues.append(
            "More @axes defined than there are " +
            "signal dimensions: " +
            "%d axes, %d dimensions." % (num_axes, ndims))

    # case of less axes than dimensions: number of axes must match
    # dimensionality defined by @interpretation
    elif ndims > num_axes:
        interpretation = get_attr_as_unicode(group[signal_name], "interpretation")
        if interpretation is None:
            interpretation = get_attr_as_unicode(group, "interpretation")
        if interpretation is None:
            issues.append("No @interpretation and not enough" +
                          " @axes defined.")

        elif interpretation not in INTERPDIM:
            issues.append("Unrecognized @interpretation=" + interpretation +
                          " for data with wrong number of defined @axes.")
        elif interpretation == "rgba-image":
            if ndims != 3 or group[signal_name].shape[-1] not in [3, 4]:
                issues.append(
                    "Inconsistent RGBA Image. Expected 3 dimensions with " +
                    "last one of length 3 or 4. Got ndim=%d " % ndims +
                    "with last dimension of length %d." % group[signal_name].shape[-1])
            if num_axes != 2:
                issues.append(
                    "Inconsistent number of axes for RGBA Image. Expected "
                    "3, but got %d." % ndims)

        elif num_axes != INTERPDIM[interpretation]:
            issues.append(
                "%d-D signal with @interpretation=%s " % (ndims, interpretation) +
                "must define %d or %d axes." % (ndims, INTERPDIM[interpretation]))
    return issues
