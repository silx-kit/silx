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

from silx.utils.deprecation import deprecated
import numpy
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
        return attr
