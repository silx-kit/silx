#/*##########################################################################
# coding: utf-8
# Copyright (C) 2016 European Synchrotron Radiation Facility
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
#############################################################################*/
""" I/O utility functions"""

from collections import defaultdict
import h5py
import sys

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "16/03/2016"

string_types = (basestring,) if sys.version_info[0] == 2 else (str,)


def repr_hdf5(h5group, lvl=0):
    """Return a string representation of an HDF5 tree structure.

    :param h5group: Any :class:`h5py.Group` or :class:`h5py.File` instance,
        or a HDF5 file name
    :param lvl: Number of tabulations added to the group. ``lvl`` is
        incremented as we recursively process sub-groups.
    :return: String representation of an HDF5 tree structure
    """
    repr = ''
    if isinstance(h5group, (h5py.File, h5py.Group)):
        h5f = h5group
    elif isinstance(h5group, string_types):
        h5f = h5py.File(h5group, "r")
    else:
        raise TypeError("h5group must be a HDF5 group object or a file name.")

    for key in h5f.keys():
        if hasattr(h5f[key], 'keys'):
            repr += '\t' * lvl + '+' + key
            repr += '\n'
            repr += repr_hdf5(h5f[key], lvl + 1)
        else:
            repr += '\t' * lvl
            repr += '-' + key + '=' + str(h5f[key])
            repr += '\n'

    if isinstance(h5group, string_types):
        h5f.close()

    return repr


def tree():
    """Initialize and return an empty nested dictionary tree-like structure.

    When accessing a non-existent key, it is automatically created, and
    initialized as an empty nested dictionary tree-like structure unless
    a value is assigned to it.

    Usage example::

        city_area = Tree()
        city_area["Europe"]["France"]["Isère"]["Grenoble"] = "18.44 km2"
        city_area["Europe"]["France"]["Nord"]["Tourcoing"] = "15.19 km2"

    is equivalent to::

        city_area = {
            "Europe": {
                "France": {
                    "Isère": {
                        "Grenoble": "18.44 km2"
                    },
                    "Nord": {
                        "Tourcoing": "15.19 km2"
                    },
                },
            },
        }
    """
    return defaultdict(tree)