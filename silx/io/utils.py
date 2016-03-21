# coding: utf-8
#/*##########################################################################
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
import numpy
import sys
import time

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "18/03/2016"

string_types = (basestring,) if sys.version_info[0] == 2 else (str,)


def repr_hdf5_tree(h5group, lvl=0):
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
            repr += repr_hdf5_tree(h5f[key], lvl + 1)
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


# todo: compare APIs (numpy.savetxt, PyMca5.PyMcaIO.SaveArray,
# PyMcaGui.plotting.PlotWindow.defaultSaveAction…)

def save_spec(specfile, x, y, xlabel=None, ylabels=None):
    """Saves any number of curves to SpecFile format

    :param specfile: Output Specfile name, or file handle open in write
        mode.
    :param x: 1D-Array (or list) of abscissa values
    :param y: 2D-array (or list of lists) of ordinates values. First index
        is the curve index, second index is the sample index. The length
        of the second dimension (number of samples) must be equal to
        ``len(x)``
    :param xlabel: Abscissa label as a string
    :param ylabels: List of columns labels or string of double space
        delimited labels
    """
    if not hasattr(specfile, write):
        f = open(specfile, "w")
    else:
        f = specfile

    x_array = numpy.asarray(x)
    y_array = numpy.asarray(y)

    # enforce list type for ylabels
    if isinstance(ylabels, string_types):
        ylabels = ylabels.split("  ")

    f.write("#F %s\n" % f.name)
    now = "#D %s\n"%(time.ctime(time.time()))
    f.write(now)
    f.write("#S 1 %s\n" % ylabels[0])
    f.write(now)
    f.write("#N 2\n")
    f.write("#L %s  %s\n" % (xlabel, ylabel[0]))

    # for i in range(len(y)):
    #     ffile.write("%.7g  %.7g\n" % (x[i], y[i]))
    # ffile.write("\n")
    if len(y_array.shape) == 1:
        # TODO: handle one curve
        raise NotImplementedError
    else:
        # TODO: handle multile curves
        raise NotImplementedError
savingDate = "#D %s\n"%(time.ctime(time.time()))


    if not hasattr(specfile, write):
        f.close()