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
import os.path
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

# TODO: support more formats (.npy numpy.save), test
def save(path, x, y, xlabel=None, ylabels=None, fmt="%.7g",
         filetype=None, csvdelimiter=";", newline="\n", header="",
         footer="", comments="#"):
    """Saves any number of curves to SpecFile format.

    The output SpecFile has one scan with two columns (`x` and `y`) per curve.

    :param specfile: Output SpecFile name, or file handle open in write
        mode.
    :param x: 1D-Array (or list) of abscissa values.
    :param y: 2D-array (or list of lists) of ordinates values. First index
        is the curve index, second index is the sample index. The length
        of the second dimension (number of samples) must be equal to
        ``len(x)``. ``y`` can be a 1D-array may be supplied in case there is
        only one curve to save.
    :param xlabel: Abscissa label
    :param ylabels: List of `y` labels, or string of labels separated by
        two spaces
    :param datafmat: Format string for data. You can specify a short format
        string that defines a single format for both ``x`` and ``y`` values,
        or a list of two different format strings (e.g. ``["%d", "%.7g"]``).
        Default is ``"%.18e"``.
    """

    available_formats = ["dat", "csv"]

    if filetype is None:
        exttypes = {"dat": "spec",
                    "csv":"csv",
                    "npy": "npy"}
        fileext = os.path.splitext(filename)
        if fileext in exttypes:
            filetype = exttypes[fileext]


    if filetype.lower() not in available_formats:
        raise IOError("File type %s is not supported" % (filetype))

    # If we decide to set fmt=None as default
    # if fmt is None:
    #     fmt = []
    #     if hasattr(x, "dtype"):
    #         if x.dtype.kind in ["i", "u"]:
    #             fmt.append("%d")
    #         elif x.dtype.kind == "f":
    #             fmt.append("%.18e")  # TODO: check float32 ou 64
    #         elif x.dtype.kind in ["S", "U"]:
    #             fmt.append("%s")
    #     elif isinstance(x[0], (int, long)):
    #         fmt.append("%d")
    #     elif isinstance(x[0], float):
    #         fmt.append("%.18e")  # TODO: check float32 ou 64
    #     elif isinstance(x[0], string_types):
    #         fmt.append("%s")  # TODO: check float32 ou 64
    #
    #     # TODO: same for each column in y

    if filetype.lower() == "spec":
        # Spec format
        savespec(path, x, y, xlabel, ylabels, datafmt=fmt)
    else:
        X = numpy.vstack((x, y))
        if filetype.lower() == "csv":
            # fixme: test if we need .T, header not compatible with older numpy
            numpy.savetxt(path, X.T, fmt=fmt, delimiter=csvdelimiter,
                          newline=newline, header=header, footer=footer,
                          comments=comments)
        elif filetype.lower() == "npy":
            # X = numpy.recarray(X.shape, X.dtype, buf=X, names=…) TODO: add names
            numpy.save(path, X)


def savespec(specfile, x, y, xlabel=None, ylabels=None, fmt="%.7g"):
    """Saves any number of curves to SpecFile format.

    The output SpecFile has one scan with two columns (`x` and `y`) per curve.

    :param specfile: Output SpecFile name, or file handle open in write
        mode.
    :param x: 1D-Array (or list) of abscissa values.
    :param y: 2D-array (or list of lists) of ordinates values. First index
        is the curve index, second index is the sample index. The length
        of the second dimension (number of samples) must be equal to
        ``len(x)``. ``y`` can be a 1D-array may be supplied in case there is
        only one curve to save.
    :param xlabel: Abscissa label
    :param ylabels: List of `y` labels, or string of labels separated by
        two spaces
    :param fmat: Format string for data. You can specify a short format
        string that defines a single format for both ``x`` and ``y`` values,
        or a list of two different format strings (e.g. ``["%d", "%.7g"]``).
        Default is ``"%.7g"``.
    """
    if not hasattr(specfile, "write"):
        f = open(specfile, "w")
    else:
        f = specfile

    x_array = numpy.asarray(x)
    y_array = numpy.asarray(y)

    if xlabel is None:
        xlabel = "X"
    if ylabels is None:
        # set labels to  ["Y0", "Y1", …]
        ylabels = ["Y" + str(j) for j in range(y_array.shape[0])]

    # enforce list type for ylabels
    if isinstance(ylabels, string_types):
        ylabels = ylabels.split("  ")

    # make sure y_array is a 2D array even for a single curve
    if len(y_array.shape) == 1:
        y_array.shape = (1, y_array.shape[0])
    elif len(y_array.shape) > 2 or len(y_array.shape) < 1:
        raise IndexError("y must be a 1D or 2D array")

    f.write("#F %s\n" % f.name)
    current_date = "#D %s\n"%(time.ctime(time.time()))
    f.write(current_date)
    f.write("\n")

    if isinstance(fmt, string_types) and fmt.count("%") == 1:
        full_fmt_string = fmt + "  " + fmt + "\n"
    elif isinstance(fmt, (list, tuple)):
        full_fmt_string = "  ".join(fmt) + "\n"
    # custom user defined format string
    # FIXME: undocumented feature: remove or document
    elif isinstance(fmt, string_types) and fmt.count("%") == 2:
        full_fmt_string = fmt if "\n" in fmt.count else fmt + "\n"

    for i in range(y_array.shape[0]):
        if y_array.shape[1] != x_array.shape[0]:
            raise IndexError("X and Y columns must have the same length")

        f.write("#S %d %s\n" % (i + 1, ylabels[i]))
        f.write(current_date)
        f.write("#N 2\n")
        f.write("#L %s  %s\n" % (xlabel, ylabels[i]))
        for j in range(y_array.shape[1]):
            f.write(full_fmt_string % (x_array[j], y[i][j]))
        f.write("\n")

    if not hasattr(specfile, "write"):
        f.close()