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

import h5py
import numpy
import os.path
import sys
import time

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "06/04/2016"

string_types = (basestring,) if sys.version_info[0] == 2 else (str,)


def save(fname, x, y, xlabel=None, ylabels=None, filetype=None,
         fmt="%.7g", csvdelim=";", newline="\n", header="",
         footer="", comments="#"):
    """Saves any number of curves to various formats: `Specfile`, `CSV`,
    `txt` or `npy`.

    :param fname: Output file name, or file handle open in write
        mode.
    :param x: 1D-Array (or list) of abscissa values.
    :param y: 2D-array (or list of lists) of ordinates values. First index
        is the curve index, second index is the sample index. The length
        of the second dimension (number of samples) must be equal to
        ``len(x)``. ``y`` can be a 1D-array in case there is only one curve
        to be saved.
    :param filetype: Filetype: ``"spec", "csv", "txt", "ndarray"``.
        If ``None``, filetype is detected from file name extension
        (``.dat, .csv, .txt, .npy``)
    :param xlabel: Abscissa label
    :param ylabels: List of `y` labels
    :param fmt: Format string for data. You can specify a short format
        string that defines a single format for both ``x`` and ``y`` values,
        or a list of two different format strings (e.g. ``["%d", "%.7g"]``).
        Default is ``"%.7g"``.
        This parameter does not apply to the `npy` format.
    :param csvdelim: String or character separating columns in `txt` and
        `CSV` formats. The user is responsible for ensuring that this
        delimiter is not used in data labels when writing a `CSV` file.
    :param newline: String or character separating lines/records in `txt`
        format (default is line break character ``\\n``).
    :param header: String that will be written at the beginning of the file in
        `txt` format.
    :param footer: String that will be written at the end of the file in `txt`
         format.
    :param comments: String that will be prepended to the ``header`` and
        ``footer`` strings, to mark them as comments. Default: ``#``.

    When saving to Specfile format, each curve is saved as a separate scan
    with two data columns (``x`` and ``y``).

    `CSV` and `txt` formats are similar, except that the `txt` format allows
    user defined header and footer text blocks, whereas the `CSV` format has
    only a single header line with columns labels separated by field
    delimiters and no footer. The `txt` format also allows defining a record
    separator different from a line break.

    The `npy` format is written with ``numpy.save`` and can be read back with
    ``numpy.load``. If ``xlabel`` and ``ylabels`` are undefined, data is saved
    as a regular 2D ``numpy.ndarray`` (contatenation of ``x`` and ``y``). If
    both ``xlabel`` and ``ylabels`` are defined, the data is saved as a
    ``numpy.recarray`` after being transposed and having labels assigned to
    columns.
    """

    available_formats = ["spec", "csv", "txt", "ndarray"]

    if filetype is None:
        exttypes = {".dat": "spec",
                    ".csv": "csv",
                    ".txt": "txt",
                    ".npy": "ndarray"}
        outfname = (fname if not hasattr(fname, "name") else
                    fname.name)
        fileext = os.path.splitext(outfname)[1]
        if fileext in exttypes:
            filetype = exttypes[fileext]
    else:
        filetype = filetype.lower()

    if filetype not in available_formats:
        raise IOError("File type %s is not supported" % (filetype))

    if filetype.lower() == "spec":
        # Spec format
        savespec(fname, x, y, xlabel, ylabels, fmt=fmt)
    else:
        if xlabel is not None and ylabels is not None and filetype == "csv":
            # csv format: single header line with labels, no footer
            header = xlabel + csvdelim + csvdelim.join(ylabels)
            comments = ""
            footer = ""
            newline = "\n"

        # Concatenate x and y in a single 2D array
        X = numpy.vstack((x, y))

        if filetype.lower() in ["csv", "txt"]:
            X = X.transpose()
            savetxt(fname, X, fmt=fmt, delimiter=csvdelim,
                    newline=newline, header=header, footer=footer,
                    comments=comments)

        elif filetype.lower() == "ndarray":
            if xlabel is not None and ylabels is not None:
                labels = [xlabel] + ylabels

                # .transpose is needed here because recarray labels
                # apply to columns
                X = numpy.core.records.fromrecords(X.transpose(),
                                                   names=labels)
            numpy.save(fname, X)


def savetxt(fname, X, fmt="%.7g", delimiter=";", newline="\n",
            header="", footer="", comments="#"):
    """``numpy.savetxt`` backport of header and footer arguments from
    numpy=1.7.0.

    Replace with ``numpy.savetxt`` when dropping support of numpy < 1.7.0

    See ``numpy.savetxt`` help:
    http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.savetxt.html
    """
    if not hasattr(fname, "name"):
        ffile = open(fname, 'wb')
    else:
        ffile = fname

    if header:
        header = comments + header.replace(newline, newline + comments) +  newline
        if sys.version_info[0] >= 3:
            header = header.encode("utf-8")
        ffile.write(header)

    numpy.savetxt(ffile, X, fmt, delimiter, newline)

    if footer:
        footer = (comments + footer.replace(newline, newline + comments) +
                  newline)
        if sys.version_info[0] >= 3:
            footer = footer.encode("utf-8")
        ffile.write(footer)

    if not hasattr(fname, "name"):
        ffile.close()


def savespec(specfile, x, y, xlabel=None, ylabels=None, fmt="%.7g"):
    """Saves any number of curves to SpecFile format.

    The output SpecFile has one scan per curve, with two data columns each
    (`x` and `y`).

    :param specfile: Output SpecFile name, or file handle open in write
        mode.
    :param x: 1D-Array (or list) of abscissa values.
    :param y: 2D-array (or list of lists) of ordinates values. First index
        is the curve index, second index is the sample index. The length
        of the second dimension (number of samples) must be equal to
        ``len(x)``. ``y`` can be a 1D-array when there is only one curve
        to be saved.
    :param xlabel: Abscissa label
    :param ylabels: List of `y` labels, or string of labels separated by
        two spaces
    :param fmt: Format string for data. You can specify a short format
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
        # set labels to  ["Y0", "Y1", …]
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
    current_date = "#D %s\n" % (time.ctime(time.time()))
    f.write(current_date)
    f.write("\n")

    if isinstance(fmt, string_types) and fmt.count("%") == 1:
        full_fmt_string = fmt + "  " + fmt + "\n"
    elif isinstance(fmt, (list, tuple)):
        full_fmt_string = "  ".join(fmt) + "\n"
    else:
        raise ValueError("fmt must be a single format string or a list of " +
                         "two format strings")

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


def h5ls(h5group, lvl=0):
    """Return a simple string representation of a HDF5 tree structure.

    :param h5group: Any :class:`h5py.Group` or :class:`h5py.File` instance,
        or a HDF5 file name
    :param lvl: Number of tabulations added to the group. ``lvl`` is
        incremented as we recursively process sub-groups.
    :return: String representation of an HDF5 tree structure


    Group names and dataset representation are printed preceded by a number of
    tabulations corresponding to their depth in the tree structure.
    Datasets are represented as :class:`h5py.Dataset` objects.

    Example::

        >>> print(h5ls("Downloads/sample.h5"))
        +fields
            +fieldB
                <HDF5 dataset "z": shape (256, 256), type "<f4">
            +fieldE
                <HDF5 dataset "x": shape (256, 256), type "<f4">
                <HDF5 dataset "y": shape (256, 256), type "<f4">
    """
    repr = ''
    if isinstance(h5group, (h5py.File, h5py.Group)):
        h5f = h5group
    elif isinstance(h5group, string_types):
        h5f = h5py.File(h5group, "r")
    else:
        raise TypeError("h5group must be a h5py.group object or a file name.")

    for key in h5f.keys():
        if hasattr(h5f[key], 'keys'):
            repr += '\t' * lvl + '+' + key
            repr += '\n'
            repr += h5ls(h5f[key], lvl + 1)
        else:
            repr += '\t' * lvl
            repr += str(h5f[key])
            repr += '\n'

    if isinstance(h5group, string_types):
        h5f.close()

    return repr
