# coding: utf-8
# /*##########################################################################
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
# ############################################################################*/
""" I/O utility functions"""

import numpy
import os.path
import sys
import time

import logging

try:
    import h5py
except ImportError as e:
    h5py_missing = True
    h5py_import_error = e
else:
    h5py_missing = False


__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "27/09/2016"


logger = logging.getLogger(__name__)

string_types = (basestring,) if sys.version_info[0] == 2 else (str,)  # noqa


def save1D(fname, x, y, xlabel=None, ylabels=None, filetype=None,
           fmt="%.7g", csvdelim=";", newline="\n", header="",
           footer="", comments="#", autoheader=False):
    """Saves any number of curves to various formats: `Specfile`, `CSV`,
    `txt` or `npy`. All curves must have the same number of points and share
    the same ``x`` values.

    :param fname: Output file path, or file handle open in write mode.
        If ``fname`` is a path, file is opened in ``w`` mode. Existing file
        with a same name will be overwritten.
    :param x: 1D-Array (or list) of abscissa values.
    :param y: 2D-array (or list of lists) of ordinates values. First index
        is the curve index, second index is the sample index. The length
        of the second dimension (number of samples) must be equal to
        ``len(x)``. ``y`` can be a 1D-array in case there is only one curve
        to be saved.
    :param filetype: Filetype: ``"spec", "csv", "txt", "ndarray"``.
        If ``None``, filetype is detected from file name extension
        (``.dat, .csv, .txt, .npy``).
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
    :param autoheader: In `CSV` or `txt`, ``True`` causes the first header
         line to be written as a standard CSV header line with column labels
         separated by the specified CSV delimiter.

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
            raise IOError("File type unspecified and could not be " +
                          "inferred from file extension (not in " +
                          "txt, dat, csv, npy)")
    else:
        filetype = filetype.lower()

    if filetype not in available_formats:
        raise IOError("File type %s is not supported" % (filetype))

    # default column headers
    if xlabel is None:
        xlabel = "x"
    if ylabels is None:
        if len(numpy.array(y).shape) > 1:
            ylabels = ["y%d" % i for i in range(len(y))]
        else:
            ylabels = ["y"]
    elif isinstance(ylabels, (list, tuple)):
        # if ylabels is provided as a list, every element must
        # be a string
        ylabels = [ylabels[i] if ylabels[i] is not None else "y%d" % i
                   for i in range(len(ylabels))]

    if filetype.lower() == "spec":
        y_array = numpy.asarray(y)

        # make sure y_array is a 2D array even for a single curve
        if len(y_array.shape) == 1:
            y_array.shape = (1, y_array.shape[0])
        elif len(y_array.shape) > 2 or len(y_array.shape) < 1:
            raise IndexError("y must be a 1D or 2D array")

        # First curve
        specf = savespec(fname, x, y_array[0], xlabel, ylabels[0], fmt=fmt,
                         scan_number=1, mode="w", write_file_header=True,
                         close_file=False)
        # Other curves
        for i in range(1, y_array.shape[0]):
            specf = savespec(specf, x, y_array[i], xlabel, ylabels[i],
                             fmt=fmt, scan_number=i + 1, mode="w",
                             write_file_header=False, close_file=False)
        # close file if we created it
        if not hasattr(fname, "write"):
            specf.close()

    else:
        autoheader_line = xlabel + csvdelim + csvdelim.join(ylabels)
        if xlabel is not None and ylabels is not None and filetype == "csv":
            # csv format: optional single header line with labels, no footer
            if autoheader:
                header = autoheader_line + newline
            else:
                header = ""
            comments = ""
            footer = ""
            newline = "\n"
        elif filetype == "txt" and autoheader:
            # Comments string is added at the beginning of header string in
            # savetxt(). We add another one after the first header line and
            # before the rest of the header.
            if header:
                header = autoheader_line + newline + comments + header
            else:
                header = autoheader_line + newline

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


# Replace with numpy.savetxt when dropping support of numpy < 1.7.0
def savetxt(fname, X, fmt="%.7g", delimiter=";", newline="\n",
            header="", footer="", comments="#"):
    """``numpy.savetxt`` backport of header and footer arguments from
    numpy=1.7.0.

    See ``numpy.savetxt`` help:
    http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.savetxt.html
    """
    if not hasattr(fname, "name"):
        ffile = open(fname, 'wb')
    else:
        ffile = fname

    if header:
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


def savespec(specfile, x, y, xlabel="X", ylabel="Y", fmt="%.7g",
             scan_number=1, mode="w", write_file_header=True,
             close_file=False):
    """Saves one curve to a SpecFile.

    The curve is saved as a scan with two data columns. To save multiple
    curves to a single SpecFile, call this function for each curve by
    providing the same file handle each time.

    :param specfile: Output SpecFile name, or file handle open in write
        or append mode. If a file name is provided, a new file is open in
        write mode (existing file with the same name will be lost)
    :param x: 1D-Array (or list) of abscissa values
    :param y: 1D-array (or list) of ordinates values
    :param xlabel: Abscissa label (default ``"X"``)
    :param ylabel: Ordinate label
    :param fmt: Format string for data. You can specify a short format
        string that defines a single format for both ``x`` and ``y`` values,
        or a list of two different format strings (e.g. ``["%d", "%.7g"]``).
        Default is ``"%.7g"``.
    :param scan_number: Scan number (default 1).
    :param mode: Mode for opening file: ``w`` (default), ``a``,  ``r+``,
        ``w+``, ``a+``. This parameter is only relevant if ``specfile`` is a
        path.
    :param write_file_header: If ``True``, write a file header before writing
        the scan (``#F`` and ``#D`` line).
    :param close_file: If ``True``, close the file after saving curve.
    :return: ``None`` if ``close_file`` is ``True``, else return the file
        handle.
    """
    x_array = numpy.asarray(x)
    y_array = numpy.asarray(y)

    if y_array.shape[0] != x_array.shape[0]:
        raise IndexError("X and Y columns must have the same length")

    if isinstance(fmt, string_types) and fmt.count("%") == 1:
        full_fmt_string = fmt + "  " + fmt + "\n"
    elif isinstance(fmt, (list, tuple)) and len(fmt) == 2:
        full_fmt_string = "  ".join(fmt) + "\n"
    else:
        raise ValueError("fmt must be a single format string or a list of " +
                         "two format strings")

    if not hasattr(specfile, "write"):
        f = open(specfile, mode)
    else:
        f = specfile

    current_date = "#D %s\n" % (time.ctime(time.time()))

    if write_file_header:
        f.write("#F %s\n" % f.name)
        f.write(current_date)
        f.write("\n")

    f.write("#S %d %s\n" % (scan_number, ylabel))
    f.write(current_date)
    f.write("#N 2\n")
    f.write("#L %s  %s\n" % (xlabel, ylabel))
    for i in range(y_array.shape[0]):
        f.write(full_fmt_string % (x_array[i], y_array[i]))
    f.write("\n")

    if close_file:
        f.close()
        return None
    return f


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

    .. note:: This function requires `h5py <http://www.h5py.org/>`_ to be
        installed.
    """
    if h5py_missing:
        logger.error("h5ls requires h5py")
        raise h5py_import_error

    h5repr = ''
    if isinstance(h5group, (h5py.File, h5py.Group)):
        h5f = h5group
    elif isinstance(h5group, string_types):
        h5f = h5py.File(h5group, "r")
    else:
        raise TypeError("h5group must be a h5py.group object or a file name.")

    for key in h5f.keys():
        if hasattr(h5f[key], 'keys'):
            h5repr += '\t' * lvl + '+' + key
            h5repr += '\n'
            h5repr += h5ls(h5f[key], lvl + 1)
        else:
            h5repr += '\t' * lvl
            h5repr += str(h5f[key])
            h5repr += '\n'

    if isinstance(h5group, string_types):
        h5f.close()

    return h5repr


def load(filename):
    """
    Load a file as an `h5py.File`-like object.

    Format supported:
    - h5 files, if `h5py` module is installed
    - Spec files if `SpecFile` module is installed

    :param str filename: A filename
    :raises: IOError if the file can't be loaded as an h5py.File like object
    :rtype: h5py.File
    """
    if not os.path.isfile(filename):
        raise IOError("Filename '%s' must be a file path" % filename)

    if not h5py_missing:
        if h5py.is_hdf5(filename):
            return h5py.File(filename)

    try:
        from . import spech5
        return spech5.SpecH5(filename)
    except ImportError:
        logger.debug("spech5 can't be loaded.", exc_info=True)
    except IOError:
        logger.debug("File '%s' can't be read as spec file.", filename, exc_info=True)

    raise IOError("File '%s' can't be read as HDF5" % filename)
