# /*##########################################################################
# Copyright (C) 2016-2022 European Synchrotron Radiation Facility
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

__authors__ = ["P. Knobel", "V. Valls"]
__license__ = "MIT"
__date__ = "03/12/2020"

import enum
import fnmatch
import os.path
import sys
import time
import logging
import collections
from typing import Generator
import urllib.parse

import numpy

from silx.utils.proxy import Proxy
import silx.io.url
from .._version import calc_hexversion

import h5py
import h5py.h5t
import h5py.h5a

try:
    import h5pyd
except ImportError as e:
    h5pyd = None

logger = logging.getLogger(__name__)

NEXUS_HDF5_EXT = [".h5", ".nx5", ".nxs", ".hdf", ".hdf5", ".cxi"]
"""List of possible extensions for HDF5 file formats."""


class H5Type(enum.Enum):
    """Identify a set of HDF5 concepts"""
    DATASET = 1
    GROUP = 2
    FILE = 3
    SOFT_LINK = 4
    EXTERNAL_LINK = 5
    HARD_LINK = 6


_CLASSES_TYPE = None
"""Store mapping between classes and types"""

string_types = (basestring,) if sys.version_info[0] == 2 else (str,)  # noqa

builtin_open = open


def supported_extensions(flat_formats=True):
    """Returns the list file extensions supported by `silx.open`.

    The result filter out formats when the expected module is not available.

    :param bool flat_formats: If true, also include flat formats like npy or
        edf (while the expected module is available)
    :returns: A dictionary indexed by file description and containing a set of
        extensions (an extension is a string like "\\*.ext").
    :rtype: Dict[str, Set[str]]
    """
    formats = collections.OrderedDict()
    formats["HDF5 files"] = set(["*.h5", "*.hdf", "*.hdf5"])
    formats["NeXus files"] = set(["*.nx", "*.nxs", "*.h5", "*.hdf", "*.hdf5"])
    formats["NeXus layout from spec files"] = set(["*.dat", "*.spec", "*.mca"])
    if flat_formats:
        try:
            from silx.io import fabioh5
        except ImportError:
            fabioh5 = None
        if fabioh5 is not None:
            formats["NeXus layout from fabio files"] = set(fabioh5.supported_extensions())

    extensions = ["*.npz"]
    if flat_formats:
        extensions.append("*.npy")

    formats["Numpy binary files"] = set(extensions)
    formats["Coherent X-Ray Imaging files"] = set(["*.cxi"])
    formats["FIO files"] = set(["*.fio"])
    return formats


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
        if numpy.array(y).ndim > 1:
            ylabels = ["y%d" % i for i in range(len(y))]
        else:
            ylabels = ["y"]
    elif isinstance(ylabels, (list, tuple)):
        # if ylabels is provided as a list, every element must
        # be a string
        ylabels = [ylabel if isinstance(ylabel, string_types) else "y%d" % i
                   for ylabel in ylabels]

    if filetype.lower() == "spec":
        # Check if we have regular data:
        ref = len(x)
        regular = True
        for one_y in y:
            regular &= len(one_y) == ref
        if regular:
            if isinstance(fmt, (list, tuple)) and len(fmt) < (len(ylabels) + 1):
                fmt = fmt + [fmt[-1] * (1 + len(ylabels) - len(fmt))]
            specf = savespec(fname, x, y, xlabel, ylabels, fmt=fmt,
                     scan_number=1, mode="w", write_file_header=True,
                     close_file=False)
        else:
            y_array = numpy.asarray(y)
            # make sure y_array is a 2D array even for a single curve
            if y_array.ndim == 1:
                y_array.shape = 1, -1
            elif y_array.ndim not in [1, 2]:
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
        ffile = builtin_open(fname, 'wb')
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
    :param y: 1D-array (or list), or list of them of ordinates values.
        All dataset must have the same length as x
    :param xlabel: Abscissa label (default ``"X"``)
    :param ylabel: Ordinate label, may be a list of labels when multiple curves 
        are to be saved together.
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
    # Make sure we use binary mode for write
    # (issue with windows: write() replaces \n with os.linesep in text mode)
    if "b" not in mode:
        first_letter = mode[0]
        assert first_letter in "rwa"
        mode = mode.replace(first_letter, first_letter + "b")

    x_array = numpy.asarray(x)
    y_array = numpy.asarray(y)
    if y_array.ndim > 2:
        raise IndexError("Y columns must have be packed as 1D")

    if y_array.shape[-1] != x_array.shape[0]:
        raise IndexError("X and Y columns must have the same length")

    if y_array.ndim == 2:
        assert isinstance(ylabel, (list, tuple))
        assert y_array.shape[0] == len(ylabel)
        labels = (xlabel, *ylabel)
    else:
        labels = (xlabel, ylabel)
    data = numpy.vstack((x_array, y_array))
    ncol = data.shape[0]
    assert len(labels) == ncol

    print(xlabel, ylabel, fmt, ncol, x_array, y_array)
    if isinstance(fmt, string_types) and fmt.count("%") == 1:
        full_fmt_string = "  ".join([fmt] * ncol)
    elif isinstance(fmt, (list, tuple)) and len(fmt) == ncol:
        full_fmt_string = "  ".join(fmt)
    else:
        raise ValueError("`fmt` must be a single format string or a list of " +
                         "format strings with as many format as ncolumns")

    if not hasattr(specfile, "write"):
        f = builtin_open(specfile, mode)
    else:
        f = specfile

    current_date = "#D %s" % (time.ctime(time.time()))
    if write_file_header:
        lines = [ "#F %s" % f.name, current_date, ""]
    else:
        lines = [""]

    lines += [  "#S %d %s" % (scan_number, labels[1]),
                current_date,
                "#N %d" % ncol,
                "#L " + "  ".join(labels)]

    for i in data.T:
        lines.append(full_fmt_string % tuple(i))
    lines.append("")
    output = "\n".join(lines)
    f.write(output.encode())

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
    h5repr = ''
    if is_group(h5group):
        h5f = h5group
    elif isinstance(h5group, string_types):
        h5f = open(h5group)  # silx.io.open
    else:
        raise TypeError("h5group must be a hdf5-like group object or a file name.")

    for key in h5f.keys():
        # group
        if hasattr(h5f[key], 'keys'):
            h5repr += '\t' * lvl + '+' + key
            h5repr += '\n'
            h5repr += h5ls(h5f[key], lvl + 1)
        # dataset
        else:
            h5repr += '\t' * lvl
            h5repr += str(h5f[key])
            h5repr += '\n'

    if isinstance(h5group, string_types):
        h5f.close()

    return h5repr


def _open_local_file(filename):
    """
    Load a file as an `h5py.File`-like object.

    Format supported:
    - h5 files, if `h5py` module is installed
    - SPEC files exposed as a NeXus layout
    - raster files exposed as a NeXus layout (if `fabio` is installed)
    - fio files exposed as a NeXus layout
    - Numpy files ('npy' and 'npz' files)

    The file is opened in read-only mode.

    :param str filename: A filename
    :raises: IOError if the file can't be loaded as an h5py.File like object
    :rtype: h5py.File
    """
    if not os.path.isfile(filename):
        raise IOError("Filename '%s' must be a file path" % filename)

    debugging_info = []
    try:
        _, extension = os.path.splitext(filename)

        if extension in [".npz", ".npy"]:
            try:
                from . import rawh5
                return rawh5.NumpyFile(filename)
            except (IOError, ValueError) as e:
                debugging_info.append((sys.exc_info(),
                                      "File '%s' can't be read as a numpy file." % filename))

        if h5py.is_hdf5(filename):
            try:
                return h5py.File(filename, "r")
            except OSError:
                return h5py.File(filename, "r", libver='latest', swmr=True)

        try:
            from . import fabioh5
            return fabioh5.File(filename)
        except ImportError:
            debugging_info.append((sys.exc_info(), "fabioh5 can't be loaded."))
        except Exception:
            debugging_info.append((sys.exc_info(),
                                   "File '%s' can't be read as fabio file." % filename))

        try:
            from . import spech5
            return spech5.SpecH5(filename)
        except ImportError:
            debugging_info.append((sys.exc_info(),
                                   "spech5 can't be loaded."))
        except IOError:
            debugging_info.append((sys.exc_info(),
                                   "File '%s' can't be read as spec file." % filename))

        try:
            from . import fioh5
            return fioh5.FioH5(filename)
        except IOError:
            debugging_info.append((sys.exc_info(),
                                   "File '%s' can't be read as fio file." % filename))

    finally:
        for exc_info, message in debugging_info:
            logger.debug(message, exc_info=exc_info)

    raise IOError("File '%s' can't be read as HDF5" % filename)


class _MainNode(Proxy):
    """A main node is a sub node of the HDF5 tree which is responsible of the
    closure of the file.

    It is a proxy to the sub node, plus support context manager and `close`
    method usually provided by `h5py.File`.

    :param h5_node: Target to the proxy.
    :param h5_file: Main file. This object became the owner of this file.
    """

    def __init__(self, h5_node, h5_file):
        super(_MainNode, self).__init__(h5_node)
        self.__node = h5_node
        self.__file = h5_file
        self.__class = get_h5_class(h5_node)

    @property
    def h5_class(self):
        """Returns the HDF5 class which is mimicked by this class.

        :rtype: H5Type
        """
        return self.__class

    @property
    def h5py_class(self):
        """Returns the h5py classes which is mimicked by this class. It can be
        one of `h5py.File, h5py.Group` or `h5py.Dataset`.

        :rtype: h5py class
        """
        return h5type_to_h5py_class(self.__class)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close the file"""
        self.__file.close()
        self.__file = None


def open(filename):  # pylint:disable=redefined-builtin
    """
    Open a file as an `h5py`-like object.

    Format supported:
    - h5 files, if `h5py` module is installed
    - SPEC files exposed as a NeXus layout
    - raster files exposed as a NeXus layout (if `fabio` is installed)
    - fio files exposed as a NeXus layout
    - Numpy files ('npy' and 'npz' files)

    The filename can be trailled an HDF5 path using the separator `::`. In this
    case the object returned is a proxy to the target node, implementing the
    `close` function and supporting `with` context.

    The file is opened in read-only mode.

    :param str filename: A filename which can containt an HDF5 path by using
        `::` separator.
    :raises: IOError if the file can't be loaded or path can't be found
    :rtype: h5py-like node
    """
    url = silx.io.url.DataUrl(filename)

    if url.scheme() in [None, "file", "silx"]:
        # That's a local file
        if not url.is_valid():
            raise IOError("URL '%s' is not valid" % filename)
        h5_file = _open_local_file(url.file_path())
    elif url.scheme() in ["fabio"]:
        raise IOError("URL '%s' containing fabio scheme is not supported" % filename)
    else:
        # That's maybe an URL supported by h5pyd
        uri = urllib.parse.urlparse(filename)
        if h5pyd is None:
            raise IOError("URL '%s' unsupported. Try to install h5pyd." % filename)
        path = uri.path
        endpoint = "%s://%s" % (uri.scheme, uri.netloc)
        if path.startswith("/"):
            path = path[1:]
        return h5pyd.File(path, 'r', endpoint=endpoint)

    if url.data_slice():
        raise IOError("URL '%s' containing slicing is not supported" % filename)

    if url.data_path() in [None, "/", ""]:
        # The full file is requested
        return h5_file
    else:
        # Only a children is requested
        if url.data_path() not in h5_file:
            msg = "File '%s' does not contain path '%s'." % (filename, url.data_path())
            raise IOError(msg)
        node = h5_file[url.data_path()]
        proxy = _MainNode(node, h5_file)
        return proxy


def _get_classes_type():
    """Returns a mapping between Python classes and HDF5 concepts.

    This function allow an lazy initialization to avoid recurssive import
    of modules.
    """
    global _CLASSES_TYPE
    from . import commonh5

    if _CLASSES_TYPE is not None:
        return _CLASSES_TYPE

    _CLASSES_TYPE = collections.OrderedDict()

    _CLASSES_TYPE[commonh5.Dataset] = H5Type.DATASET
    _CLASSES_TYPE[commonh5.File] = H5Type.FILE
    _CLASSES_TYPE[commonh5.Group] = H5Type.GROUP
    _CLASSES_TYPE[commonh5.SoftLink] = H5Type.SOFT_LINK

    _CLASSES_TYPE[h5py.Dataset] = H5Type.DATASET
    _CLASSES_TYPE[h5py.File] = H5Type.FILE
    _CLASSES_TYPE[h5py.Group] = H5Type.GROUP
    _CLASSES_TYPE[h5py.SoftLink] = H5Type.SOFT_LINK
    _CLASSES_TYPE[h5py.HardLink] = H5Type.HARD_LINK
    _CLASSES_TYPE[h5py.ExternalLink] = H5Type.EXTERNAL_LINK

    if h5pyd is not None:
        _CLASSES_TYPE[h5pyd.Dataset] = H5Type.DATASET
        _CLASSES_TYPE[h5pyd.File] = H5Type.FILE
        _CLASSES_TYPE[h5pyd.Group] = H5Type.GROUP
        _CLASSES_TYPE[h5pyd.SoftLink] = H5Type.SOFT_LINK
        _CLASSES_TYPE[h5pyd.HardLink] = H5Type.HARD_LINK
        _CLASSES_TYPE[h5pyd.ExternalLink] = H5Type.EXTERNAL_LINK

    return _CLASSES_TYPE


def get_h5_class(obj=None, class_=None):
    """
    Returns the HDF5 type relative to the object or to the class.

    :param obj: Instance of an object
    :param class_: A class
    :rtype: H5Type
    """
    if class_ is None:
        class_ = obj.__class__

    classes = _get_classes_type()
    t = classes.get(class_, None)
    if t is not None:
        return t

    if obj is not None:
        if hasattr(obj, "h5_class"):
            return obj.h5_class

    for referencedClass_, type_ in classes.items():
        if issubclass(class_, referencedClass_):
            classes[class_] = type_
            return type_

    classes[class_] = None
    return None


def h5type_to_h5py_class(type_):
    """
    Returns an h5py class from an H5Type. None if nothing found.

    :param H5Type type_:
    :rtype: H5py class
    """
    if type_ == H5Type.FILE:
        return h5py.File
    if type_ == H5Type.GROUP:
        return h5py.Group
    if type_ == H5Type.DATASET:
        return h5py.Dataset
    if type_ == H5Type.SOFT_LINK:
        return h5py.SoftLink
    if type_ == H5Type.HARD_LINK:
        return h5py.HardLink
    if type_ == H5Type.EXTERNAL_LINK:
        return h5py.ExternalLink
    return None


def get_h5py_class(obj):
    """Returns the h5py class from an object.

    If it is an h5py object or an h5py-like object, an h5py class is returned.
    If the object is not an h5py-like object, None is returned.

    :param obj: An object
    :return: An h5py object
    """
    if hasattr(obj, "h5py_class"):
        return obj.h5py_class
    type_ = get_h5_class(obj)
    return h5type_to_h5py_class(type_)


def is_file(obj):
    """
    True is the object is an h5py.File-like object.

    :param obj: An object
    """
    t = get_h5_class(obj)
    return t == H5Type.FILE


def is_group(obj):
    """
    True if the object is a h5py.Group-like object. A file is a group.

    :param obj: An object
    """
    t = get_h5_class(obj)
    return t in [H5Type.GROUP, H5Type.FILE]


def is_dataset(obj):
    """
    True if the object is a h5py.Dataset-like object.

    :param obj: An object
    """
    t = get_h5_class(obj)
    return t == H5Type.DATASET


def is_softlink(obj):
    """
    True if the object is a h5py.SoftLink-like object.

    :param obj: An object
    """
    t = get_h5_class(obj)
    return t == H5Type.SOFT_LINK


def is_externallink(obj):
    """
    True if the object is a h5py.ExternalLink-like object.

    :param obj: An object
    """
    t = get_h5_class(obj)
    return t == H5Type.EXTERNAL_LINK


def is_link(obj):
    """
    True if the object is a h5py link-like object.

    :param obj: An object
    """
    t = get_h5_class(obj)
    return t in {H5Type.SOFT_LINK, H5Type.EXTERNAL_LINK}


def _visitall(item, path=''):
    """Helper function for func:`visitall`.

    :param item: Item to visit
    :param str path: Relative path of the item
    """
    if not is_group(item):
        return

    for name, child_item in item.items():
        if isinstance(child_item, (h5py.Group, h5py.Dataset)):
            link = item.get(name, getlink=True)
        else:
            link = child_item
        child_path = '/'.join((path, name))

        ret = link if link is not None and is_link(link) else child_item
        yield child_path, ret
        yield from _visitall(child_item, child_path)


def visitall(item):
    """Visit entity recursively including links.

    It does not follow links.
    This is a generator yielding (relative path, object) for visited items.

    :param item: The item to visit.
    """
    yield from _visitall(item, '')



def match(group, path_pattern: str) -> Generator[str, None, None]:
    """Generator of paths inside given h5py-like `group` matching `path_pattern`"""
    if not is_group(group):
        raise ValueError(f"Not a h5py-like group: {group}")

    path_parts = path_pattern.strip("/").split("/", 1)
    for matching_path in fnmatch.filter(group.keys(), path_parts[0]):
        if len(path_parts) == 1:  # No more sub-path, stop recursion
            yield matching_path
            continue

        entity = group.get(matching_path)
        if is_group(entity):
            for matching_subpath in match(entity, path_parts[1]):
                yield f"{matching_path}/{matching_subpath}"


def get_data(url):
    """Returns a numpy data from an URL.

    Examples:

    >>> # 1st frame from an EDF using silx.io.open
    >>> data = silx.io.get_data("silx:/users/foo/image.edf::/scan_0/instrument/detector_0/data[0]")

    >>> # 1st frame from an EDF using fabio
    >>> data = silx.io.get_data("fabio:/users/foo/image.edf::[0]")

    Yet 2 schemes are supported by the function.

    - If `silx` scheme is used, the file is opened using
        :meth:`silx.io.open`
        and the data is reach using usually NeXus paths.
    - If `fabio` scheme is used, the file is opened using :meth:`fabio.open`
        from the FabIO library.
        No data path have to be specified, but each frames can be accessed
        using the data slicing.
        This shortcut of :meth:`silx.io.open` allow to have a faster access to
        the data.

    .. seealso:: :class:`silx.io.url.DataUrl`

    :param Union[str,silx.io.url.DataUrl]: A data URL
    :rtype: Union[numpy.ndarray, numpy.generic]
    :raises ImportError: If the mandatory library to read the file is not
        available.
    :raises ValueError: If the URL is not valid or do not match the data
    :raises IOError: If the file is not found or in case of internal error of
        :meth:`fabio.open` or :meth:`silx.io.open`. In this last case more
        informations are displayed in debug mode.
    """
    if not isinstance(url, silx.io.url.DataUrl):
        url = silx.io.url.DataUrl(url)

    if not url.is_valid():
        raise ValueError("URL '%s' is not valid" % url.path())

    if not os.path.exists(url.file_path()):
        raise IOError("File '%s' not found" % url.file_path())

    if url.scheme() == "silx":
        data_path = url.data_path()
        data_slice = url.data_slice()

        with open(url.file_path()) as h5:
            if data_path not in h5:
                raise ValueError("Data path from URL '%s' not found" % url.path())
            data = h5[data_path]

            if not silx.io.is_dataset(data):
                raise ValueError("Data path from URL '%s' is not a dataset" % url.path())

            if data_slice is not None:
                data = h5py_read_dataset(data, index=data_slice)
            else:
                # works for scalar and array
                data = h5py_read_dataset(data)

    elif url.scheme() == "fabio":
        import fabio
        data_slice = url.data_slice()
        if data_slice is None:
            data_slice = (0,)
        if data_slice is None or len(data_slice) != 1:
            raise ValueError("Fabio slice expect a single frame, but %s found" % data_slice)
        index = data_slice[0]
        if not isinstance(index, int):
            raise ValueError("Fabio slice expect a single integer, but %s found" % data_slice)

        try:
            fabio_file = fabio.open(url.file_path())
        except Exception:
            logger.debug("Error while opening %s with fabio", url.file_path(), exc_info=True)
            raise IOError("Error while opening %s with fabio (use debug for more information)" % url.path())

        if fabio_file.nframes == 1:
            if index != 0:
                raise ValueError("Only a single frame available. Slice %s out of range" % index)
            data = fabio_file.data
        else:
            data = fabio_file.getframe(index).data

        # There is no explicit close
        fabio_file = None

    else:
        raise ValueError("Scheme '%s' not supported" % url.scheme())

    return data


def rawfile_to_h5_external_dataset(bin_file, output_url, shape, dtype,
                                   overwrite=False):
    """
    Create a HDF5 dataset at `output_url` pointing to the given vol_file.

    Either `shape` or `info_file` must be provided.

    :param str bin_file: Path to the .vol file
    :param DataUrl output_url: HDF5 URL where to save the external dataset
    :param tuple shape: Shape of the volume
    :param numpy.dtype dtype: Data type of the volume elements (default: float32)
    :param bool overwrite: True to allow overwriting (default: False).
    """
    assert isinstance(output_url, silx.io.url.DataUrl)
    assert isinstance(shape, (tuple, list))
    v_majeur, v_mineur, v_micro = [int(i) for i in h5py.version.version.split('.')[:3]]
    if calc_hexversion(v_majeur, v_mineur, v_micro)< calc_hexversion(2,9,0):
        raise Exception('h5py >= 2.9 should be installed to access the '
                        'external feature.')

    with h5py.File(output_url.file_path(), mode="a") as _h5_file:
        if output_url.data_path() in _h5_file:
            if overwrite is False:
                raise ValueError('data_path already exists')
            else:
                logger.warning('will overwrite path %s' % output_url.data_path())
                del _h5_file[output_url.data_path()]
        external = [(bin_file, 0, h5py.h5f.UNLIMITED)]
        _h5_file.create_dataset(output_url.data_path(),
                                shape,
                                dtype=dtype,
                                external=external)


def vol_to_h5_external_dataset(vol_file, output_url, info_file=None,
                               vol_dtype=numpy.float32, overwrite=False):
    """
    Create a HDF5 dataset at `output_url` pointing to the given vol_file.

    If the vol_file.info containing the shape is not on the same folder as the
     vol-file then you should specify her location.

    :param str vol_file: Path to the .vol file
    :param DataUrl output_url: HDF5 URL where to save the external dataset
    :param Union[str,None] info_file:
        .vol.info file name written by pyhst and containing the shape information
    :param numpy.dtype vol_dtype: Data type of the volume elements (default: float32)
    :param bool overwrite: True to allow overwriting (default: False).
    :raises ValueError: If fails to read shape from the .vol.info file
    """
    _info_file = info_file
    if _info_file is None:
        _info_file = vol_file + '.info'
        if not os.path.exists(_info_file):
            logger.error('info_file not given and %s does not exists, please'
                         'specify .vol.info file' % _info_file)
            return

    def info_file_to_dict():
        ddict = {}
        with builtin_open(info_file, "r") as _file:
            lines = _file.readlines()
            for line in lines:
                if not '=' in line:
                    continue
                l = line.rstrip().replace(' ', '')
                l = l.split('#')[0]
                key, value = l.split('=')
                ddict[key.lower()] = value
        return ddict

    ddict = info_file_to_dict()
    if 'num_x' not in ddict or 'num_y' not in ddict or 'num_z' not in ddict:
        raise ValueError(
            'Unable to retrieve volume shape from %s' % info_file)

    dimX = int(ddict['num_x'])
    dimY = int(ddict['num_y'])
    dimZ = int(ddict['num_z'])
    shape = (dimZ, dimY, dimX)

    return rawfile_to_h5_external_dataset(bin_file=vol_file,
                                          output_url=output_url,
                                          shape=shape,
                                          dtype=vol_dtype,
                                          overwrite=overwrite)


def h5py_decode_value(value, encoding="utf-8", errors="surrogateescape"):
    """Keep bytes when value cannot be decoded

    :param value: bytes or array of bytes
    :param encoding str:
    :param errors str:
    """
    try:
        if numpy.isscalar(value):
            return value.decode(encoding, errors=errors)
        str_item = [b.decode(encoding, errors=errors) for b in value.flat]
        return numpy.array(str_item, dtype=object).reshape(value.shape)
    except UnicodeDecodeError:
        return value


def h5py_encode_value(value, encoding="utf-8", errors="surrogateescape"):
    """Keep string when value cannot be encoding

    :param value: string or array of strings
    :param encoding str:
    :param errors str:
    """
    try:
        if numpy.isscalar(value):
            return value.encode(encoding, errors=errors)
        bytes_item = [s.encode(encoding, errors=errors) for s in value.flat]
        return numpy.array(bytes_item, dtype=object).reshape(value.shape)
    except UnicodeEncodeError:
        return value


class H5pyDatasetReadWrapper:
    """Wrapper to handle H5T_STRING decoding on-the-fly when reading
    a dataset. Uniform behaviour for h5py 2.x and h5py 3.x

    h5py abuses H5T_STRING with ASCII character set
    to store `bytes`: dset[()] = b"..."
    Therefore an H5T_STRING with ASCII encoding is not decoded by default.
    """

    H5PY_AUTODECODE_NONASCII = int(h5py.version.version.split(".")[0]) < 3

    def __init__(self, dset, decode_ascii=False):
        """
        :param h5py.Dataset dset:
        :param bool decode_ascii:
        """
        try:
            string_info = h5py.h5t.check_string_dtype(dset.dtype)
        except AttributeError:
            # h5py < 2.10
            try:
                idx = dset.id.get_type().get_cset()
            except AttributeError:
                # Not an H5T_STRING
                encoding = None
            else:
                encoding = ["ascii", "utf-8"][idx]
        else:
            # h5py >= 2.10
            try:
                encoding = string_info.encoding
            except AttributeError:
                # Not an H5T_STRING
                encoding = None
        if encoding == "ascii" and not decode_ascii:
            encoding = None
        if encoding != "ascii" and self.H5PY_AUTODECODE_NONASCII:
            # Decoding is already done by the h5py library
            encoding = None
        if encoding == "ascii":
            # ASCII can be decoded as UTF-8
            encoding = "utf-8"
        self._encoding = encoding
        self._dset = dset

    def __getitem__(self, args):
        value = self._dset[args]
        if self._encoding:
            return h5py_decode_value(value, encoding=self._encoding)
        else:
            return value


class H5pyAttributesReadWrapper:
    """Wrapper to handle H5T_STRING decoding on-the-fly when reading
    an attribute. Uniform behaviour for h5py 2.x and h5py 3.x

    h5py abuses H5T_STRING with ASCII character set
    to store `bytes`: dset[()] = b"..."
    Therefore an H5T_STRING with ASCII encoding is not decoded by default.
    """

    H5PY_AUTODECODE = int(h5py.version.version.split(".")[0]) >= 3

    def __init__(self, attrs, decode_ascii=False):
        """
        :param h5py.Dataset dset:
        :param bool decode_ascii:
        """
        self._attrs = attrs
        self._decode_ascii = decode_ascii

    def __getitem__(self, args):
        value = self._attrs[args]

        # Get the string encoding (if a string)
        try:
            dtype = self._attrs.get_id(args).dtype
        except AttributeError:
            # h5py < 2.10
            attr_id = h5py.h5a.open(self._attrs._id, self._attrs._e(args))
            try:
                idx = attr_id.get_type().get_cset()
            except AttributeError:
                # Not an H5T_STRING
                return value
            else:
                encoding = ["ascii", "utf-8"][idx]
        else:
            # h5py >= 2.10
            try:
                encoding = h5py.h5t.check_string_dtype(dtype).encoding
            except AttributeError:
                # Not an H5T_STRING
                return value

        if self.H5PY_AUTODECODE:
            if encoding == "ascii" and not self._decode_ascii:
                # Undo decoding by the h5py library
                return h5py_encode_value(value, encoding="utf-8")
        else:
            if encoding == "ascii" and self._decode_ascii:
                # Decode ASCII as UTF-8 for consistency
                return h5py_decode_value(value, encoding="utf-8")

        # Decoding is already done by the h5py library
        return value

    def items(self):
        for k in self._attrs.keys():
            yield k, self[k]


def h5py_read_dataset(dset, index=tuple(), decode_ascii=False):
    """Read data from dataset object. UTF-8 strings will be
    decoded while ASCII strings will only be decoded when
    `decode_ascii=True`.

    :param h5py.Dataset dset:
    :param index: slicing (all by default)
    :param bool decode_ascii:
    """
    return H5pyDatasetReadWrapper(dset, decode_ascii=decode_ascii)[index]


def h5py_read_attribute(attrs, name, decode_ascii=False):
    """Read data from attributes. UTF-8 strings will be
    decoded while ASCII strings will only be decoded when
    `decode_ascii=True`.

    :param h5py.AttributeManager attrs:
    :param str name: attribute name
    :param bool decode_ascii:
    """
    return H5pyAttributesReadWrapper(attrs, decode_ascii=decode_ascii)[name]


def h5py_read_attributes(attrs, decode_ascii=False):
    """Read data from attributes. UTF-8 strings will be
    decoded while ASCII strings will only be decoded when
    `decode_ascii=True`.

    :param h5py.AttributeManager attrs:
    :param bool decode_ascii:
    """
    return dict(H5pyAttributesReadWrapper(attrs, decode_ascii=decode_ascii).items())
