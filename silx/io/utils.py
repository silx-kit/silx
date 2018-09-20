# coding: utf-8
# /*##########################################################################
# Copyright (C) 2016-2018 European Synchrotron Radiation Facility
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
__date__ = "18/04/2018"

import numpy
import os.path
import sys
import time
import logging
import collections

from silx.utils.proxy import Proxy
from silx.third_party import six
from silx.third_party import enum
import silx.io.url

try:
    import h5py
except ImportError as e:
    h5py = None
    h5py_import_error = e

try:
    import h5pyd
except ImportError as e:
    h5pyd = None
    h5py_import_error = e


logger = logging.getLogger(__name__)


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
        extensions (an extension is a string like "\*.ext").
    :rtype: Dict[str, Set[str]]
    """
    formats = {}
    if h5py is not None:
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
            y_array = y_array.reshape(1, y_array.shape[0])
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
    # Make sure we use binary mode for write
    # (issue with windows: write() replaces \n with os.linesep in text mode)
    if "b" not in mode:
        first_letter = mode[0]
        assert first_letter in "rwa"
        mode = mode.replace(first_letter, first_letter + "b")

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
        f = builtin_open(specfile, mode)
    else:
        f = specfile

    output = ""

    current_date = "#D %s\n" % (time.ctime(time.time()))

    if write_file_header:
        output += "#F %s\n" % f.name
        output += current_date
        output += "\n"

    output += "#S %d %s\n" % (scan_number, ylabel)
    output += current_date
    output += "#N 2\n"
    output += "#L %s  %s\n" % (xlabel, ylabel)
    for i in range(y_array.shape[0]):
        output += full_fmt_string % (x_array[i], y_array[i])
    output += "\n"

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
    if h5py is None:
        logger.error("h5ls requires h5py")
        raise h5py_import_error

    h5repr = ''
    if is_group(h5group):
        h5f = h5group
    elif isinstance(h5group, string_types):
        h5f = open(h5group)      # silx.io.open
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

        if h5py is not None:
            if h5py.is_hdf5(filename):
                return h5py.File(filename, "r")

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
        uri = six.moves.urllib.parse.urlparse(filename)
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

    if h5py is not None:
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
    if h5py is None:
        logger.error("get_h5py_class/is_file/is_group/is_dataset requires h5py")
        raise h5py_import_error
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
                data = data[data_slice]
            else:
                # works for scalar and array
                data = data[()]

    elif url.scheme() == "fabio":
        import fabio
        data_slice = url.data_slice()
        if data_slice is None:
            data_slice = (0, )
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
