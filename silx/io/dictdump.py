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
"""This module offers a set of functions to dump a python dictionary indexed
by text strings to following file formats: `HDF5, INI, JSON`
"""

from collections import OrderedDict
import json
import logging
import numpy
import os.path
import sys

try:
    import h5py
except ImportError as e:
    h5py_missing = True
    h5py_import_error = e
else:
    h5py_missing = False

from .configdict import ConfigDict
from .utils import is_group
from .utils import is_file as is_h5_file_like
from .utils import open as h5open

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "17/07/2018"

logger = logging.getLogger(__name__)

string_types = (basestring,) if sys.version_info[0] == 2 else (str,)    # noqa


def _prepare_hdf5_dataset(array_like):
    """Cast a python object into a numpy array in a HDF5 friendly format.

    :param array_like: Input dataset in a type that can be digested by
        ``numpy.array()`` (`str`, `list`, `numpy.ndarray`…)
    :return: ``numpy.ndarray`` ready to be written as an HDF5 dataset
    """
    # simple strings
    if isinstance(array_like, string_types):
        array_like = numpy.string_(array_like)

    # Ensure our data is a numpy.ndarray
    if not isinstance(array_like, (numpy.ndarray, numpy.string_)):
        array = numpy.array(array_like)
    else:
        array = array_like

    # handle list of strings or numpy array of strings
    if not isinstance(array, numpy.string_):
        data_kind = array.dtype.kind
        # unicode: convert to byte strings
        # (http://docs.h5py.org/en/latest/strings.html)
        if data_kind.lower() in ["s", "u"]:
            array = numpy.asarray(array, dtype=numpy.string_)

    return array


class _SafeH5FileWrite(object):
    """Context manager returning a :class:`h5py.File` object.

    If this object is initialized with a file path, we open the file
    and then we close it on exiting.

    If a :class:`h5py.File` instance is provided to :meth:`__init__` rather
    than a path, we assume that the user is responsible for closing the
    file.

    This behavior is well suited for handling h5py file in a recursive
    function. The object is created in the initial call if a path is provided,
    and it is closed only at the end when all the processing is finished.
    """
    def __init__(self, h5file, mode="w"):
        """

        :param h5file:  HDF5 file path or :class:`h5py.File` instance
        :param str mode:  Can be ``"r+"`` (read/write, file must exist),
            ``"w"`` (write, existing file is lost), ``"w-"`` (write, fail if
            exists) or ``"a"`` (read/write if exists, create otherwise).
            This parameter is ignored if ``h5file`` is a file handle.
        """
        self.raw_h5file = h5file
        self.mode = mode

    def __enter__(self):
        if not isinstance(self.raw_h5file, h5py.File):
            self.h5file = h5py.File(self.raw_h5file, self.mode)
            self.close_when_finished = True
        else:
            self.h5file = self.raw_h5file
            self.close_when_finished = False
        return self.h5file

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.close_when_finished:
            self.h5file.close()


class _SafeH5FileRead(object):
    """Context manager returning a :class:`h5py.File` or a
    :class:`silx.io.spech5.SpecH5` or a :class:`silx.io.fabioh5.File` object.

    The general behavior is the same as :class:`_SafeH5FileWrite` except
    that SPEC files and all formats supported by fabio can also be opened,
    but in read-only mode.
    """
    def __init__(self, h5file):
        """

        :param h5file:  HDF5 file path or h5py.File-like object
        """
        self.raw_h5file = h5file

    def __enter__(self):
        if not is_h5_file_like(self.raw_h5file):
            self.h5file = h5open(self.raw_h5file)
            self.close_when_finished = True
        else:
            self.h5file = self.raw_h5file
            self.close_when_finished = False

        return self.h5file

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.close_when_finished:
            self.h5file.close()


def dicttoh5(treedict, h5file, h5path='/',
             mode="w", overwrite_data=False,
             create_dataset_args=None):
    """Write a nested dictionary to a HDF5 file, using keys as member names.

    If a dictionary value is a sub-dictionary, a group is created. If it is
    any other data type, it is cast into a numpy array and written as a
    :mod:`h5py` dataset. Dictionary keys must be strings and cannot contain
    the ``/`` character.

    .. note::

        This function requires `h5py <http://www.h5py.org/>`_ to be installed.

    :param treedict: Nested dictionary/tree structure with strings as keys
         and array-like objects as leafs. The ``"/"`` character is not allowed
         in keys.
    :param h5file: HDF5 file name or handle. If a file name is provided, the
        function opens the file in the specified mode and closes it again
        before completing.
    :param h5path: Target path in HDF5 file in which scan groups are created.
        Default is root (``"/"``)
    :param mode: Can be ``"r+"`` (read/write, file must exist),
        ``"w"`` (write, existing file is lost), ``"w-"`` (write, fail if
        exists) or ``"a"`` (read/write if exists, create otherwise).
        This parameter is ignored if ``h5file`` is a file handle.
    :param overwrite_data: If ``True``, existing groups and datasets can be
        overwritten, if ``False`` they are skipped. This parameter is only
        relevant if ``h5file_mode`` is ``"r+"`` or ``"a"``.
    :param create_dataset_args: Dictionary of args you want to pass to
        ``h5f.create_dataset``. This allows you to specify filters and
        compression parameters. Don't specify ``name`` and ``data``.

    Example::

        from silx.io.dictdump import dicttoh5

        city_area = {
            "Europe": {
                "France": {
                    "Isère": {
                        "Grenoble": "18.44 km2"
                    },
                    "Nord": {
                        "Tourcoing": "15.19 km2"
                    },
                },
            },
        }

        create_ds_args = {'compression': "gzip",
                          'shuffle': True,
                          'fletcher32': True}

        dicttoh5(city_area, "cities.h5", h5path="/area",
                 create_dataset_args=create_ds_args)
    """
    if h5py_missing:
        raise h5py_import_error

    if not h5path.endswith("/"):
        h5path += "/"

    with _SafeH5FileWrite(h5file, mode=mode) as h5f:
        for key in treedict:
            if isinstance(treedict[key], dict) and len(treedict[key]):
                # non-empty group: recurse
                dicttoh5(treedict[key], h5f, h5path + key,
                         overwrite_data=overwrite_data,
                         create_dataset_args=create_dataset_args)

            elif treedict[key] is None or (isinstance(treedict[key], dict) and
                                           not len(treedict[key])):
                if (h5path + key) in h5f:
                    if overwrite_data is True:
                        del h5f[h5path + key]
                    else:
                        logger.warning('key (%s) already exists. '
                                       'Not overwriting.' % (h5path + key))
                        continue
                # Create empty group
                h5f.create_group(h5path + key)

            else:
                ds = _prepare_hdf5_dataset(treedict[key])
                # can't apply filters on scalars (datasets with shape == () )
                if ds.shape == () or create_dataset_args is None:
                    if h5path + key in h5f:
                        if overwrite_data is True:
                            del h5f[h5path + key]
                        else:
                            logger.warning('key (%s) already exists. '
                                           'Not overwriting.' % (h5path + key))
                            continue

                    h5f.create_dataset(h5path + key,
                                       data=ds)
                else:
                    if h5path + key in h5f:
                        if overwrite_data is True:
                            del h5f[h5path + key]
                        else:
                            logger.warning('key (%s) already exists. '
                                           'Not overwriting.' % (h5path + key))
                            continue

                    h5f.create_dataset(h5path + key,
                                       data=ds,
                                       **create_dataset_args)


def _name_contains_string_in_list(name, strlist):
    if strlist is None:
        return False
    for filter_str in strlist:
        if filter_str in name:
            return True
    return False


def h5todict(h5file, path="/", exclude_names=None):
    """Read a HDF5 file and return a nested dictionary with the complete file
    structure and all data.

    Example of usage::

        from silx.io.dictdump import h5todict

        # initialize dict with file header and scan header
        header94 = h5todict("oleg.dat",
                            "/94.1/instrument/specfile")
        # add positioners subdict
        header94["positioners"] = h5todict("oleg.dat",
                                           "/94.1/instrument/positioners")
        # add scan data without mca data
        header94["detector data"] = h5todict("oleg.dat",
                                             "/94.1/measurement",
                                             exclude_names="mca_")


    .. note:: This function requires `h5py <http://www.h5py.org/>`_ to be
        installed.

    .. note:: If you write a dictionary to a HDF5 file with
        :func:`dicttoh5` and then read it back with :func:`h5todict`, data
        types are not preserved. All values are cast to numpy arrays before
        being written to file, and they are read back as numpy arrays (or
        scalars). In some cases, you may find that a list of heterogeneous
        data types is converted to a numpy array of strings.

    :param h5file: File name or :class:`h5py.File` object or spech5 file or
        fabioh5 file.
    :param str path: Name of HDF5 group to use as dictionary root level,
        to read only a sub-group in the file
    :param List[str] exclude_names: Groups and datasets whose name contains
        a string in this list will be ignored. Default is None (ignore nothing)
    :return: Nested dictionary
    """
    if h5py_missing:
        raise h5py_import_error

    with _SafeH5FileRead(h5file) as h5f:
        ddict = {}
        for key in h5f[path]:
            if _name_contains_string_in_list(key, exclude_names):
                continue
            if is_group(h5f[path + "/" + key]):
                ddict[key] = h5todict(h5f,
                                      path + "/" + key,
                                      exclude_names=exclude_names)
            else:
                # Convert HDF5 dataset to numpy array
                ddict[key] = h5f[path + "/" + key][...]

    return ddict


def dicttojson(ddict, jsonfile, indent=None, mode="w"):
    """Serialize ``ddict`` as a JSON formatted stream to ``jsonfile``.

    :param ddict: Dictionary (or any object compatible with ``json.dump``).
    :param jsonfile: JSON file name or file-like object.
        If a file name is provided, the function opens the file in the
        specified mode and closes it again.
    :param indent: If indent is a non-negative integer, then JSON array
        elements and object members will be pretty-printed with that indent
        level. An indent level of ``0`` will only insert newlines.
        ``None`` (the default) selects the most compact representation.
    :param mode: File opening mode (``w``, ``a``, ``w+``…)
    """
    if not hasattr(jsonfile, "write"):
        jsonf = open(jsonfile, mode)
    else:
        jsonf = jsonfile

    json.dump(ddict, jsonf, indent=indent)

    if not hasattr(jsonfile, "write"):
        jsonf.close()


def dicttoini(ddict, inifile, mode="w"):
    """Output dict as configuration file (similar to Microsoft Windows INI).

    :param dict: Dictionary of configuration parameters
    :param inifile: INI file name or file-like object.
        If a file name is provided, the function opens the file in the
        specified mode and closes it again.
    :param mode: File opening mode (``w``, ``a``, ``w+``…)
    """
    if not hasattr(inifile, "write"):
        inif = open(inifile, mode)
    else:
        inif = inifile

    ConfigDict(initdict=ddict).write(inif)

    if not hasattr(inifile, "write"):
        inif.close()


def dump(ddict, ffile, mode="w", fmat=None):
    """Dump dictionary to a file

    :param ddict: Dictionary with string keys
    :param ffile: File name or file-like object with a ``write`` method
    :param str fmat: Output format: ``"json"``, ``"hdf5"`` or ``"ini"``.
        When None (the default), it uses the filename extension as the format.
        Dumping to a HDF5 file requires `h5py <http://www.h5py.org/>`_ to be
        installed.
    :param str mode: File opening mode (``w``, ``a``, ``w+``…)
        Default is *"w"*, write mode, overwrite if exists.
    :raises IOError: if file format is not supported
    """
    if fmat is None:
        # If file-like object get its name, else use ffile as filename
        filename = getattr(ffile, 'name', ffile)
        fmat = os.path.splitext(filename)[1][1:]  # Strip extension leading '.'
    fmat = fmat.lower()

    if fmat == "json":
        dicttojson(ddict, ffile, indent=2, mode=mode)
    elif fmat in ["hdf5", "h5"]:
        if h5py_missing:
            logger.error("Cannot dump to HDF5 format, missing h5py library")
            raise h5py_import_error
        dicttoh5(ddict, ffile, mode=mode)
    elif fmat in ["ini", "cfg"]:
        dicttoini(ddict, ffile, mode=mode)
    else:
        raise IOError("Unknown format " + fmat)


def load(ffile, fmat=None):
    """Load dictionary from a file

    When loading from a JSON or INI file, an OrderedDict is returned to
    preserve the values' insertion order.

    :param ffile: File name or file-like object with a ``read`` method
    :param fmat: Input format: ``json``, ``hdf5`` or ``ini``.
        When None (the default), it uses the filename extension as the format.
        Loading from a HDF5 file requires `h5py <http://www.h5py.org/>`_ to be
        installed.
    :return: Dictionary (ordered dictionary for JSON and INI)
    :raises IOError: if file format is not supported
    """
    must_be_closed = False
    if not hasattr(ffile, "read"):
        f = open(ffile, "r")
        fname = ffile
        must_be_closed = True
    else:
        f = ffile
        fname = ffile.name

    try:
        if fmat is None:  # Use file extension as format
            fmat = os.path.splitext(fname)[1][1:]  # Strip extension leading '.'
        fmat = fmat.lower()

        if fmat == "json":
            return json.load(f, object_pairs_hook=OrderedDict)
        if fmat in ["hdf5", "h5"]:
            if h5py_missing:
                logger.error("Cannot load from HDF5 format, missing h5py library")
                raise h5py_import_error
            return h5todict(fname)
        elif fmat in ["ini", "cfg"]:
            return ConfigDict(filelist=[fname])
        else:
            raise IOError("Unknown format " + fmat)
    finally:
        if must_be_closed:
            f.close()
