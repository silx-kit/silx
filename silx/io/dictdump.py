# coding: utf-8
# /*##########################################################################
# Copyright (C) 2016-2020 European Synchrotron Radiation Facility
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
import h5py

from .configdict import ConfigDict
from .utils import is_group, is_link, is_softlink, is_externallink
from .utils import is_file as is_h5_file_like
from .utils import open as h5open
from .utils import h5py_read_dataset
from .utils import H5pyAttributesReadWrapper

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "17/07/2018"

logger = logging.getLogger(__name__)

vlen_utf8 = h5py.special_dtype(vlen=str)
vlen_bytes = h5py.special_dtype(vlen=bytes)


def _prepare_hdf5_write_value(array_like):
    """Cast a python object into a numpy array in a HDF5 friendly format.

    :param array_like: Input dataset in a type that can be digested by
        ``numpy.array()`` (`str`, `list`, `numpy.ndarray`…)
    :return: ``numpy.ndarray`` ready to be written as an HDF5 dataset
    """
    array = numpy.asarray(array_like)
    if numpy.issubdtype(array.dtype, numpy.bytes_):
        return numpy.array(array_like, dtype=vlen_bytes)
    elif numpy.issubdtype(array.dtype, numpy.unicode):
        return numpy.array(array_like, dtype=vlen_utf8)
    else:
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
    
    If dictionary keys are tuples they are interpreted to set h5 attributes.
    The tuples should have the format (dataset_name,attr_name)

    .. note::

        This function requires `h5py <http://www.h5py.org/>`_ to be installed.

    :param treedict: Nested dictionary/tree structure with strings or tuples as
        keys and array-like objects as leafs. The ``"/"`` character can be used
        to define sub trees. If tuples are used as keys they should have the
        format (dataset_name,attr_name) and will add a 5h attribute with the
        corresponding value.
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
                        "Grenoble": 18.44,
                        ("Grenoble","unit"): "km2"
                    },
                    "Nord": {
                        "Tourcoing": 15.19,
                        ("Tourcoing","unit"): "km2"
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

    if not h5path.endswith("/"):
        h5path += "/"

    with _SafeH5FileWrite(h5file, mode=mode) as h5f:
        if isinstance(treedict, dict) and h5path != "/":
            if h5path not in h5f:
                h5f.create_group(h5path)

        for key in filter(lambda k: not isinstance(k, tuple), treedict):
            key_is_group = isinstance(treedict[key], dict)
            h5name = h5path + key

            if key_is_group and treedict[key]:
                # non-empty group: recurse
                dicttoh5(treedict[key], h5f, h5name,
                         overwrite_data=overwrite_data,
                         create_dataset_args=create_dataset_args)
                continue

            if h5name in h5f:
                # key already exists: delete or skip
                if overwrite_data is True:
                    del h5f[h5name]
                else:
                    logger.warning('key (%s) already exists. '
                                    'Not overwriting.' % (h5name))
                    continue

            value = treedict[key]

            if value is None or key_is_group:
                # Create empty group
                h5f.create_group(h5name)
            elif is_link(value):
                h5f[h5name] = value
            else:
                data = _prepare_hdf5_write_value(value)
                # can't apply filters on scalars (datasets with shape == () )
                if data.shape == () or create_dataset_args is None:
                    h5f.create_dataset(h5name,
                                       data=data)
                else:
                    h5f.create_dataset(h5name,
                                       data=data,
                                       **create_dataset_args)

        # deal with h5 attributes which have tuples as keys in treedict
        for key in filter(lambda k: isinstance(k, tuple), treedict):
            assert len(key) == 2, "attribute must be defined by 2 values"
            h5name = h5path + key[0]
            attr_name = key[1]

            if h5name not in h5f:
                # Create empty group if key for attr does not exist
                h5f.create_group(h5name)
                logger.warning(
                    "key (%s) does not exist. attr %s "
                    "will be written to ." % (h5name, attr_name)
                )

            if attr_name in h5f[h5name].attrs:
                if not overwrite_data:
                    logger.warning(
                        "attribute %s@%s already exists. Not overwriting."
                        "" % (h5name, attr_name)
                    )
                    continue

            # Write attribute
            value = treedict[key]
            data = _prepare_hdf5_write_value(value)
            h5f[h5name].attrs[attr_name] = data


def nexus_to_h5_dict(treedict, parents=tuple()):
    """The following conversions are applied:
        * key with "{name}@{attr_name}" notation: key converted to 2-tuple
        * key with ">{url}" notation: strip ">" and convert value to
                                      h5py.SoftLink or h5py.ExternalLink 

    :param treedict: Nested dictionary/tree structure with strings as keys
         and array-like objects as leafs. The ``"/"`` character can be used
         to define sub tree. The ``"@"`` character is used to write attributes.
         The ``">"`` prefix is used to define links.
    :param parents: Needed to resolve up-links (tuple of HDF5 group names)

    :rtype dict:
    """
    copy = dict()
    for key, value in treedict.items():
        if "@" in key:
            key = tuple(key.rsplit("@", 1))
        elif key.startswith(">"):
            if isinstance(value, str):
                key = key[1:]
                first, sep, second = value.partition("::")
                if sep:
                    value = h5py.ExternalLink(first, second)
                else:
                    if ".." in first:
                        # Up-links not supported: make absolute
                        parts = []
                        for p in list(parents) + first.split("/"):
                            if not p or p == ".":
                                continue
                            elif p == "..":
                                parts.pop(-1)
                            else:
                                parts.append(p)
                        first = "/" + "/".join(parts)
                    value = h5py.SoftLink(first)
            elif is_link(value):
                key = key[1:]
        if isinstance(value, dict):
            copy[key] = nexus_to_h5_dict(value, parents=parents+(key,))
        else:
            copy[key] = value
    return copy


def h5_to_nexus_dict(treedict):
    """The following conversions are applied:
        * 2-tuple key: converted to string ("@" notation)
        * h5py.Softlink value: converted to string (">" key prefix)
        * h5py.ExternalLink value: converted to string (">" key prefix)

    :param treedict: Nested dictionary/tree structure with strings as keys
         and array-like objects as leafs. The ``"/"`` character can be used
         to define sub tree.

    :rtype dict:
    """
    copy = dict()
    for key, value in treedict.items():
        if isinstance(key, tuple):
            assert len(key)==2, "attribute must be defined by 2 values"
            key = "%s@%s" % (key[0], key[1])
        elif is_softlink(value):
            key = ">" + key
            value = value.path
        elif is_externallink(value):
            key = ">" + key
            value = value.filename + "::" + value.path
        if isinstance(value, dict):
            copy[key] = h5_to_nexus_dict(value)
        else:
            copy[key] = value
    return copy


def _name_contains_string_in_list(name, strlist):
    if strlist is None:
        return False
    for filter_str in strlist:
        if filter_str in name:
            return True
    return False


def _handle_error(mode: str, exception, msg: str, *args) -> None:
    """Handle errors.

    :param str mode: 'raise', 'log', 'ignore'
    :param type exception: Exception class to use in 'raise' mode
    :param str msg: Error message template
    :param List[str] args: Arguments for error message template
    """
    if mode == 'ignore':
        return  # no-op
    elif mode == 'log':
        logger.error(msg, *args)
    elif mode == 'raise':
        raise exception(msg % args)
    else:
        raise ValueError("Unsupported error handling: %s" % mode)


def h5todict(h5file,
             path="/",
             exclude_names=None,
             asarray=True,
             dereference_links=True,
             include_attributes=False,
             errors='raise'):
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

    .. note:: If you write a dictionary to a HDF5 file with
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
    :param bool asarray: True (default) to read scalar as arrays, False to
        read them as scalar
    :param bool dereference_links: True (default) to dereference links, False
        to preserve the link itself
    :param bool include_attributes: False (default)
    :param str errors: Handling of errors (HDF5 access issue, broken link,...):
        - 'raise' (default): Raise an exception
        - 'log': Log as errors
        - 'ignore': Ignore errors
    :return: Nested dictionary
    """
    with _SafeH5FileRead(h5file) as h5f:
        ddict = {}
        if path not in h5f:
            _handle_error(
                errors, KeyError, 'Path "%s" does not exist in file.', path)
            return ddict

        try:
            root = h5f[path]
        except KeyError as e:
            if not isinstance(h5f.get(path, getlink=True), h5py.HardLink):
                _handle_error(errors,
                              KeyError,
                              'Cannot retrieve path "%s" (broken link)',
                              path)
            else:
                _handle_error(errors, KeyError, ', '.join(e.args))
            return ddict

        # Read the attributes of the group
        if include_attributes:
            attrs = H5pyAttributesReadWrapper(root.attrs)
            for aname, avalue in attrs.items():
                ddict[("", aname)] = avalue
        # Read the children of the group
        for key in root:
            if _name_contains_string_in_list(key, exclude_names):
                continue
            h5name = path + "/" + key
            # Preserve HDF5 link when requested
            if not dereference_links:
                lnk = h5f.get(h5name, getlink=True)
                if is_link(lnk):
                    ddict[key] = lnk
                    continue

            try:
                h5obj = h5f[h5name]
            except KeyError as e:
                if not isinstance(h5f.get(h5name, getlink=True), h5py.HardLink):
                    _handle_error(errors,
                                  KeyError,
                                  'Cannot retrieve path "%s" (broken link)',
                                  h5name)
                else:
                    _handle_error(errors, KeyError, ', '.join(e.args))
                continue

            if is_group(h5obj):
                # Child is an HDF5 group
                ddict[key] = h5todict(h5f,
                                      h5name,
                                      exclude_names=exclude_names,
                                      asarray=asarray,
                                      dereference_links=dereference_links,
                                      include_attributes=include_attributes)
            else:
                # Child is an HDF5 dataset
                try:
                    data = h5py_read_dataset(h5obj)
                except OSError:
                    _handle_error(errors,
                                  OSError,
                                  'Cannot retrieve dataset "%s"',
                                  h5name)
                else:
                    if asarray:  # Convert HDF5 dataset to numpy array
                        data = numpy.array(data, copy=False)
                    ddict[key] = data
                    # Read the attributes of the child
                    if include_attributes:
                        attrs = H5pyAttributesReadWrapper(h5obj.attrs)
                        for aname, avalue in attrs.items():
                            ddict[(key, aname)] = avalue
    return ddict


def dicttonx(treedict, h5file, h5path="/", **kw):
    """
    Write a nested dictionary to a HDF5 file, using string keys as member names.
    The NeXus convention is used to identify attributes with ``"@"`` character,
    therefore the dataset_names should not contain ``"@"``.

    Similarly, links are identified by keys starting with the ``">"`` character.
    The corresponding value can be a soft or external link.

    :param treedict: Nested dictionary/tree structure with strings as keys
         and array-like objects as leafs. The ``"/"`` character can be used
         to define sub tree. The ``"@"`` character is used to write attributes.
         The ``">"`` prefix is used to define links.

    The named parameters are passed to dicttoh5.

    Example::

        import numpy
        from silx.io.dictdump import dicttonx

        gauss = {
            "entry":{
                "title":u"A plot of a gaussian",
                "instrument": {
                    "@NX_class": u"NXinstrument",
                    "positioners": {
                        "@NX_class": u"NXCollection",
                        "x": numpy.arange(0,1.1,.1)
                    }
                }
                "plot": {
                    "y": numpy.array([0.08, 0.19, 0.39, 0.66, 0.9, 1.,
                                  0.9, 0.66, 0.39, 0.19, 0.08]),
                    ">x": "../instrument/positioners/x",
                    "@signal": "y",
                    "@axes": "x",
                    "@NX_class":u"NXdata",
                    "title:u"Gauss Plot",
                 },
                 "@NX_class": u"NXentry",
                 "default":"plot",
            }
            "@NX_class": u"NXroot",
            "@default": "entry",
        }

        dicttonx(gauss,"test.h5")
    """
    parents = tuple(p for p in h5path.split("/") if p)
    nxtreedict = nexus_to_h5_dict(treedict, parents=parents)
    dicttoh5(nxtreedict, h5file, h5path=h5path, **kw)


def nxtodict(h5file, **kw):
    """Read a HDF5 file and return a nested dictionary with the complete file
    structure and all data.

    As opposed to h5todict, all keys will be strings and no h5py objects are
    present in the tree.

    The named parameters are passed to h5todict.
    """
    nxtreedict = h5todict(h5file, **kw)
    return h5_to_nexus_dict(nxtreedict)


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
            return h5todict(fname)
        elif fmat in ["ini", "cfg"]:
            return ConfigDict(filelist=[fname])
        else:
            raise IOError("Unknown format " + fmat)
    finally:
        if must_be_closed:
            f.close()
