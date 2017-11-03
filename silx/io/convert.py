# coding: utf-8
# /*##########################################################################
# Copyright (C) 2016-2017 European Synchrotron Radiation Facility
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
"""This module provides classes and function to convert file formats supported
by *silx* into HDF5 file. Currently, SPEC file and fabio images are the
supported formats.

Read the documentation of :mod:`silx.io.spech5` and :mod:`silx.io.fabioh5` for
information on the structure of the output HDF5 files.

Strings are written to the HDF5 datasets as fixed-length ASCII (NumPy *S* type).
This is done in order to produce files that have maximum compatibility with
other HDF5 libraries, as recommended in the
`h5py documentation <http://docs.h5py.org/en/latest/strings.html#how-to-store-text-strings>`_.

If you read the files back with *h5py* in Python 3, you will recover strings
as bytes, which you should decode to transform them into python strings::

    >>> import h5py
    >>> f = h5py.File("myfile.h5")
    >>> f["/1.1/instrument/specfile/scan_header"][0]
    b'#S 94  ascan  del -0.5 0.5  20 1'
    >>> f["/1.1/instrument/specfile/scan_header"][0].decode()
    '#S 94  ascan  del -0.5 0.5  20 1'

Arrays of strings, such as file and scan headers, are stored as fixed-length
strings. The length of all strings in an array is equal to the length of the
longest string. Shorter strings are right-padded with blank spaces.

.. note:: This module has a dependency on the `h5py <http://www.h5py.org/>`_
    library, which is not a mandatory dependency for `silx`. You might need
    to install it if you don't already have it.
"""

import numpy
import logging

import silx.io
from silx.io import is_dataset, is_group, is_softlink

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "14/09/2017"

_logger = logging.getLogger(__name__)

try:
    import h5py
except ImportError as e:
    _logger.error("Module " + __name__ + " requires h5py")
    raise e


def _create_link(h5f, link_name, target_name,
                 link_type="soft", overwrite_data=False):
    """Create a link in a HDF5 file

    If member with name ``link_name`` already exists, delete it first or
    ignore link depending on global param ``overwrite_data``.

    :param h5f: :class:`h5py.File` object
    :param link_name: Link path
    :param target_name: Handle for target group or dataset
    :param str link_type: "soft" or "hard"
    :param bool overwrite_data: If True, delete existing member (group,
        dataset or link) with the same name. Default is False.
    """
    if link_name not in h5f:
        _logger.debug("Creating link " + link_name + " -> " + target_name)
    elif overwrite_data:
        _logger.warn("Overwriting " + link_name + " with link to " +
                     target_name)
        del h5f[link_name]
    else:
        _logger.warn(link_name + " already exist. Can't create link to " +
                     target_name)
        return None

    if link_type == "hard":
        h5f[link_name] = h5f[target_name]
    elif link_type == "soft":
        h5f[link_name] = h5py.SoftLink(target_name)
    else:
        raise ValueError("link_type  must be 'hard' or 'soft'")


class Hdf5Writer(object):
    """Converter class to write the content of a data file to a HDF5 file.
    """
    def __init__(self,
                 h5path='/',
                 overwrite_data=False,
                 link_type="soft",
                 create_dataset_args=None,
                 min_size=500):
        """

        :param h5path: Target path where the scan groups will be written
            in the output HDF5 file.
        :param bool overwrite_data:
            See documentation of :func:`write_to_h5`
        :param str link_type: ``"hard"`` or ``"soft"`` (default)
        :param dict create_dataset_args: Dictionary of args you want to pass to
            ``h5py.File.create_dataset``.
            See documentation of :func:`write_to_h5`
        :param int min_size:
            See documentation of :func:`write_to_h5`
        """
        self.h5path = h5path
        if not h5path.startswith("/"):
            # target path must be absolute
            self.h5path = "/" + h5path
        if not self.h5path.endswith("/"):
            self.h5path += "/"

        self._h5f = None
        """h5py.File object, assigned in :meth:`write`"""

        if create_dataset_args is None:
            create_dataset_args = {}
        self.create_dataset_args = create_dataset_args

        self.min_size = min_size

        self.overwrite_data = overwrite_data   # boolean

        self.link_type = link_type
        """'soft' or 'hard' """

        self._links = []
        """List of *(link_path, target_path)* tuples."""

    def write(self, infile, h5f):
        """Do the conversion from :attr:`sfh5` (Spec file) to *h5f* (HDF5)

        All the parameters needed for the conversion have been initialized
        in the constructor.

        :param infile: :class:`SpecH5` object
        :param h5f: :class:`h5py.File` instance
        """
        # Recurse through all groups and datasets to add them to the HDF5
        self._h5f = h5f
        infile.visititems(self.append_member_to_h5, visit_links=True)

        # Handle the attributes of the root group
        root_grp = h5f[self.h5path]
        for key in infile.attrs:
            if self.overwrite_data or key not in root_grp.attrs:
                root_grp.attrs.create(key,
                                      numpy.string_(infile.attrs[key]))

        # Handle links at the end, when their targets are created
        for link_name, target_name in self._links:
            _create_link(self._h5f, link_name, target_name,
                         link_type=self.link_type,
                         overwrite_data=self.overwrite_data)
        self._links = []

    def append_member_to_h5(self, h5like_name, obj):
        """Add one group or one dataset to :attr:`h5f`"""
        h5_name = self.h5path + h5like_name.lstrip("/")

        if is_softlink(obj):
            # links to be created after all groups and datasets
            h5_target = self.h5path + obj.path.lstrip("/")
            self._links.append((h5_name, h5_target))

        elif is_dataset(obj):
            _logger.debug("Saving dataset: " + h5_name)

            member_initially_exists = h5_name in self._h5f

            if self.overwrite_data and member_initially_exists:
                _logger.warn("Overwriting dataset: " + h5_name)
                del self._h5f[h5_name]

            if self.overwrite_data or not member_initially_exists:
                # fancy arguments don't apply to small dataset
                if obj.size < self.min_size:
                    ds = self._h5f.create_dataset(h5_name, data=obj.value)
                else:
                    ds = self._h5f.create_dataset(h5_name, data=obj.value,
                                                  **self.create_dataset_args)
            else:
                ds = self._h5f[h5_name]

            # add HDF5 attributes
            for key in obj.attrs:
                if self.overwrite_data or key not in ds.attrs:
                    ds.attrs.create(key, numpy.string_(obj.attrs[key]))

            if not self.overwrite_data and member_initially_exists:
                _logger.warn("Ignoring existing dataset: " + h5_name)

        elif is_group(obj):
            if h5_name not in self._h5f:
                _logger.debug("Creating group: " + h5_name)
                grp = self._h5f.create_group(h5_name)
            else:
                grp = self._h5f[h5_name]

            # add HDF5 attributes
            for key in obj.attrs:
                if self.overwrite_data or key not in grp.attrs:
                    grp.attrs.create(key, numpy.string_(obj.attrs[key]))


def write_to_h5(infile, h5file, h5path='/', mode="a",
                overwrite_data=False, link_type="soft",
                create_dataset_args=None, min_size=500):
    """Write content of a h5py-like object into a HDF5 file.

    :param infile: Path of input file, or :class:`commonh5.File` object
        or :class:`commonh5.Group` object
    :param h5file: Path of output HDF5 file or HDF5 file handle
        (`h5py.File` object)
    :param str h5path: Target path in HDF5 file in which scan groups are created.
        Default is root (``"/"``)
    :param str mode: Can be ``"r+"`` (read/write, file must exist),
        ``"w"`` (write, existing file is lost), ``"w-"`` (write, fail
        if exists) or ``"a"`` (read/write if exists, create otherwise).
        This parameter is ignored if ``h5file`` is a file handle.
    :param bool overwrite_data: If ``True``, existing groups and datasets can be
        overwritten, if ``False`` they are skipped. This parameter is only
        relevant if ``file_mode`` is ``"r+"`` or ``"a"``.
    :param str link_type: *"soft"* (default) or *"hard"*
    :param dict create_dataset_args: Dictionary of args you want to pass to
        ``h5py.File.create_dataset``. This allows you to specify filters and
        compression parameters. Don't specify ``name`` and ``data``.
        These arguments are only applied to datasets larger than 1MB.
    :param int min_size: Minimum number of elements in a dataset to apply
        chunking and compression. Default is 500.

    The structure of the spec data in an HDF5 file is described in the
    documentation of :mod:`silx.io.spech5`.
    """
    writer = Hdf5Writer(h5path=h5path,
                        overwrite_data=overwrite_data,
                        link_type=link_type,
                        create_dataset_args=create_dataset_args,
                        min_size=min_size)

    # both infile and h5file can be either file handle or a file name: 4 cases
    if not isinstance(h5file, h5py.File) and not is_group(infile):
        with silx.io.open(infile) as h5pylike:
            with h5py.File(h5file, mode) as h5f:
                writer.write(h5pylike, h5f)
    elif isinstance(h5file, h5py.File) and not is_group(infile):
        with silx.io.open(infile) as h5pylike:
            writer.write(h5pylike, h5file)
    elif is_group(infile) and not isinstance(h5file, h5py.File):
        with h5py.File(h5file, mode) as h5f:
            writer.write(infile, h5f)
    else:
        writer.write(infile, h5file)


def convert(infile, h5file, mode="w-", create_dataset_args=None):
    """Convert a supported file into an HDF5 file, write scans into the
    root group (``/``).

    This is a convenience shortcut to call::

        write_to_h5(h5like, h5file, h5path='/',
                    mode="w-", link_type="soft")

    :param infile: Path of input file or :class:`commonh5.File` object
        or :class:`commonh5.Group` object
    :param h5file: Path of output HDF5 file, or h5py.File object
    :param mode: Can be ``"w"`` (write, existing file is
        lost), ``"w-"`` (write, fail if exists). This is ignored
        if ``h5file`` is a file handle.
    :param create_dataset_args: Dictionary of args you want to pass to
        ``h5py.File.create_dataset``. This allows you to specify filters and
        compression parameters. Don't specify ``name`` and ``data``.
    """
    if mode not in ["w", "w-"]:
        raise IOError("File mode must be 'w' or 'w-'. Use write_to_h5" +
                      " to append data to an existing HDF5 file.")
    write_to_h5(infile, h5file, h5path='/', mode=mode,
                create_dataset_args=create_dataset_args)
