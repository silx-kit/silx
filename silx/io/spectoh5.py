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
"""This module provides functions to convert a SpecFile into a HDF5 file.

Read the documentation of :mod:`silx.io.spech5` for information on the
structure of the output HDF5 files.

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
import re

_logger = logging.getLogger(__name__)

try:
    import h5py
except ImportError as e:
    _logger.error("Module " + __name__ + " requires h5py")
    raise e

from .spech5 import SpecH5, SpecH5Group, SpecH5Dataset, \
     SpecH5LinkToGroup, SpecH5LinkToDataset


__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "24/11/2016"


def _create_link(h5f, link_name, target,
                 link_type="hard", overwrite_data=False):
    """Create a link in a HDF5 file

    If member with name ``link_name`` already exists, delete it first or
    ignore link depending on global param ``overwrite_data``.

    :param h5f: :class:`h5py.File` object
    :param link_name: Link path
    :param target: Handle for target group or dataset
    """
    if link_name not in h5f:
        _logger.debug("Creating link " + link_name + " -> " + target.name)
    elif overwrite_data:
        _logger.warn("Overwriting " + link_name + " with link to" +
                     target.name)
        del h5f[link_name]
    else:
        _logger.warn(link_name + " already exist. Can't create link to " +
                     target.name)
        return None

    if link_type == "hard":
        h5f[link_name] = target
    elif link_type == "soft":
        h5f[link_name] = h5py.SoftLink(target.name)
    else:
        raise ValueError("link_type  must be 'hard' or 'soft'")


class SpecToHdf5Writer(object):
    """Converter class to write a Spec file to a HDF5 file."""
    def __init__(self,
                 h5path='/',
                 overwrite_data=False,
                 link_type="hard",
                 create_dataset_args=None):
        """

        :param h5path: Target path where the scan groups will be written
            in the output HDF5 file.
        :param bool overwrite_data:
            See documentation of :func:`write_spec_to_h5`
        :param str link_type: ``"hard"`` (default) or ``"soft"``
        :param dict create_dataset_args: Dictionary of args you want to pass to
            ``h5py.File.create_dataset``.
            See documentation of :func:`write_spec_to_h5`
        """
        self.h5path = h5path

        if create_dataset_args is None:
            create_dataset_args = {}
        self.create_dataset_args = create_dataset_args

        self.overwrite_data = overwrite_data

        self.link_type = link_type

    def write(self, sfh5, h5f):
        """Do the conversion from :attr:`sfh5` (Spec file) to *h5f* (HDF5)

        All the parameters needed for the conversion have been initialized
        in the constructor.

        :param sfh5: :class:`SpecH5` object
        :param h5f: :class:`h5py.File` instance
        """
        # Recurse through all groups and datasets to add them to the HDF5
        self.sfh5 = sfh5
        self.h5f = h5f
        self.sfh5.visititems(self.append_spec_member_to_h5)

        # Handle the attributes of the root group
        root_grp = h5f[self.h5path]
        for key in self.sfh5.attrs:
            if self.overwrite_data or key not in root_grp.attrs:
                root_grp.attrs.create(key,
                                      numpy.string_(self.sfh5.attrs[key]))

    def append_spec_member_to_h5(self, spec_h5_name, obj):
        """Add one group or one dataset to :attr:`h5f`"""
        h5_name = self.h5path + spec_h5_name.lstrip("/")

        if isinstance(obj, SpecH5LinkToGroup) or\
                isinstance(obj, SpecH5LinkToDataset):
            # links are created at the same time as their targets
            _logger.debug("Ignoring link: " + h5_name)
            pass

        elif isinstance(obj, SpecH5Dataset):
            _logger.debug("Saving dataset: " + h5_name)

            member_initially_exists = h5_name in self.h5f

            if self.overwrite_data and member_initially_exists:
                _logger.warn("Overwriting dataset: " + h5_name)
                del self.h5f[h5_name]

            if self.overwrite_data or not member_initially_exists:
                # fancy arguments don't apply to scalars (shape==())
                if obj.shape == ():
                    ds = self.h5f.create_dataset(h5_name, data=obj.value)
                else:
                    ds = self.h5f.create_dataset(h5_name, data=obj.value,
                                                 **self.create_dataset_args)
            else:
                ds = self.h5f[h5_name]

            # add HDF5 attributes
            for key in obj.attrs:
                if self.overwrite_data or key not in ds.attrs:
                    ds.attrs.create(key, numpy.string_(obj.attrs[key]))

            # link:
            #  /1.1/measurement/mca_0/data  --> /1.1/instrument/mca_0/data
            if re.match(r".*/([0-9]+\.[0-9]+)/instrument/mca_([0-9]+)/?data$",
                        h5_name):
                link_name = h5_name.replace("instrument", "measurement")
                _create_link(self.h5f, link_name, ds,
                             link_type=self.link_type,
                             overwrite_data=self.overwrite_data)

            # this has to be at the end if we want link creation and
            # dataset creation to remain independent for odd cases
            # where dataset exists but not the link
            if not self.overwrite_data and member_initially_exists:
                _logger.warn("Ignoring existing dataset: " + h5_name)

        elif isinstance(obj, SpecH5Group):
            if h5_name not in self.h5f:
                _logger.debug("Creating group: " + h5_name)
                grp = self.h5f.create_group(h5_name)
            else:
                grp = self.h5f[h5_name]

            # add HDF5 attributes
            for key in obj.attrs:
                if self.overwrite_data or key not in grp.attrs:
                    grp.attrs.create(key, numpy.string_(obj.attrs[key]))

            # link:
            # /1.1/measurement/mca_0/info  --> /1.1/instrument/mca_0/
            if re.match(r".*/([0-9]+\.[0-9]+)/instrument/mca_([0-9]+)/?$",
                        h5_name):
                link_name = h5_name.replace("instrument", "measurement")
                link_name += "/info"
                _create_link(self.h5f, link_name, grp,
                             link_type=self.link_type,
                             overwrite_data=self.overwrite_data)


def write_spec_to_h5(specfile, h5file, h5path='/',
                     mode="a", overwrite_data=False,
                     link_type="hard", create_dataset_args=None):
    """Write content of a SpecFile in a HDF5 file.

    :param specfile: Path of input SpecFile or :class:`SpecH5` object
    :param h5file: Path of output HDF5 file or HDF5 file handle
        (`h5py.File` object)
    :param h5path: Target path in HDF5 file in which scan groups are created.
        Default is root (``"/"``)
    :param mode: Can be ``"r+"`` (read/write, file must exist),
        ``"w"`` (write, existing file is lost), ``"w-"`` (write, fail
        if exists) or ``"a"`` (read/write if exists, create otherwise).
        This parameter is ignored if ``h5file`` is a file handle.
    :param overwrite_data: If ``True``, existing groups and datasets can be
        overwritten, if ``False`` they are skipped. This parameter is only
        relevant if ``file_mode`` is ``"r+"`` or ``"a"``.
    :param link_type: ``"hard"`` (default) or ``"soft"``
    :param create_dataset_args: Dictionary of args you want to pass to
        ``h5py.File.create_dataset``. This allows you to specify filters and
        compression parameters. Don't specify ``name`` and ``data``.
        These arguments don't apply to scalar datasets.

    The structure of the spec data in an HDF5 file is described in the
    documentation of :mod:`silx.io.spech5`.
    """
    if not isinstance(specfile, SpecH5):
        sfh5 = SpecH5(specfile)
    else:
        sfh5 = specfile

    if not h5path.endswith("/"):
        h5path += "/"

    writer = SpecToHdf5Writer(h5path=h5path,
                              overwrite_data=overwrite_data,
                              link_type=link_type,
                              create_dataset_args=create_dataset_args)

    if not isinstance(h5file, h5py.File):
        # If h5file is a file path, open and close it
        with h5py.File(h5file, mode) as h5f:
            writer.write(sfh5, h5f)
    else:
        writer.write(sfh5, h5file)


def convert(specfile, h5file, mode="w-",
            create_dataset_args=None):
    """Convert a SpecFile into an HDF5 file, write scans into the root (``/``)
    group.

    :param specfile: Path of input SpecFile or :class:`SpecH5` object
    :param h5file: Path of output HDF5 file, or h5py.File object
    :param mode: Can be ``"w"`` (write, existing file is
        lost), ``"w-"`` (write, fail if exists). This is ignored
        if ``h5file`` is a file handle.
    :param create_dataset_args: Dictionary of args you want to pass to
        ``h5py.File.create_dataset``. This allows you to specify filters and
        compression parameters. Don't specify ``name`` and ``data``.

    This is a convenience shortcut to call::

        write_spec_to_h5(specfile, h5file, h5path='/',
                         mode="w-", link_type="hard")
    """
    if mode not in ["w", "w-"]:
        raise IOError("File mode must be 'w' or 'w-'. Use write_spec_to_h5" +
                      " to append Spec data to an existing HDF5 file.")
    write_spec_to_h5(specfile, h5file, h5path='/', mode=mode,
                     create_dataset_args=create_dataset_args)
