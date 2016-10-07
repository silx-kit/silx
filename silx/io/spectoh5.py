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
__date__ = "15/09/2016"


def write_spec_to_h5(specfile, h5file, h5path='/',
                     mode="a", overwrite_data=False,
                     link_type="hard", create_dataset_args=None):
    """Write content of a SpecFile in a HDF5 file.

    :param specfile: Path of input SpecFile or :class:`SpecH5` instance
    :param h5file: Path of output HDF5 file or HDF5 file handle
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
        ``h5f.create_dataset``. This allows you to specify filters and
        compression parameters. Don't specify ``name`` and ``data``.
        These arguments don't apply to scalar datasets.

    The structure of the spec data in an HDF5 file is described in the
    documentation of :mod:`silx.io.spech5`.
    """
    if not isinstance(specfile, SpecH5):
        sfh5 = SpecH5(specfile)
    else:
        sfh5 = specfile

    if not isinstance(h5file, h5py.File):
        h5f = h5py.File(h5file, mode)
    else:
        h5f = h5file

    if not h5path.endswith("/"):
        h5path += "/"

    if create_dataset_args is None:
        create_dataset_args = {}

    def create_link(link_name, target):
        """Create link

        If member with name ``link_name`` already exists, delete it first or
        ignore link depending on global param ``overwrite_data``.

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

    def append_spec_member_to_h5(spec_h5_name, obj):
        h5_name = h5path + spec_h5_name.lstrip("/")

        if isinstance(obj, SpecH5LinkToGroup) or\
                isinstance(obj, SpecH5LinkToDataset):
            # links are created at the same time as their targets
            _logger.debug("Ignoring link: " + h5_name)
            pass

        elif isinstance(obj, SpecH5Dataset):
            _logger.debug("Saving dataset: " + h5_name)

            member_initially_exists = h5_name in h5f

            if overwrite_data and member_initially_exists:
                _logger.warn("Overwriting dataset: " + h5_name)
                del h5f[h5_name]

            if overwrite_data or not member_initially_exists:
                # fancy arguments don't apply to scalars (shape==())
                if obj.shape == ():
                    ds = h5f.create_dataset(h5_name, data=obj.value)
                else:
                    ds = h5f.create_dataset(h5_name, data=obj.value,
                                            **create_dataset_args)
            else:
                ds = h5f[h5_name]

            # add HDF5 attributes
            for key in obj.attrs:
                if overwrite_data or key not in ds.attrs:
                    ds.attrs.create(key, numpy.string_(obj.attrs[key]))

            # link:
            #  /1.1/measurement/mca_0/data  --> /1.1/instrument/mca_0/data
            if re.match(r".*/([0-9]+\.[0-9]+)/instrument/mca_([0-9]+)/?data$",
                        h5_name):
                link_name = h5_name.replace("instrument", "measurement")
                create_link(link_name, ds)

            # this has to be at the end if we want link creation and
            # dataset creation to remain independent for odd cases
            # where dataset exists but not the link
            if not overwrite_data and member_initially_exists:
                _logger.warn("Ignoring existing dataset: " + h5_name)

        elif isinstance(obj, SpecH5Group):
            if h5_name not in h5f:
                _logger.debug("Creating group: " + h5_name)
                grp = h5f.create_group(h5_name)
            else:
                grp = h5f[h5_name]

            # add HDF5 attributes
            for key in obj.attrs:
                if overwrite_data or key not in grp.attrs:
                    grp.attrs.create(key, numpy.string_(obj.attrs[key]))

            # link:
            # /1.1/measurement/mca_0/info  --> /1.1/instrument/mca_0/
            if re.match(r".*/([0-9]+\.[0-9]+)/instrument/mca_([0-9]+)/?$",
                        h5_name):
                link_name = h5_name.replace("instrument", "measurement")
                link_name += "/info"
                create_link(link_name, grp)

    sfh5.visititems(append_spec_member_to_h5)

    # visititems didn't create attributes for the root group
    root_grp = h5f[h5path]
    for key in sfh5.attrs:
        if overwrite_data or key not in root_grp.attrs:
            root_grp.attrs.create(key, numpy.string_(sfh5.attrs[key]))

    # Close file if it was opened in this function
    if not isinstance(h5file, h5py.File):
        h5f.close()


def convert(specfile, h5file, mode="w-",
            create_dataset_args=None):
    """Convert a SpecFile into an HDF5 file, write scans into the root (``/``)
    group.

    :param specfile: Path of input SpecFile or :class:`SpecH5` instance
    :param h5file: Path of output HDF5 file or HDF5 file handle
    :param mode: Can be ``"w"`` (write, existing file is
        lost), ``"w-"`` (write, fail if exists). This is ignored
        if ``h5file`` is a file handle.
    :param create_dataset_args: Dictionary of args you want to pass to
        ``h5f.create_dataset``. This allows you to specify filters and
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
