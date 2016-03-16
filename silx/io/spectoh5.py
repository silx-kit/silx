#/*##########################################################################
# coding: utf-8
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
"""This module provides functions to convert a SpecFile into a HDF5 file"""

import h5py
import logging
import re
from .specfileh5 import SpecFileH5, SpecFileH5Group, SpecFileH5Dataset, \
    SpecFileH5LinkToGroup, SpecFileH5LinkToDataset

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "16/03/2016"

logger = logging.getLogger('silx.io.convert_spec_h5')
#logger.setLevel(logging.DEBUG)

# TODO:
#  - test "write_spec_to_h5


def write_spec_to_h5(spec_file, h5_file, h5path='/',
                     h5_file_mode="a", overwrite_data=False,
                     link_type="hard"):
    """Write content of a SpecFile in a HDF5 file.

    :param spec_file: Path of input SpecFile or :class:`SpecFileH5` instance
    :param h5_file: Path of output HDF5 file or HDF5 file handle
    :param h5_path: Target path in HDF5 file in which scan groups are created.
        Default is root (``"/"``)
    :param h5_file_mode: Can be ``"r+"`` (read/write, file must exist),
        ``"w"`` (write, existing file is lost), ``"w-"`` (write, fail if
         exists) or ``"a"`` (read/write if exists, create otherwise).
         This parameter is ignored if ``h5_file`` is a file handle.
    :param overwrite_data: If ``True``, existing groups and datasets can be
        overwritten, if ``False`` they are skipped. This parameter is only
        relevant if ``file_mode`` is ``"r+"`` or ``"a"``.
    :param link_type: ``"hard"`` (default) or ``"soft"``

    The structure of the spec data in an HDF5 file is described in the
    documentation of :mod:`silx.io.specfileh5`.
    """
    if not isinstance(spec_file, SpecFileH5):
        sfh5 = SpecFileH5(spec_file)
    else:
        sfh5 = spec_file

    if not isinstance(h5_file, h5py.File):
        h5f = h5py.File(h5_file, h5_file_mode)
    else:
        h5f = h5_file

    if not h5path.endswith("/"):
        h5path += "/"

    def create_link(link_name, target):
        """Create link or print  warning

        :param link: Link path
        :param target: Handle for target group or dataset
        :param type: Link type (``soft`` or ``hard``)
        """
        if not link_name in h5f:
            logger.debug("Creating link " + link_name + " -> " + target.name)
        elif overwrite_data:
            logger.warn("Overwriting " + link_name + " with link to" +
                        target.name)
            del h5f[link_name]
        else:
            logger.warn(link_name + " already exist. Can't create link to " +
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

        if isinstance(obj, SpecFileH5LinkToGroup) or\
                isinstance(obj, SpecFileH5LinkToDataset):
            # links are created at the same time as their targets
            logger.debug("Ignoring link: " + h5_name)
            pass

        elif isinstance(obj, SpecFileH5Dataset):
            logger.debug("Saving dataset: " + h5_name)
            #h5f[h5_name] = obj
            if not h5_name in h5f:
                ds = h5f.create_dataset(h5_name, data=obj)
            elif overwrite_data:
                logger.warn("Overwriting dataset: " + h5_name)
                del h5f[h5_name]
                ds = h5f.create_dataset(h5_name, data=obj)
            else:
                logger.warn("Ignoring existing dataset: " + h5_name)

            # link:
            #  /1.1/measurement/mca_0/data  --> /1.1/instrument/mca_0/data
            if re.match(r".*/([0-9]+\.[0-9]+)/instrument/mca_([0-9]+)/?data$",
                        h5_name):
                link_name = h5_name.replace("instrument", "measurement")
                create_link(link_name, ds)

        elif isinstance(obj, SpecFileH5Group):
            if not h5_name in h5f:
                logger.debug("Creating group: " + h5_name)
                grp = h5f.create_group(h5_name)

            # link:
            # /1.1/measurement/mca_0/info  --> /1.1/instrument/mca_0/
            if re.match(r".*/([0-9]+\.[0-9]+)/instrument/mca_([0-9]+)/?$",
                        h5_name):
                link_name = h5_name.replace("instrument", "measurement")
                link_name +=  "/info"
                create_link(link_name, grp)

    sfh5.visititems(append_spec_member_to_h5)

    # Close file if it was opened in this function
    if not isinstance(h5_file, h5py.File):
        h5f.close()


def convert(spec_file, h5_file,
            h5_file_mode="w-", link_type="hard"):
    """Convert a SpecFile into an HDF5 file, write scans into the root (``/``)
     group.

    This is a convenience shortcut to call::
        append_spec_to_h5(spec_file, h5_file, h5path='/',
                          h5_file_mode="w-", link_type="hard")

    :param spec_file: Path of input SpecFile or :class:`SpecFileH5` instance
    :param h5_file: Path of output HDF5 file or HDF5 file handle
    :param h5_file_mode: Can be ``"w"`` (write, existing file is lost),
        ``"w-"`` (write, fail if exists).
         This parameter is ignored if ``h5_file`` is a file handle.
    """
    if h5_file_mode not in ["w", "w-"]:
        raise IOError("File mode must be 'w' or 'w-'. Use write_spec_to_h5" +
                      " to append Spec data to an existing HDF5 file.")
    write_spec_to_h5(spec_file, h5_file, h5path='/',
                     h5_file_mode=h5_file_mode, link_type=link_type)

