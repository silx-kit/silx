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
"""Convert a SpecFile into a HDF5 file"""

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "09/03/2016"

import h5py
import logging
import re
from .specfileh5 import SpecFileH5, SpecFileH5Group, SpecFileH5Dataset, \
    SpecFileH5LinkToGroup

logger = logging.getLogger('silx.io.convert_spec_h5')
logger.setLevel(logging.DEBUG)


def convert(spec_filename, hdf5_filename):
    sfh5 = SpecFileH5(spec_filename)
    with h5py.File(hdf5_filename, 'w') as h5file:

        def append_spec_member_to_h5(name, obj):
            if isinstance(obj, SpecFileH5Dataset):
                logger.debug("Saving dataset: " + name)
                h5file[name] = obj
                # alternative: sfh5.create_dataset(name, data=obj, dtype=np.float32)

            elif isinstance(obj, SpecFileH5LinkToGroup):
                # link will be created at the same time as the target group
                pass

            elif isinstance(obj, SpecFileH5Group):
                logger.debug("Creating group: " + name)
                if not name in h5file:
                    grp = h5file.create_group(name)

                # link: /1.1/measurement/mca_0/info  --> /1.1/instrument/mca_0/
                if re.match(r"/([0-9]+\.[0-9]+)/instrument/mca_([0-9]+)/?$",
                                name):
                    link_name = name.replace("instrument", "measurement") + "/info"
                    logger.debug("Creating link: " + link_name +
                                 " --> " + name)
                    h5file[link_name] = name    # hard link
                    # h5file[link_name] = h5py.SoftLink(name)

        sfh5.visititems(append_spec_member_to_h5)





