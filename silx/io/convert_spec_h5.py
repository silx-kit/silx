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
"""Convert a SpecFile into a HDF5 file"""

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "08/03/2016"

import h5py
import re
from .specfileh5 import SpecFileH5, SpecFileH5Group, SpecFileH5Dataset

# link: /1.1/measurement/mca_0/  --> /1.1/instrument/mca_0/
mca_group_link_pattern = re.compile(r"/([0-9]+\.[0-9]+)/measurement/mca_([0-9]+)/?$")
mca_group_pattern = re.compile(r"/([0-9]+\.[0-9]+)/instrument/mca_([0-9]+)/?$")

# every subitem recursively accessed through the link should be ignored to
# avoid data duplication
ignore_mca_link_pattern = re.compile(r"/[0-9]+\.[0-9]+/measurement/mca_[0-9]+/.+$")


mca_data_pattern = re.compile(r"/[0-9]+\.[0-9]+/measurement/mca_([0-9]+)/data$")
mca_calib_pattern = re.compile(r"/[0-9]+\.[0-9]+/measurement/mca_[0-9]+/info/calibration$")

def convert(spec_filename, hdf5_filename):
    sfh5 = SpecFileH5(spec_filename)
    with h5py.File(hdf5_filename, 'w') as h5file:

        def append_to_h5(name, obj):
            if (ignore_mca_link_pattern.match(name) or
                mca_group_link_pattern.match(name) or
                name == "/"):                                      # root already exists -> valueError
                pass

            elif isinstance(obj, SpecFileH5Dataset):
                h5file[name] = obj
                # sfh5.create_dataset(name, data=obj, dtype=np.float32)

            # filling all datasets should be enough, as HDF5 will create all
            # the intermediate groups when necessary and SpecFileH5 doesn't
            # have empty groups (without datasets)
            elif isinstance(obj, SpecFileH5Group):
                grp = h5file.create_group(name)
                # after creating a MCA group, immediately create a link
                if mca_group_pattern.match(name):
                    h5file[name.replace("instrument", "measurement")] = grp   # hard link
                    # h5file[name.replace("instrument", "measurement")] = h5py.SoftLink(name)

        sfh5.visititems(append_to_h5)





