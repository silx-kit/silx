# coding: utf-8
#/*##########################################################################
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
"""
Python h5 module and octave h5 module have differente ways to deal without
h5 files.
This module is used to make the link between octave and python using such files.
(python is using a dictionnay and octave a struct )

This module provides to prepare fasttomo input as HDF5 file.

.. note:: These functions depend on the `h5py <http://www.h5py.org/>`_ 
    library, which is not a mandatory dependency for `silx`.
"""

import sys
import os.path

import logging
logger = logging.getLogger(__name__)
import numpy as np
from silx.gui import qt

try:
    import h5py
except ImportError as e:
    logger.error("Module " + __name__ + " requires h5py")
    raise e

__authors__ = ["C. Nemoz", "H. Payno"]
__license__ = "MIT"
__date__ = "25/05/2016"


class Octaveh5():
    """
    This class allows communication between octave and python using hdf5 format.
    """  

    def __init__(self, octave_targetted_version=3.8):
            self.file = None
            self.octave_targetted_version = octave_targetted_version

    def open(self, h5file, mode='r'):
        """Open the h5 file which has been write by octave"""
        try:
            self.file = h5py.File(h5file, mode)
            return self
        except:
            if mode == 'a' : 
                reason = "\n %s: Can t find or create " % h5file
            else :
                reason = "\n %s: File not found" % h5file
            self.file = None

            print reason
            logger.info(reason)
            return

    def get(self, struct_name):
        """Read octave equivalent structures in hdf5 file"""   
        data_dict= {}

        grr=(self.file[struct_name].items()[1])[1]
        try:
            gr_level2=grr.items()
        except:
            reason = "no gr_level2"
            print reason
            return

        for key, val in dict(gr_level2).iteritems():
            data_dict[str(key)] = (val.items()[1])[1].value

            # In the case Octave have added a 0 at the end
            if (val.items()[0])[1].value == 'sq_string' and self.octave_targetted_version < 3.8 :
                data_dict[str(key)] = data_dict[str(key)][:-1]

        return data_dict

    def write(self, struct_name, data_dict):
        if not self.file : 
            info = "No file currently openned"
            print info
            logger.info(info)
        # write
        group_l1 = self.file.create_group(struct_name)
        group_l1.attrs['OCTAVE_GLOBAL'] = np.uint8(1)
        group_l1.attrs['OCTAVE_NEW_FORMAT'] = np.uint8(1)
        data_l1 = group_l1.create_dataset("type", data=np.string0('scalar struct'), dtype="|S14")
        group_l2 = group_l1.create_group('value')
        for ftparams in data_dict:
            group_l3 = group_l2.create_group(ftparams)
            group_l3.attrs['OCTAVE_NEW_FORMAT'] = np.uint8(1)
            if type(data_dict[ftparams]) == str:
                data_l2 = group_l3.create_dataset("type",(), data=np.string0('sq_string'), dtype="|S10")
                if self.octave_targetted_version < 3.8 :
                    data_l3 = group_l3.create_dataset("value", data=np.string0(data_dict[ftparams]+'0'))
                else :
                    data_l3 = group_l3.create_dataset("value", data=np.string0(data_dict[ftparams]))
            else:
                data_l2 = group_l3.create_dataset("type",(), data=np.string0('scalar'), dtype="|S7")
                data_l3 = group_l3.create_dataset("value", data=data_dict[ftparams])

    def close(self):
        """Close the file after calling read function"""
        if self.file :
            self.file.close()

    def __del__(self):
        self.close()

        