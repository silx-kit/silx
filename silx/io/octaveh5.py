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
"""
Python h5 module and octave h5 module have different ways to deal with
h5 files.
This module is used to make the link between octave and python using such files.
(python is using a dictionary and octave a struct )

This module provides tool to set HDF5 file for fasttomo input.

Here is an example of a simple read and write :

.. code-block:: python
    :emphasize-lines: 3,5    

    # writing a structure
    myStruct = {'MKEEP_MASK': 0.0, 'UNSHARP_SIGMA': 0.80000000000000004 }
    writer = Octaveh5().open("my_h5file", 'a')
    writer.write('mt_struct_name', myStruct)

    # reading a h5 file
    reader = Octaveh5().open("my_h5file")
    strucDict = reader.get('mt_struct_name')

.. note:: These functions depend on the `h5py <http://www.h5py.org/>`_ 
    library, which is a mandatory dependency for `silx`.

"""

import logging
logger = logging.getLogger(__name__)
import numpy as np
import h5py

__authors__ = ["C. Nemoz", "H. Payno"]
__license__ = "MIT"
__date__ = "05/10/2016"


class Octaveh5(object):
    """This class allows communication between octave and python using hdf5 format.
    """

    def __init__(self, octave_targetted_version=3.8):
        """Constructor

        :param octave_targetted_version: the version of Octave for which we want to write this hdf5 file.
        
        This is needed because for old Octave version we need to had a hack(adding one extra character)
        """
        self.file = None
        self.octave_targetted_version = octave_targetted_version

    def open(self, h5file, mode='r'):
        """Open the h5 file which has been write by octave

        :param h5file: The path of the file to read
        :param mode: the opening mode of the file :'r', 'w'...
        """
        try:
            self.file = h5py.File(h5file, mode)
            return self
        except IOError as e:
            if mode == 'a':
                reason = "\n %s: Can t find or create " % h5file
            else:
                reason = "\n %s: File not found" % h5file
            self.file = None

            logger.info(reason)
            raise e

    def get(self, struct_name):
        """Read octave equivalent structures in hdf5 file

        :param struct_name: the identification of the top level identity we want to get from an hdf5 structure
        :return: the dictionnary of the requested struct. None if can t find it
        """
        if self.file is None:
            info = "No file currently open"
            logger.info(info)
            return None

        data_dict = {}
        grr = (list(self.file[struct_name].items())[1])[1]
        try:
            gr_level2 = grr.items()
        except AttributeError:
            reason = "no gr_level2"
            logger.info(reason)
            return None

        for key, val in iter(dict(gr_level2).items()):
            data_dict[str(key)] = list(val.items())[1][1][()]

            if list(val.items())[0][1][()] != np.string_('sq_string'):
                data_dict[str(key)] = float(data_dict[str(key)])
            else:
                if list(val.items())[0][1][()] == np.string_('sq_string'):
                    # in the case the string has been stored as an nd-array of char
                    if type(data_dict[str(key)]) is np.ndarray:
                        data_dict[str(key)] = "".join(chr(item) for item in data_dict[str(key)])
                    else:
                        data_dict[str(key)] = data_dict[str(key)].decode('UTF-8')

                # In the case Octave have added an extra character at the end
                if self.octave_targetted_version < 3.8:
                    data_dict[str(key)] = data_dict[str(key)][:-1]

        return data_dict

    def write(self, struct_name, data_dict):
        """write data_dict under the group struct_name in the open hdf5 file

        :param struct_name: the identificatioon of the structure to write in the hdf5
        :param data_dict: The python dictionnary containing the informations to write
        """
        if self.file is None:
            info = "No file currently open"
            logger.info(info)
            return

        group_l1 = self.file.create_group(struct_name)
        group_l1.attrs['OCTAVE_GLOBAL'] = np.uint8(1)
        group_l1.attrs['OCTAVE_NEW_FORMAT'] = np.uint8(1)
        group_l1.create_dataset("type", data=np.string_('scalar struct'), dtype="|S14")
        group_l2 = group_l1.create_group('value')
        for ftparams in data_dict:
            group_l3 = group_l2.create_group(ftparams)
            group_l3.attrs['OCTAVE_NEW_FORMAT'] = np.uint8(1)
            if type(data_dict[ftparams]) == str:
                group_l3.create_dataset("type", (), data=np.string_('sq_string'), dtype="|S10")
                if self.octave_targetted_version < 3.8:
                    group_l3.create_dataset("value", data=np.string_(data_dict[ftparams] + '0'))
                else:
                    group_l3.create_dataset("value", data=np.string_(data_dict[ftparams]))
            else:
                group_l3.create_dataset("type", (), data=np.string_('scalar'), dtype="|S7")
                group_l3.create_dataset("value", data=data_dict[ftparams])

    def close(self):
        """Close the file after calling read function
        """
        if self.file:
            self.file.close()

    def __del__(self):
        """Destructor
        """
        self.close()
