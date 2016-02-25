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
"""h5py-like API to SpecFile

API structure:

::

  /
      1.1/
          title = "…"
          start_time = "2016-02-23T22:49:05Z"
          instrument/
              positioners/
                  motor_name = value
                  …
          measurement/
              colname0 = 1D_data_array
              colname1 = …
              mca0/
                  data = (nlines, nchannels) 2D array
                  info = {"CALIB": ""
                          …
                         }
            mca1/
      2.1/
          …

"""
import os.path
import re
from .specfile import SpecFile

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "25/02/2016"


scan_subgroups = ["title", "start_time", "instrument", "measurement"]
instrument_subgroups = ["positioners"]
mca_subgroups = ["data", "info"]

def is_group(name):
    return name == "/" or \
           re.match(r"/[0-9]\.[0-9]/?$", name) or\
           re.match(r"/[0-9]\.[0-9]/instrument/?$", name) or\
           re.match(r"/[0-9]\.[0-9]/instrument/positioners/?$", name) or\
           re.match(r"/[0-9]\.[0-9]/measurement/?$", name) or\
           re.match(r"/[0-9]\.[0-9]/measurement/mca[0-9]+/?$", name)

def is_dataset(name):
    return re.match(r"/[0-9]\.[0-9]/title$", name) or\
           re.match(r"/[0-9]\.[0-9]/date$", name) or\
           re.match(r"/[0-9]\.[0-9]/instrument/positioners/[\w\d]+$", name) or\
           re.match(r"/[0-9]\.[0-9]/measurement/[\w\d]+$", name) or\
           re.match(r"/[0-9]\.[0-9]/measurement/mca[0-9]+/data$", name) or\
           re.match(r"/[0-9]\.[0-9]/measurement/mca[0-9]+/info$", name)


class SpecFileH5Group(object):
    """Emulate HDF5 groups for a SpecFile object

    :param name: Group full name (posix path format, starting with /)
    :type name: str
    :param specfileh5: parent :class:`SpecFileH5` object
    :type specfileh5: :class:`SpecFileH5`

    """

    def __init__(self, name, specfileh5):
        if not name.startswith("/"):
            raise AttributeError("Invalid group name " + name)

        self.name = name
        """Full name/path of group"""

        self._sfh5 = specfileh5
        """Parent SpecFileH5 object"""

        if self.name == "/":
             self.parent = None
             self._scan = None
        else:
            parent_name = os.path.dirname(self.name.rstrip("/"))
            parent = SpecFileH5Group(parent_name, self._sfh5)
            match = re.match(r"/([0-9]+\.[0-9]+)", name)
            self._scan = self._sfh5._sf[match.group(1)]

            number_of_MCA_spectra = len(self._scan.mca)
            number_of_data_lines = self._scan.data.shape[0]

            self.number_of_data_columns = self._scan.data.shape[1]
            assert self.number_of_data_columns == len(self._scan.labels)

            self.number_of_MCA_analysers = number_of_MCA_spectra // number_of_data_lines

    def __repr__(self):
        return '<SpecFileH5Group "%s" (%d members)>' % (self.name, len(self))

    def __len__(self):
        return len(self.keys())

    def keys(self):
        if self.name == "/":
            # return scan keys
            return self._sfh5._sf.keys()

        if re.match(r"/[0-9]\.[0-9]/?$", self.name):
            return scan_subgroups

        if re.match(r"/[0-9]\.[0-9]/instrument/?$", self.name):
            return instrument_subgroups

        if re.match(r"/[0-9]\.[0-9]/instrument/positioners/?$", self.name):
            return self._scan.motor_names()

        if re.match(r"/[0-9]\.[0-9]/measurement/?$", self.name):
            mca_list = ["mca%d" % (i) for i in range(self.number_of_MCA_analysers)]
            return self._scan.labels + mca_list

        if re.match(r"/[0-9]\.[0-9]/measurement/mca[0-9]+/?$", self.name):
            return mca_subgroups


class SpecFileH5Dataset(object):
    """Emulate HDF5 dataset for a SpecFile object

    :param name: Datatset full name (posix path format, starting with /)
    :type name: str
    :param specfileh5: parent :class:`SpecFileH5` object
    :type specfileh5: :class:`SpecFileH5`

    """
    def __init__(self, name, specfileh5):
        raise NotImplementedError


class SpecFileH5(object):
    def __init__(self, filename):
        self._sf = SpecFile(filename)

    def __len__(self):
        """Return number of scans"""
        return len(self._sf)

    def keys(self):
        return self._sf.keys()

    def __getitem__(self, key):
        # TODO: return group or dataset
        """Fetch the requested piece of data or header.

        :param key: Path to data (e.g. "/3.1/measurement/column_name",
            or "/1.1/instrument/positioners/motor_name", or "/1.2/title")
        :type key: str
        """
        try:
            split_path = key.split("/")
        except AttributeError:
            raise TypeError("key must be a string")

        if not key.startswith("/"):
            raise KeyError("key must be a HDF5 like key, starting with /")

        if is_group(key):
            return SpecFileH5Group(key, self)
        elif is_dataset(key):
            return SpecFileH5Dataset(key, self)


        # # first element must be an empty string if key started with "/"
        # else:
        #     # remove empty string
        #     split_path.pop(0)
        #
        # if key == "/":
        #     return ["/" + scan_key for scan_key in self._sf.keys()]
        #
        # # next element should be a scan index such as "1.1"
        # scan_key = split_path.pop(0)
        # scan = self._sf[scan_key]
        #
        # parent_path = "/" + scan_key + "/"
        #
        # # if there is nothing remaining in split_path, list available subfolders
        # if not split_path:
        #     possible_sub_dirs = ["title", "start_time", "instrument", "measurement"]
        #     return [parent_path + item for item in possible_sub_dirs]
        #
        # next_level = split_path.pop(0)
        #
        # if next_level == "title":
        #     return scan.scan_header['S']
        # elif next_level == "start_time":
        #     # TODO: format date
        #     return scan.scan_header['D']
        # elif next_level == "instrument":
        #     if not split_path:
        #         return ["/" + scan_key + "/instrument/positioners"]
        #     else:
        #         next_level = split_path.pop(0)
        #         if next_level == "positioners":
        #             if not split_path:
        #                 return ["/" + scan_key + "/instrument/positioners/" + motor
        #                         for motor in scan.motor_names]
        #             # request for a specific motor position
        #             elif len(split_path) == 1:
        #                 return scan.motor_position_by_name(split_path.pop())
        #             else:
        #                 raise KeyError("invalid key: " + key)
        #         else:
        #             raise KeyError("invalid key: " + key)
        # elif next_level == "measurement":
        #     raise NotImplementedError()
        #     # TODO: implement
