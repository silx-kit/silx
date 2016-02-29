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
__date__ = "29/02/2016"


scan_subgroups = ["title", "start_time", "instrument", "measurement"]
instrument_subgroups = ["positioners"]
mca_subgroups = ["data", "info"]

scan_pattern = re.compile(r"/[0-9]+\.[0-9]+/?$")
instrument_pattern = re.compile(r"/[0-9]+\.[0-9]+/instrument/?$")
positioners_group_pattern = re.compile(r"/[0-9]+\.[0-9]+/instrument/positioners/?$")
measurement_group_pattern  = re.compile(r"/[0-9]+\.[0-9]+/measurement/?$")
mca_group_pattern  = re.compile(r"/[0-9]+\.[0-9]+/measurement/mca_[0-9]+/?$")

title_pattern = re.compile(r"/[0-9]+\.[0-9]+/title$")
start_time_pattern = re.compile(r"/[0-9]+\.[0-9]+/start_time$")
positioners_data_pattern = re.compile(r"/[0-9]+\.[0-9]+/instrument/positioners/.+$")
measurement_data_pattern = re.compile(r"/[0-9]+\.[0-9]+/measurement/.+$")
mca_data_pattern = re.compile(r"/[0-9]+\.[0-9]+/measurement/mca_[0-9]+/data$")
mca_info_pattern = re.compile(r"/[0-9]+\.[0-9]+/measurement/mca_[0-9]+/info$")

def is_group(name):
    """Check if ``name`` is a valid group in a generic :class:`SpecFileH5`.

    :return: True or False
    :rtype: boolean
    """
    return name == "/" or \
           scan_pattern.match(name) or\
           instrument_pattern.match(name) or\
           positioners_group_pattern.match(name) or\
           measurement_group_pattern.match(name) or\
           mca_group_pattern.match(name)

def is_dataset(name):
    """Check if ``name`` is a valid dataset in a generic :class:`SpecFileH5`.

    :return: True or False
    :rtype: boolean
    """
    return title_pattern.match(name) or\
           start_time_pattern.match(name) or\
           positioners_data_pattern.match(name) or\
           measurement_data_pattern.match(name) or\
           mca_data_pattern.match(name) or\
           mca_info_pattern.match(name)  # FIXME: this one is probably a group


def specDateToIso8601(self, date, zone=None):
    """Convert SpecFile date to Iso8601.

    Example:

        specDateToIso8601("Thu Feb 11 09:54:35 2016") -> "2016-02-11T09:54:35"
    """
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'Jun', 'Jul',
              'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    items = date.split()
    year = items[-1]
    hour = items[-2]
    day  = items[-3]
    month = "%02d" % (int(months.index(items[-4])) + 1)
    if zone is None:
        return "%s-%s-%sT%s" % (year, month, day, hour)
    else:
        return "%s-%s-%sT%s%s" % (year, month, day, hour, zone)

class SpecFileH5Group(object):
    """Emulate :class:`h5py.Group` for a SpecFile object

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


        match = re.match(r"/([0-9]+\.[0-9]+)", name)
        if name != "/":
            self._scan = self._sfh5._sf[match.group(1)]

    def __repr__(self):
        return '<SpecFileH5Group "%s" (%d members)>' % (self.name, len(self))

    def __len__(self):
        """Return number of members attached to this group"""
        return len(self.keys())

    def __getitem__(self, key):
        """Return a :class:`SpecFileH5Group` or a :class:`SpecFileH5Dataset`
        if ``key`` is a valid member attached to this group . If ``key`` is
        not valid, raise a KeyError.

        :param key: Name of member
        :type key: str
        """
        if not key in self.keys():
            msg = key + " is not a valid member of " + self.__repr__() + "."
            msg += " List of valid keys: " + ", ".join(self.keys())
            raise KeyError(msg)

        full_key = os.path.join(self.name, key)

        if is_group(full_key):
            return SpecFileH5Group(full_key, self._sfh5)

        elif is_dataset(full_key):
            return SpecFileH5Dataset(full_key, self._sfh5)

        else:
            raise KeyError("unrecognized group or dataset: " + full_key)

    def __iter__(self):
        for key in self.keys():
            yield key

    def keys(self):
        """:return: List of all names of members attached to this group
        :rtype: list of strings
        """
        if scan_pattern.match(self.name):
            return scan_subgroups

        if instrument_pattern.match(self.name):
            return instrument_subgroups

        if positioners_group_pattern.match(self.name):
            return self._scan.motor_names

        if mca_group_pattern.match(self.name):
            return mca_subgroups

        # consistency check: number of data columns must be equal to number of
        # labels
        assert self._scan.data.shape[1] == len(self._scan.labels)

        number_of_MCA_spectra = len(self._scan.mca)
        number_of_data_lines = self._scan.data.shape[0]
        number_of_MCA_analysers = number_of_MCA_spectra // number_of_data_lines


        if measurement_group_pattern.match(self.name):
            mca_list = ["mca_%d" % (i) for i in range(number_of_MCA_analysers)]
            return self._scan.labels + mca_list


    def visit(self, func):
        """Recursively visit all names in this group and subgroups.

        :param func: Callable (function, method or callable object)
        :type func: function

        You supply a callable (function, method or callable object); it
        will be called exactly once for each link in this group and every
        group below it. Your callable must conform to the signature:

            func(<member name>) => <None or return value>

        Returning None continues iteration, returning anything else stops
        and immediately returns that value from the visit method.  No
        particular order of iteration within groups is guaranteed.

        Example:

        .. code-block:: python

            # Get a list of all contents in a SpecFile
            mylist = []
            f = File('foo.dat')
            f.visit(mylist.append)
        """
        for member_name in self.keys():
            ret = func(member_name)
            if ret is not None:
                return ret
            # recurse into subgroups
            if isinstance(self[member_name], SpecFileH5Group):
                self[member_name].visit(func)

    def visititems(self, func):
        """Recursively visit names and objects in this group.

        You supply a callable (function, method or callable object); it
        will be called exactly once for each link in this group and every
        group below it. Your callable must conform to the signature:

            func(<member name>, <object>) => <None or return value>

        Returning None continues iteration, returning anything else stops
        and immediately returns that value from the visit method.  No
        particular order of iteration within groups is guaranteed.

        Example:

        .. code-block:: python

            # Get a list of all datasets in a specific scan
            mylist = []
            def func(name, obj):
                if isinstance(obj, SpecFileH5Dataset):
                    mylist.append(name)

            f = File('foo.dat')
            f["1.1"].visititems(func)
        """
        for member_name in self.keys():
            ret = func(member_name, self[member_name])
            if ret is not None:
                return ret
            # recurse into subgroups
            if isinstance(self[member_name], SpecFileH5Group):
                self[member_name].visititems(func)


class SpecFileH5(SpecFileH5Group):
    """Special :class:`SpecFileH5Group` representing the root of a SpecFile.

    Members of this group are SpecFile scans.

    In addition to :class:`SpecFileH5Group` attributes, this class keeps a
    reference to the original :class:`SpecFile` object.
    """
    def __init__(self, filename):
        #
        super(SpecFileH5, self).__init__("/", self)

        self._filename = filename
        self._sf = SpecFile(filename)
        #self._scan = None

    def keys(self):
        """
        :return: List of all scan keys in this SpecFile (e.g. 1.1, 2.1...)
        :rtype: list of strings
        """
        # return scan keys
        return self._sf.keys()

    def __repr__(self):
        return '<SpecFileH5 "%s" (%d members)>' % (self._filename, len(self))



class SpecFileH5Dataset(object):
    """Emulate :class:`h5py.Dataset` for a SpecFile object

    :param name: Datatset full name (posix path format, starting with /)
    :type name: str
    :param specfileh5: parent :class:`SpecFileH5` object
    :type specfileh5: :class:`SpecFileH5`
    """
    def __init__(self, name, specfileh5):
        self.name = name
        """Full name/path of group"""

        self._sfh5 = specfileh5
        """Parent SpecFileH5 object"""