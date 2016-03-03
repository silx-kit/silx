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

API description
===============
Specfile data structure exposed by this API:

::

  /
      1.1/
          title = "…"
          start_time = "…"
          instrument/
              positioners/
                  motor_name = value
                  …
              mca_0/
                  data = …
                  info/
                      calibration = …
                      channels = …

              mca_1/
                  …
              …
          measurement/
              colname0 = …
              colname1 = …
              mca_0 -> /1.1/instrument/mca_0/ (link)
              …
      2.1/
          …

The title is the content of the ``#S`` scan header line without the leading
``#S`` (e.g ``"1  ascan  ss1vo -4.55687 -0.556875  40 0.2"``).

The start time is in the ISO8601 format (``"2016-02-23T22:49:05Z"``)

All datasets that are not strings are formatted as `float32`.

Motor positions (e.g. ``/1.1/instrument/positioners/motor_name``) can be
scalars as defined in ``#P`` scan header lines, or 1D numpy arrays if they
are measured as scan data. A simple test is done to check if the motor name
is also a data column header defined in the ``#L`` scan header line.

Scan data  (e.g. ``/1.1/measurement/colname0``) is accessed by column,
the dataset name ``colname0`` being the column label as defined in the ``#L``
scan header line.

MCA data is exposed as a 2D numpy array containing all spectra for a given
analyser. The number of analysers is calculated as the number of MCA spectra
per scan data line. Demultiplexing is then performed to assign the correct
spectra to a given analyser.

MCA calibration is an array of 3 scalars, from the ``#@CALIB`` header line.
It is considered identical for all MCA analysers, as there can be only one
``#@CALIB`` line per scan.

MCA channels is an array containing all channel numbers. This information is
computed from the ``#@CHANN`` scan header line (if present), or computed from
the shape of the first spectrum in a scan (``[0, … len(first_spectrum] - 1]``).

Classes
=======

- :class:`SpecFileH5`
- :class:`SpecFileH5Group`
- :class:`SpecFileH5Dataset`
"""
# make all strings unicode
from __future__ import unicode_literals
import logging
import numpy
import re

from .specfile import SpecFile

logger1 = logging.getLogger('silx.io.specfileh5')

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "03/03/2016"

# Static subgroups
scan_subgroups = ["title", "start_time", "instrument", "measurement"]
instrument_subgroups = ["positioners"]  # also dynamic subgroups: mca_0…
mca_subgroups = ["data", "info"]
mca_info_subgroups = ["calibration", "channels"]

# Patterns for group keys
scan_pattern = re.compile(r"/[0-9]+\.[0-9]+/?$")
instrument_pattern = re.compile(r"/[0-9]+\.[0-9]+/instrument/?$")
positioners_group_pattern = re.compile(r"/[0-9]+\.[0-9]+/instrument/positioners/?$")
measurement_group_pattern = re.compile(r"/[0-9]+\.[0-9]+/measurement/?$")
mca_group_pattern = re.compile(r"/[0-9]+\.[0-9]+/measurement/mca_[0-9]+/?$")
mca_group_pattern2 = re.compile(r"/[0-9]+\.[0-9]+/instrument/mca_[0-9]+/?$")
mca_info_pattern = re.compile(r"/[0-9]+\.[0-9]+/measurement/mca_([0-9]+)/info$")
mca_info_pattern2 = re.compile(r"/[0-9]+\.[0-9]+/instrument/mca_([0-9]+)/info$")

# Patterns for dataset keys
title_pattern = re.compile(r"/[0-9]+\.[0-9]+/title$")
start_time_pattern = re.compile(r"/[0-9]+\.[0-9]+/start_time$")
positioners_data_pattern = re.compile(r"/[0-9]+\.[0-9]+/instrument/positioners/(.+)$")
measurement_data_pattern = re.compile(r"/[0-9]+\.[0-9]+/measurement/([^/]+)$")
mca_data_pattern = re.compile(r"/[0-9]+\.[0-9]+/measurement/mca_([0-9]+)/data$")
mca_data_pattern2 = re.compile(r"/[0-9]+\.[0-9]+/instrument/mca_([0-9]+)/data$")
mca_calib_pattern = re.compile(r"/[0-9]+\.[0-9]+/measurement/mca_[0-9]+/info/calibration$")
mca_calib_pattern2 = re.compile(r"/[0-9]+\.[0-9]+/instrument/mca_[0-9]+/info/calibration$")
mca_chann_pattern = re.compile(r"/[0-9]+\.[0-9]+/measurement/mca_[0-9]+/info/channels$")
mca_chann_pattern2 = re.compile(r"/[0-9]+\.[0-9]+/instrument/mca_[0-9]+/info/channels$")


def is_group(name):
    """Check if ``name`` is a valid group in a :class:`SpecFileH5`.

    :param name: Full name of member
    :type name: str
    :return: ``True`` if this member is a group
    :rtype: boolean
    """
    return (name == "/" or
            scan_pattern.match(name) or
            instrument_pattern.match(name) or
            positioners_group_pattern.match(name) or
            measurement_group_pattern.match(name) or
            mca_group_pattern.match(name) or
            mca_group_pattern2.match(name) or
            mca_info_pattern.match(name) or
            mca_info_pattern2.match(name))


def is_dataset(name):
    """Check if ``name`` is a valid dataset in a :class:`SpecFileH5`.

    :param name: Full name of member
    :type name: str
    :return: ``True`` if this member is a dataset
    :rtype: boolean
    """
    return (title_pattern.match(name) or
            start_time_pattern.match(name) or
            positioners_data_pattern.match(name) or
            measurement_data_pattern.match(name) or
            mca_data_pattern.match(name) or
            mca_data_pattern2.match(name) or
            mca_calib_pattern.match(name) or
            mca_calib_pattern2.match(name) or
            mca_chann_pattern.match(name) or
            mca_chann_pattern2.match(name))


def spec_date_to_iso8601(date, zone=None):
    """Convert SpecFile date to Iso8601.

    :param date: Date in SpecFile format
    :type date: str

    Example:

        ``spec_date_to_iso8601("Thu Feb 11 09:54:35 2016")``

        `` => "2016-02-11T09:54:35"``
    """
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'Jun', 'Jul',
              'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    items = date.split()
    year = items[-1]
    hour = items[-2]
    day = items[-3]
    month = "%02d" % (int(months.index(items[-4])) + 1)
    if zone is None:
        return "%s-%s-%sT%s" % (year, month, day, hour)
    else:
        return "%s-%s-%sT%s%s" % (year, month, day, hour, zone)


# For documentation on subclassing numpy.ndarray,
# see http://docs.scipy.org/doc/numpy-1.10.1/user/basics.subclassing.html
class SpecFileH5Dataset(numpy.ndarray):
    """Emulate :class:`h5py.Dataset` for a SpecFile object

    :param array_like: Input dataset in an array like format (string, list…)
    :param name: Dataset full name (posix path format, starting with ``/``)
    :type name: str

    This class inherits from :class:`numpy.ndarray` and adds ``name`` and
    ``value`` attributes for HDF5 compatibility. ``value`` is a reference
    to the class instance (``value = self``).

    Data is stored in float32 format, unless it is a string.
    """
    def __new__(cls, array_like, name):
        if not isinstance(array_like, numpy.ndarray):
            # Ensure our data is a numpy.ndarray
            array = numpy.array(array_like)
        else:
            array = array_like

        # general kind of data
        data_kind = array.dtype.kind
        # byte-string or unicode: leave unchanged
        if data_kind in ["S", "U"]:
            obj = array.view(cls)
        # enforce float32 for int, unsigned int, float
        elif data_kind in ["i", "u", "f"]:
            obj = numpy.asarray(array, dtype=numpy.float32).view(cls)
        # reject boolean (b), complex (c), object (O), void/data block (V)
        else:
            raise TypeError("Unexpected data type " + data_kind +
                            " (expected int-, string- or float-like data)")

        obj.name = name
        obj.value = obj
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.name = getattr(obj, 'name', None)
        self.value = getattr(obj, 'value', None)


def _dataset_builder(name, specfileh5):
    """Retrieve dataset from :class:`SpecFile`, based on dataset name, as a
    subclass of :class:`numpy.ndarray`.

    :param name: Datatset full name (posix path format, starting with ``/``)
    :type name: str
    :param specfileh5: parent :class:`SpecFileH5` object
    :type specfileh5: :class:`SpecFileH5`

    :return: Array with the requested data
    :rtype: :class:`SpecFileH5Dataset`.
    """
    scan_match = re.match(r"/([0-9]+\.[0-9]+)", name)
    if not scan_match:
        raise KeyError("Cannot parse scan key (e.g. '1.1') in dataset name " +
                       name)
    scan = specfileh5._sf[scan_match.group(1)]

    # get dataset in an array-like format (ndarray, str, list…)
    array_like = None
    if title_pattern.match(name):
        array_like = scan.scan_header["S"]

    elif start_time_pattern.match(name):
        try:
            spec_date = scan.scan_header["D"]
        except KeyError:
            logger1.warn("No #D line in scan header. Trying file header.")
            spec_date = scan.file_header["D"]
        array_like = spec_date_to_iso8601(spec_date)

    elif positioners_data_pattern.match(name):
        m = positioners_data_pattern.match(name)
        motor_name = m.group(1)
        # if a motor is recorded as a data column, ignore its position in
        # header and return the data column instead
        if motor_name in scan.labels:
            array_like = scan.data_column_by_name(motor_name)
        else:
            # may return float("inf") if #P line is missing from scan hdr
            array_like = scan.motor_position_by_name(motor_name)

    elif measurement_data_pattern.match(name):
        m = measurement_data_pattern.match(name)
        column_name = m.group(1)
        array_like = scan.data_column_by_name(column_name)

    elif (mca_data_pattern.match(name) or
          mca_data_pattern2.match(name)):
        m = mca_data_pattern.match(name)
        if not m:
            m = mca_data_pattern2.match(name)

        analyser_index = int(m.group(1))
        # retrieve 2D array of all MCA spectra from one analyser
        array_like = _demultiplex_mca(scan, analyser_index)

    elif (mca_calib_pattern.match(name) or
          mca_calib_pattern2.match(name)):
        array_like = scan.mca.calibration

    elif (mca_chann_pattern.match(name) or
          mca_chann_pattern2.match(name)):
        array_like = scan.mca.channels

    if array_like is None:
        raise KeyError("Name " + name + " does not match any known dataset.")

    return SpecFileH5Dataset(array_like, name)


def _demultiplex_mca(scan, analyser_index):
    """Return MCA data for a single analyser.

    Each MCA spectrum is a 1D array. For each analyser, there is one
    spectrum recorded per scan data line. When there are more than a single
    MCA analyser in a scan, the data will be multiplexed. For instance if
    there are 3 analysers, the consecutive spectra for the first analyser must
    be accessed as ``mca[0], mca[3], mca[6]…``.

    :param scan: :class:`Scan` instance containing the MCA data
    :param analyser_index: 0-based index referencing the analyser
    :type analyser_index: int
    :return: 2D numpy array containing all spectra for one analyser
    """
    mca_data = scan.mca

    number_of_MCA_spectra = len(mca_data)
    number_of_scan_data_lines = scan.data.shape[0]

    # Number of MCA spectra must be a multiple of number of scan data lines
    assert number_of_MCA_spectra % number_of_scan_data_lines == 0
    number_of_analysers = number_of_MCA_spectra // number_of_scan_data_lines

    list_of_1D_arrays = []
    for i in range(analyser_index,
                   number_of_MCA_spectra,
                   number_of_analysers):
        list_of_1D_arrays.append(mca_data[i])
    # convert list to 2D array
    return numpy.array(list_of_1D_arrays)


class SpecFileH5Group(object):
    """Emulate :class:`h5py.Group` for a SpecFile object

    :param name: Group full name (posix path format, starting with ``/``)
    :type name: str
    :param specfileh5: parent :class:`SpecFileH5` instance

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

    def __eq__(self, other):
        return isinstance(other, SpecFileH5Group) and \
               self.name == other.name and \
               self._sfh5.filename == other._sfh5.filename and \
               self.keys() == other.keys()

    def __len__(self):
        """Return number of members attached to this group,
        subgroups and datasets."""
        return len(self.keys())

    def __getitem__(self, key):
        """Return a :class:`SpecFileH5Group` or a :class:`SpecFileH5Dataset`
        if ``key`` is a valid member attached to this group . If ``key`` is
        not valid, raise a KeyError.

        :param key: Name of member
        :type key: str
        """
        if key not in self.keys():
            msg = key + " is not a valid member of " + self.__repr__() + "."
            msg += " List of valid keys: " + ", ".join(self.keys())
            if key.startswith("/"):
                msg += "\nYou can access a dataset using its full path from "
                msg += "a SpecFileH5 object, but not from a SpecFileH5Group."
            raise KeyError(msg)

        full_key = self.name.rstrip("/") + "/" + key

        if is_group(full_key):
            return SpecFileH5Group(full_key, self._sfh5)
        elif is_dataset(full_key):
            return _dataset_builder(full_key, self._sfh5)
        else:
            raise KeyError("unrecognized group or dataset: " + full_key)

    def __iter__(self):
        for key in self.keys():
            yield key

    def keys(self):
        """:return: List of all names of members attached to this group
        """
        if scan_pattern.match(self.name):
            return scan_subgroups

        if positioners_group_pattern.match(self.name):
            return self._scan.motor_names

        if mca_group_pattern.match(self.name) or\
           mca_group_pattern2.match(self.name):
            return mca_subgroups

        if mca_info_pattern.match(self.name) or \
           mca_info_pattern2.match(self.name):
            return mca_info_subgroups

        # number of data columns must be equal to number of labels
        assert self._scan.data.shape[1] == len(self._scan.labels)

        number_of_MCA_spectra = len(self._scan.mca)
        number_of_data_lines = self._scan.data.shape[0]

        # Number of MCA spectra must be a multiple of number of data lines
        assert number_of_MCA_spectra % number_of_data_lines == 0
        number_of_MCA_analysers = number_of_MCA_spectra // number_of_data_lines
        mca_list = ["mca_%d" % i for i in range(number_of_MCA_analysers)]

        if measurement_group_pattern.match(self.name):
            return self._scan.labels + mca_list

        if instrument_pattern.match(self.name):
            return instrument_subgroups + mca_list

    def visit(self, func):
        """Recursively visit all names in this group and subgroups.

        :param func: Callable (function, method or callable object)
        :type func: function

        You supply a callable (function, method or callable object); it
        will be called exactly once for each link in this group and every
        group below it. Your callable must conform to the signature:

            ``func(<member name>) => <None or return value>``

        Returning ``None`` continues iteration, returning anything else stops
        and immediately returns that value from the visit method.  No
        particular order of iteration within groups is guaranteed.

        Example:

        .. code-block:: python

            # Get a list of all contents (groups and datasets) in a SpecFile
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

            ``func(<member name>, <object>) => <None or return value>``

        Returning ``None`` continues iteration, returning anything else stops
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

    :param filename: Path to SpecFile in filesystem
    :type filename: str

    In addition to all generic :class:`SpecFileH5Group` attributes, this class
    keeps a reference to the original :class:`SpecFile` object.

    Its immediate children are scans, but it also allows access to any group
    or dataset in the entire SpecFile tree using the full path.

    Example:

    .. code-block:: python

        sfh5 = SpecFileH5("test.dat")

        # method 1: using SpecFileH5 as a regular group
        scan1group = sfh5["1.1"]
        instrument_group = scan1group["instrument"]

        # method 2: full path access
        instrument_group = sfh5["/1.1/instrument"]
    """
    def __init__(self, filename):
        super(SpecFileH5, self).__init__("/", self)

        self.filename = filename
        self._sf = SpecFile(filename)
        # self._scan = None

    def keys(self):
        """
        :return: List of all scan keys in this SpecFile
            (e.g. ``["1.1", "2.1"…]``)
        """
        return self._sf.keys()

    def __repr__(self):
        return '<SpecFileH5 "%s" (%d members)>' % (self.filename, len(self))

    def __eq__(self, other):
        return isinstance(other, SpecFileH5) and \
               self.filename == other.filename and \
               self.keys() == other.keys()

    def __getitem__(self, key):
        """In addition to :func:`SpecFileH5Group.__getitem__` (inherited),
        :func:`SpecFileH5.__getitem__` allows access to groups or datasets
        using their full path.

        :param key: Scan key (e.g ``"1.1"``) or full name of group or dataset
            (e.g. ``"/2.1/instrument/positioners"``)
        :return: Requested :class:`SpecFileH5Group` or  :class:`SpecFileH5Dataset`
        """
        try:
            # access a scan as defined in self.keys()
            return SpecFileH5Group.__getitem__(self, key)
        except KeyError:
            # access a member using an absolute path
            if key.startswith("/"):
                if is_group(key):
                    return SpecFileH5Group(key, self._sfh5)
                elif is_dataset(key):
                    return _dataset_builder(key, self._sfh5)
                else:
                    raise KeyError("unrecognized group or dataset: " + key)
