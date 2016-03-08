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
It is identical for all MCA analysers, as there can be only one
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

from __future__ import unicode_literals
import logging
import numpy
import re

from .specfile import SpecFile

logger1 = logging.getLogger('silx.io.specfileh5')

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "04/03/2016"

# Static subgroups
scan_subgroups = ["title", "start_time", "instrument", "measurement"]
instrument_subgroups = ["positioners"]  # also dynamic subgroups: mca_0…
mca_subgroups = ["data", "info"]
mca_info_subgroups = ["calibration", "channels"]

# Patterns for group keys
root_pattern = re.compile(r"/$")
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

# MCA pattern used to find MCA analyser index
general_mca_pattern = re.compile(r"/.*/mca_([0-9]+)[^0-9]*")

def _bulk_match(string_, list_of_patterns):
    """Check whether a string matches any regular expression pattern in a list

    :param string_: String to match
    :param list_of_patterns: List of regular expressions
    :return: True or False
    """
    for pattern in list_of_patterns:
        if pattern.match(string_):
            return True
    return False

def is_group(name):
    """Check if ``name`` is a valid group in a :class:`SpecFileH5`.

    :param name: Full name of member
    :type name: str
    :return: ``True`` if this member is a group
    :rtype: boolean
    """
    list_of_group_patterns = (
        scan_pattern, instrument_pattern,
        positioners_group_pattern, measurement_group_pattern,
        mca_group_pattern, mca_group_pattern2,
        mca_info_pattern, mca_info_pattern2
    )
    return _bulk_match(name, list_of_group_patterns)


def is_dataset(name):
    """Check if ``name`` is a valid dataset in a :class:`SpecFileH5`.

    :param name: Full name of member
    :type name: str
    :return: ``True`` if this member is a dataset
    :rtype: boolean
    """
    # /1.1/measurement/mca_0 could be interpreted as a data column
    # with label "mca_0"
    if mca_group_pattern.match(name):
        return False

    list_of_data_patterns = (
        title_pattern, start_time_pattern,
        positioners_data_pattern, measurement_data_pattern,
        mca_data_pattern, mca_data_pattern2,
        mca_calib_pattern, mca_calib_pattern2,
        mca_chann_pattern, mca_chann_pattern2
    )
    return _bulk_match(name, list_of_data_patterns)


# Associate group and dataset patterns to their attributes
pattern_attrs = {
    root_pattern:
        {"NX_class": "NXroot",
         },
    scan_pattern:
        {"NX_class": "NXentry", },
    instrument_pattern:
        {"NX_class": "NXinstrument", },
    positioners_group_pattern:
        {"NX_class": "", },
    measurement_group_pattern:
        {"NX_class": "measurement", },
    mca_group_pattern:
        {"NX_class": "NXsubentry", },
    mca_group_pattern2:
        {"NX_class": "NXsubentry", },
    mca_info_pattern:
        {"NX_class": "", },
    mca_info_pattern2:
        {"NX_class": "", },
    title_pattern:
        {},
    start_time_pattern:
        {},
    positioners_data_pattern:
        {},
    measurement_data_pattern:
        {},
    mca_data_pattern:
        {"intepretation": "spectrum", },
    mca_data_pattern2:
        {"intepretation": "spectrum", },
    mca_calib_pattern:
        {},
    mca_calib_pattern2:
        {},
    mca_chann_pattern:
        {},
    mca_chann_pattern2:
        {},
}


def _get_attrs_dict(name):
    """Return attributes dictionary corresponding to the group or dataset
    pointed to by name.

    :param name: Full name/path to data or group
    :return: attributes dictionary
    """
    for pattern in pattern_attrs:
        if pattern.match(name):
            return pattern_attrs[pattern]


def _get_scan_key_in_name(item_name):
    """
    :param item_name: Name of a group or dataset
    :return: Scan identification key (e.g. ``"1.1"``)
    :rtype: str on None
    """
    scan_match = re.match(r"/([0-9]+\.[0-9]+)", item_name)
    if not scan_match:
        return None
    return scan_match.group(1)


def _get_mca_index_in_name(item_name):
    """
    :param item_name: Name of a group or dataset
    :return: MCA analyser index, ``None`` if item name does not reference
        a mca dataset
    :rtype: int or None
    """
    mca_match = general_mca_pattern.match(item_name)
    if not mca_match:
        return None
    return int(mca_match.group(1))


def _get_motor_in_name(item_name):
    """
    :param item_name: Name of a group or dataset
    :return: Motor name or ``None``
    :rtype: str on None
    """
    motor_match = positioners_data_pattern.match(item_name)
    if not motor_match:
        return None
    return motor_match.group(1)

def _get_data_column_label_in_name(item_name):
    """
    :param item_name: Name of a group or dataset
    :return: Data column label or ``None``
    :rtype: str on None
    """
    # /1.1/measurement/mca_0 should not be interpreted as the label of a
    # data column (let's hope no-one ever uses mca_0 as a label)
    if mca_group_pattern.match(item_name):
        return None
    data_column_match = measurement_data_pattern.match(item_name)
    if not data_column_match:
        return None
    return data_column_match.group(1)


def scan_in_specfile(sf, scan_key):
    """
    :param sf: :class:`SpecFile` instance
    :param scan_key: Scan identification key (e.g. ``1.1``)
    :return: ``True`` if scan exists in SpecFile, else ``False``
    """
    return scan_key in sf.keys()


def mca_analyser_in_scan(sf, scan_key, mca_analyser_index):
    """
    :param sf: :class:`SpecFile` instance
    :param scan_key: Scan identification key (e.g. ``1.1``)
    :param mca_analyser_index: 0-based index of MCA analyser
    :return: ``True`` if MCA analyser exists in Scan, else ``False``
    :raise: ``KeyError`` if scan_key not found in SpecFile
    :raise: ``AssertionError`` if number of MCA spectra is not a multiple
          of the number of data lines
    """
    if not scan_in_specfile(sf, scan_key):
        raise KeyError("Scan key %s " % scan_key +
                       "does not exist in SpecFile %s" % sf.filename)

    number_of_MCA_spectra = len(sf[scan_key].mca)
    number_of_data_lines = sf[scan_key].data.shape[0]

    # Number of MCA spectra must be a multiple of number of data lines
    assert number_of_MCA_spectra % number_of_data_lines == 0
    number_of_MCA_analysers = number_of_MCA_spectra // number_of_data_lines

    return 0 <= mca_analyser_index < number_of_MCA_analysers

def motor_in_scan(sf, scan_key, motor_name):
    """
    :param sf: :class:`SpecFile` instance
    :param scan_key: Scan identification key (e.g. ``1.1``)
    :param motor_name: Name of motor as defined in file header lines
    :return: ``True`` if motor exists in scan, else ``False``
    :raise: ``KeyError`` if scan_key not found in SpecFile
    """
    if not scan_in_specfile(sf, scan_key):
        raise KeyError("Scan key %s " % scan_key +
                       "does not exist in SpecFile %s" % sf.filename)
    return motor_name in sf[scan_key].motor_names


def column_label_in_scan(sf, scan_key, column_label):
    """
    :param sf: :class:`SpecFile` instance
    :param scan_key: Scan identification key (e.g. ``1.1``)
    :param column_label: Column label as defined in scan header
    :return: ``True`` if data column label exists in scan, else ``False``
    :raise: ``KeyError`` if scan_key not found in SpecFile
    """
    if not scan_in_specfile(sf, scan_key):
        raise KeyError("Scan key %s " % scan_key +
                       "does not exist in SpecFile %s" % sf.filename)
    return column_label in sf[scan_key].labels


def spec_date_to_iso8601(date, zone=None):
    """Convert SpecFile date to Iso8601.

    :param date: Date in SpecFile format
    :type date: str
    :param zone: Time zone as it appears in a ISO8601 date

    Example:

        ``spec_date_to_iso8601("Thu Feb 11 09:54:35 2016") => "2016-02-11T09:54:35"``
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


class SpecFileH5Dataset(numpy.ndarray):
    """Emulate :class:`h5py.Dataset` for a SpecFile object

    :param array_like: Input dataset in a type that can be digested by
        ``numpy.array()`` (`str`, `list`, `numpy.ndarray`…)
    :param name: Dataset full name (posix path format, starting with ``/``)
    :type name: str

    This class inherits from :class:`numpy.ndarray` and adds ``name`` and
    ``value`` attributes for HDF5 compatibility. ``value`` is a reference
    to the class instance (``value = self``).

    Data is stored in float32 format, unless it is a string.
    """
    # For documentation on subclassing numpy.ndarray,
    # see http://docs.scipy.org/doc/numpy-1.10.1/user/basics.subclassing.html
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

        obj.attrs = _get_attrs_dict(name)

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.name = getattr(obj, 'name', None)
        self.value = getattr(obj, 'value', None)
        self.attrs = getattr(obj, 'attrs', None)


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
    scan_key = _get_scan_key_in_name(name)
    scan = specfileh5._sf[scan_key]

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
        if (not name.startswith("/") or not name in specfileh5):
            raise KeyError("Invalid group name " + name)

        self.name = name
        """Full name/path of group"""

        self._sfh5 = specfileh5
        """Parent SpecFileH5 object"""

        self.attrs = _get_attrs_dict(name)
        """Attributes dictionary"""

        if name != "/":
            scan_key = _get_scan_key_in_name(name)
            self._scan = self._sfh5._sf[scan_key]

    def __repr__(self):
        return '<SpecFileH5Group "%s" (%d members)>' % (self.name, len(self))

    def __eq__(self, other):
        return (isinstance(other, SpecFileH5Group) and
                self.name == other.name and
                self._sfh5.filename == other._sfh5.filename and
                self.keys() == other.keys())

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
            # should never happen thanks to ``key in self.keys()`` test
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

        if (mca_group_pattern.match(self.name) or
            mca_group_pattern2.match(self.name)):
            return mca_subgroups

        if (mca_info_pattern.match(self.name) or
            mca_info_pattern2.match(self.name)):
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
        super(SpecFileH5, self).__init__(name="/",
                                         specfileh5=self)

        self.filename = filename
        self.attrs = _get_attrs_dict("/")
        self._sf = SpecFile(filename)

    def keys(self):
        """
        :return: List of all scan keys in this SpecFile
            (e.g. ``["1.1", "2.1"…]``)
        """
        return self._sf.keys()

    def __repr__(self):
        return '<SpecFileH5 "%s" (%d members)>' % (self.filename, len(self))

    def __eq__(self, other):
        return (isinstance(other, SpecFileH5) and
                self.filename == other.filename and
                self.keys() == other.keys())

    def __getitem__(self, key):
        """In addition to :func:`SpecFileH5Group.__getitem__` (inherited),
        :func:`SpecFileH5.__getitem__` allows access to groups or datasets
        using their full path.

        :param key: Scan key (e.g ``"1.1"``) or full name of group or dataset
            (e.g. ``"/2.1/instrument/positioners"``)
        :return: Requested :class:`SpecFileH5Group` or  :class:`SpecFileH5Dataset`
        """
        if not key.startswith("/"):
            # raises a KeyError if key not in self.keys
            return SpecFileH5Group.__getitem__(self, key)

        if is_group(key):
            return SpecFileH5Group(name=key,
                                   specfileh5=self)
        if is_dataset(key):
            return _dataset_builder(name=key,
                                    specfileh5=self)

        raise KeyError(key + " is not a valid group or dataset in " +
                       self.filename)

    def __contains__(self, key):
        """

        :param key:  Scan key (e.g ``"1.1"``) or full name of group or dataset
            (e.g. ``"/2.1/instrument/positioners"``)
        :return: True if key refers to a valid group or dataset in this SpecFile,
            else False
        """
        # root
        if key == "/":
            return True

        # scan key without leading /
        if not key.startswith("/"):
            return key in self.keys()

        # invalid key
        if not is_group(key) and not is_dataset(key):
            print 1
            return False

        #  nonexistent scan in specfile
        scan_key = _get_scan_key_in_name(key)
        if not scan_in_specfile(self._sf, scan_key):
            print 2
            return False

        #  nonexistent MCA analyser in scan
        mca_analyser_index = _get_mca_index_in_name(key)
        if mca_analyser_index is not None:
            if not mca_analyser_in_scan(self._sf,
                                        scan_key,
                                        mca_analyser_index):
                print 3
                return False

        #  nonexistent motor name
        motor_name = _get_motor_in_name(key)
        if motor_name is not None:
            if not motor_in_scan(self._sf,
                                 scan_key,
                                 motor_name):
                print 4
                return False

        #  nonexistent data column
        column_label = _get_data_column_label_in_name(key)
        if column_label is not None:
            if not column_label_in_scan(self._sf,
                                        scan_key,
                                        column_label):
                print 5
                return False

        # title, start_time, existing scan/mca/motor/measurement
        return True
