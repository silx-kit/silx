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
"""This module provides a h5py-like API to access SpecFile data.

API description
===============
Specfile data structure exposed by this API:

::

  /
      1.1/
          title = "…"
          start_time = "…"
          instrument/
              specfile/
                  file_header = ["…", "…", …]
                  scan_header = ["…", "…", …]
              positioners/
                  motor_name = value
                  …
              mca_0/
                  data = …
                  calibration = …
                  channels = …
                  preset_time = …
                  elapsed_time = …
                  live_time = …

              mca_1/
                  …
              …
          measurement/
              colname0 = …
              colname1 = …
              …
              mca_0/
                   data -> /1.1/instrument/mca_0/data
                   info -> /1.1/instrument/mca_0/
              …
      2.1/
          …

``file_header`` and ``scan_header`` are numpy arrays of fixed-length strings
containing raw header lines relevant to the scan.

The title is the content of the ``#S`` scan header line without the leading
``#S`` (e.g ``"1  ascan  ss1vo -4.55687 -0.556875  40 0.2"``).

The start time is in ISO8601 format (``"2016-02-23T22:49:05Z"``)

All numeric datasets store values in `float32` format.

Motor positions (e.g. ``/1.1/instrument/positioners/motor_name``) can be
1D numpy arrays if they are measured as scan data, or else scalars as defined
on ``#P`` scan header lines. A simple test is done to check if the motor name
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

Accessing data
==============

Data and groups are accessed in :mod:`h5py` fashion::

    from silx.io.spech5 import SpecH5

    # Open a SpecFile
    sfh5 = SpecH5("test.dat")

    # using SpecH5 as a regular group to access scans
    scan1group = sfh5["1.1"]
    instrument_group = scan1group["instrument"]

    # altenative: full path access
    measurement_group = sfh5["/1.1/measurement"]

    # accessing a scan data column by name as a 1D numpy array
    data_array = measurement_group["Pslit HGap"]

    # accessing all mca-spectra for one MCA device
    mca_0_spectra = measurement_group["mca_0/data"]

:class:`SpecH5` and :class:`SpecH5Group` provide a :meth:`SpecH5Group.keys` method::

    >>> sfh5.keys()
    ['96.1', '97.1', '98.1']
    >>> sfh5['96.1'].keys()
    ['title', 'start_time', 'instrument', 'measurement']

They can also be treated as iterators:

.. code-block:: python

    for scan_group in SpecH5("test.dat"):
        dataset_names = [item.name in scan_group["measurement"] if
                         isinstance(item, SpecH5Dataset)]
        print("Found data columns in scan " + scan_group.name)
        print(", ".join(dataset_names))

You can test for existence of data or groups::

    >>> "/1.1/measurement/Pslit HGap" in sfh5
    True
    >>> "positioners" in sfh5["/2.1/instrument"]
    True
    >>> "spam" in sfh5["1.1"]
    False

Classes
=======

- :class:`SpecH5`
- :class:`SpecH5Group`
- :class:`SpecH5Dataset`
- :class:`SpecH5LinkToGroup`
- :class:`SpecH5LinkToDataset`
"""

import logging
import numpy
import posixpath
import re
import sys
import time

from .specfile import SpecFile

__authors__ = ["P. Knobel", "D. Naudet"]
__license__ = "MIT"
__date__ = "30/03/2016"

logging.basicConfig()
logger1 = logging.getLogger(__name__)

string_types = (basestring,) if sys.version_info[0] == 2 else (str,)

# Static subitems: all groups and datasets that are present in any
# scan (excludes list of scans, data columns, list of mca devices,
# optional mca headers)
static_items = {
    "scan": [u"title", u"start_time", u"instrument",
             u"measurement"],
    "scan/instrument": [u"specfile", u"positioners"],
    "scan/instrument/specfile": [u"file_header", u"scan_header"],
    "scan/measurement/mca": [u"data", u"info"],
    "scan/instrument/mca": [u"data", u"calibration", u"channels"],
}

# Patterns for group keys
root_pattern = re.compile(r"/$")
scan_pattern = re.compile(r"/[0-9]+\.[0-9]+/?$")
instrument_pattern = re.compile(r"/[0-9]+\.[0-9]+/instrument/?$")
specfile_group_pattern = re.compile(r"/[0-9]+\.[0-9]+/instrument/specfile/?$")
positioners_group_pattern = re.compile(r"/[0-9]+\.[0-9]+/instrument/positioners/?$")
measurement_group_pattern = re.compile(r"/[0-9]+\.[0-9]+/measurement/?$")
measurement_mca_group_pattern = re.compile(r"/[0-9]+\.[0-9]+/measurement/mca_[0-9]+/?$")
instrument_mca_group_pattern = re.compile(r"/[0-9]+\.[0-9]+/instrument/mca_[0-9]+/?$")

# Link to group
measurement_mca_info_pattern = re.compile(r"/[0-9]+\.[0-9]+/measurement/mca_([0-9]+)/info/?$")

# Patterns for dataset keys
header_pattern = re.compile(r"/[0-9]+\.[0-9]+/header$")
title_pattern = re.compile(r"/[0-9]+\.[0-9]+/title$")
start_time_pattern = re.compile(r"/[0-9]+\.[0-9]+/start_time$")
file_header_data_pattern = re.compile(r"/[0-9]+\.[0-9]+/instrument/specfile/file_header$")
scan_header_data_pattern = re.compile(r"/[0-9]+\.[0-9]+/instrument/specfile/scan_header$")
positioners_data_pattern = re.compile(r"/[0-9]+\.[0-9]+/instrument/positioners/([^/]+)$")
measurement_data_pattern = re.compile(r"/[0-9]+\.[0-9]+/measurement/([^/]+)$")
instrument_mca_data_pattern = re.compile(r"/[0-9]+\.[0-9]+/instrument/mca_([0-9]+)/data$")
instrument_mca_calib_pattern = re.compile(r"/[0-9]+\.[0-9]+/instrument/mca_[0-9]+/calibration$")
instrument_mca_chann_pattern = re.compile(r"/[0-9]+\.[0-9]+/instrument/mca_[0-9]+/channels$")
instrument_mca_preset_t_pattern = re.compile(r"/[0-9]+\.[0-9]+/instrument/mca_[0-9]+/preset_time$")
instrument_mca_elapsed_t_pattern = re.compile(r"/[0-9]+\.[0-9]+/instrument/mca_[0-9]+/elapsed_time$")
instrument_mca_live_t_pattern = re.compile(r"/[0-9]+\.[0-9]+/instrument/mca_[0-9]+/live_time$")

# Links to dataset
measurement_mca_data_pattern = re.compile(r"/[0-9]+\.[0-9]+/measurement/mca_([0-9]+)/data$")
measurement_mca_info_dataset_pattern = re.compile(r"/[0-9]+\.[0-9]+/measurement/mca_[0-9]+/info/([^/]+)$")


def _bulk_match(string_, list_of_patterns):
    """Check whether a string matches any regular expression pattern in a list
    """
    for pattern in list_of_patterns:
        if pattern.match(string_):
            return True
    return False


def is_group(name):
    """Check if ``name`` matches a valid group name pattern in a
    :class:`SpecH5`.

    :param name: Full name of member
    :type name: str

    For example:

        - ``is_group("/123.456/instrument/")`` returns ``True``.
        - ``is_group("spam")`` returns ``False`` because :literal:`\"spam\"`
          is not at all a valid group name.
        - ``is_group("/1.2/instrument/positioners/xyz")`` returns ``False``
          because this key would point to a motor position, which is a
          dataset and not a group.
    """
    group_patterns = (
        root_pattern, scan_pattern, instrument_pattern,
        specfile_group_pattern, positioners_group_pattern,
        measurement_group_pattern, measurement_mca_group_pattern,
        instrument_mca_group_pattern
    )
    return _bulk_match(name, group_patterns)


def is_dataset(name):
    """Check if ``name`` matches a valid dataset name pattern in a
    :class:`SpecH5`.

    :param name: Full name of member
    :type name: str

    For example:

        - ``is_dataset("/1.2/instrument/positioners/xyz")`` returns ``True``
          because this name could be the key to the dataset recording motor
          positions for motor ``xyz`` in scan ``1.2``.
        - ``is_dataset("/123.456/instrument/")`` returns ``False`` because
          this name points to a group.
        - ``is_dataset("spam")`` returns ``False`` because :literal:`\"spam\"`
          is not at all a valid dataset name.
    """
    # /1.1/measurement/mca_0 could be interpreted as a data column
    # with label "mca_0"
    if measurement_mca_group_pattern.match(name):
        return False

    data_patterns = (
        header_pattern, title_pattern, start_time_pattern,
        file_header_data_pattern, scan_header_data_pattern,
        positioners_data_pattern, measurement_data_pattern,
        instrument_mca_data_pattern, instrument_mca_calib_pattern,
        instrument_mca_chann_pattern,
        instrument_mca_preset_t_pattern, instrument_mca_elapsed_t_pattern,
        instrument_mca_live_t_pattern
    )
    return _bulk_match(name, data_patterns)


def is_link_to_group(name):
    """Check if ``name`` is a valid link to a group in a :class:`SpecH5`.
    Return ``True`` or ``False``

    :param name: Full name of member
    :type name: str
    """
    # so far we only have one type of link to a group
    if measurement_mca_info_pattern.match(name):
        return True
    return False


def is_link_to_dataset(name):
    """Check if ``name`` is a valid link to a dataset in a :class:`SpecH5`.
    Return ``True`` or ``False``

    :param name: Full name of member
    :type name: str
    """
    list_of_link_patterns = (
        measurement_mca_data_pattern, measurement_mca_info_dataset_pattern,
    )
    return _bulk_match(name, list_of_link_patterns)


def _get_attrs_dict(name):
    """Return attributes dictionary corresponding to the group or dataset
    pointed to by name.

    :param name: Full name/path to data or group
    :return: attributes dictionary
    """
    # Associate group and dataset patterns to their attributes
    pattern_attrs = {
        root_pattern:
            {"NX_class": "NXroot",
             },
        scan_pattern:
            {"NX_class": "NXentry", },
        title_pattern:
            {},
        start_time_pattern:
            {},
        instrument_pattern:
            {"NX_class": "NXinstrument", },
        specfile_group_pattern:
            {"NX_class": "NXcollection", },
        file_header_data_pattern:
            {},
        scan_header_data_pattern:
            {},
        positioners_group_pattern:
            {"NX_class": "NXcollection", },
        positioners_data_pattern:
            {},
        instrument_mca_group_pattern:
            {"NX_class": "NXdetector", },
        instrument_mca_data_pattern:
            {"interpretation": "spectrum", },
        instrument_mca_calib_pattern:
            {},
        instrument_mca_chann_pattern:
            {},
        instrument_mca_preset_t_pattern:
            {},
        instrument_mca_elapsed_t_pattern:
            {},
        instrument_mca_live_t_pattern:
            {},
        measurement_group_pattern:
            {"NX_class": "NXcollection", },
        measurement_data_pattern:
            {},
        measurement_mca_group_pattern:
            {},
        measurement_mca_data_pattern:
            {"interpretation": "spectrum", },
        measurement_mca_info_pattern:
            {"NX_class": "NXdetector", }
    }

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
    mca_match = re.match(r"/.*/mca_([0-9]+)[^0-9]*", item_name)
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
    if measurement_mca_group_pattern.match(item_name):
        return None
    data_column_match = measurement_data_pattern.match(item_name)
    if not data_column_match:
        return None
    return data_column_match.group(1)


def _mca_analyser_in_scan(sf, scan_key, mca_analyser_index):
    """
    :param sf: :class:`SpecFile` instance
    :param scan_key: Scan identification key (e.g. ``1.1``)
    :param mca_analyser_index: 0-based index of MCA analyser
    :return: ``True`` if MCA analyser exists in Scan, else ``False``
    :raise: ``KeyError`` if scan_key not found in SpecFile
    :raise: ``AssertionError`` if number of MCA spectra is not a multiple
          of the number of data lines
    """
    if scan_key not in sf:
        raise KeyError("Scan key %s " % scan_key +
                       "does not exist in SpecFile %s" % sf.filename)

    number_of_MCA_spectra = len(sf[scan_key].mca)
    # Scan.data is transposed
    number_of_data_lines = sf[scan_key].data.shape[1]

    # Number of MCA spectra must be a multiple of number of data lines
    assert number_of_MCA_spectra % number_of_data_lines == 0
    number_of_MCA_analysers = number_of_MCA_spectra // number_of_data_lines

    return 0 <= mca_analyser_index < number_of_MCA_analysers


def _motor_in_scan(sf, scan_key, motor_name):
    """
    :param sf: :class:`SpecFile` instance
    :param scan_key: Scan identification key (e.g. ``1.1``)
    :param motor_name: Name of motor as defined in file header lines
    :return: ``True`` if motor exists in scan, else ``False``
    :raise: ``KeyError`` if scan_key not found in SpecFile
    """
    if scan_key not in sf:
        raise KeyError("Scan key %s " % scan_key +
                       "does not exist in SpecFile %s" % sf.filename)
    return motor_name in sf[scan_key].motor_names


def _column_label_in_scan(sf, scan_key, column_label):
    """
    :param sf: :class:`SpecFile` instance
    :param scan_key: Scan identification key (e.g. ``1.1``)
    :param column_label: Column label as defined in scan header
    :return: ``True`` if data column label exists in scan, else ``False``
    :raise: ``KeyError`` if scan_key not found in SpecFile
    """
    if scan_key not in sf:
        raise KeyError("Scan key %s " % scan_key +
                       "does not exist in SpecFile %s" % sf.filename)
    return column_label in sf[scan_key].labels


def _parse_ctime(ctime_line):
    """
    :param ctime_line: e.g ``@CTIME %f %f %f``, first word ``@CTIME`` optional
    :return: (preset_time, live_time, elapsed_time)
    """
    ctime_line = ctime_line.lstrip("@CTIME ")
    if not len(ctime_line.split()) == 3:
        raise ValueError("Incorrect format for @CTIME header line " +
                         '(expected "@CTIME %f %f %f").')
    return map(float, ctime_line.split())


def spec_date_to_iso8601(date, zone=None):
    """Convert SpecFile date to Iso8601.

    :param date: Date (see supported formats below)
    :type date: str
    :param zone: Time zone as it appears in a ISO8601 date

    Supported formats:

    * ``DDD MMM dd hh:mm:ss YYYY``
    * ``DDD YYYY/MM/dd hh:mm:ss YYYY``

    where `DDD` is the abbreviated weekday, `MMM` is the month abbreviated
    name, `MM` is the month number (zero padded), `dd` is the weekday number
    (zero padded) `YYYY` is the year, `hh` the hour (zero padded), `mm` the
    minute (zero padded) and `ss` the second (zero padded).
    All names are expected to be in english.

    Examples::

        >>> spec_date_to_iso8601("Thu Feb 11 09:54:35 2016")
        '2016-02-11T09:54:35'

        >>> spec_date_to_iso8601("Sat 2015/03/14 03:53:50")
        '2015-03-14T03:53:50'
    """
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul',
              'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    days_rx = '(?P<day>' + '|'.join(days) + ')'
    months_rx = '(?P<month>' + '|'.join(months) + ')'
    year_rx = '(?P<year>\d{4})'
    day_nb_rx = '(?P<day_nb>[0-3]\d)'
    month_nb_rx = '(?P<month_nb>[0-1]\d)'
    hh_rx = '(?P<hh>[0-2]\d)'
    mm_rx = '(?P<mm>[0-5]\d)'
    ss_rx = '(?P<ss>[0-5]\d)'
    tz_rx = '(?P<tz>[+-]\d\d:\d\d){0,1}'

    # date formats must have either month_nb (1..12) or month (Jan, Feb, ...)
    re_tpls = ['{days} {months} {day_nb} {hh}:{mm}:{ss}{tz} {year}',
               '{days} {year}/{month_nb}/{day_nb} {hh}:{mm}:{ss}{tz}']

    grp_d = None

    for rx in re_tpls:
        full_rx = rx.format(days=days_rx,
                            months=months_rx,
                            year=year_rx,
                            day_nb=day_nb_rx,
                            month_nb=month_nb_rx,
                            hh=hh_rx,
                            mm=mm_rx,
                            ss=ss_rx,
                            tz=tz_rx)
        m = re.match(full_rx, date)

        if m:
            grp_d = m.groupdict()
            break

    if not grp_d:
        raise ValueError('Date format not recognized : {0}'.format(date))

    year = grp_d['year']

    month = grp_d.get('month_nb')

    if not month:
        month = '{0:02d}'.format(months.index(grp_d.get('month')) + 1)

    day = grp_d['day_nb']

    tz = grp_d['tz']
    if not tz:
        tz = zone

    time = '{0}:{1}:{2}'.format(grp_d['hh'],
                                grp_d['mm'],
                                grp_d['ss'])

    full_date = '{0}-{1}-{2}T{3}{4}'.format(year,
                                            month,
                                            day,
                                            time,
                                            tz if tz else '')
    return full_date


def _fixed_length_strings(strings, length=0):
    """Return list of fixed length strings, left-justified and right-padded
    with spaces.

    :param strings: List of variable length strings
    :param length: Length of strings in returned list, defaults to the maximum
         length in the original list if set to 0.
    :type length: int or None
    """
    if length == 0 and strings:
        length = max(len(s) for s in strings)
    return [s.ljust(length) for s in strings]


class SpecH5Dataset(numpy.ndarray):
    """Emulate :class:`h5py.Dataset` for a SpecFile object

    :param array_like: Input dataset in a type that can be digested by
        ``numpy.array()`` (`str`, `list`, `numpy.ndarray`…)
    :param name: Dataset full name (posix path format, starting with ``/``)
    :type name: str
    :param file_: Parent :class:`SpecH5`
    :param parent: Parent :class:`SpecH5Group` which contains this dataset

    This class inherits from :class:`numpy.ndarray` and adds ``name`` and
    ``value`` attributes for HDF5 compatibility. ``value`` is a reference
    to the class instance (``value = self``).

    Data is stored in float32 format, unless it is a string.
    """
    # For documentation on subclassing numpy.ndarray,
    # see http://docs.scipy.org/doc/numpy-1.10.1/user/basics.subclassing.html
    def __new__(cls, array_like, name, file_, parent):
        # unicode can't be stored in hdf5, we need to use bytes
        if isinstance(array_like, string_types):
            array_like = numpy.string_(array_like)

        # Ensure our data is a numpy.ndarray
        if not isinstance(array_like, numpy.ndarray):
            array = numpy.array(array_like)
        else:
            array = array_like

        data_kind = array.dtype.kind
        # unicode: convert to byte strings
        # (http://docs.h5py.org/en/latest/strings.html)
        if data_kind in ["S", "U"]:
            obj = numpy.asarray(array, dtype=numpy.string_).view(cls)
        # enforce float32 for int, unsigned int, float
        elif data_kind in ["i", "u", "f"]:
            obj = numpy.asarray(array, dtype=numpy.float32).view(cls)
        # reject boolean (b), complex (c), object (O), void/data block (V)
        else:
            raise TypeError("Unexpected data type " + data_kind +
                            " (expected int-, string- or float-like data)")

        obj.name = name
        obj.value = obj
        obj.parent = parent
        obj.file = file_

        obj.attrs = _get_attrs_dict(name)

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.name = getattr(obj, 'name', None)
        self.value = getattr(obj, 'value', None)
        self.parent = getattr(obj, 'parent', None)
        self.file = getattr(obj, 'file', None)
        self.attrs = getattr(obj, 'attrs', None)


class SpecH5LinkToDataset(SpecH5Dataset):
    """Special :class:`SpecH5Dataset` representing a link to a dataset. It
    works exactly like a regular dataset, but :meth:`SpecH5Group.visit`
    and :meth:`SpecH5Group.visititems` methods will recognize that it is
    a link and will ignore it.
    """
    pass


def _dataset_builder(name, specfileh5, parent_group):
    """Retrieve dataset from :class:`SpecFile`, based on dataset name, as a
    subclass of :class:`numpy.ndarray`.

    :param name: Datatset full name (posix path format, starting with ``/``)
    :type name: str
    :param specfileh5: parent :class:`SpecH5` object
    :type specfileh5: :class:`SpecH5`
    :param parent_group: Parent :class:`SpecH5Group`

    :return: Array with the requested data
    :rtype: :class:`SpecH5Dataset`.
    """
    scan_key = _get_scan_key_in_name(name)
    scan = specfileh5._sf[scan_key]

    # get dataset in an array-like format (ndarray, str, list…)
    array_like = None

    if title_pattern.match(name):
        array_like = scan.scan_header_dict["S"]

    elif start_time_pattern.match(name):
        if "D" in scan.scan_header_dict:
            array_like = spec_date_to_iso8601(scan.scan_header_dict["D"])
        elif "D" in scan.file_header_dict:
            logger1.warn("No #D line in scan header. " +
                         "Using file header for start_time.")
            array_like = spec_date_to_iso8601(scan.file_header["D"])
        else:
            logger1.warn("No #D line in header. " +
                         "Using current system time for start_time.")
            array_like = time.ctime(time.time())

    elif file_header_data_pattern.match(name):
        array_like = _fixed_length_strings(scan.file_header)

    elif scan_header_data_pattern.match(name):
        array_like = _fixed_length_strings(scan.scan_header)

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

    elif instrument_mca_data_pattern.match(name):
        m = instrument_mca_data_pattern.match(name)

        analyser_index = int(m.group(1))
        # retrieve 2D array of all MCA spectra from one analyser
        array_like = _demultiplex_mca(scan, analyser_index)

    elif instrument_mca_calib_pattern.match(name):
        array_like = scan.mca.calibration

    elif instrument_mca_chann_pattern.match(name):
        array_like = scan.mca.channels

    elif "CTIME" in scan.mca_header_dict:
        ctime_line = scan.mca_header_dict['CTIME']
        (preset_time, live_time, elapsed_time) = _parse_ctime(ctime_line)
        if instrument_mca_preset_t_pattern.match(name):
            array_like = preset_time
        elif instrument_mca_live_t_pattern.match(name):
            array_like = live_time
        elif instrument_mca_elapsed_t_pattern.match(name):
            array_like = elapsed_time

    if array_like is None:
        raise KeyError("Name " + name + " does not match any known dataset.")

    return SpecH5Dataset(array_like, name,
                         file_=specfileh5, parent=parent_group)


def _link_to_dataset_builder(name, specfileh5, parent_group):
    """Same as :func:`_dataset_builder`, but returns a
    :class:`SpecH5LinkToDataset`

    :param name: Datatset full name (posix path format, starting with ``/``)
    :type name: str
    :param specfileh5: parent :class:`SpecH5` object
    :type specfileh5: :class:`SpecH5`
    :param parent_group: Parent :class:`SpecH5Group`

    :return: Array with the requested data
    :rtype: :class:`SpecH5LinkToDataset`.
    """
    scan_key = _get_scan_key_in_name(name)
    scan = specfileh5._sf[scan_key]

    # get dataset in an array-like format (ndarray, str, list…)
    array_like = None

    if measurement_mca_data_pattern.match(name):
        m = measurement_mca_data_pattern.match(name)
        analyser_index = int(m.group(1))
        array_like = _demultiplex_mca(scan, analyser_index)

    elif measurement_mca_info_dataset_pattern:
        m = measurement_mca_info_dataset_pattern.match(name)

        mca_hdr_type = m.group(1)
        if mca_hdr_type == "calibration":
            array_like = scan.mca.calibration
        elif mca_hdr_type == "channels":
            array_like = scan.mca.channels
        elif "CTIME" in scan.mca_header_dict:
            ctime_line = scan.mca_header_dict['CTIME']
            (preset_time, live_time, elapsed_time) = _parse_ctime(ctime_line)
            if instrument_mca_preset_t_pattern.match(name):
                array_like = preset_time
            elif instrument_mca_live_t_pattern.match(name):
                array_like = live_time
            elif instrument_mca_elapsed_t_pattern.match(name):
                array_like = elapsed_time

    if array_like is None:
        raise KeyError("Name " + name + " does not match any known dataset.")

    return SpecH5LinkToDataset(array_like, name,
                               file_=specfileh5, parent=parent_group)


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
    number_of_scan_data_lines = scan.data.shape[1]

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


class SpecH5Group(object):
    """Emulate :class:`h5py.Group` for a SpecFile object

    :param name: Group full name (posix path format, starting with ``/``)
    :type name: str
    :param specfileh5: parent :class:`SpecH5` instance

    """
    def __init__(self, name, specfileh5):
        self.name = name
        """Full name/path of group"""

        self.file = specfileh5
        """Parent SpecH5 object"""

        self.attrs = _get_attrs_dict(name)
        """Attributes dictionary"""

        if name != "/":
            scan_key = _get_scan_key_in_name(name)
            self._scan = self.file._sf[scan_key]

    @property
    def parent(self):
        """Parent group (group that contains this group)"""
        if not self.name.strip("/"):
            return None

        parent_name = posixpath.dirname(self.name.rstrip("/"))
        return SpecH5Group(parent_name, self.file)

    def __contains__(self, key):
        """
        :param key: Path to child element (e.g. ``"mca_0/info"``) or full name
            of group or dataset (e.g. ``"/2.1/instrument/positioners"``)
        :return: True if key refers to a valid member of this group,
            else False
        """
        # Absolute path to an item outside this group
        if key.startswith("/"):
            if not key.startswith(self.name):
                return False
        # Make sure key is an absolute path by prepending this group's name
        else:
            key = self.name.rstrip("/") + "/" + key

        # key not matching any known pattern
        if not is_group(key) and not is_dataset(key) and\
           not is_link_to_group(key) and not is_link_to_dataset(key):
            return False

        # nonexistent scan in specfile
        scan_key = _get_scan_key_in_name(key)
        if scan_key not in self.file._sf:
            return False

        # nonexistent MCA analyser in scan
        mca_analyser_index = _get_mca_index_in_name(key)
        if mca_analyser_index is not None:
            if not _mca_analyser_in_scan(self.file._sf,
                                         scan_key,
                                         mca_analyser_index):
                return False

        # nonexistent motor name
        motor_name = _get_motor_in_name(key)
        if motor_name is not None:
            if not _motor_in_scan(self.file._sf,
                                  scan_key,
                                  motor_name):
                return False

        # nonexistent data column
        column_label = _get_data_column_label_in_name(key)
        if column_label is not None:
            if not _column_label_in_scan(self.file._sf,
                                         scan_key,
                                         column_label):
                return False

        if key.endswith("preset_time") or\
           key.endswith("elapsed_time") or\
           key.endswith("live_time"):
            return "CTIME" in self.file._sf[scan_key].mca_header_dict

        # header, title, start_time, existing scan/mca/motor/measurement
        return True

    def __eq__(self, other):
        return (isinstance(other, SpecH5Group) and
                self.name == other.name and
                self.file.filename == other.file.filename and
                self.keys() == other.keys())

    def __getitem__(self, key):
        """Return a :class:`SpecH5Group` or a :class:`SpecH5Dataset`
        if ``key`` is a valid name of a group or dataset.

        ``key`` can be a member of ``self.keys()``, i.e. an immediate child of
        the group, or a path reaching into subgroups (e.g.
        ``"instrument/positioners"``)

        In the special case were this group is the root group, ``key`` can
        start with a ``/`` character.

        :param key: Name of member
        :type key: str
        :raise: KeyError if ``key`` is not a known member of this group.
        """
        # Relative path starting from this group (e.g "mca_0/info")
        if not key.startswith("/"):
            full_key = self.name.rstrip("/") + "/" + key
        # Absolute path called from the root group or from a parent group
        elif key.startswith(self.name):
            full_key = key
        # Absolute path to an element called from a non-parent group
        else:
            raise KeyError(key + " is not a child of " + self.__repr__())

        if is_group(full_key):
            return SpecH5Group(full_key, self.file)
        elif is_dataset(full_key):
            return _dataset_builder(full_key, self.file, self)
        elif is_link_to_group(full_key):
            return SpecH5LinkToGroup(full_key, self.file)
        elif is_link_to_dataset(full_key):
            return _link_to_dataset_builder(full_key, self.file, self)
        else:
            raise KeyError("unrecognized group or dataset: " + full_key)

    def __iter__(self):
        for key in self.keys():
            yield key

    def __len__(self):
        """Return number of members,subgroups and datasets, attached to this
         group.
         """
        return len(self.keys())

    def __repr__(self):
        return '<SpecH5Group "%s" (%d members)>' % (self.name, len(self))

    def keys(self):
        """:return: List of all names of members attached to this group
        """
        # keys in hdf5 are unicode
        if self.name == "/":
            return self.file.keys()

        if scan_pattern.match(self.name):
            return static_items["scan"]

        if positioners_group_pattern.match(self.name):
            return self._scan.motor_names

        if specfile_group_pattern.match(self.name):
            return static_items["scan/instrument/specfile"]

        if measurement_mca_group_pattern.match(self.name):
            return static_items["scan/measurement/mca"]

        if instrument_mca_group_pattern.match(self.name):
            ret = static_items["scan/instrument/mca"]
            if "CTIME" in self._scan.mca_header_dict:
                ret += ["preset_time", "elapsed_time", "live_time"]
            return ret

        # number of data columns must be equal to number of labels
        assert self._scan.data.shape[0] == len(self._scan.labels)

        number_of_MCA_spectra = len(self._scan.mca)
        number_of_data_lines = self._scan.data.shape[1]

        # Number of MCA spectra must be a multiple of number of data lines
        assert number_of_MCA_spectra % number_of_data_lines == 0
        number_of_MCA_analysers = number_of_MCA_spectra // number_of_data_lines
        mca_list = ["mca_%d" % i for i in range(number_of_MCA_analysers)]

        if measurement_group_pattern.match(self.name):
            return self._scan.labels + mca_list

        if instrument_pattern.match(self.name):
            return static_items["scan/instrument"] + mca_list

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
            member = self[member_name]
            ret = None
            if not is_link_to_dataset(member.name) and\
               not is_link_to_group(member.name):
                ret = func(member.name)
            if ret is not None:
                return ret
            # recurse into subgroups
            if isinstance(self[member_name], SpecH5Group) and\
               not isinstance(self[member_name], SpecH5LinkToGroup):
                self[member_name].visit(func)

    def visititems(self, func):
        """Recursively visit names and objects in this group.

        :param func: Callable (function, method or callable object)
        :type func: function

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
                if isinstance(obj, SpecH5Dataset):
                    mylist.append(name)

            f = File('foo.dat')
            f["1.1"].visititems(func)
        """
        for member_name in self.keys():
            member = self[member_name]
            ret = None
            if not is_link_to_dataset(member.name):
                ret = func(member.name, member)
            if ret is not None:
                return ret
            # recurse into subgroups
            if isinstance(self[member_name], SpecH5Group) and\
               not isinstance(self[member_name], SpecH5LinkToGroup):
                self[member_name].visititems(func)


class SpecH5LinkToGroup(SpecH5Group):
    """Special :class:`SpecH5Group` representing a link to a group.

    It works exactly like a regular group but :meth:`SpecH5Group.visit`
    and :meth:`SpecH5Group.visititems` methods will recognize it as a
    link and will ignore it.
    """
    def keys(self):
        """:return: List of all names of members attached to the target group
        """
        # we only have a single type of link to a group:
        # /1.1/measurement/mca_0/info/ -> /1.1/instrument/mca_0/
        if measurement_mca_info_pattern.match(self.name):
            link_target = self.name.replace("measurement", "instrument").rstrip("/")[:-4]
            return SpecH5Group(link_target, self.file).keys()


class SpecH5(SpecH5Group):
    """Special :class:`SpecH5Group` representing the root of a SpecFile.

    :param filename: Path to SpecFile in filesystem
    :type filename: str

    In addition to all generic :class:`SpecH5Group` attributes, this class
    also keeps a reference to the original :class:`SpecFile` object and
    has a :attr:`filename` attribute.

    Its immediate children are scans, but it also gives access to any group
    or dataset in the entire SpecFile tree by specifying the full path.
    """
    def __init__(self, filename):
        self.filename = filename
        self.attrs = _get_attrs_dict("/")
        self._sf = SpecFile(filename)

        SpecH5Group.__init__(self, name="/", specfileh5=self)

    def keys(self):
        """
        :return: List of all scan keys in this SpecFile
            (e.g. ``["1.1", "2.1"…]``)
        """
        return self._sf.keys()

    def __repr__(self):
        return '<SpecH5 "%s" (%d members)>' % (self.filename, len(self))

    def __eq__(self, other):
        return (isinstance(other, SpecH5) and
                self.filename == other.filename and
                self.keys() == other.keys())

