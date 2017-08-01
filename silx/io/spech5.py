# coding: utf-8
# /*##########################################################################
# Copyright (C) 2016-2017 European Synchrotron Radiation Facility
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
                  file_header = "…"
                  scan_header = "…"
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

``file_header`` and ``scan_header`` are the raw headers as they
appear in the original file, as a string of lines separated by newline (``\\n``) characters.

The title is the content of the ``#S`` scan header line without the leading
``#S`` (e.g ``"1  ascan  ss1vo -4.55687 -0.556875  40 0.2"``).

The start time is converted to ISO8601 format (``"2016-02-23T22:49:05Z"``),
if the original date format is standard.

Numeric datasets are stored in *float32* format, except for scalar integers
which are stored as *int64*.

Motor positions (e.g. ``/1.1/instrument/positioners/motor_name``) can be
1D numpy arrays if they are measured as scan data, or else scalars as defined
on ``#P`` scan header lines. A simple test is done to check if the motor name
is also a data column header defined in the ``#L`` scan header line.

Scan data  (e.g. ``/1.1/measurement/colname0``) is accessed by column,
the dataset name ``colname0`` being the column label as defined in the ``#L``
scan header line.

If a ``/`` character is present in a column label or in a motor name in the
original SPEC file, it will be substituted with a ``%`` character in the
corresponding dataset name.

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

    # alternative: full path access
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

Strings are stored encoded as ``numpy.string_``, as recommended by
`the h5py documentation <http://docs.h5py.org/en/latest/strings.html>`_.
This ensures maximum compatibility with third party software libraries,
when saving a :class:`SpecH5` to a HDF5 file using :mod:`silx.io.spectoh5`.

The type ``numpy.string_`` is a byte-string format. The consequence of this
is that you should decode strings before using them in **Python 3**::

    >>> from silx.io.spech5 import SpecH5
    >>> sfh5 = SpecH5("31oct98.dat")
    >>> sfh5["/68.1/title"]
    b'68  ascan  tx3 -28.5 -24.5  20 0.5'
    >>> sfh5["/68.1/title"].decode()
    '68  ascan  tx3 -28.5 -24.5  20 0.5'


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
import io

from silx.third_party import six
from .specfile import SpecFile

__authors__ = ["P. Knobel", "D. Naudet"]
__license__ = "MIT"
__date__ = "01/08/2017"

logger1 = logging.getLogger(__name__)

try:
    import h5py
except ImportError:
    h5py = None
    logger1.debug("Module h5py optional.", exc_info=True)


string_types = (basestring,) if sys.version_info[0] == 2 else (str,)  # noqa

# Static subitems: all groups and datasets that are present in any
# scan (excludes list of scans, data columns, list of mca devices,
# optional mca headers, optional sample group)
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
sample_pattern = re.compile(r"/[0-9]+\.[0-9]+/sample/?$")
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
instrument_mca_calib_pattern = re.compile(r"/[0-9]+\.[0-9]+/instrument/mca_([0-9]+)/calibration$")
instrument_mca_chann_pattern = re.compile(r"/[0-9]+\.[0-9]+/instrument/mca_([0-9])+/channels$")
instrument_mca_preset_t_pattern = re.compile(r"/[0-9]+\.[0-9]+/instrument/mca_[0-9]+/preset_time$")
instrument_mca_elapsed_t_pattern = re.compile(r"/[0-9]+\.[0-9]+/instrument/mca_[0-9]+/elapsed_time$")
instrument_mca_live_t_pattern = re.compile(r"/[0-9]+\.[0-9]+/instrument/mca_[0-9]+/live_time$")
ub_matrix_pattern = re.compile(r"/[0-9]+\.[0-9]+/sample/ub_matrix$")
unit_cell_pattern = re.compile(r"/[0-9]+\.[0-9]+/sample/unit_cell$")
unit_cell_abc_pattern = re.compile(r"/[0-9]+\.[0-9]+/sample/unit_cell_abc$")
unit_cell_alphabetagamma_pattern = re.compile(r"/[0-9]+\.[0-9]+/sample/unit_cell_alphabetagamma$")

# Links to dataset
measurement_mca_data_pattern = re.compile(r"/[0-9]+\.[0-9]+/measurement/mca_([0-9]+)/data$")
# info/ + calibration, channel, preset_time, live_time, elapsed_time (not data)
measurement_mca_info_dataset_pattern = re.compile(r"/[0-9]+\.[0-9]+/measurement/mca_([0-9]+)/info/([^d/][^/]+)$")
# info/data
measurement_mca_info_data_pattern = re.compile(r"/[0-9]+\.[0-9]+/measurement/mca_([0-9]+)/info/data$")


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
          because this key would point to a motor position, which is a
          dataset and not a group.
    """
    group_patterns = (
        root_pattern, scan_pattern, instrument_pattern,
        specfile_group_pattern, positioners_group_pattern,
        measurement_group_pattern, measurement_mca_group_pattern,
        instrument_mca_group_pattern, sample_pattern
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
    # Check groups first because /1.1/measurement/mca_0 could be interpreted
    # as a data column with label "mca_0"
    if measurement_mca_group_pattern.match(name):
        return False

    data_patterns = (
        header_pattern, title_pattern, start_time_pattern,
        file_header_data_pattern, scan_header_data_pattern,
        positioners_data_pattern, measurement_data_pattern,
        instrument_mca_data_pattern, instrument_mca_calib_pattern,
        instrument_mca_chann_pattern,
        instrument_mca_preset_t_pattern, instrument_mca_elapsed_t_pattern,
        instrument_mca_live_t_pattern,
        ub_matrix_pattern, unit_cell_pattern, unit_cell_abc_pattern, unit_cell_alphabetagamma_pattern
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
        measurement_mca_info_data_pattern
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
            {"NX_class": "NXdetector", },
        measurement_mca_info_dataset_pattern:
            {},
        measurement_mca_info_data_pattern:
            {"interpretation": "spectrum"},
        sample_pattern:
            {"NX_class": "NXsample", },
        ub_matrix_pattern:
            {"interpretation": "scalar"},
        unit_cell_pattern:
            {"interpretation": "scalar"},
        unit_cell_abc_pattern:
            {"interpretation": "scalar"},
        unit_cell_alphabetagamma_pattern:
            {"interpretation": "scalar"},
    }

    for pattern in pattern_attrs:
        if pattern.match(name):
            return pattern_attrs[pattern]

    logger1.warning("%s not a known pattern, assigning empty dict to attrs",
                    name)
    return {}


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


def _get_number_of_mca_analysers(scan):
    """
    :param SpecFile sf: :class:`SpecFile` instance
    :param str scan_key: Scan identification key (e.g. ``1.1``)
    """
    number_of_MCA_spectra = len(scan.mca)
    # Scan.data is transposed
    number_of_data_lines = scan.data.shape[1]

    if not number_of_data_lines == 0:
        # Number of MCA spectra must be a multiple of number of data lines
        assert number_of_MCA_spectra % number_of_data_lines == 0
        return number_of_MCA_spectra // number_of_data_lines
    elif number_of_MCA_spectra:
        # Case of a scan without data lines, only MCA.
        # Our only option is to assume that the number of analysers
        # is the number of #@CHANN lines
        return len(scan.mca.channels)
    else:
        return 0


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

    number_of_analysers = _get_number_of_mca_analysers(sf[scan_key])

    return 0 <= mca_analyser_index < number_of_analysers


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
    ret = motor_name in sf[scan_key].motor_names
    if not ret and "%" in motor_name:
        motor_name = motor_name.replace("%", "/")
        ret = motor_name in sf[scan_key].motor_names
    return ret


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
    ret = column_label in sf[scan_key].labels
    if not ret and "%" in column_label:
        column_label = column_label.replace("%", "/")
        ret = column_label in sf[scan_key].labels
    return ret


def _parse_UB_matrix(header_line):
    """Parse G3 header line and return UB matrix

    :param str header_line: G3 header line
    :return: UB matrix
    """
    return numpy.array(list(map(float, header_line.split()))).reshape((1, 3, 3))


def _ub_matrix_in_scan(scan):
    """Return True if scan header has a G3 line and all values are not 0.

    :param scan: specfile.Scan instance
    :return: True or False
    """
    if "G3" not in scan.scan_header_dict:
        return False
    return numpy.any(_parse_UB_matrix(scan.scan_header_dict["G3"]))


def _parse_unit_cell(header_line):
    return numpy.array(list(map(float, header_line.split()))[0:6]).reshape((1, 6))


def _unit_cell_in_scan(scan):
    """Return True if scan header has a G1 line and all values are not 0.

    :param scan: specfile.Scan instance
    :return: True or False
    """
    if "G1" not in scan.scan_header_dict:
        return False
    return numpy.any(_parse_unit_cell(scan.scan_header_dict["G1"]))


def _parse_ctime(ctime_lines, analyser_index=0):
    """
    :param ctime_lines: e.g ``@CTIME %f %f %f``, first word ``@CTIME`` optional
        When multiple CTIME lines are present in a scan header, this argument
        is a concatenation of them separated by a ``\n`` character.
    :param analyser_index: MCA device/analyser index, when multiple devices
        are in a scan.
    :return: (preset_time, live_time, elapsed_time)
    """
    ctime_lines = ctime_lines.lstrip("@CTIME ")
    ctimes_lines_list = ctime_lines.split("\n")
    if len(ctimes_lines_list) == 1:
        # single @CTIME line for all devices
        ctime_line = ctimes_lines_list[0]
    else:
        ctime_line = ctimes_lines_list[analyser_index]
    if not len(ctime_line.split()) == 3:
        raise ValueError("Incorrect format for @CTIME header line " +
                         '(expected "@CTIME %f %f %f").')
    return list(map(float, ctime_line.split()))


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
    day_nb_rx = '(?P<day_nb>[0-3 ]\d)'
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


class SpecH5Dataset(object):
    """Emulate :class:`h5py.Dataset` for a SpecFile object.

    A :class:`SpecH5Dataset` instance is basically  a proxy for the numpy
    array :attr:`value` attribute, with additional attributes for
    compatibility  with *h5py* datasets.

    :param value: Actual dataset value
    :param name: Dataset full name (posix path format, starting with ``/``)
    :type name: str
    :param file_: Parent :class:`SpecH5`
    :param parent: Parent :class:`SpecH5Group` which contains this dataset
    """
    def __init__(self, value, name, file_, parent):
        object.__init__(self)

        self.value = None
        """Actual dataset, can be a *numpy array*, a *numpy.string_*,
        a *numpy.int_* or a *numpy.float32*

        All operations applied to an instance of the class use this."""

        # get proper value types, to inherit from numpy
        # attributes (dtype, shape, size)
        if isinstance(value, string_types):
            # use bytes for maximum compatibility
            # (see http://docs.h5py.org/en/latest/strings.html)
            self.value = numpy.string_(value)
        elif isinstance(value, float):
            # use 32 bits for float scalars
            self.value = numpy.float32(value)
        elif isinstance(value, int):
            self.value = numpy.int_(value)
        else:
            # Enforce numpy array
            array = numpy.array(value)
            data_kind = array.dtype.kind

            if data_kind in ["S", "U"]:
                self.value = numpy.asarray(array, dtype=numpy.string_)
            elif data_kind in ["f"]:
                self.value = numpy.asarray(array, dtype=numpy.float32)
            else:
                self.value = array

        # numpy array attributes (more attributes handled in __getattribute__)
        self.shape = self.value.shape
        """Dataset shape, as a tuple with the length of each dimension
        of the dataset."""

        self.dtype = self.value.dtype
        """Dataset dtype"""

        self.size = self.value.size
        """Dataset size (number of elements)"""

        # h5py dataset specific attributes
        self.name = name
        """"Dataset name (posix path format, starting with ``/``)"""

        self.parent = parent
        """Parent :class:`SpecH5Group` object which contains this dataset"""

        self.file = file_
        """Parent :class:`SpecH5` object"""

        self.attrs = _get_attrs_dict(name)
        """Attributes dictionary"""

        self.compression = None
        """Compression attribute as provided by h5py.Dataset"""

        self.compression_opts = None
        """Compression options attribute as provided by h5py.Dataset"""

        self.chunks = None

    @property
    def h5py_class(self):
        """Return h5py class which is mimicked by this class:
        :class:`h5py.dataset`.

        Accessing this attribute if :mod:`h5py` is not installed causes
        an ``ImportError`` to be raised
        """
        if h5py is None:
            raise ImportError("Cannot return h5py.Dataset class, " +
                              "unable to import h5py module")
        return h5py.Dataset

    def __getattribute__(self, item):
        if item in ["value", "name", "parent", "file", "attrs",
                    "shape", "dtype", "size", "h5py_class",
                    "chunks", "compression", "compression_opts",
                    "target"]:
            return object.__getattribute__(self, item)

        if hasattr(self.value, item):
            return getattr(self.value, item)

        raise AttributeError("SpecH5Dataset has no attribute %s" % item)

    def __len__(self):
        return len(self.value)

    def __getitem__(self, item):
        if not isinstance(self.value, numpy.ndarray):
            if item == Ellipsis:
                return numpy.array(self.value)
            elif item == tuple():
                return self.value
            else:
                raise ValueError("Scalar can only be reached with an ellipsis or an empty tuple")
        return self.value.__getitem__(item)

    def __getslice__(self, i, j):
        # deprecated but still in use for python 2.7
        return self.__getitem__(slice(i, j, None))

    def __iter__(self):
        return self.value.__iter__()

    def __dir__(self):
        attrs = set(dir(self.value) +
                    ["value", "name", "parent", "file",
                     "attrs", "shape", "dtype", "size",
                     "h5py_class", "chunks", "compression",
                     "compression_opts"])
        return sorted(attrs)

    # casting
    def __repr__(self):
        return '<SpecH5Dataset "%s": shape %s, type "%s">' % \
               (self.name, self.shape, self.dtype.str)

    def __float__(self):
        return float(self.value)

    def __int__(self):
        return int(self.value)

    def __str__(self):
        basename = self.name.split("/")[-1]
        return '<SPEC dataset "%s": shape %s, type "%s">' % \
               (basename, self.shape, self.dtype.str)

    def __bool__(self):
        if self.value:
            return True
        return False

    def __nonzero__(self):
        # python 2
        return self.__bool__()

    def __array__(self, dtype=None):
        if dtype is None:
            return numpy.array(self.value)
        else:
            return numpy.array(self.value, dtype=dtype)

    # comparisons
    def __eq__(self, other):
        if hasattr(other, "value"):
            return self.value == other.value
        else:
            return self.value == other

    def __ne__(self, other):
        if hasattr(other, "value"):
            return self.value != other.value
        else:
            return self.value != other

    def __lt__(self, other):
        if hasattr(other, "value"):
            return self.value < other.value
        else:
            return self.value < other

    def __le__(self, other):
        if hasattr(other, "value"):
            return self.value <= other.value
        else:
            return self.value <= other

    def __gt__(self, other):
        if hasattr(other, "value"):
            return self.value > other.value
        else:
            return self.value > other

    def __ge__(self, other):
        if hasattr(other, "value"):
            return self.value >= other.value
        else:
            return self.value >= other

    # operations
    def __add__(self, other):
        return self.value + other

    def __radd__(self, other):
        return other + self.value

    def __sub__(self, other):
        return self.value - other

    def __rsub__(self, other):
        return other - self.value

    def __mul__(self, other):
        return self.value * other

    def __rmul__(self, other):
        return other * self.value

    def __truediv__(self, other):
        return self.value / other

    def __rtruediv__(self, other):
        return other / self.value

    def __floordiv__(self, other):
        return self.value // other

    def __rfloordiv__(self, other):
        return other // self.value

    # unary operations
    def __neg__(self):
        return -self.value

    def __abs__(self):
        return abs(self.value)


class SpecH5LinkToDataset(SpecH5Dataset):
    """Special :class:`SpecH5Dataset` representing a link to a dataset. It
    works like a regular dataset, but :meth:`SpecH5Group.visit`
    and :meth:`SpecH5Group.visititems` methods will recognize that it is
    a link and will ignore it.

    A special attribute contains the name of the target dataset:
    :attr:`target`
    """
    def __init__(self, value, name, file_, parent, target):
        """
        :param value: Actual dataset value
        :param name: Dataset full name (posix path format, starting with ``/``)
        :type name: str
        :param file_: Parent :class:`SpecH5`
        :param parent: Parent :class:`SpecH5Group` which contains this dataset
        :param str target: Name of linked dataset
        """
        SpecH5Dataset.__init__(self, value, name, file_, parent)
        self.target = target
        """Name of the target dataset"""


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
            try:
                array_like = spec_date_to_iso8601(scan.scan_header_dict["D"])
            except (IndexError, ValueError):
                logger1.warn("Could not parse date format in scan %s header." +
                             " Using original date not converted to ISO-8601",
                             scan_key)
                array_like = scan.scan_header_dict["D"]
        elif "D" in scan.file_header_dict:
            logger1.warn("No #D line in scan %s header. " +
                         "Using file header for start_time.",
                         scan_key)
            try:
                array_like = spec_date_to_iso8601(scan.file_header_dict["D"])
            except (IndexError, ValueError):
                logger1.warn("Could not parse date format in scan %s header. " +
                             "Using original date not converted to ISO-8601",
                             scan_key)
                array_like = scan.file_header_dict["D"]
        else:
            logger1.warn("No #D line in %s header. Setting date to empty string.",
                         scan_key)
            array_like = ""

    elif file_header_data_pattern.match(name):
        # array_like = _fixed_length_strings(scan.file_header)
        array_like = "\n".join(scan.file_header)

    elif scan_header_data_pattern.match(name):
        # array_like = _fixed_length_strings(scan.scan_header)
        array_like = "\n".join(scan.scan_header)

    elif positioners_data_pattern.match(name):
        m = positioners_data_pattern.match(name)
        motor_name = m.group(1)
        if motor_name not in (scan.labels + scan.motor_names):
            if "%" in motor_name:
                motor_name = motor_name.replace("%", "/")
        # if a motor is recorded as a data column, ignore its position in
        # header and return the data column instead
        if motor_name in scan.labels and scan.data.shape[0] > 0:
            array_like = scan.data_column_by_name(motor_name)
        else:
            # may return float("inf") if #P line is missing from scan hdr
            array_like = scan.motor_position_by_name(motor_name)

    elif measurement_data_pattern.match(name):
        m = measurement_data_pattern.match(name)
        column_name = m.group(1)
        if column_name not in scan.labels:
            if "%" in column_name:
                column_name = column_name.replace("%", "/")
        array_like = scan.data_column_by_name(column_name)

    elif instrument_mca_data_pattern.match(name):
        m = instrument_mca_data_pattern.match(name)

        analyser_index = int(m.group(1))
        # retrieve 2D array of all MCA spectra from one analyser
        array_like = _demultiplex_mca(scan, analyser_index)

    elif instrument_mca_calib_pattern.match(name):
        m = instrument_mca_calib_pattern.match(name)
        analyser_index = int(m.group(1))
        if len(scan.mca.channels) == 1:
            # single @CALIB line applying to multiple devices
            analyser_index = 0
        array_like = scan.mca.calibration[analyser_index]

    elif instrument_mca_chann_pattern.match(name):
        m = instrument_mca_chann_pattern.match(name)
        analyser_index = int(m.group(1))
        if len(scan.mca.channels) == 1:
            # single @CHANN line applying to multiple devices
            analyser_index = 0
        array_like = scan.mca.channels[analyser_index]

    elif ub_matrix_pattern.match(name):
        if not _ub_matrix_in_scan(scan):
            raise KeyError("No UB matrix in a scan without a #G3 header line"
                           " or all zeros.")
        array_like = _parse_UB_matrix(scan.scan_header_dict["G3"])
    elif unit_cell_pattern.match(name):
        if not _unit_cell_in_scan(scan):
            raise KeyError(
                    "No unit_cell matrix in a scan without a #G1 header line")
        array_like = _parse_unit_cell(scan.scan_header_dict["G1"])
    elif unit_cell_abc_pattern.match(name):
        if not _unit_cell_in_scan(scan):
            raise KeyError(
                    "No unit_cell matrix in a scan without a #G1 header line")
        array_like = _parse_unit_cell(scan.scan_header_dict["G1"])[0, 0:3]
    elif unit_cell_alphabetagamma_pattern.match(name):
        if not _unit_cell_in_scan(scan):
            raise KeyError(
                    "No unit_cell matrix in a scan without a #G1 header line")
        array_like = _parse_unit_cell(scan.scan_header_dict["G1"])[0, 3:6]
    elif "CTIME" in scan.mca_header_dict and "mca_" in name:
        m = re.compile(r"/.*/mca_([0-9]+)/.*").match(name)
        analyser_index = int(m.group(1))

        ctime_line = scan.mca_header_dict['CTIME']
        (preset_time, live_time, elapsed_time) = _parse_ctime(ctime_line, analyser_index)
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

    # /1.1/measurement/mca_0/data -> /1.1/instrument/mca_0/data
    if measurement_mca_data_pattern.match(name):
        m = measurement_mca_data_pattern.match(name)
        analyser_index = int(m.group(1))
        array_like = _demultiplex_mca(scan, analyser_index)

    # /1.1/measurement/mca_0/info/X -> /1.1/instrument/mca_0/X
    # X: calibration, channels, preset_time, live_time, elapsed_time
    elif measurement_mca_info_dataset_pattern.match(name):
        m = measurement_mca_info_dataset_pattern.match(name)
        analyser_index = int(m.group(1))
        mca_hdr_type = m.group(2)

        if mca_hdr_type == "calibration":
            if len(scan.mca.calibration) == 1:
                # single @CALIB line for multiple devices
                analyser_index = 0
            array_like = scan.mca.calibration[analyser_index]

        elif mca_hdr_type == "channels":
            if len(scan.mca.channels) == 1:
                # single @CHANN line for multiple devices
                analyser_index = 0
            array_like = scan.mca.channels[analyser_index]

        elif "CTIME" in scan.mca_header_dict:
            ctime_line = scan.mca_header_dict['CTIME']
            (preset_time, live_time, elapsed_time) = _parse_ctime(ctime_line,
                                                                  analyser_index)
            if mca_hdr_type == "preset_time":
                array_like = preset_time
            elif mca_hdr_type == "live_time":
                array_like = live_time
            elif mca_hdr_type == "elapsed_time":
                array_like = elapsed_time

    # /1.1/measurement/mca_0/info/data -> /1.1/instrument/mca_0/data
    elif measurement_mca_info_data_pattern.match(name):
        m = measurement_mca_info_data_pattern.match(name)
        analyser_index = int(m.group(1))
        array_like = _demultiplex_mca(scan, analyser_index)

    if array_like is None:
        raise KeyError("Name " + name + " does not match any known dataset.")

    target = name.replace("measurement", "instrument")
    target = target.replace("info/", "")

    return SpecH5LinkToDataset(array_like, name,
                               file_=specfileh5, parent=parent_group,
                               target=target)


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
    number_of_analysers = _get_number_of_mca_analysers(scan)
    number_of_spectra = len(scan.mca)
    number_of_spectra_per_analyser = number_of_spectra // number_of_analysers
    len_spectrum = len(scan.mca[analyser_index])

    mca_array = numpy.empty((number_of_spectra_per_analyser, len_spectrum))

    for i in range(number_of_spectra_per_analyser):
        mca_array[i, :] = scan.mca[analyser_index + i * number_of_analysers]

    return mca_array


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
            if name not in specfileh5:
                raise KeyError("File %s does not contain group %s" %
                               (specfileh5, name))
            scan_key = _get_scan_key_in_name(name)
            self._scan = self.file._sf[scan_key]

    @property
    def h5py_class(self):
        """Return h5py class which is mimicked by this class:
        :class:`h5py.Group`.

        Accessing this attribute if :mod:`h5py` is not installed causes
        an ``ImportError`` to be raised
        """
        if h5py is None:
            raise ImportError("Cannot return h5py.Group class, " +
                              "unable to import h5py module")
        return h5py.Group

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

        if sample_pattern.match(key):
            return (_ub_matrix_in_scan(self.file._sf[scan_key]) or
                    _unit_cell_in_scan(self.file._sf[scan_key]))

        if key.endswith("sample/ub_matrix"):
            return _ub_matrix_in_scan(self.file._sf[scan_key])

        if (key.endswith("sample/unit_cell") or
            key.endswith("sample/unit_cell_abc") or
            key.endswith("sample/unit_cell_alphabetagamma")):
            return _unit_cell_in_scan(self.file._sf[scan_key])

        # header, title, start_time, existing scan/mca/motor/measurement
        return True

    def __eq__(self, other):
        return (isinstance(other, SpecH5Group) and
                self.name == other.name and
                self.file.filename == other.file.filename and
                self.keys() == other.keys())

    def get(self, name, default=None, getclass=False, getlink=False):
        """Retrieve an item by name, or a default value if name does not
        point to an existing item.

        :param name str: name of the item
        :param default: Default value returned if the name is not found
        :param bool getclass: if *True*, the returned object is the class of
            the item, instead of the item instance.
        :param bool getlink: if *True*, the returned object is based on links
            instead of the hdf5 items.
        :return: The requested item, or its class if *getclass* is *True*,
            or the specified *default* value if the group does not contain
            an item with the requested name.
        """
        if name not in self:
            return default

        full_key = self._get_full_key(name)
        if getlink:
            # TODO It may be interesting to implement soft links here
            if getclass:
                return h5py.HardLink
            else:
                return h5py.HardLink()
        else:
            if getclass:
                # Return the kind of the object instead of loading it
                if is_group(full_key):
                    return h5py.Group
                elif is_dataset(full_key):
                    return h5py.Dataset
                elif is_link_to_group(full_key):
                    return h5py.Group
                elif is_link_to_dataset(full_key):
                    return h5py.Dataset
                raise KeyError("unrecognized group or dataset: " + full_key)
            else:
                return self[name]

    def _get_full_key(self, key):
        """Return the full key if ``key`` is a valid name of a group or
        dataset.

        ``key`` can be a member of ``self.keys()``, i.e. an immediate child of
        the group, or a path reaching into subgroups (e.g.
        ``"instrument/positioners"``)

        In the special case were this group is the root group, ``key`` can
        start with a ``/`` character.

        :param key: Name of member
        :type key: str
        :raise: KeyError if ``key`` is not a known member of this group.
        """
        # accept numbers for scan indices
        if isinstance(key, int):
            number = self.file._sf.number(key)
            order = self.file._sf.order(key)
            full_key = "/%d.%d" % (number, order)
        # Relative path starting from this group (e.g "mca_0/info")
        elif not key.startswith("/"):
            full_key = self.name.rstrip("/") + "/" + key
        # Absolute path called from the root group or from a parent group
        elif key.startswith(self.name):
            full_key = key
        # Absolute path to an element called from a non-parent group
        else:
            raise KeyError(key + " is not a child of " + self.__repr__())
        return full_key

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
        full_key = self._get_full_key(key)
        if full_key in self.file._cached_items:
            return self.file._cached_items[full_key]

        if is_group(full_key):
            self.file.cache(full_key, SpecH5Group(full_key, self.file))
        elif is_dataset(full_key):
            self.file.cache(full_key, _dataset_builder(full_key, self.file, self))
        elif is_link_to_group(full_key):
            link_target = full_key.replace("measurement", "instrument").rstrip("/")[:-4]
            self.file.cache(full_key, SpecH5LinkToGroup(full_key, self.file, link_target))
        elif is_link_to_dataset(full_key):
            self.file.cache(full_key, _link_to_dataset_builder(full_key, self.file, self))
        else:
            raise KeyError("unrecognized group or dataset: " + full_key)

        return self.file._cached_items[full_key]

    def __iter__(self):
        for key in self.keys():
            yield key

    def items(self):
        """Returns a generator yielding 2-tuples *(key, value)*,
        where key is the name of a group or dataset, and value is the instance
        of that group or dataset.

        If a list is required, it must be requested explicitly::

            itemslist = list(group.items())
        """
        for key in self.keys():
            yield key, self[key]

    def __len__(self):
        """Return number of members, subgroups and datasets, attached to this
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
            ret = static_items["scan"]
            if _unit_cell_in_scan(self._scan) or _ub_matrix_in_scan(self._scan):
                return ret + [u"sample"]
            return ret

        if positioners_group_pattern.match(self.name):
            motor_names = self._scan.motor_names
            return [name.replace("/", "%") for name in motor_names]

        if specfile_group_pattern.match(self.name):
            return static_items["scan/instrument/specfile"]

        if measurement_mca_group_pattern.match(self.name):
            return static_items["scan/measurement/mca"]

        if instrument_mca_group_pattern.match(self.name):
            ret = static_items["scan/instrument/mca"]
            if "CTIME" in self._scan.mca_header_dict:
                return ret + [u"preset_time", u"elapsed_time", u"live_time"]
            return ret

        if sample_pattern.match(self.name):
            ret = []
            if _unit_cell_in_scan(self._scan):
                ret.append(u"unit_cell")
                ret.append(u"unit_cell_abc")
                ret.append(u"unit_cell_alphabetagamma")
            if _ub_matrix_in_scan(self._scan):
                ret.append(u"ub_matrix")
            return ret

        number_of_MCA_spectra = len(self._scan.mca)
        number_of_data_lines = self._scan.data.shape[1]

        if not number_of_data_lines == 0:
            # Number of MCA spectra must be a multiple of number of data lines
            assert number_of_MCA_spectra % number_of_data_lines == 0
            number_of_MCA_analysers = number_of_MCA_spectra // number_of_data_lines
        elif number_of_MCA_spectra:
            # Case of a scan without data lines, only MCA.
            # Our only option is to assume that the number of analysers
            # is the number of #@CHANN lines
            number_of_MCA_analysers = len(self._scan.mca.channels)
        else:
            number_of_MCA_analysers = 0

        mca_list = ["mca_%d" % i for i in range(number_of_MCA_analysers)]

        if measurement_group_pattern.match(self.name):
            scan_labels = self._scan.labels
            return [label.replace("/", "%") for label in scan_labels] + mca_list

        if instrument_pattern.match(self.name):
            return static_items["scan/instrument"] + mca_list

    def values(self):
        """New in silx 0.6"""
        return list(map(self.__getitem__, self.keys()))

    def visit(self, func, follow_links=False):
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
        base_name = self.name
        return self._visit(func, base_name, follow_links)

    def _visit(self, func, base_name, follow_links=False):
        """

        :param func:
        :param base_name: name of first group that initiated the recursion
            This is used to compute the relative path from each item's
            absolute path.
        :param follow_links:
        :return:
        """
        for member_name in self.keys():
            member = self[member_name]
            ret = None
            if (not is_link_to_dataset(member.name) and
                    not is_link_to_group(member.name)) or follow_links:
                assert member.name.startswith(base_name)
                # left strip to remove the base name
                relative_name = member.name[len(base_name):]
                # remove leading slash and unnecessary trailing slash
                relative_name = relative_name.strip("/")
                ret = func(relative_name)
            if ret is not None:
                return ret
            # recurse into subgroups
            if isinstance(member, SpecH5Group):
                if not isinstance(member, SpecH5LinkToGroup) or follow_links:
                    self[member_name]._visit(func, base_name, follow_links)

    def visititems(self, func, follow_links=False):
        """Recursively visit names and objects in this group.

        :param func: Callable (function, method or callable object)
        :type func: function

        You supply a callable (function, method or callable object); it
        will be called exactly once for each
        member in this group and every group below it. Your callable must
        conform to the signature:

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
        base_name = self.name
        return self._visititems(func, base_name, follow_links)

    def _visititems(self, func, base_name, follow_links=False):
        """

        :param func:
        :param base_name: Name of first group that initiated the recursion.
            This is used to compute the relative path from each item's
            absolute path.
        :param follow_links:
        :return:
        """
        for member_name in self.keys():
            member = self[member_name]
            ret = None
            if (not is_link_to_dataset(member.name) and
                    not is_link_to_group(member.name)) or follow_links:
                assert member.name.startswith(base_name)
                # left strip to remove the base name
                relative_name = member.name[len(base_name):]
                # remove leading slash and unnecessary trailing slash
                relative_name = relative_name.strip("/")
                ret = func(relative_name, member)
            if ret is not None:
                return ret
            # recurse into subgroups
            if isinstance(self[member_name], SpecH5Group):
                if not isinstance(self[member_name], SpecH5LinkToGroup) or follow_links:
                    self[member_name]._visititems(func, base_name, follow_links)


class SpecH5LinkToGroup(SpecH5Group):
    """Special :class:`SpecH5Group` representing a link to a group.

    It works like a regular group but :meth:`SpecH5Group.visit`
    and :meth:`SpecH5Group.visititems` methods will recognize it as a
    link and will ignore it.

    An additional attribute indicates the name of the target group:
    :attr:`target`
    """
    def __init__(self, name, specfileh5, target):
        SpecH5Group.__init__(self, name, specfileh5)
        self.target = target
        """Name of the target group."""

    def keys(self):
        """:return: List of all names of members attached to the target group
        """
        # we only have a single type of link to a group:
        # /1.1/measurement/mca_0/info/ -> /1.1/instrument/mca_0/
        if measurement_mca_info_pattern.match(self.name):
            # link_target = self.name.replace("measurement", "instrument").rstrip("/")[:-4]
            # return SpecH5Group(link_target, self.file).keys()
            return SpecH5Group(self.target, self.file).keys()
        else:
            raise NameError("Unknown link to SpecH5Group: "
                            "%s -> %s" % (self.name, self.target))


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
        if isinstance(filename, six.string_types + (six.binary_type, )):
            # silx.io.SpecFile can process unicode and bytes
            self.filename = filename
        elif isinstance(filename, io.IOBase):
            self.filename = filename.name
        else:
            raise TypeError("SpecH5 filename must be a string or a " +
                            "file handle.")
        self.attrs = _get_attrs_dict("/")
        self._sf = SpecFile(self.filename)
        self._cached_items = {}
        """Keeps a reference to created items, to avoid regenerating
        them again."""

        SpecH5Group.__init__(self, name="/", specfileh5=self)
        if len(self) == 0:
            # SpecFile library do not raise exception for non specfiles
            raise IOError("Empty specfile. Not a valid spec format.")

    def cache(self, name, item):
        self._cached_items[name] = item

    def uncache(self, name):
        if name in self._cached_items:
            del self._cached_items[name]
        else:
            raise KeyError("Key %s not in cache" % name)

    def keys(self):
        """
        :return: List of all scan keys in this SpecFile
            (e.g. ``["1.1", "2.1"…]``)
        """
        return self._sf.keys()

    def __enter__(self):
        """Context manager enter"""
        return self

    def __exit__(self, type, value, tb):  # pylint: disable=W0622
        """Context manager exit"""
        self.close()

    def close(self):
        """Close the object, and free up associated resources.

        After calling this method, attempts to use the object may fail.
        """
        self._sf = None

    def __repr__(self):
        return '<SpecH5 "%s" (%d members)>' % (self.filename, len(self))

    def __eq__(self, other):
        return (isinstance(other, SpecH5) and
                self.filename == other.filename and
                self.keys() == other.keys())

    @property
    def h5py_class(self):
        """h5py class which is mimicked by this class"""
        if h5py is None:
            raise ImportError("Cannot return h5py.File class, " +
                              "unable to import h5py module")
        return h5py.File
