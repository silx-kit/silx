# coding: utf-8
# /*##########################################################################
# Copyright (C) 2016-2018 European Synchrotron Radiation Facility
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
+++++++++++++++

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
          sample/
              ub_matrix = …
              unit_cell = …
              unit_cell_abc = …
              unit_cell_alphabetagamma = …
      2.1/
          …

``file_header`` and ``scan_header`` are the raw headers as they
appear in the original file, as a string of lines separated by newline (``\\n``) characters.

The title is the content of the ``#S`` scan header line without the leading
``#S`` and without the scan number (e.g ``"ascan  ss1vo -4.55687 -0.556875  40 0.2"``).

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
++++++++++++++

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

:class:`SpecH5` files and groups provide a :meth:`keys` method::

    >>> sfh5.keys()
    ['96.1', '97.1', '98.1']
    >>> sfh5['96.1'].keys()
    ['title', 'start_time', 'instrument', 'measurement']

They can also be treated as iterators:

.. code-block:: python

    from silx.io import is_dataset

    for scan_group in SpecH5("test.dat"):
        dataset_names = [item.name in scan_group["measurement"] if
                         is_dataset(item)]
        print("Found data columns in scan " + scan_group.name)
        print(", ".join(dataset_names))

You can test for existence of data or groups::

    >>> "/1.1/measurement/Pslit HGap" in sfh5
    True
    >>> "positioners" in sfh5["/2.1/instrument"]
    True
    >>> "spam" in sfh5["1.1"]
    False

.. note::

    Text used to be stored with a dtype ``numpy.string_`` in silx versions
    prior to *0.7.0*. The type ``numpy.string_`` is a byte-string format.
    The consequence of this is that you had to decode strings before using
    them in **Python 3**::

        >>> from silx.io.spech5 import SpecH5
        >>> sfh5 = SpecH5("31oct98.dat")
        >>> sfh5["/68.1/title"]
        b'68  ascan  tx3 -28.5 -24.5  20 0.5'
        >>> sfh5["/68.1/title"].decode()
        '68  ascan  tx3 -28.5 -24.5  20 0.5'

    From silx version *0.7.0* onwards, text is now stored as unicode. This
    corresponds to the default text type in python 3, and to the *unicode*
    type in Python 2.

    To be on the safe side, you can test for the presence of a *decode*
    attribute, to ensure that you always work with unicode text::

        >>> title = sfh5["/68.1/title"]
        >>> if hasattr(title, "decode"):
        ...     title = title.decode()

"""

import datetime
import logging
import numpy
import re
import io
import h5py

from silx import version as silx_version
from .specfile import SpecFile
from . import commonh5
from silx.third_party import six

__authors__ = ["P. Knobel", "D. Naudet"]
__license__ = "MIT"
__date__ = "17/07/2018"

logger1 = logging.getLogger(__name__)


text_dtype = h5py.special_dtype(vlen=six.text_type)


def to_h5py_utf8(str_list):
    """Convert a string or a list of strings to a numpy array of
    unicode strings that can be written to HDF5 as utf-8.

    This ensures that the type will be consistent between python 2 and
    python 3, if attributes or datasets are saved to an HDF5 file.
    """
    return numpy.array(str_list, dtype=text_dtype)


def _get_number_of_mca_analysers(scan):
    """
    :param SpecFile sf: :class:`SpecFile` instance
    """
    number_of_mca_spectra = len(scan.mca)
    # Scan.data is transposed
    number_of_data_lines = scan.data.shape[1]

    if not number_of_data_lines == 0:
        # Number of MCA spectra must be a multiple of number of data lines
        assert number_of_mca_spectra % number_of_data_lines == 0
        return number_of_mca_spectra // number_of_data_lines
    elif number_of_mca_spectra:
        # Case of a scan without data lines, only MCA.
        # Our only option is to assume that the number of analysers
        # is the number of #@CHANN lines
        return len(scan.mca.channels)
    else:
        return 0


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
        is a concatenation of them separated by a ``\\n`` character.
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
    year_rx = r'(?P<year>\d{4})'
    day_nb_rx = r'(?P<day_nb>[0-3 ]\d)'
    month_nb_rx = r'(?P<month_nb>[0-1]\d)'
    hh_rx = r'(?P<hh>[0-2]\d)'
    mm_rx = r'(?P<mm>[0-5]\d)'
    ss_rx = r'(?P<ss>[0-5]\d)'
    tz_rx = r'(?P<tz>[+-]\d\d:\d\d){0,1}'

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


# Node classes
class SpecH5Dataset(object):
    """This convenience class is to be inherited by all datasets, for
    compatibility purpose with code that tests for
    ``isinstance(obj, SpecH5Dataset)``.

    This legacy behavior is deprecated. The correct way to test
    if an object is a dataset is to use :meth:`silx.io.utils.is_dataset`.

    Datasets must also inherit :class:`SpecH5NodeDataset` or
    :class:`SpecH5LazyNodeDataset` which actually implement all the
    API."""
    pass


class SpecH5NodeDataset(commonh5.Dataset, SpecH5Dataset):
    """This class inherits :class:`commonh5.Dataset`, to which it adds
    little extra functionality. The main additional functionality is the
    proxy behavior that allows to mimic the numpy array stored in this
    class.
    """
    def __init__(self, name, data, parent=None, attrs=None):
        # get proper value types, to inherit from numpy
        # attributes (dtype, shape, size)
        if isinstance(data, six.string_types):
            # use unicode (utf-8 when saved to HDF5 output)
            value = to_h5py_utf8(data)
        elif isinstance(data, float):
            # use 32 bits for float scalars
            value = numpy.float32(data)
        elif isinstance(data, int):
            value = numpy.int_(data)
        else:
            # Enforce numpy array
            array = numpy.array(data)
            data_kind = array.dtype.kind

            if data_kind in ["S", "U"]:
                value = numpy.asarray(array,
                                      dtype=text_dtype)
            elif data_kind in ["f"]:
                value = numpy.asarray(array, dtype=numpy.float32)
            else:
                value = array
        commonh5.Dataset.__init__(self, name, value, parent, attrs)

    def __getattr__(self, item):
        """Proxy to underlying numpy array methods.
        """
        if hasattr(self[()], item):
            return getattr(self[()], item)

        raise AttributeError("SpecH5Dataset has no attribute %s" % item)


class SpecH5LazyNodeDataset(commonh5.LazyLoadableDataset, SpecH5Dataset):
    """This class inherits :class:`commonh5.LazyLoadableDataset`,
    to which it adds a proxy behavior that allows to mimic the numpy
    array stored in this class.

    The class has to be inherited and the :meth:`_create_data` method has to be
    implemented to return the numpy data exposed by the dataset. This factory
    method is only called once, when the data is needed.
    """
    def __getattr__(self, item):
        """Proxy to underlying numpy array methods.
        """
        if hasattr(self[()], item):
            return getattr(self[()], item)

        raise AttributeError("SpecH5Dataset has no attribute %s" % item)

    def _create_data(self):
        """
        Factory to create the data exposed by the dataset when it is needed.

        It has to be implemented for the class to work.

        :rtype: numpy.ndarray
        """
        raise NotImplementedError()


class SpecH5Group(object):
    """This convenience class is to be inherited by all groups, for
    compatibility purposes with code that tests for
    ``isinstance(obj, SpecH5Group)``.

    This legacy behavior is deprecated. The correct way to test
    if an object is a group is to use :meth:`silx.io.utils.is_group`.

    Groups must also inherit :class:`silx.io.commonh5.Group`, which
    actually implements all the methods and attributes."""
    pass


class SpecH5(commonh5.File, SpecH5Group):
    """This class opens a SPEC file and exposes it as a *h5py.File*.

    It inherits :class:`silx.io.commonh5.Group` (via :class:`commonh5.File`),
    which implements most of its API.
    """

    def __init__(self, filename):
        """
        :param filename: Path to SpecFile in filesystem
        :type filename: str
        """
        if isinstance(filename, io.IOBase):
            # see https://github.com/silx-kit/silx/issues/858
            filename = filename.name

        self._sf = SpecFile(filename)

        attrs = {"NX_class": to_h5py_utf8("NXroot"),
                 "file_time": to_h5py_utf8(
                         datetime.datetime.now().isoformat()),
                 "file_name": to_h5py_utf8(filename),
                 "creator": to_h5py_utf8("silx spech5 %s" % silx_version)}
        commonh5.File.__init__(self, filename, attrs=attrs)

        for scan_key in self._sf.keys():
            scan = self._sf[scan_key]
            scan_group = ScanGroup(scan_key, parent=self, scan=scan)
            self.add_node(scan_group)

    def close(self):
        self._sf.close()
        self._sf = None


class ScanGroup(commonh5.Group, SpecH5Group):
    def __init__(self, scan_key, parent, scan):
        """

        :param parent: parent Group
        :param str scan_key: Scan key (e.g. "1.1")
        :param scan: specfile.Scan object
        """
        commonh5.Group.__init__(self, scan_key, parent=parent,
                                attrs={"NX_class": to_h5py_utf8("NXentry")})

        # take title in #S after stripping away scan number and spaces
        s_hdr_line = scan.scan_header_dict["S"]
        title = s_hdr_line.lstrip("0123456789").lstrip()
        self.add_node(SpecH5NodeDataset(name="title",
                                        data=to_h5py_utf8(title),
                                        parent=self))

        if "D" in scan.scan_header_dict:
            try:
                start_time_str = spec_date_to_iso8601(scan.scan_header_dict["D"])
            except (IndexError, ValueError):
                logger1.warning("Could not parse date format in scan %s header." +
                                " Using original date not converted to ISO-8601",
                                scan_key)
                start_time_str = scan.scan_header_dict["D"]
        elif "D" in scan.file_header_dict:
            logger1.warning("No #D line in scan %s header. " +
                            "Using file header for start_time.",
                            scan_key)
            try:
                start_time_str = spec_date_to_iso8601(scan.file_header_dict["D"])
            except (IndexError, ValueError):
                logger1.warning("Could not parse date format in scan %s header. " +
                                "Using original date not converted to ISO-8601",
                                scan_key)
                start_time_str = scan.file_header_dict["D"]
        else:
            logger1.warning("No #D line in %s header. Setting date to empty string.",
                            scan_key)
            start_time_str = ""
        self.add_node(SpecH5NodeDataset(name="start_time",
                                        data=to_h5py_utf8(start_time_str),
                                        parent=self))

        self.add_node(InstrumentGroup(parent=self, scan=scan))
        self.add_node(MeasurementGroup(parent=self, scan=scan))
        if _unit_cell_in_scan(scan) or _ub_matrix_in_scan(scan):
            self.add_node(SampleGroup(parent=self, scan=scan))


class InstrumentGroup(commonh5.Group, SpecH5Group):
    def __init__(self, parent, scan):
        """

        :param parent: parent Group
        :param scan: specfile.Scan object
        """
        commonh5.Group.__init__(self, name="instrument", parent=parent,
                                attrs={"NX_class": to_h5py_utf8("NXinstrument")})

        self.add_node(InstrumentSpecfileGroup(parent=self, scan=scan))
        self.add_node(PositionersGroup(parent=self, scan=scan))

        num_analysers = _get_number_of_mca_analysers(scan)
        for anal_idx in range(num_analysers):
            self.add_node(InstrumentMcaGroup(parent=self,
                                             analyser_index=anal_idx,
                                             scan=scan))


class InstrumentSpecfileGroup(commonh5.Group, SpecH5Group):
    def __init__(self, parent, scan):
        commonh5.Group.__init__(self, name="specfile", parent=parent,
                                attrs={"NX_class": to_h5py_utf8("NXcollection")})
        self.add_node(SpecH5NodeDataset(
                name="file_header",
                data=to_h5py_utf8(scan.file_header),
                parent=self,
                attrs={}))
        self.add_node(SpecH5NodeDataset(
                name="scan_header",
                data=to_h5py_utf8(scan.scan_header),
                parent=self,
                attrs={}))


class PositionersGroup(commonh5.Group, SpecH5Group):
    def __init__(self, parent, scan):
        commonh5.Group.__init__(self, name="positioners", parent=parent,
                                attrs={"NX_class": to_h5py_utf8("NXcollection")})
        for motor_name in scan.motor_names:
            safe_motor_name = motor_name.replace("/", "%")
            if motor_name in scan.labels and scan.data.shape[0] > 0:
                # return a data column if one has the same label as the motor
                motor_value = scan.data_column_by_name(motor_name)
            else:
                # Take value from #P scan header.
                # (may return float("inf") if #P line is missing from scan hdr)
                motor_value = scan.motor_position_by_name(motor_name)
            self.add_node(SpecH5NodeDataset(name=safe_motor_name,
                                            data=motor_value,
                                            parent=self))


class InstrumentMcaGroup(commonh5.Group, SpecH5Group):
    def __init__(self, parent, analyser_index, scan):
        name = "mca_%d" % analyser_index
        commonh5.Group.__init__(self, name=name, parent=parent,
                                attrs={"NX_class": to_h5py_utf8("NXdetector")})

        mcaDataDataset = McaDataDataset(parent=self,
                                     analyser_index=analyser_index,
                                     scan=scan)
        self.add_node(mcaDataDataset)
        spectrum_length = mcaDataDataset.shape[-1]
        mcaDataDataset = None

        if len(scan.mca.channels) == 1:
            # single @CALIB line applying to multiple devices
            calibration_dataset = scan.mca.calibration[0]
            channels_dataset = scan.mca.channels[0]
        else:
            calibration_dataset = scan.mca.calibration[analyser_index]
            channels_dataset = scan.mca.channels[analyser_index]

        channels_length = len(channels_dataset) 
        if (channels_length > 1) and (spectrum_length > 0):
            logger1.info("Spectrum and channels length mismatch")
            # this should always be the case
            if channels_length > spectrum_length:
                channels_dataset = channels_dataset[:spectrum_length]
            elif channels_length < spectrum_length:
                # only trust first channel and increment
                channel0 = channels_dataset[0]
                increment = channels_dataset[1] - channels_dataset[0]
                channels_dataset = numpy.linspace(channel0,
                                        channel0 + increment * spectrum_length,
                                        spectrum_length, endpoint=False)

        self.add_node(SpecH5NodeDataset(name="calibration",
                                        data=calibration_dataset,
                                        parent=self))
        self.add_node(SpecH5NodeDataset(name="channels",
                                        data=channels_dataset,
                                        parent=self))

        if "CTIME" in scan.mca_header_dict:
            ctime_line = scan.mca_header_dict['CTIME']
            preset_time, live_time, elapsed_time = _parse_ctime(ctime_line, analyser_index)
            self.add_node(SpecH5NodeDataset(name="preset_time",
                                            data=preset_time,
                                            parent=self))
            self.add_node(SpecH5NodeDataset(name="live_time",
                                            data=live_time,
                                            parent=self))
            self.add_node(SpecH5NodeDataset(name="elapsed_time",
                                            data=elapsed_time,
                                            parent=self))


class McaDataDataset(SpecH5LazyNodeDataset):
    """Lazy loadable dataset for MCA data"""
    def __init__(self, parent, analyser_index, scan):
        commonh5.LazyLoadableDataset.__init__(
            self, name="data", parent=parent,
            attrs={"interpretation": to_h5py_utf8("spectrum"),})
        self._scan = scan
        self._analyser_index = analyser_index
        self._shape = None
        self._num_analysers = _get_number_of_mca_analysers(self._scan)

    def _create_data(self):
        return _demultiplex_mca(self._scan, self._analyser_index)

    @property
    def shape(self):
        if self._shape is None:
            num_spectra_in_file = len(self._scan.mca)
            num_spectra_per_analyser = num_spectra_in_file // self._num_analysers
            len_spectrum = len(self._scan.mca[self._analyser_index])
            self._shape = num_spectra_per_analyser, len_spectrum
        return self._shape

    @property
    def size(self):
        return numpy.prod(self.shape, dtype=numpy.intp)

    @property
    def dtype(self):
        # we initialize the data with numpy.empty() without specifying a dtype
        # in _demultiplex_mca()
        return numpy.empty((1, )).dtype

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, item):
        # optimization for fetching a single spectrum if data not already loaded
        if not self._is_initialized:
            if isinstance(item, six.integer_types):
                if item < 0:
                    # negative indexing
                    item += len(self)
                return self._scan.mca[self._analyser_index +
                                      item * self._num_analysers]
            # accessing a slice or element of a single spectrum [i, j:k]
            try:
                spectrum_idx, channel_idx_or_slice = item
                assert isinstance(spectrum_idx, six.integer_types)
            except (ValueError, TypeError, AssertionError):
                pass
            else:
                if spectrum_idx < 0:
                    item += len(self)
                idx = self._analyser_index + spectrum_idx * self._num_analysers
                return self._scan.mca[idx][channel_idx_or_slice]

        return super(McaDataDataset, self).__getitem__(item)


class MeasurementGroup(commonh5.Group, SpecH5Group):
    def __init__(self, parent, scan):
        """

        :param parent: parent Group
        :param scan: specfile.Scan object
        """
        commonh5.Group.__init__(self, name="measurement", parent=parent,
                                attrs={"NX_class": to_h5py_utf8("NXcollection"),})
        for label in scan.labels:
            safe_label = label.replace("/", "%")
            self.add_node(SpecH5NodeDataset(name=safe_label,
                                            data=scan.data_column_by_name(label),
                                            parent=self))

        num_analysers = _get_number_of_mca_analysers(scan)
        for anal_idx in range(num_analysers):
            self.add_node(MeasurementMcaGroup(parent=self, analyser_index=anal_idx))


class MeasurementMcaGroup(commonh5.Group, SpecH5Group):
    def __init__(self, parent, analyser_index):
        basename = "mca_%d" % analyser_index
        commonh5.Group.__init__(self, name=basename, parent=parent,
                                attrs={})

        target_name = self.name.replace("measurement", "instrument")
        self.add_node(commonh5.SoftLink(name="data",
                                        path=target_name + "/data",
                                        parent=self))
        self.add_node(commonh5.SoftLink(name="info",
                                        path=target_name,
                                        parent=self))


class SampleGroup(commonh5.Group, SpecH5Group):
    def __init__(self, parent, scan):
        """

        :param parent: parent Group
        :param scan: specfile.Scan object
        """
        commonh5.Group.__init__(self, name="sample", parent=parent,
                                attrs={"NX_class": to_h5py_utf8("NXsample"),})

        if _unit_cell_in_scan(scan):
            self.add_node(SpecH5NodeDataset(name="unit_cell",
                                            data=_parse_unit_cell(scan.scan_header_dict["G1"]),
                                            parent=self,
                                            attrs={"interpretation": to_h5py_utf8("scalar")}))
            self.add_node(SpecH5NodeDataset(name="unit_cell_abc",
                                            data=_parse_unit_cell(scan.scan_header_dict["G1"])[0, 0:3],
                                            parent=self,
                                            attrs={"interpretation": to_h5py_utf8("scalar")}))
            self.add_node(SpecH5NodeDataset(name="unit_cell_alphabetagamma",
                                            data=_parse_unit_cell(scan.scan_header_dict["G1"])[0, 3:6],
                                            parent=self,
                                            attrs={"interpretation": to_h5py_utf8("scalar")}))
        if _ub_matrix_in_scan(scan):
            self.add_node(SpecH5NodeDataset(name="ub_matrix",
                                            data=_parse_UB_matrix(scan.scan_header_dict["G3"]),
                                            parent=self,
                                            attrs={"interpretation": to_h5py_utf8("scalar")}))
