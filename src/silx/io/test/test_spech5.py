# /*##########################################################################
# Copyright (C) 2016-2021 European Synchrotron Radiation Facility
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
"""Tests for spech5"""
import numpy
import os
import io
import sys
import tempfile
import unittest
import datetime
from functools import partial

from silx.utils import testutils

from .. import spech5
from ..spech5 import (SpecH5, SpecH5Dataset, spec_date_to_iso8601)
from .. import specfile

import h5py

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "12/02/2018"

sftext = """#F /tmp/sf.dat
#E 1455180875
#D Thu Feb 11 09:54:35 2016
#C imaging  User = opid17
#O0 Pslit HGap  MRTSlit UP  MRTSlit DOWN
#O1 Sslit1 VOff  Sslit1 HOff  Sslit1 VGap
#o0 pshg mrtu mrtd
#o2 ss1vo ss1ho ss1vg

#J0 Seconds  IA  ion.mono  Current
#J1 xbpmc2  idgap1  Inorm

#S 1  ascan  ss1vo -4.55687 -0.556875  40 0.2
#D Thu Feb 11 09:55:20 2016
#T 0.2  (Seconds)
#P0 180.005 -0.66875 0.87125
#P1 14.74255 16.197579 12.238283
#N 4
#L MRTSlit UP  second column  3rd_col
-1.23 5.89  8
8.478100E+01  5 1.56
3.14 2.73 -3.14
1.2 2.3 3.4

#S 25  ascan  c3th 1.33245 1.52245  40 0.15
#D Sat 2015/03/14 03:53:50
#P0 80.005 -1.66875 1.87125
#P1 4.74255 6.197579 2.238283
#N 5
#L column0  column1  col2  col3
0.0 0.1 0.2 0.3
1.0 1.1 1.2 1.3
2.0 2.1 2.2 2.3
3.0 3.1 3.2 3.3

#S 1 aaaaaa
#D Thu Feb 11 10:00:32 2016
#@MCADEV 1
#@MCA %16C
#@CHANN 3 0 2 1
#@CALIB 1 2 3
#@CTIME 123.4 234.5 345.6
#N 3
#L uno  duo
1 2
@A 0 1 2
@A 10 9 8
@A 1 1 1.1
3 4
@A 3.1 4 5
@A 7 6 5
@A 1 1 1
5 6
@A 6 7.7 8
@A 4 3 2
@A 1 1 1

#S 1000 bbbbb
#G1 3.25 3.25 5.207 90 90 120 2.232368448 2.232368448 1.206680489 90 90 60 1 1 2 -1 2 2 26.132 7.41 -88.96 1.11 1.000012861 15.19 26.06 67.355 -88.96 1.11 1.000012861 15.11 0.723353 0.723353
#G3 0.0106337923671 0.027529133 1.206191273 -1.43467075 0.7633438883 0.02401568018 -1.709143587 -2.097621783 0.02456954971
#L a  b
1 2

#S 1001 ccccc
#G1 0. 0. 0. 0 0 0 2.232368448 2.232368448 1.206680489 90 90 60 1 1 2 -1 2 2 26.132 7.41 -88.96 1.11 1.000012861 15.19 26.06 67.355 -88.96 1.11 1.000012861 15.11 0.723353 0.723353
#G3 0. 0. 0. 0. 0.0 0. 0. 0. 0.
#L a  b
1 2

"""


class TestSpecDate(unittest.TestCase):
    """
    Test of the spec_date_to_iso8601 function.
    """
    # TODO : time zone tests
    # TODO : error cases

    @classmethod
    def setUpClass(cls):
        import locale
        # FYI : not threadsafe
        cls.locale_saved = locale.setlocale(locale.LC_TIME)
        locale.setlocale(locale.LC_TIME, 'C')

    @classmethod
    def tearDownClass(cls):
        import locale
        # FYI : not threadsafe
        locale.setlocale(locale.LC_TIME, cls.locale_saved)

    def setUp(self):
        # covering all week days
        self.n_days = range(1, 10)
        # covering all months
        self.n_months = range(1, 13)

        self.n_years = [1999, 2016, 2020]
        self.n_seconds = [0, 5, 26, 59]
        self.n_minutes = [0, 9, 42, 59]
        self.n_hours = [0, 2, 17, 23]

        self.formats = ['%a %b %d %H:%M:%S %Y', '%a %Y/%m/%d %H:%M:%S']

        self.check_date_formats = partial(self.__check_date_formats,
                                          year=self.n_years[0],
                                          month=self.n_months[0],
                                          day=self.n_days[0],
                                          hour=self.n_hours[0],
                                          minute=self.n_minutes[0],
                                          second=self.n_seconds[0],
                                          msg=None)

    def __check_date_formats(self,
                             year,
                             month,
                             day,
                             hour,
                             minute,
                             second,
                             msg=None):
        dt = datetime.datetime(year, month, day, hour, minute, second)
        expected_date = dt.isoformat()

        for i_fmt, fmt in enumerate(self.formats):
            spec_date = dt.strftime(fmt)
            iso_date = spec_date_to_iso8601(spec_date)
            self.assertEqual(iso_date,
                             expected_date,
                             msg='Testing {0}. format={1}. '
                                 'Expected "{2}", got "{3} ({4})" (dt={5}).'
                                 ''.format(msg,
                                           i_fmt,
                                           expected_date,
                                           iso_date,
                                           spec_date,
                                           dt))

    def testYearsNominal(self):
        for year in self.n_years:
            self.check_date_formats(year=year, msg='year')

    def testMonthsNominal(self):
        for month in self.n_months:
            self.check_date_formats(month=month, msg='month')

    def testDaysNominal(self):
        for day in self.n_days:
            self.check_date_formats(day=day, msg='day')

    def testHoursNominal(self):
        for hour in self.n_hours:
            self.check_date_formats(hour=hour, msg='hour')

    def testMinutesNominal(self):
        for minute in self.n_minutes:
            self.check_date_formats(minute=minute, msg='minute')

    def testSecondsNominal(self):
        for second in self.n_seconds:
            self.check_date_formats(second=second, msg='second')


class TestSpecH5(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fd, cls.fname = tempfile.mkstemp()
        if sys.version_info < (3, ):
            os.write(fd, sftext)
        else:
            os.write(fd, bytes(sftext, 'ascii'))
        os.close(fd)

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls.fname)

    def setUp(self):
        self.sfh5 = SpecH5(self.fname)

    def tearDown(self):
        self.sfh5.close()

    def testContainsFile(self):
        self.assertIn("/1.2/measurement", self.sfh5)
        self.assertIn("/25.1", self.sfh5)
        self.assertIn("25.1", self.sfh5)
        self.assertNotIn("25.2", self.sfh5)
        # measurement is a child of a scan, full path would be required to
        # access from root level
        self.assertNotIn("measurement", self.sfh5)
        # Groups may or may not have a trailing /
        self.assertIn("/1.2/measurement/mca_1/", self.sfh5)
        self.assertIn("/1.2/measurement/mca_1", self.sfh5)
        # Datasets can't have a trailing /
        self.assertNotIn("/1.2/measurement/mca_0/info/calibration/ ", self.sfh5)
        # No mca_8
        self.assertNotIn("/1.2/measurement/mca_8/info/calibration", self.sfh5)
        # Link
        self.assertIn("/1.2/measurement/mca_0/info/calibration", self.sfh5)

    def testContainsGroup(self):
        self.assertIn("measurement", self.sfh5["/1.2/"])
        self.assertIn("measurement", self.sfh5["/1.2"])
        self.assertIn("25.1", self.sfh5["/"])
        self.assertNotIn("25.2", self.sfh5["/"])
        self.assertIn("instrument/positioners/Sslit1 HOff", self.sfh5["/1.1"])
        # illegal trailing "/" after dataset name
        self.assertNotIn("instrument/positioners/Sslit1 HOff/",
                         self.sfh5["/1.1"])
        # full path to element in group (OK)
        self.assertIn("/1.1/instrument/positioners/Sslit1 HOff",
                      self.sfh5["/1.1/instrument"])

    def testDataColumn(self):
        self.assertAlmostEqual(sum(self.sfh5["/1.2/measurement/duo"]),
                               12.0)
        self.assertAlmostEqual(
                sum(self.sfh5["1.1"]["measurement"]["MRTSlit UP"]),
                87.891, places=4)

    def testDate(self):
        # start time is in Iso8601 format
        self.assertEqual(self.sfh5["/1.1/start_time"],
                         u"2016-02-11T09:55:20")
        self.assertEqual(self.sfh5["25.1/start_time"],
                         u"2015-03-14T03:53:50")

    def assertRaisesRegex(self, *args, **kwargs):
        # Python 2 compatibility
        if sys.version_info.major >= 3:
            return super(TestSpecH5, self).assertRaisesRegex(*args, **kwargs)
        else:
            return self.assertRaisesRegexp(*args, **kwargs)

    def testDatasetInstanceAttr(self):
        """The SpecH5Dataset objects must implement some dummy attributes
        to improve compatibility with widgets dealing with h5py datasets."""
        self.assertIsNone(self.sfh5["/1.1/start_time"].compression)
        self.assertIsNone(self.sfh5["1.1"]["measurement"]["MRTSlit UP"].chunks)

        # error message must be explicit
        with self.assertRaisesRegex(
                AttributeError,
                "SpecH5Dataset has no attribute tOTo"):
            dummy = self.sfh5["/1.1/start_time"].tOTo

    def testGet(self):
        """Test :meth:`SpecH5Group.get`"""
        # default value of param *default* is None
        self.assertIsNone(self.sfh5.get("toto"))
        self.assertEqual(self.sfh5["25.1"].get("toto", default=-3),
                         -3)

        self.assertEqual(self.sfh5.get("/1.1/start_time", default=-3),
                         u"2016-02-11T09:55:20")

    def testGetClass(self):
        """Test :meth:`SpecH5Group.get`"""
        self.assertIs(self.sfh5["1.1"].get("start_time", getclass=True),
                      h5py.Dataset)
        self.assertIs(self.sfh5["1.1"].get("instrument", getclass=True),
                      h5py.Group)

        # spech5 does not define external link, so there is no way
        # a group can *get* a SpecH5 class

    def testGetApi(self):
        result = self.sfh5.get("1.1", getclass=True, getlink=True)
        self.assertIs(result, h5py.HardLink)
        result = self.sfh5.get("1.1", getclass=False, getlink=True)
        self.assertIsInstance(result, h5py.HardLink)
        result = self.sfh5.get("1.1", getclass=True, getlink=False)
        self.assertIs(result, h5py.Group)
        result = self.sfh5.get("1.1", getclass=False, getlink=False)
        self.assertIsInstance(result, spech5.SpecH5Group)

    def testGetItemGroup(self):
        group = self.sfh5["25.1"]["instrument"]
        self.assertEqual(list(group["positioners"].keys()),
                         ["Pslit HGap", "MRTSlit UP", "MRTSlit DOWN",
                          "Sslit1 VOff", "Sslit1 HOff", "Sslit1 VGap"])
        with self.assertRaises(KeyError):
            group["Holy Grail"]

    def testGetitemSpecH5(self):
        self.assertEqual(self.sfh5["/1.2/instrument/positioners"],
                         self.sfh5["1.2"]["instrument"]["positioners"])

    def testH5pyClass(self):
        """Test :attr:`h5py_class` returns the corresponding h5py class
        (h5py.File, h5py.Group, h5py.Dataset)"""
        a_file = self.sfh5
        self.assertIs(a_file.h5py_class,
                      h5py.File)

        a_group = self.sfh5["/1.2/measurement"]
        self.assertIs(a_group.h5py_class,
                      h5py.Group)

        a_dataset = self.sfh5["/1.1/instrument/positioners/Sslit1 HOff"]
        self.assertIs(a_dataset.h5py_class,
                      h5py.Dataset)

    def testHeader(self):
        file_header = self.sfh5["/1.2/instrument/specfile/file_header"]
        scan_header = self.sfh5["/1.2/instrument/specfile/scan_header"]

        # File header has 10 lines
        self.assertEqual(len(file_header), 10)
        # 1.2 has 9 scan & mca header lines
        self.assertEqual(len(scan_header), 9)

        # line 4 of file header
        self.assertEqual(
                file_header[3],
                u"#C imaging  User = opid17")
        # line 4 of scan header
        scan_header = self.sfh5["25.1/instrument/specfile/scan_header"]

        self.assertEqual(
                scan_header[3],
                u"#P1 4.74255 6.197579 2.238283")

    def testLinks(self):
        self.assertTrue(numpy.array_equal(
            self.sfh5["/1.2/measurement/mca_0/data"],
            self.sfh5["/1.2/instrument/mca_0/data"])
        )
        self.assertTrue(numpy.array_equal(
            self.sfh5["/1.2/measurement/mca_0/info/data"],
            self.sfh5["/1.2/instrument/mca_0/data"])
        )
        self.assertTrue(numpy.array_equal(
            self.sfh5["/1.2/measurement/mca_0/info/channels"],
            self.sfh5["/1.2/instrument/mca_0/channels"])
        )
        self.assertEqual(self.sfh5["/1.2/measurement/mca_0/info/"].keys(),
                         self.sfh5["/1.2/instrument/mca_0/"].keys())

        self.assertEqual(self.sfh5["/1.2/measurement/mca_0/info/preset_time"],
                         self.sfh5["/1.2/instrument/mca_0/preset_time"])
        self.assertEqual(self.sfh5["/1.2/measurement/mca_0/info/live_time"],
                         self.sfh5["/1.2/instrument/mca_0/live_time"])
        self.assertEqual(self.sfh5["/1.2/measurement/mca_0/info/elapsed_time"],
                         self.sfh5["/1.2/instrument/mca_0/elapsed_time"])

    def testListScanIndices(self):
        self.assertEqual(list(self.sfh5.keys()),
                         ["1.1", "25.1", "1.2", "1000.1", "1001.1"])
        self.assertEqual(self.sfh5["1.2"].attrs,
                         {"NX_class": "NXentry", })

    def testMcaAbsent(self):
        def access_absent_mca():
            """This must raise a KeyError, because scan 1.1 has no MCA"""
            return self.sfh5["/1.1/measurement/mca_0/"]
        self.assertRaises(KeyError, access_absent_mca)

    def testMcaCalib(self):
        mca0_calib = self.sfh5["/1.2/measurement/mca_0/info/calibration"]
        mca1_calib = self.sfh5["/1.2/measurement/mca_1/info/calibration"]
        self.assertEqual(mca0_calib.tolist(),
                         [1, 2, 3])
        # calibration is unique in this scan and applies to all analysers
        self.assertEqual(mca0_calib.tolist(),
                         mca1_calib.tolist())

    def testMcaChannels(self):
        mca0_chann = self.sfh5["/1.2/measurement/mca_0/info/channels"]
        mca1_chann = self.sfh5["/1.2/measurement/mca_1/info/channels"]
        self.assertEqual(mca0_chann.tolist(),
                         [0, 1, 2])
        self.assertEqual(mca0_chann.tolist(),
                         mca1_chann.tolist())

    def testMcaCtime(self):
        """Tests for #@CTIME mca header"""
        datasets = ["preset_time", "live_time", "elapsed_time"]
        for ds in datasets:
            self.assertNotIn("/1.1/instrument/mca_0/" + ds, self.sfh5)
            self.assertIn("/1.2/instrument/mca_0/" + ds, self.sfh5)

        mca0_preset_time = self.sfh5["/1.2/instrument/mca_0/preset_time"]
        mca1_preset_time = self.sfh5["/1.2/instrument/mca_1/preset_time"]
        self.assertLess(mca0_preset_time - 123.4,
                        10**-5)
        # ctime is unique in a this scan and applies to all analysers
        self.assertEqual(mca0_preset_time,
                         mca1_preset_time)

        mca0_live_time = self.sfh5["/1.2/instrument/mca_0/live_time"]
        mca1_live_time = self.sfh5["/1.2/instrument/mca_1/live_time"]
        self.assertLess(mca0_live_time - 234.5,
                        10**-5)
        self.assertEqual(mca0_live_time,
                         mca1_live_time)

        mca0_elapsed_time = self.sfh5["/1.2/instrument/mca_0/elapsed_time"]
        mca1_elapsed_time = self.sfh5["/1.2/instrument/mca_1/elapsed_time"]
        self.assertLess(mca0_elapsed_time - 345.6,
                        10**-5)
        self.assertEqual(mca0_elapsed_time,
                         mca1_elapsed_time)

    def testMcaData(self):
        # sum 1st MCA in scan 1.2 over rows
        mca_0_data = self.sfh5["/1.2/measurement/mca_0/data"]
        for summed_row, expected in zip(mca_0_data.sum(axis=1).tolist(),
                                        [3.0, 12.1, 21.7]):
            self.assertAlmostEqual(summed_row, expected, places=4)

        # sum 3rd MCA in scan 1.2 along both axis
        mca_2_data = self.sfh5["1.2"]["measurement"]["mca_2"]["data"]
        self.assertAlmostEqual(sum(sum(mca_2_data)), 9.1, places=5)
        # attrs
        self.assertEqual(mca_0_data.attrs, {"interpretation": "spectrum"})

    def testMotorPosition(self):
        positioners_group = self.sfh5["/1.1/instrument/positioners"]
        # MRTSlit DOWN position is defined in #P0 san header line
        self.assertAlmostEqual(float(positioners_group["MRTSlit DOWN"]),
                               0.87125)
        # MRTSlit UP position is defined in first data column
        for a, b in zip(positioners_group["MRTSlit UP"].tolist(),
                        [-1.23, 8.478100E+01, 3.14, 1.2]):
            self.assertAlmostEqual(float(a), b, places=4)

    def testNumberMcaAnalysers(self):
        """Scan 1.2 has 2 data columns + 3 mca spectra per data line."""
        self.assertEqual(len(self.sfh5["1.2"]["measurement"]), 5)

    def testTitle(self):
        self.assertEqual(self.sfh5["/25.1/title"],
                         u"ascan  c3th 1.33245 1.52245  40 0.15")

    def testValues(self):
        group = self.sfh5["/25.1"]
        self.assertTrue(hasattr(group, "values"))
        self.assertTrue(callable(group.values))
        self.assertIn(self.sfh5["/25.1/title"],
                      self.sfh5["/25.1"].values())

    # visit and visititems ignore links
    def testVisit(self):
        name_list = []
        self.sfh5.visit(name_list.append)
        self.assertIn('1.2/instrument/positioners/Pslit HGap', name_list)
        self.assertIn("1.2/instrument/specfile/scan_header", name_list)
        self.assertEqual(len(name_list), 117)

        # test also visit of a subgroup, with various group name formats
        name_list_leading_and_trailing_slash = []
        self.sfh5['/1.2/instrument/'].visit(name_list_leading_and_trailing_slash.append)
        name_list_leading_slash = []
        self.sfh5['/1.2/instrument'].visit(name_list_leading_slash.append)
        name_list_trailing_slash = []
        self.sfh5['1.2/instrument/'].visit(name_list_trailing_slash.append)
        name_list_no_slash = []
        self.sfh5['1.2/instrument'].visit(name_list_no_slash.append)

        # no differences expected in the output names
        self.assertEqual(name_list_leading_and_trailing_slash,
                         name_list_leading_slash)
        self.assertEqual(name_list_leading_slash,
                         name_list_trailing_slash)
        self.assertEqual(name_list_leading_slash,
                         name_list_no_slash)
        self.assertIn("positioners/Pslit HGap", name_list_no_slash)
        self.assertIn("positioners", name_list_no_slash)

    def testVisitItems(self):
        dataset_name_list = []

        def func_generator(l):
            """return a function appending names to list l"""
            def func(name, obj):
                if isinstance(obj, SpecH5Dataset):
                    l.append(name)
            return func

        self.sfh5.visititems(func_generator(dataset_name_list))
        self.assertIn('1.2/instrument/positioners/Pslit HGap', dataset_name_list)
        self.assertEqual(len(dataset_name_list), 85)

        # test also visit of a subgroup, with various group name formats
        name_list_leading_and_trailing_slash = []
        self.sfh5['/1.2/instrument/'].visititems(func_generator(name_list_leading_and_trailing_slash))
        name_list_leading_slash = []
        self.sfh5['/1.2/instrument'].visititems(func_generator(name_list_leading_slash))
        name_list_trailing_slash = []
        self.sfh5['1.2/instrument/'].visititems(func_generator(name_list_trailing_slash))
        name_list_no_slash = []
        self.sfh5['1.2/instrument'].visititems(func_generator(name_list_no_slash))

        # no differences expected in the output names
        self.assertEqual(name_list_leading_and_trailing_slash,
                         name_list_leading_slash)
        self.assertEqual(name_list_leading_slash,
                         name_list_trailing_slash)
        self.assertEqual(name_list_leading_slash,
                         name_list_no_slash)
        self.assertIn("positioners/Pslit HGap", name_list_no_slash)

    def testNotSpecH5(self):
        fd, fname = tempfile.mkstemp()
        os.write(fd, b"Not a spec file!")
        os.close(fd)
        self.assertRaises(specfile.SfErrFileOpen, SpecH5, fname)
        self.assertRaises(IOError, SpecH5, fname)
        os.unlink(fname)

    def testSample(self):
        self.assertNotIn("sample", self.sfh5["/1.1"])
        self.assertIn("sample", self.sfh5["/1000.1"])
        self.assertIn("ub_matrix", self.sfh5["/1000.1/sample"])
        self.assertIn("unit_cell", self.sfh5["/1000.1/sample"])
        self.assertIn("unit_cell_abc", self.sfh5["/1000.1/sample"])
        self.assertIn("unit_cell_alphabetagamma", self.sfh5["/1000.1/sample"])

        # All 0 values
        self.assertNotIn("sample", self.sfh5["/1001.1"])
        with self.assertRaises(KeyError):
            self.sfh5["/1001.1/sample/unit_cell"]

    @testutils.validate_logging(spech5.logger1.name, warning=2)
    def testOpenFileDescriptor(self):
        """Open a SpecH5 file from a file descriptor"""
        with io.open(self.sfh5.filename) as f:
            sfh5 = SpecH5(f)
            self.assertIsNotNone(sfh5)
            name_list = []
            # check if the object is working
            self.sfh5.visit(name_list.append)
            sfh5.close()


sftext_multi_mca_headers = """
#S 1 aaaaaa
#@MCA %16C
#@CHANN 3 0 2 1
#@CALIB 1 2 3
#@CTIME 123.4 234.5 345.6
#@MCA %16C
#@CHANN 3 1 3 1
#@CALIB 5.5 6.6 7.7
#@CTIME 10 11 12
#N 3
#L uno  duo
1 2
@A 0 1 2
@A 10 9 8
3 4
@A 3.1 4 5
@A 7 6 5
5 6
@A 6 7.7 8
@A 4 3 2

"""


class TestSpecH5MultiMca(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fd, cls.fname = tempfile.mkstemp(text=False)
        if sys.version_info < (3, ):
            os.write(fd, sftext_multi_mca_headers)
        else:
            os.write(fd, bytes(sftext_multi_mca_headers, 'ascii'))
        os.close(fd)

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls.fname)

    def setUp(self):
        self.sfh5 = SpecH5(self.fname)

    def tearDown(self):
        self.sfh5.close()

    def testMcaCalib(self):
        mca0_calib = self.sfh5["/1.1/measurement/mca_0/info/calibration"]
        mca1_calib = self.sfh5["/1.1/measurement/mca_1/info/calibration"]
        self.assertEqual(mca0_calib.tolist(),
                         [1, 2, 3])
        self.assertAlmostEqual(sum(mca1_calib.tolist()),
                               sum([5.5, 6.6, 7.7]),
                               places=5)

    def testMcaChannels(self):
        mca0_chann = self.sfh5["/1.1/measurement/mca_0/info/channels"]
        mca1_chann = self.sfh5["/1.1/measurement/mca_1/info/channels"]
        self.assertEqual(mca0_chann.tolist(),
                         [0., 1., 2.])
        # @CHANN is unique in this scan and applies to all analysers
        self.assertEqual(mca1_chann.tolist(),
                         [1., 2., 3.])

    def testMcaCtime(self):
        """Tests for #@CTIME mca header"""
        mca0_preset_time = self.sfh5["/1.1/instrument/mca_0/preset_time"]
        mca1_preset_time = self.sfh5["/1.1/instrument/mca_1/preset_time"]
        self.assertLess(mca0_preset_time - 123.4,
                        10**-5)
        self.assertLess(mca1_preset_time - 10,
                        10**-5)

        mca0_live_time = self.sfh5["/1.1/instrument/mca_0/live_time"]
        mca1_live_time = self.sfh5["/1.1/instrument/mca_1/live_time"]
        self.assertLess(mca0_live_time - 234.5,
                        10**-5)
        self.assertLess(mca1_live_time - 11,
                        10**-5)

        mca0_elapsed_time = self.sfh5["/1.1/instrument/mca_0/elapsed_time"]
        mca1_elapsed_time = self.sfh5["/1.1/instrument/mca_1/elapsed_time"]
        self.assertLess(mca0_elapsed_time - 345.6,
                        10**-5)
        self.assertLess(mca1_elapsed_time - 12,
                        10**-5)


sftext_no_cols = r"""#F C:/DATA\test.mca
#D Thu Jul  7 08:40:19 2016

#S 1 31oct98.dat 22.1 If4
#D Thu Jul  7 08:40:19 2016
#C no data cols, one mca analyser, single spectrum
#@MCA %16C
#@CHANN 151 0 150 1
#@CALIB 0 2 0
@A 789 784 788 814 847 862 880 904 925 955 987 1015 1031 1070 1111 1139 \
1203 1236 1290 1392 1492 1558 1688 1813 1977 2119 2346 2699 3121 3542 4102 4970 \
6071 7611 10426 16188 28266 40348 50539 55555 56162 54162 47102 35718 24588 17034 12994 11444 \
11808 13461 15687 18885 23827 31578 41999 49556 58084 59415 59456 55698 44525 28219 17680 12881 \
9518 7415 6155 5246 4646 3978 3612 3299 3020 2761 2670 2472 2500 2310 2286 2106 \
1989 1890 1782 1655 1421 1293 1135 990 879 757 672 618 532 488 445 424 \
414 373 351 325 307 284 270 247 228 213 199 187 183 176 164 156 \
153 140 142 130 118 118 103 101 97 86 90 86 87 81 75 82 \
80 76 77 75 76 77 62 69 74 60 65 68 65 58 63 64 \
63 59 60 56 57 60 55

#S 2 31oct98.dat 22.1 If4
#D Thu Jul  7 08:40:19 2016
#C no data cols, one mca analyser, multiple spectra
#@MCA %16C
#@CHANN 3 0 2 1
#@CALIB 1 2 3
#@CTIME 123.4 234.5 345.6
@A 0 1 2
@A 10 9 8
@A 1 1 1.1
@A 3.1 4 5
@A 7 6 5
@A 1 1 1
@A 6 7.7 8
@A 4 3 2
@A 1 1 1

#S 3 31oct98.dat 22.1 If4
#D Thu Jul  7 08:40:19 2016
#C no data cols, 3 mca analysers, multiple spectra
#@MCADEV 1
#@MCA %16C
#@CHANN 3 0 2 1
#@CALIB 1 2 3
#@CTIME 123.4 234.5 345.6
#@MCADEV 2
#@MCA %16C
#@CHANN 3 0 2 1
#@CALIB 1 2 3
#@CTIME 123.4 234.5 345.6
#@MCADEV 3
#@MCA %16C
#@CHANN 3 0 2 1
#@CALIB 1 2 3
#@CTIME 123.4 234.5 345.6
@A 0 1 2
@A 10 9 8
@A 1 1 1.1
@A 3.1 4 5
@A 7 6 5
@A 1 1 1
@A 6 7.7 8
@A 4 3 2
@A 1 1 1
"""


class TestSpecH5NoDataCols(unittest.TestCase):
    """Test reading SPEC files with only MCA data"""
    @classmethod
    def setUpClass(cls):
        fd, cls.fname = tempfile.mkstemp()
        if sys.version_info < (3, ):
            os.write(fd, sftext_no_cols)
        else:
            os.write(fd, bytes(sftext_no_cols, 'ascii'))
        os.close(fd)

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls.fname)

    def setUp(self):
        self.sfh5 = SpecH5(self.fname)

    def tearDown(self):
        self.sfh5.close()

    def testScan1(self):
        # 1.1: single analyser, single spectrum, 151 channels
        self.assertIn("mca_0",
                      self.sfh5["1.1/instrument/"])
        self.assertEqual(self.sfh5["1.1/instrument/mca_0/data"].shape,
                         (1, 151))
        self.assertNotIn("mca_1",
                         self.sfh5["1.1/instrument/"])

    def testScan2(self):
        # 2.1: single analyser, 9 spectra, 3 channels
        self.assertIn("mca_0",
                      self.sfh5["2.1/instrument/"])
        self.assertEqual(self.sfh5["2.1/instrument/mca_0/data"].shape,
                         (9, 3))
        self.assertNotIn("mca_1",
                         self.sfh5["2.1/instrument/"])

    def testScan3(self):
        # 3.1: 3 analysers, 3 spectra/analyser, 3 channels
        for i in range(3):
            self.assertIn("mca_%d" % i,
                          self.sfh5["3.1/instrument/"])
            self.assertEqual(
                self.sfh5["3.1/instrument/mca_%d/data" % i].shape,
                (3, 3))

        self.assertNotIn("mca_3",
                         self.sfh5["3.1/instrument/"])


sf_text_slash = r"""#F /data/id09/archive/logspecfiles/laue/2016/scan_231_laue_16-11-29.dat
#D Sat Dec 10 22:20:59 2016
#O0 Pslit/HGap  MRTSlit%UP

#S 1 laue_16-11-29.log 231.1 PD3/A
#D Sat Dec 10 22:20:59 2016
#P0 180.005 -0.66875
#N 2
#L GONY/mm  PD3%A
-2.015  5.250424e-05
-2.01  5.30798e-05
-2.005  5.281903e-05
-2  5.220436e-05
"""


class TestSpecH5SlashInLabels(unittest.TestCase):
    """Test reading SPEC files with labels containing a / character

    The / character must be substituted with a %
    """
    @classmethod
    def setUpClass(cls):
        fd, cls.fname = tempfile.mkstemp()
        if sys.version_info < (3, ):
            os.write(fd, sf_text_slash)
        else:
            os.write(fd, bytes(sf_text_slash, 'ascii'))
        os.close(fd)

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls.fname)

    def setUp(self):
        self.sfh5 = SpecH5(self.fname)

    def tearDown(self):
        self.sfh5.close()

    def testLabels(self):
        """Ensure `/` is substituted with `%` and
        ensure legitimate `%` in names are still working"""
        self.assertEqual(list(self.sfh5["1.1/measurement/"].keys()),
                         ["GONY%mm", "PD3%A"])

        # substituted "%"
        self.assertIn("GONY%mm",
                      self.sfh5["1.1/measurement/"])
        self.assertNotIn("GONY/mm",
                         self.sfh5["1.1/measurement/"])
        self.assertAlmostEqual(self.sfh5["1.1/measurement/GONY%mm"][0],
                               -2.015, places=4)
        # legitimate "%"
        self.assertIn("PD3%A",
                      self.sfh5["1.1/measurement/"])

    def testMotors(self):
        """Ensure `/` is substituted with `%` and
        ensure legitimate `%` in names are still working"""
        self.assertEqual(list(self.sfh5["1.1/instrument/positioners"].keys()),
                         ["Pslit%HGap", "MRTSlit%UP"])
        # substituted "%"
        self.assertIn("Pslit%HGap",
                      self.sfh5["1.1/instrument/positioners"])
        self.assertNotIn("Pslit/HGap",
                         self.sfh5["1.1/instrument/positioners"])
        self.assertAlmostEqual(
                self.sfh5["1.1/instrument/positioners/Pslit%HGap"],
                180.005, places=4)
        # legitimate "%"
        self.assertIn("MRTSlit%UP",
                      self.sfh5["1.1/instrument/positioners"])


def testUnitCellUBMatrix(tmp_path):
    """Test unit cell (#G1) and UB matrix (#G3)"""
    file_path = tmp_path / "spec.dat"
    file_path.write_bytes(bytes("""
#S 1 OK
#G1 0 1 2 3 4 5
#G3 0 1 2 3 4 5 6 7 8
""", encoding="ascii"))
    with SpecH5(str(file_path)) as spech5:
        assert numpy.array_equal(
            spech5["/1.1/sample/ub_matrix"],
            numpy.arange(9).reshape(1, 3, 3))
        assert numpy.array_equal(
            spech5["/1.1/sample/unit_cell"], [[0, 1, 2, 3, 4, 5]])
        assert numpy.array_equal(
            spech5["/1.1/sample/unit_cell_abc"], [0, 1, 2])
        assert numpy.array_equal(
            spech5["/1.1/sample/unit_cell_alphabetagamma"], [3, 4, 5])


def testMalformedUnitCellUBMatrix(tmp_path):
    """Test malformed unit cell (#G1) and UB matrix (#G3): 1 value"""
    file_path = tmp_path / "spec.dat"
    file_path.write_bytes(bytes("""
#S 1 all malformed=0
#G1 0
#G3 0
""", encoding="ascii"))
    with SpecH5(str(file_path)) as spech5:
        assert "sample" not in spech5["1.1"]


def testMalformedUBMatrix(tmp_path):
    """Test malformed UB matrix (#G3): all zeros"""
    file_path = tmp_path / "spec.dat"
    file_path.write_bytes(bytes("""
#S 1 G3 all 0
#G1 0 1 2 3 4 5
#G3 0 0 0 0 0 0 0 0 0
""", encoding="ascii"))
    with SpecH5(str(file_path)) as spech5:
        assert "ub_matrix" not in spech5["/1.1/sample"]
        assert numpy.array_equal(
            spech5["/1.1/sample/unit_cell"], [[0, 1, 2, 3, 4, 5]])
        assert numpy.array_equal(
            spech5["/1.1/sample/unit_cell_abc"], [0, 1, 2])
        assert numpy.array_equal(
            spech5["/1.1/sample/unit_cell_alphabetagamma"], [3, 4, 5])


def testMalformedUnitCell(tmp_path):
    """Test malformed unit cell (#G1): missing values"""
    file_path = tmp_path / "spec.dat"
    file_path.write_bytes(bytes("""
#S 1 G1 malformed missing values
#G1 0 1 2
#G3 0 1 2 3 4 5 6 7 8
""", encoding="ascii"))
    with SpecH5(str(file_path)) as spech5:
        assert "unit_cell" not in spech5["/1.1/sample"]
        assert "unit_cell_abc" not in spech5["/1.1/sample"]
        assert "unit_cell_alphabetagamma" not in spech5["/1.1/sample"]
        assert numpy.array_equal(
            spech5["/1.1/sample/ub_matrix"],
            numpy.arange(9).reshape(1, 3, 3))
