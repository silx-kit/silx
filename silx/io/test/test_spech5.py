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
"""Tests for spech5"""
import gc
from numpy import float32, array_equal
import os
import sys
import tempfile
import unittest
import datetime
from functools import partial

from ..spech5 import (SpecH5, SpecH5Group,
                      SpecH5Dataset, spec_date_to_iso8601)

try:
    import h5py
except ImportError:
    h5py = None

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "19/08/2016"

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
        fd, cls.fname = tempfile.mkstemp(text=False)
        if sys.version < '3.0':
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
        # fix Win32 permission error when deleting temp file
        del self.sfh5
        gc.collect()

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
        # full path to element outside group (illegal)
        self.assertNotIn("/1.1/instrument/positioners/Sslit1 HOff",
                         self.sfh5["/1.1/measurement"])

    def testDataColumn(self):
        self.assertAlmostEqual(sum(self.sfh5["/1.2/measurement/duo"]),
                               12.0)
        self.assertAlmostEqual(
                sum(self.sfh5["1.1"]["measurement"]["MRTSlit UP"]),
                87.891, places=4)

    def testDate(self):
        # start time is in Iso8601 format
        self.assertEqual(self.sfh5["/1.1/start_time"],
                         b"2016-02-11T09:55:20")
        self.assertEqual(self.sfh5["25.1/start_time"],
                         b"2015-03-14T03:53:50")

    def testGet(self):
        """Test :meth:`SpecH5Group.get`"""
        # default value of param *default* is None
        self.assertIsNone(self.sfh5.get("toto"))
        self.assertEqual(self.sfh5["25.1"].get("toto", default=-3),
                         -3)

        self.assertEqual(self.sfh5.get("/1.1/start_time", default=-3),
                         b"2016-02-11T09:55:20")

        self.assertIs(self.sfh5["1.1"].get("start_time", getclass=True),
                      SpecH5Dataset)
        self.assertIs(self.sfh5["1.1"].get("instrument", getclass=True),
                      SpecH5Group)

        # spech5 does not define external link, so there is no way
        # a group can *get* a SpecH5 class

    def testGetItemGroup(self):
        group = self.sfh5["25.1"]["instrument"]
        self.assertEqual(group["positioners"].keys(),
                         ["Pslit HGap", "MRTSlit UP", "MRTSlit DOWN",
                          "Sslit1 VOff", "Sslit1 HOff", "Sslit1 VGap"])
        with self.assertRaises(KeyError):
            group["Holy Grail"]

    def testGetitemSpecH5(self):
        self.assertEqual(self.sfh5["/1.2/instrument/positioners"],
                         self.sfh5["1.2"]["instrument"]["positioners"])

    @unittest.skipIf(h5py is None, "test requires h5py (not installed)")
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
        # File header has 10 lines
        self.assertEqual(len(self.sfh5["/1.2/instrument/specfile/file_header"]), 10)
        # 1.2 has 9 scan & mca header lines
        self.assertEqual(len(self.sfh5["/1.2/instrument/specfile/scan_header"]), 9)
        # line 4 of file header
        self.assertEqual(
                self.sfh5["1.2/instrument/specfile/file_header"][3].rstrip(),
                b"#C imaging  User = opid17")
        # line 4 of scan header
        self.assertEqual(
                self.sfh5["25.1/instrument/specfile/scan_header"][3].rstrip(),
                b"#P1 4.74255 6.197579 2.238283")

    def testLinks(self):
        self.assertTrue(
            array_equal(self.sfh5["/1.2/measurement/mca_0/data"],
                        self.sfh5["/1.2/instrument/mca_0/data"])
        )
        self.assertTrue(
            array_equal(self.sfh5["/1.2/measurement/mca_0/info/data"],
                        self.sfh5["/1.2/instrument/mca_0/data"])
        )
        self.assertTrue(
            array_equal(self.sfh5["/1.2/measurement/mca_0/info/channels"],
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
        self.assertEqual(self.sfh5.keys(),
                         ["1.1", "25.1", "1.2"])
        self.assertEqual(self.sfh5["1.2"].attrs,
                         {"NX_class": "NXentry", })

    def testMcaCalib(self):
        mca0_calib = self.sfh5["/1.2/measurement/mca_0/info/calibration"]
        mca1_calib = self.sfh5["/1.2/measurement/mca_1/info/calibration"]
        self.assertEqual(mca0_calib.tolist(),
                         [1, 2, 3])
        # calibration is unique in a given scan and applies to all analysers
        self.assertEqual(mca0_calib.tolist(),
                         mca1_calib.tolist())

    def testMcaChannels(self):
        mca0_chann = self.sfh5["/1.2/measurement/mca_0/info/channels"]
        mca1_chann = self.sfh5["/1.2/measurement/mca_1/info/channels"]
        self.assertEqual(mca0_chann.tolist(),
                         [0., 1., 2.])
        # channels is unique in a given scan and applies to all analysers
        self.assertEqual(mca0_chann.tolist(),
                         mca1_chann.tolist())

        self.assertIs(mca0_chann.dtype.type,
                      float32)

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
        # ctime is unique in a given scan and applies to all analysers
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
                         b"25  ascan  c3th 1.33245 1.52245  40 0.15")

    # visit and visititems ignore links
    def testVisit(self):
        name_list = []
        self.sfh5.visit(name_list.append)
        self.assertIn('/1.2/instrument/positioners/Pslit HGap', name_list)
        self.assertIn("/1.2/instrument/specfile/scan_header", name_list)
        self.assertEqual(len(name_list), 78)

    def testVisitItems(self):
        dataset_name_list = []

        def func(name, obj):
            if isinstance(obj, SpecH5Dataset):
                dataset_name_list.append(name)

        self.sfh5.visititems(func)
        self.assertIn('/1.2/instrument/positioners/Pslit HGap', dataset_name_list)
        self.assertEqual(len(dataset_name_list), 57)


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestSpecH5))
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestSpecDate))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest="suite")
