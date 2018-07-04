# coding: utf-8
# /*##########################################################################
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
# ############################################################################*/
"""Tests for old specfile wrapper"""

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "15/05/2017"

import locale
import logging
import numpy
import os
import sys
import tempfile
import unittest

logger1 = logging.getLogger(__name__)

from ..specfilewrapper import Specfile

sftext = """#F /tmp/sf.dat
#E 1455180875
#D Thu Feb 11 09:54:35 2016
#C imaging  User = opid17
#U00 user comment first line
#U01 This is a dummy file to test SpecFile parsing
#U02
#U03 last line

#O0 Pslit HGap  MRTSlit UP  MRTSlit DOWN
#O1 Sslit1 VOff  Sslit1 HOff  Sslit1 VGap
#o0 pshg mrtu mrtd
#o2 ss1vo ss1ho ss1vg

#J0 Seconds  IA  ion.mono  Current
#J1 xbpmc2  idgap1  Inorm

#S 1  ascan  ss1vo -4.55687 -0.556875  40 0.2
#D Thu Feb 11 09:55:20 2016
#T 0.2  (Seconds)
#G0 0
#G1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#G3 0 0 0 0 0 0 0 0 0
#G4 0
#Q
#P0 180.005 -0.66875 0.87125
#P1 14.74255 16.197579 12.238283
#UMI0     Current AutoM      Shutter
#UMI1      192.51   OFF     FE open
#UMI2 Refill in 39883 sec, Fill Mode: uniform multibunch / Message: Feb 11 08:00 Delivery:Next Refill at 21:00;
#N 4
#L first column  second column  3rd_col
-1.23 5.89  8
8.478100E+01  5 1.56
3.14 2.73 -3.14
1.2 2.3 3.4

#S 25  ascan  c3th 1.33245 1.52245  40 0.15
#D Thu Feb 11 10:00:31 2016
#P0 80.005 -1.66875 1.87125
#P1 4.74255 6.197579 2.238283
#N 5
#L column0  column1  col2  col3
0.0 0.1 0.2 0.3
1.0 1.1 1.2 1.3
2.0 2.1 2.2 2.3
3.0 3.1 3.2 3.3

#F /tmp/sf.dat
#E 1455180876
#D Thu Feb 11 09:54:36 2016

#S 1 aaaaaa
#U first duplicate line
#U second duplicate line
#@MCADEV 1
#@MCA %16C
#@CHANN 3 0 2 1
#@CALIB 1 2 3
#N 3
#L uno  duo
1 2
@A 0 1 2
3 4
@A 3.1 4 5
5 6
@A 6 7.7 8
"""


class TestSpecfilewrapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fd, cls.fname1 = tempfile.mkstemp(text=False)
        if sys.version_info < (3, ):
            os.write(fd, sftext)
        else:
            os.write(fd, bytes(sftext, 'ascii'))
        os.close(fd)

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls.fname1)

    def setUp(self):
        self.sf = Specfile(self.fname1)
        self.scan1 = self.sf[0]
        self.scan1_2 = self.sf.select("1.2")
        self.scan25 = self.sf.select("25.1")

    def tearDown(self):
        self.sf.close()

    def test_number_of_scans(self):
        self.assertEqual(3, len(self.sf))

    def test_list_of_scan_indices(self):
        self.assertEqual(self.sf.list(),
                         '1,25,1')
        self.assertEqual(self.sf.keys(),
                         ["1.1", "25.1", "1.2"])

    def test_scan_headers(self):
        self.assertEqual(self.scan25.header('S'),
                         ["#S 25  ascan  c3th 1.33245 1.52245  40 0.15"])
        self.assertEqual(self.scan1.header("G0"), ['#G0 0'])
        # parsing headers with long keys
        # parsing empty headers
        self.assertEqual(self.scan1.header('Q'), ['#Q '])

    def test_file_headers(self):
        self.assertEqual(self.scan1.header("E"),
                         ['#E 1455180875'])
        self.assertEqual(self.sf.title(),
                         "imaging")
        self.assertEqual(self.sf.epoch(),
                         1455180875)
        self.assertEqual(self.sf.allmotors(),
                         ["Pslit HGap", "MRTSlit UP", "MRTSlit DOWN",
                          "Sslit1 VOff", "Sslit1 HOff", "Sslit1 VGap"])

    def test_scan_labels(self):
        self.assertEqual(self.scan1.alllabels(),
                         ['first column', 'second column', '3rd_col'])

    def test_data(self):
        self.assertAlmostEqual(self.scan1.dataline(3)[2],
                               -3.14)
        self.assertAlmostEqual(self.scan1.datacol(1)[2],
                               3.14)
        # tests for data transposition between original file and .data attr
        self.assertAlmostEqual(self.scan1.data()[2, 0],
                               8)
        self.assertEqual(self.scan1.data().shape, (3, 4))
        self.assertAlmostEqual(numpy.sum(self.scan1.data()), 113.631)

    def test_date(self):
        self.assertEqual(self.scan1.date(),
                         "Thu Feb 11 09:55:20 2016")

    def test_motors(self):
        self.assertEqual(len(self.sf.allmotors()), 6)
        self.assertEqual(len(self.scan1.allmotorpos()), 6)
        self.assertAlmostEqual(sum(self.scan1.allmotorpos()),
                               223.385912)
        self.assertEqual(self.sf.allmotors()[1], 'MRTSlit UP')

    def test_mca(self):
        self.assertEqual(self.scan1_2.mca(2)[2], 5)
        self.assertEqual(sum(self.scan1_2.mca(3)), 21.7)

    def test_mca_header(self):
        self.assertEqual(self.scan1_2.header("CALIB"),
                         ["#@CALIB 1 2 3"])


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestSpecfilewrapper))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest="suite")
