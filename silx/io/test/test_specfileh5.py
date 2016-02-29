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
"""Tests for specfileh5r"""

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "25/02/2016"

import logging
import os
import sys
import tempfile
import unittest

logger1 = logging.getLogger('silx.io.test')

from silx.io.specfileh5 import SpecFileH5, SpecFileH5Group, SpecFileH5Dataset

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

#S 1 aaaaaa
#@MCADEV 1
#@MCA %16C
#@CHANN 3 0 2 1
#@CALIB 1 2 3
#N 3
#L uno  duo
1 2
@A 0 1 2
@A 10 9 8
@A 1 1 1
3 4
@A 3.1 4 5
@A 7 6 5
@A 1 1 1
5 6
@A 6 7.7 8
@A 4 3 2
@A 1 1 1
"""


class TestSpecFileH5(unittest.TestCase):
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
        self.sfh5 = SpecFileH5(self.fname)

    def tearDown(self):
        del self.sfh5
    #
    # def test_title(self):
    #     self.assertEqual(self.sfh5["/25.1/title"],
    #                      "25  ascan  c3th 1.33245 1.52245  40 0.15")
        
    def test_list_of_scan_indices(self):
        self.assertEqual(self.sfh5.keys(),
                         ["1.1", "25.1", "1.2"])

    def test_number_of_mca_analysers(self):
        """Scan 1.2 has 2 data columns + 3 mca spectra per data line"""
        self.assertEqual(len(self.sfh5["1.2"]["measurement"]),
                         5)

    def test_group_get_item(self):
        group = self.sfh5["1.2"]["instrument"]
        self.assertEqual(group["positioners"].keys(),
                         ["Pslit HGap", "MRTSlit UP", "MRTSlit DOWN",
                          "Sslit1 VOff", "Sslit1 HOff", "Sslit1 VGap"])
        with self.assertRaises(KeyError):
            group["Holy Grail"]

    def test_visit(self):
        name_list = []
        self.sfh5.visit(name_list.append)
        self.assertIn(u"Pslit HGap", name_list)
        self.assertEqual(len(name_list), 54)

    def test_visit_items(self):
        dataset_name_list = []
        def func(name, obj):
            if isinstance(obj, SpecFileH5Dataset):
                dataset_name_list.append(name)

        self.sfh5.visititems(func)
        self.assertIn(u"Pslit HGap", dataset_name_list)
        self.assertEqual(len(dataset_name_list), 39)


    #
    # def test_date(self):
    #     self.assertEqual(self.sfh5["/1.1/start_time"],
    #                     "Thu Feb 11 09:55:20 2016")



def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestSpecFileH5))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest="suite")
