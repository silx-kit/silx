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
"""Tests for SpecFile to HDF5 converter"""


__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "09/03/2016"

import gc
import h5py
from numpy import array_equal
import os
import re
import sys
import tempfile
import unittest

from silx.io.specfileh5 import SpecFileH5
from silx.io.convert_spec_h5 import convert

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

#S 1 aaaaaa
#D Thu Feb 11 10:00:32 2016
#@MCADEV 1
#@MCA %16C
#@CHANN 3 0 2 1
#@CALIB 1 2 3
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

mca_link_pattern = re.compile(r"/[0-9]+\.[0-9]+/measurement/mca_[0-9]+")

class TestConvertSpecHDF5(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fd, cls.spec_fname = tempfile.mkstemp(text=False)
        if sys.version < '3.0':
            os.write(fd, sftext)
        else:
            os.write(fd, bytes(sftext, 'ascii'))
        os.close(fd)

        fd, cls.h5_fname = tempfile.mkstemp(text=False)
        # Close and delete (we just need the name)
        os.close(fd)
        os.unlink(cls.h5_fname)

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls.spec_fname)
        os.unlink(cls.h5_fname)

    def setUp(self):
        convert(self.spec_fname, self.h5_fname)

        self.sfh5 = SpecFileH5(self.spec_fname)
        self.h5f = h5py.File(self.h5_fname)

    def tearDown(self):
        del self.sfh5
        self.h5f.close()
        del self.h5f
        gc.collect()

    def test_HDF5_has_same_members(self):
        spec_member_list = []
        def append_spec_members(name):
            spec_member_list.append(name)
        self.sfh5.visit(append_spec_members)

        hdf5_member_list = []
        def append_hdf5_members(name):
            hdf5_member_list.append(name)
        self.h5f.visit(append_hdf5_members)

        # 1. For some reason, h5py visit method doesn't include the leading
        # "/" character when it passes the member name to the function,
        # even though an explicit the .name attribute of a member will
        # have a leading "/"
        # 2. h5py visit method apparently does not follow links
        spec_member_list = [m.lstrip("/") for m in spec_member_list
                            if not mca_link_pattern.match(m)]

        self.assertEqual(set(spec_member_list),
                         set(hdf5_member_list))

    def test_links(self):
        self.assertTrue(
            array_equal(self.sfh5["/1.2/measurement/mca_0/data"],
                        self.h5f["/1.2/measurement/mca_0/data"])
        )
        self.assertTrue(
            array_equal(self.h5f["/1.2/instrument/mca_1/channels"],
                        self.h5f["/1.2/measurement/mca_1/info/channels"])
        )


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestConvertSpecHDF5))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest="suite")
