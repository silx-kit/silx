# /*##########################################################################
# Copyright (C) 2021 Timo Fuchs
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
"""Tests for fioh5"""
import numpy
import os
import io
import sys
import tempfile
import unittest
import datetime
import logging

from silx.utils import testutils

from .. import fioh5
from ..fioh5 import (FioH5, FioH5NodeDataset, is_fiofile, logger1, dtypeConverter)

import h5py

__authors__ = ["T. Fuchs"]
__license__ = "MIT"
__date__ = "15/10/2021"

fioftext = """
!
! Comments
!
%c
ascan omega 180.0 180.5 3:10/1 4
user username, acquisition started at Thu Dec 12 18:00:00 2021
sweep motor lag: 1.0e-03
channel 3: Detector
!
! Parameter
!
%p
channel3_exposure = 1.000000e+00
ScanName = ascan
!
! Data
!
%d
 Col 1 omega(encoder) DOUBLE
 Col 2 channel INTEGER
 Col 3 filename STRING
 Col 4 type STRING
 Col 5 unix time DOUBLE
 Col 6 enable BOOLEAN
 Col 7 time_s FLOAT
      179.998418821    3    00001    exposure    1576165741.20308    1    1.243
      180.048418821    3    00002    exposure    1576165742.20308    1    1.243
      180.098418821    3    00003    exposure    1576165743.20308    1    1.243
      180.148418821    3    00004    exposure    1576165744.20308    1    1.243
      180.198418821    3    00005    exposure    1576165745.20308    1    1.243
      180.248418821    3    00006    exposure    1576165746.20308    1    1.243
      180.298418821    3    00007    exposure    1576165747.20308    1    1.243
      180.348418821    3    00008    exposure    1576165748.20308    1    1.243
      180.398418821    3    00009    exposure    1576165749.20308    1    1.243
      180.448418821    3    00010    exposure    1576165750.20308    1    1.243
"""



class TestFioH5(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        #fd, cls.fname = tempfile.mkstemp()
        cls.fname_numbered = os.path.join(cls.temp_dir.name, "eh1scan_00005.fio")
        
        with open(cls.fname_numbered, 'w') as fiof:
            fiof.write(fioftext)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()
        del cls.temp_dir

    def setUp(self):
        self.fioh5 = FioH5(self.fname_numbered)
        
    def tearDown(self):
        self.fioh5.close()
        
    def testScanNumber(self):
        # scan number is parsed from the file name.
        self.assertIn("/5.1", self.fioh5)
        self.assertIn("5.1", self.fioh5)

    def testContainsFile(self):
        self.assertIn("/5.1/measurement", self.fioh5)
        self.assertNotIn("25.2", self.fioh5)
        # measurement is a child of a scan, full path would be required to
        # access from root level
        self.assertNotIn("measurement", self.fioh5)
        # Groups may or may not have a trailing /
        self.assertIn("/5.1/measurement/", self.fioh5)
        self.assertIn("/5.1/measurement", self.fioh5)
        # Datasets can't have a trailing /
        self.assertIn("/5.1/measurement/omega(encoder)", self.fioh5)
        self.assertNotIn("/5.1/measurement/omega(encoder)/", self.fioh5)
        # No gamma
        self.assertNotIn("/5.1/measurement/gamma", self.fioh5)
        
    def testContainsGroup(self):
        self.assertIn("measurement", self.fioh5["/5.1/"])
        self.assertIn("measurement", self.fioh5["/5.1"])
        self.assertIn("5.1", self.fioh5["/"])
        self.assertNotIn("5.2", self.fioh5["/"])
        self.assertIn("measurement/filename", self.fioh5["/5.1"])
        # illegal trailing "/" after dataset name
        self.assertNotIn("measurement/filename/",
                         self.fioh5["/5.1"])
        # full path to element in group (OK)
        self.assertIn("/5.1/measurement/filename",
                      self.fioh5["/5.1/measurement"])
                      
    def testDataType(self):
        meas = self.fioh5["/5.1/measurement/"]
        self.assertEqual(meas["omega(encoder)"].dtype, dtypeConverter['DOUBLE'])
        self.assertEqual(meas["channel"].dtype, dtypeConverter['INTEGER'])
        self.assertEqual(meas["filename"].dtype, dtypeConverter['STRING'])
        self.assertEqual(meas["time_s"].dtype, dtypeConverter['FLOAT'])
        self.assertEqual(meas["enable"].dtype, dtypeConverter['BOOLEAN'])
        
    def testDataColumn(self):
        self.assertAlmostEqual(sum(self.fioh5["/5.1/measurement/omega(encoder)"]),
                               1802.23418821)
        self.assertTrue(numpy.all(self.fioh5["/5.1/measurement/enable"]))
    
    # --- comment section tests ---
    
    def testComment(self):
        # should hold the complete comment section
        self.assertEqual(self.fioh5["/5.1/instrument/fiofile/comments"],
"""ascan omega 180.0 180.5 3:10/1 4
user username, acquisition started at Thu Dec 12 18:00:00 2021
sweep motor lag: 1.0e-03
channel 3: Detector
""")
    
    def testDate(self):
        # there is no convention on how to format the time. So just check its existence.
        self.assertEqual(self.fioh5["/5.1/start_time"],
                         u"Thu Dec 12 18:00:00 2021")
    
    def testTitle(self):
        self.assertEqual(self.fioh5["/5.1/title"],
                         u"ascan omega 180.0 180.5 3:10/1 4")
                         
                         
    # --- parameter section tests ---
    
    def testParameter(self):
        # should hold the complete parameter section
        self.assertEqual(self.fioh5["/5.1/instrument/fiofile/parameter"],
"""channel3_exposure = 1.000000e+00
ScanName = ascan
""")
    
    def testParsedParameter(self):
        # no dtype is given, so everything is str.
        self.assertEqual(self.fioh5["/5.1/instrument/parameter/channel3_exposure"],
                                    u"1.000000e+00")
        self.assertEqual(self.fioh5["/5.1/instrument/parameter/ScanName"], u"ascan")
    
    def testNotFioH5(self):
        testfilename = os.path.join(self.temp_dir.name, "eh1scan_00010.fio")
        with open(testfilename, 'w') as fiof:
            fiof.write("!Not a fio file!")

        self.assertRaises(IOError, FioH5, testfilename)
        
        self.assertTrue(is_fiofile(self.fname_numbered))
        self.assertFalse(is_fiofile(testfilename))
        
        os.unlink(testfilename)
    

class TestUnnumberedFioH5(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.fname_nosuffix = os.path.join(cls.temp_dir.name, "eh1scan_nosuffix.fio")
        
        with open(cls.fname_nosuffix, 'w') as fiof:
            fiof.write(fioftext)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()
        del cls.temp_dir
        
    def setUp(self):
        self.fioh5 = FioH5(self.fname_nosuffix)
        
    def testLogMissingScanno(self):
        with self.assertLogs(logger1,level='WARNING') as cm:
            fioh5 = FioH5(self.fname_nosuffix)
        self.assertIn("Cannot parse scan number of file", cm.output[0])
        
    def testFallbackName(self):
        self.assertIn("/eh1scan_nosuffix", self.fioh5)
        
brokenHeaderText = """
!
! Comments
!
%c
ascan omega 180.0 180.5 3:10/1 4
user username, acquisited at Thu Dec 12 100 2021
sweep motor lavgvf.0e-03
channel 3: Detector
!
! Parameter
!
%p
channel3_exposu65 1.000000e+00
ScanName = ascan
!
! Data
!
%d
 Col 1 omega(encoder) DOUBLE
 Col 2 channel INTEGER
 Col 3 filename STRING
 Col 4 type STRING
 Col 5 unix time DOUBLE
      179.998418821    3    00001    exposure    1576165741.20308
      180.048418821    3    00002    exposure    1576165742.20308
      180.098418821    3    00003    exposure    1576165743.20308
      180.148418821    3    00004    exposure    1576165744.20308
      180.198418821    3    00005    exposure    1576165745.20308
      180.248418821    3    00006    exposure    1576165746.20308
      180.298418821    3    00007    exposure    1576165747.20308
      180.348418821    3    00008    exposure    1576165748.20308
      180.398418821    3    00009    exposure    1576165749.20308
      180.448418821    3    00010    exposure    1576165750.20308
"""

class TestBrokenHeaderFioH5(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.fname_numbered = os.path.join(cls.temp_dir.name, "eh1scan_00005.fio")
        
        with open(cls.fname_numbered, 'w') as fiof:
            fiof.write(brokenHeaderText)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()
        del cls.temp_dir
        
    def setUp(self):
        self.fioh5 = FioH5(self.fname_numbered)
        
    def testLogBrokenHeader(self):
        with self.assertLogs(logger1,level='WARNING') as cm:
            fioh5 = FioH5(self.fname_numbered)
        self.assertIn("Cannot parse parameter section", cm.output[0])
        self.assertIn("Cannot parse default comment section", cm.output[1])
        
    def testComment(self):
        # should hold the complete comment section
        self.assertEqual(self.fioh5["/5.1/instrument/fiofile/comments"],
"""ascan omega 180.0 180.5 3:10/1 4
user username, acquisited at Thu Dec 12 100 2021
sweep motor lavgvf.0e-03
channel 3: Detector
""")

    def testParameter(self):
        # should hold the complete parameter section
        self.assertEqual(self.fioh5["/5.1/instrument/fiofile/parameter"],
"""channel3_exposu65 1.000000e+00
ScanName = ascan
""")
