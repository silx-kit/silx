"""Tests for specfile wrapper"""
import unittest
import tempfile
import os
import sys
import gc
import numpy

from silx.io.specfile import SpecFile, Scan

sftext = """
#F /tmp/sf.dat
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
#L column0  column1  col2 col3
0.0 0.1 0.2 0.3
1.0 1.1 1.2 1.3
2.0 2.1 2.2 2.3
3.0 3.1 3.2 3.3

#F /tmp/sf.dat
#E 1455180876
#D Thu Feb 11 09:54:36 2016

#S 1 aaaaaa
#N 2
#L uno duo
1 2
3 4
"""

class TestSpecFile(unittest.TestCase):
    def setUp(self):
        fd, tmp_path = tempfile.mkstemp(text=False)
        if sys.version < '3.0':
            os.write(fd, sftext)
        else:
            os.write(fd, bytes(sftext, 'utf-8'))
        os.close(fd)
        self.fname =tmp_path
        self.sf = SpecFile(self.fname)
        self.scan1 = self.sf[0]
        self.scan1_2 = self.sf["1.2"]
        self.scan25 = self.sf["25.1"]

    def tearDown(self):
        self.sf = None
        self.scan1 = None
        self.scan1_2 = None
        self.scan25 = None
        gc.collect()

    def test_open(self):
        self.assertIsInstance(self.sf, SpecFile)
        with self.assertRaises(IOError):
            sf2 = SpecFile("doesnt_exist.dat")
        
    def test_number_of_scans(self):
        self.assertEqual(3, len(self.sf))
        
    def test_list_of_scan_indices(self):
        # self.assertEqual(len(self.sf.list()), 3)
        # self.assertEqual(max(self.sf.list()), 25)
        # self.assertEqual(min(self.sf.list()), 1)
        self.assertEqual(self.sf.list(),
                         [1, 25, 1])

    def test_index_number_order(self):
        self.assertEqual(self.sf.index(1, 2), 2)  #sf["1.2"]==sf[2]
        self.assertEqual(self.sf.number(1), 25)   #sf[1]==sf["25"]
        self.assertEqual(self.sf.order(2), 2)     #sf[2]==sf["1.2"]
        with self.assertRaises(IndexError):
            self.sf.index(3, 2)
        with self.assertRaises(IndexError):
            self.sf.index(99)
        
    def test_getitem(self):
        self.assertIsInstance(self.sf[2], Scan)
        self.assertIsInstance(self.sf["1.2"], Scan)
        # int out of range
        with self.assertRaisesRegexp(IndexError, 'Scan index must be in ran'):
            self.sf[107]
        # float indexing not allowed
        with self.assertRaisesRegexp(TypeError, 'The scan identification k'):
            self.sf[1.2]
        # non existant scan with "N.M" indexing 
        with self.assertRaises(KeyError):
            self.sf["3.2"]

    def test_iterator(self):
        i=0
        for scan in self.sf:
            if i == 1:
                self.assertEqual(scan.motor_positions, self.sf[1].motor_positions)
            i += 1
        # number of returned scans
        self.assertEqual(i, len(self.sf))

    def test_scan_index(self):
        self.assertEqual(self.scan1.index, 0)
        self.assertEqual(self.scan1_2.index, 2)
        self.assertEqual(self.scan25.index, 1)

    def test_scan_headers(self):
        self.assertEqual(self.sf[0].header_dict['S'],
                         self.sf["1.1"].header_dict['S'])
        self.assertEqual(self.scan25.header_dict['S'],
                         "25  ascan  c3th 1.33245 1.52245  40 0.15")
        self.assertEqual(self.scan1.header_dict['N'], '4')
        self.assertEqual(self.scan1.header_lines[3], '#G0 0')
        self.assertEqual(len(self.scan1.header_lines), 15)
        
    def test_file_headers(self):
        self.assertEqual(self.scan1.file_header_lines[1],
                         '#E 1455180875')
        self.assertEqual(self.scan25.file_header_lines[1],   # fails. why?
                         '#E 1455180876')
        self.assertEqual(len(self.scan1.file_header_lines), 14)
        # parsing headers with single character key 
        self.assertEqual(self.scan1.file_header_dict['F'],
                         '/tmp/sf.dat')
        # parsing headers with long keys  
        self.assertEqual(self.scan1.file_header_dict['UMI0'],
                         'Current AutoM      Shutter')
        # parsing empty headers
        self.assertEqual(self.scan1.file_header_dict['Q'], '')
        
    def test_scan_labels(self):
        self.assertEqual(self.scan1.labels,
                         ['first column', 'second column', '3rd_col'])

    def test_data(self):
        # assertAlmostEqual compares 7 decimal places by default
        self.assertAlmostEqual(self.scan1.data_line(1)[2],
                               1.56)
        self.assertEqual(self.scan1.data.shape, (4, 3))
        self.assertEqual(self.scan1.nlines, 4)
        self.assertEqual(self.scan1.ncolumns, 3)
        self.assertAlmostEqual(numpy.sum(self.scan1.data), 113.631)
        
    def test_motors(self):
        self.assertEqual(len(self.scan1.motor_names), 6)
        self.assertEqual(len(self.scan1.motor_positions), 6)
        self.assertAlmostEqual(sum(self.scan1.motor_positions),
                               223.385912)
        self.assertEqual(self.scan1.motor_names[1], 'MRTSlit UP')


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestSpecFile))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest="suite")
