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
"""
Tests for the octaveh5 module
"""

__authors__ = ["C. Nemoz", "H. Payno"]
__license__ = "MIT"
__date__ = "12/07/2016"

import unittest
import os
import tempfile

try:
    from ..octaveh5 import Octaveh5
except ImportError:
    Octaveh5 = None


@unittest.skipIf(Octaveh5 is None, "Could not import h5py")
class TestOctaveH5(unittest.TestCase):
    @staticmethod
    def _get_struct_FT():
        return { 
            'NO_CHECK': 0.0, 'SHOWSLICE': 1.0, 'DOTOMO': 1.0, 'DATABASE': 0.0, 'ANGLE_OFFSET': 0.0, 
            'VOLSELECTION_REMEMBER': 0.0, 'NUM_PART': 4.0, 'VOLOUTFILE': 0.0, 'RINGSCORRECTION': 0.0, 
            'DO_TEST_SLICE': 1.0, 'ZEROOFFMASK': 1.0, 'VERSION': 'fastomo3 version 2.0', 
            'CORRECT_SPIKES_THRESHOLD': 0.040000000000000001, 'SHOWPROJ': 0.0, 'HALF_ACQ': 0.0, 
            'ANGLE_OFFSET_VALUE': 0.0, 'FIXEDSLICE': 'middle', 'VOLSELECT': 'total' }
    @staticmethod
    def _get_struct_PYHSTEXE():
        return {
            'EXE': 'PyHST2_2015d', 'VERBOSE': 0.0, 'OFFV': 'PyHST2_2015d', 'TOMO': 0.0, 
            'VERBOSE_FILE': 'pyhst_out.txt', 'DIR': '/usr/bin/', 'OFFN': 'pyhst2'}

    @staticmethod
    def _get_struct_FTAXIS():
        return {
            'POSITION_VALUE': 12345.0, 'COR_ERROR': 0.0, 'FILESDURINGSCAN': 0.0, 'PLOTFIGURE': 1.0,
            'DIM1': 0.0, 'OVERSAMPLING': 5.0, 'TO_THE_CENTER': 1.0, 'POSITION': 'fixed', 
            'COR_POSITION': 0.0, 'HA': 0.0 }
            
    @staticmethod
    def _get_struct_PAGANIN():
        return {
            'MKEEP_MASK': 0.0, 'UNSHARP_SIGMA': 0.80000000000000004, 'DILATE': 2.0, 'UNSHARP_COEFF': 3.0, 
            'MEDIANR': 4.0, 'DB': 500.0, 'MKEEP_ABS': 0.0, 'MODE': 0.0, 'THRESHOLD': 0.5, 
            'MKEEP_BONE': 0.0, 'DB2': 100.0, 'MKEEP_CORR': 0.0, 'MKEEP_SOFT': 0.0 }

    @staticmethod
    def _get_struct_BEAMGEO():
        return {'DIST': 55.0, 'SY': 0.0, 'SX': 0.0, 'TYPE': 'p'}


    def setUp(self):
        self.tempdir = tempfile.mkdtemp()        
        self.test_3_6_fname = os.path.join(self.tempdir, "silx_tmp_t00_octaveTest_3_6.h5")
        self.test_3_8_fname = os.path.join(self.tempdir, "silx_tmp_t00_octaveTest_3_8.h5")

    def tearDown(self):
        if os.path.isfile(self.test_3_6_fname):
            os.unlink(self.test_3_6_fname)
        if os.path.isfile(self.test_3_8_fname):
            os.unlink(self.test_3_8_fname)

    def testWritedIsReaded(self):
        """
        Simple test to write and reaf the structure compatible with the octave h5 using structure.
        This test is for # test for octave version > 3.8
        """   
        writer = Octaveh5()

        writer.open(self.test_3_8_fname, 'a')
        # step 1 writing the file
        writer.write('FT', self._get_struct_FT())
        writer.write('PYHSTEXE', self._get_struct_PYHSTEXE())
        writer.write('FTAXIS', self._get_struct_FTAXIS())
        writer.write('PAGANIN', self._get_struct_PAGANIN())
        writer.write('BEAMGEO', self._get_struct_BEAMGEO())
        writer.close()

        # step 2 reading the file
        reader = Octaveh5().open(self.test_3_8_fname)
        #  2.1 check FT
        data_readed = reader.get('FT')
        self.assertEqual(data_readed, self._get_struct_FT() )
        #  2.2 check PYHSTEXE
        data_readed = reader.get('PYHSTEXE')
        self.assertEqual(data_readed, self._get_struct_PYHSTEXE() )
        #  2.3 check FTAXIS
        data_readed = reader.get('FTAXIS')
        self.assertEqual(data_readed, self._get_struct_FTAXIS() )
        #  2.4 check PAGANIN
        data_readed = reader.get('PAGANIN')
        self.assertEqual(data_readed, self._get_struct_PAGANIN() )
        #  2.5 check BEAMGEO
        data_readed = reader.get('BEAMGEO')
        self.assertEqual(data_readed, self._get_struct_BEAMGEO() )
        reader.close()

    def testWritedIsReadedOldOctaveVersion(self):
        """The same test as testWritedIsReaded but for octave version < 3.8
        """
        # test for octave version < 3.8
        writer = Octaveh5(3.6)

        writer.open(self.test_3_6_fname, 'a')

        # step 1 writing the file
        writer.write('FT', self._get_struct_FT())
        writer.write('PYHSTEXE', self._get_struct_PYHSTEXE())
        writer.write('FTAXIS', self._get_struct_FTAXIS())
        writer.write('PAGANIN', self._get_struct_PAGANIN())
        writer.write('BEAMGEO', self._get_struct_BEAMGEO())
        writer.close()

        # step 2 reading the file
        reader = Octaveh5(3.6).open(self.test_3_6_fname)
        #  2.1 check FT
        data_readed = reader.get('FT')
        self.assertEqual(data_readed, self._get_struct_FT() )
        #  2.2 check PYHSTEXE
        data_readed = reader.get('PYHSTEXE')
        self.assertEqual(data_readed, self._get_struct_PYHSTEXE() )
        #  2.3 check FTAXIS
        data_readed = reader.get('FTAXIS')
        self.assertEqual(data_readed, self._get_struct_FTAXIS() )
        #  2.4 check PAGANIN
        data_readed = reader.get('PAGANIN')
        self.assertEqual(data_readed, self._get_struct_PAGANIN() )
        #  2.5 check BEAMGEO
        data_readed = reader.get('BEAMGEO')
        self.assertEqual(data_readed, self._get_struct_BEAMGEO() )
        reader.close()

def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestOctaveH5))
    return test_suite

if __name__ == '__main__':
    unittest.main(defaultTest="suite")
