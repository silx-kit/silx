# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016 European Synchrotron Radiation Facility
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
# ###########################################################################*/
__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "13/10/2016"

import unittest
import numpy
import tempfile
import os

from .. import ArrayTableWidget
from ...testutils import TestCaseQt
from silx.gui import qt

try:
    import h5py
except ImportError:
    h5py = None


class TestNumpyArrayWidget(TestCaseQt):
    """Basic test for ArrayTableWidget with a numpy array"""
    def setUp(self):
        super(TestNumpyArrayWidget, self).setUp()
        self.aw = ArrayTableWidget.ArrayTableWidget()

    def tearDown(self):
        del self.aw
        super(TestNumpyArrayWidget, self).tearDown()

    def testShow(self):
        """test for errors"""
        self.aw.show()
        self.qWaitForWindowExposed(self.aw)

    def testSetData0D(self):
        """test for errors"""
        a = 1
        self.aw.setArrayData(a)
        b = self.aw.getData(copy=True)

        self.assertTrue(numpy.array_equal(a, b))

    def testSetData1D(self):
        """test for errors"""
        a = [1, 2]
        self.aw.setArrayData(a)
        b = self.aw.getData(copy=True)

        self.assertTrue(numpy.array_equal(a, b))

    def testSetData4D(self):
        """test for errors"""
        a = numpy.reshape(numpy.linspace(0.213, 1.234, 10000),
                          (10, 10, 10, 10))
        self.aw.setArrayData(a)
        self.aw.setPerspective((1, 3))
        b = self.aw.getData(copy=True)

        self.assertTrue(numpy.array_equal(a, b))

    def testFlagEditable(self):
        self.aw.setArrayData([[0]])
        idx = self.aw.model.createIndex(0, 0)
        # model is editable
        self.assertTrue(
                self.aw.model.flags(idx) & qt.Qt.ItemIsEditable)


@unittest.skipIf(h5py is None, "Could not import h5py")
class TestH5pyArrayWidget(TestCaseQt):
    """Basic test for ArrayTableWidget with a dataset.

    Test flags, for dataset open in read-only or read-write modes"""
    def setUp(self):
        super(TestH5pyArrayWidget, self).setUp()
        self.aw = ArrayTableWidget.ArrayTableWidget()
        self.data = numpy.reshape(numpy.linspace(0.213, 1.234, 10000),
                                  (10, 10, 10, 10))
        # create an h5py file with a dataset
        self.tempdir = tempfile.mkdtemp()
        self.h5_fname = os.path.join(self.tempdir, "array.h5")
        h5f = h5py.File(self.h5_fname)
        h5f["my_array"] = self.data
        h5f.close()

    def tearDown(self):
        del self.aw
        os.unlink(self.h5_fname)
        os.rmdir(self.tempdir)
        super(TestH5pyArrayWidget, self).tearDown()

    def testShow(self):
        self.aw.show()
        self.qWaitForWindowExposed(self.aw)

    def _readAndSetData(self, mode):
        h5f = h5py.File(self.h5_fname, mode)
        a = h5f["my_array"]
        self.aw.setArrayData(a, copy=False)

    def testReadOnly(self):
        """Open H5 dataset in read-only mode, use a reference and not
        a copy. Ensure the model is not editable."""
        self._readAndSetData(mode="r")
        b = self.aw.getData(copy=False)
        self.assertTrue(numpy.array_equal(self.data, b))

        # model must not be editable
        idx = self.aw.model.createIndex(0, 0)
        self.assertFalse(
                self.aw.model.flags(idx) & qt.Qt.ItemIsEditable)

    def testReadWrite(self):
        self._readAndSetData(mode="r+")
        b = self.aw.getData(copy=True)
        self.assertTrue(numpy.array_equal(self.data, b))

        idx = self.aw.model.createIndex(0, 0)
        # model is editable
        self.assertTrue(
                self.aw.model.flags(idx) & qt.Qt.ItemIsEditable)


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestNumpyArrayWidget))
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestH5pyArrayWidget))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
