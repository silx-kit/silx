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
"""Test for silx.gui.hdf5 module"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "21/09/2017"


import unittest
import tempfile
import numpy
import shutil
from ..import rawh5


class TestNumpyFile(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmpDirectory = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpDirectory)

    def testNumpyFile(self):
        filename = "%s/%s.npy" % (self.tmpDirectory, self.id())
        c = numpy.random.rand(5, 5)
        numpy.save(filename, c)
        h5 = rawh5.NumpyFile(filename)
        self.assertIn("data", h5)
        self.assertEqual(h5["data"].dtype.kind, "f")

    def testNumpyZFile(self):
        filename = "%s/%s.npz" % (self.tmpDirectory, self.id())
        a = numpy.array(u"aaaaa")
        b = numpy.array([1, 2, 3, 4])
        c = numpy.random.rand(5, 5)
        d = numpy.array(b"aaaaa")
        e = numpy.array(u"i \u2661 my mother")
        numpy.savez(filename, a, b=b, c=c, d=d, e=e)
        h5 = rawh5.NumpyFile(filename)
        self.assertIn("arr_0", h5)
        self.assertIn("b", h5)
        self.assertIn("c", h5)
        self.assertIn("d", h5)
        self.assertIn("e", h5)
        self.assertEqual(h5["arr_0"].dtype.kind, "U")
        self.assertEqual(h5["b"].dtype.kind, "i")
        self.assertEqual(h5["c"].dtype.kind, "f")
        self.assertEqual(h5["d"].dtype.kind, "S")
        self.assertEqual(h5["e"].dtype.kind, "U")

    def testNumpyZFileContainingDirectories(self):
        filename = "%s/%s.npz" % (self.tmpDirectory, self.id())
        data = {}
        data['a/b/c'] = numpy.arange(10)
        data['a/b/e'] = numpy.arange(10)
        numpy.savez(filename, **data)
        h5 = rawh5.NumpyFile(filename)
        self.assertIn("a/b/c", h5)
        self.assertIn("a/b/e", h5)


def suite():
    test_suite = unittest.TestSuite()
    loadTests = unittest.defaultTestLoader.loadTestsFromTestCase
    test_suite.addTest(loadTests(TestNumpyFile))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
