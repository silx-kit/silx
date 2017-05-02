# coding: utf-8
# /*##########################################################################
# Copyright (C) 2017 European Synchrotron Radiation Facility
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
"""Tests of the median filter"""

__authors__ = ["H. Payno"]
__license__ = "MIT"
__date__ = "02/05/2017"

import unittest
import numpy
import tempfile
import os
from  silx.math.medianfilter import medfilt2d
from silx.test.utils import ParametricTestCase

try:
    import scipy
except:
    scipy = None
else:
    import scipy.ndimage

import logging
_logger = logging.getLogger(__name__)


class Test2DFilter(unittest.TestCase):
    """Some unit tests for the median filter"""

    def testFilter3_100(self):
        """Test median filter on a 10x10 matrix with a 3x3 kernel."""
        dataIn = numpy.arange(100, dtype=numpy.int32)
        dataIn = dataIn.reshape((10,10))

        dataOut = medfilt2d(image=dataIn,
                            kernel_size=(3, 3),
                            conditional=False)
        self.assertTrue(dataOut[0, 0] == 1)
        self.assertTrue(dataOut[9, 0] == 90)
        self.assertTrue(dataOut[9, 9] == 98)

        self.assertTrue(dataOut[0, 9] == 9)
        self.assertTrue(dataOut[0, 4] == 5)
        self.assertTrue(dataOut[9, 4] == 93)
        self.assertTrue(dataOut[4, 4] == 44)

    def testFilter3_9(self):
        "Test median filter on a 3x3 matrix a 3x3 kernel."
        dataIn = numpy.array([0, -1, 1,
                              12, 6, -2,
                              100, 4, 12],
                             dtype=numpy.int16)
        dataIn = dataIn.reshape((3, 3))
        dataOut = medfilt2d(image=dataIn,
                            kernel_size=(3, 3),
                            conditional=False)
        self.assertTrue(dataOut.shape == dataIn.shape)
        self.assertTrue(dataOut[1, 1] == 4)
        self.assertTrue(dataOut[0, 0] == 0)
        self.assertTrue(dataOut[0, 1] == 0)
        self.assertTrue(dataOut[1, 0] == 6)


    def testFilterWidthOne(self):
        """Make sure a filter of one by one give the same result as the input"""
        dataIn = numpy.arange(100, dtype=numpy.int32)
        dataIn = dataIn.reshape((10,10))

        dataOut = medfilt2d(image=dataIn,
                            kernel_size=(1, 1),
                            conditional=False)

        self.assertTrue(numpy.array_equal(dataIn, dataOut))

    def testInputDataIsNotModify(self):
        """Make sure input data is not modify by the median filter"""
        dataIn = numpy.arange(100, dtype=numpy.int32)
        dataIn = dataIn.reshape((10,10))
        dataInCopy = dataIn.copy()

        dataOut = medfilt2d(image=dataIn,
                            kernel_size=(3, 3),
                            conditional=False)
        self.assertTrue(numpy.array_equal(dataIn, dataInCopy))

    def testThreads(self):
        """Make sure the result doesn't depends on the number of threads used"""
        dataIn = numpy.random.rand(100, 100)

        former = os.environ.get("OMP_NUM_THREADS")
        os.environ["OMP_NUM_THREADS"] = "1"
        dataOut1Thr = medfilt2d(image=dataIn,
                                kernel_size=(3, 3),
                                conditional=False,
                                )
        os.environ["OMP_NUM_THREADS"] = "2"
        dataOut2Thr = medfilt2d(image=dataIn,
                                kernel_size=(3, 3),
                                conditional=False)
        os.environ["OMP_NUM_THREADS"] = "4"
        dataOut4Thr = medfilt2d(image=dataIn,
                                kernel_size=(3, 3),
                                conditional=False)
        os.environ["OMP_NUM_THREADS"] = "8"
        dataOut8Thr = medfilt2d(image=dataIn,
                                kernel_size=(3, 3),
                                conditional=False)
        if former is None:
            os.environ.pop("OMP_NUM_THREADS")
        else:
            os.environ["OMP_NUM_THREADS"] = former

        self.assertTrue(numpy.array_equal(dataOut1Thr, dataOut2Thr))
        self.assertTrue(numpy.array_equal(dataOut1Thr, dataOut4Thr))
        self.assertTrue(numpy.array_equal(dataOut1Thr, dataOut8Thr))


class Testconditional2DFilter(unittest.TestCase):
    """Test that the conditional filter apply correctly"""

    def testFilter3(self):
        dataIn = numpy.arange(100, dtype=numpy.int32)
        dataIn = dataIn.reshape((10, 10))

        dataOut = medfilt2d(image=dataIn,
                            kernel_size=(3, 3),
                            conditional=True)
        self.assertTrue(dataOut[0, 0] == 1)
        self.assertTrue(dataOut[0, 1] == 1)
        self.assertTrue(numpy.array_equal(dataOut[1:8, 1:8], dataIn[1:8, 1:8]))
        self.assertTrue(dataOut[9, 9] == 98)


class Test2DFilterInputTypes(ParametricTestCase):
    """Test that all needed types have their implementation of the median filter
    """

    def testTypes(self):
        for testType in [numpy.float32, numpy.float64, numpy.int16, numpy.uint16,
                         numpy.int32, numpy.int64, numpy.uint64]:
            data = (numpy.random.rand(10, 10) * 65000).astype(testType)
            out = medfilt2d(image=data,
                            kernel_size=(3, 3),
                            conditional=False)
            self.assertTrue(out.dtype.type is testType)


class Test1DFilter(unittest.TestCase):
    """Some unit tests for the median filter"""

    def testFilter3(self):
        """Simple test of a three by three kernel median filter"""
        dataIn = numpy.arange(100, dtype=numpy.int32)

        dataOut = medfilt2d(image=dataIn,
                            kernel_size=(5),
                            conditional=False)
        
        self.assertTrue(dataOut[0] == 0)
        self.assertTrue(dataOut[9] == 9)
        self.assertTrue(dataOut[99] == 99)

@unittest.skipUnless(scipy, "scipy not available")
class TestCompareScipy(unittest.TestCase):
    """Make sure the result between scipy and medfilt2d are equivalent"""

    def test100(self):
        data = numpy.arange(100, dtype=numpy.int32)
        data = data.reshape(10, 10)
        resScipy = scipy.ndimage.median_filter(data, size=5, mode='nearest')
        resSilx = medfilt2d(image=data,
                            kernel_size=(5),
                            conditional=False)

        self.assertTrue(numpy.array_equal(resScipy, resSilx))

    def test1000(self):
        data = numpy.arange(10000, dtype=numpy.int32)
        data = data.reshape(100, 100)
        resScipy = scipy.ndimage.median_filter(data, size=(3, 7), mode='nearest')
        resSilx = medfilt2d(image=data,
                            kernel_size=(3, 7),
                            conditional=False)

        self.assertTrue(numpy.array_equal(resScipy, resSilx))

    def test25(self):
        data = numpy.arange(25, dtype=numpy.int32)
        data = data.reshape(5, 5)
        resScipy = scipy.ndimage.median_filter(data, size=(9, 7), mode='nearest')
        resSilx = medfilt2d(image=data,
                            kernel_size=(9, 7),
                            conditional=False)

        self.assertTrue(numpy.array_equal(resScipy, resSilx))



def suite():
    test_suite = unittest.TestSuite()
    for test in [Test2DFilter, Testconditional2DFilter, Test2DFilterInputTypes,
        Test1DFilter, TestCompareScipy]:
        test_suite.addTest(
            unittest.defaultTestLoader.loadTestsFromTestCase(test))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
