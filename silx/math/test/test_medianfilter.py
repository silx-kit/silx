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
__date__ = "10/02/2017"

import unittest
import numpy
import tempfile
import os
from  silx.math.medianfilter import medianfilter
from silx.test.utils import ParametricTestCase

import logging
_logger = logging.getLogger(__name__)

class Test2DFilter(unittest.TestCase):
    """Some unit tests for the median filter"""

    def testFilter3(self):
        """Simple test of a three by three kernel median filter"""
        dataIn = numpy.arange(100, dtype=numpy.int32)
        dataIn = dataIn.reshape((10,10))

        dataOut = medianfilter(input_buffer=dataIn,
                               kernel_dim=(3, 3),
                               conditionnal=False)
        
        self.assertTrue(dataOut[0, 0] == 10)
        self.assertTrue(dataOut[9, 0] == 90)
        self.assertTrue(dataOut[9, 9] == 98)
        self.assertTrue(dataOut[0, 9] == 18)

        self.assertTrue(dataOut[0, 4] == 13)
        self.assertTrue(dataOut[9, 4] == 93)
        self.assertTrue(dataOut[4, 4] == 44)


        dataIn = numpy.array([0, -1, 1,
                              12, 6, -2,
                              100, 4, 12],
                             dtype=numpy.int16)
        dataIn = dataIn.reshape((3, 3))
        dataOut = medianfilter(input_buffer=dataIn,
                               kernel_dim=(3, 3),
                               conditionnal=False)
        self.assertTrue(dataOut.shape == dataIn.shape)
        self.assertTrue(dataOut[1, 1] == 4)
        self.assertTrue(dataOut[0, 0] == 6)
        self.assertTrue(dataOut[0, 1] == 1)
        self.assertTrue(dataOut[1, 0] == 6)


    def testFilterWidthOne(self):
        """Make sure a filter of one by one give the same result as the input"""
        dataIn = numpy.arange(100, dtype=numpy.int32)
        dataIn = dataIn.reshape((10,10))

        dataOut = medianfilter(input_buffer=dataIn,
                               kernel_dim=(1, 1),
                               conditionnal=False)

        self.assertTrue(numpy.array_equal(dataIn, dataOut))

    def testInputDataIsNotModify(self):
        """Make sure input data is not modify by the median filter"""
        dataIn = numpy.arange(100, dtype=numpy.int32)
        dataIn = dataIn.reshape((10,10))
        dataInCopy = dataIn.copy()

        dataOut = medianfilter(input_buffer=dataIn,
                               kernel_dim=(3, 3),
                               conditionnal=False)
        self.assertTrue(numpy.array_equal(dataIn, dataInCopy))

    def testThreads(self):
        """Make sure the result doesn't depends on the number of threads used"""
        dataIn = numpy.random.rand(100, 100)

        dataOut1Thr = medianfilter(input_buffer=dataIn,
                                   kernel_dim=(3, 3),
                                   conditionnal=False,
                                   nthread=1)
        dataOut2Thr = medianfilter(input_buffer=dataIn,
                                   kernel_dim=(3, 3),
                                   conditionnal=False,
                                   nthread=2)
        dataOut4Thr = medianfilter(input_buffer=dataIn,
                                   kernel_dim=(3, 3),
                                   conditionnal=False,
                                   nthread=4)
        dataOut8Thr = medianfilter(input_buffer=dataIn,
                                   kernel_dim=(3, 3),
                                   conditionnal=False,
                                   nthread=8)

        self.assertTrue(numpy.array_equal(dataOut1Thr, dataOut2Thr))
        self.assertTrue(numpy.array_equal(dataOut1Thr, dataOut4Thr))
        self.assertTrue(numpy.array_equal(dataOut1Thr, dataOut8Thr))

class TestConditionnal2DFilter(unittest.TestCase):
    """Test that the conditionnal filter apply correctly"""

    def testFilter3(self):
        dataIn = numpy.arange(100, dtype=numpy.int32)
        dataIn = dataIn.reshape((10,10))

        dataOut = medianfilter(input_buffer=dataIn,
                               kernel_dim=(3, 3),
                               conditionnal=True)
        
        self.assertTrue(dataOut[0, 0] == 10)
        self.assertTrue(dataOut[0, 1] == 1)
        self.assertTrue(numpy.array_equal(dataOut[1:8, 1:8], dataIn[1:8, 1:8]))
        self.assertTrue(dataOut[9, 9] == 98)

class Test2DFilterInputTypes(ParametricTestCase):
    """Test that all needed types have their implementation of the median filter
    """

    def testTypes(self):
        for testType in [numpy.float32, numpy.float64, numpy.int16, numpy.uint16,
            numpy.int32, numpy.int64, numpy.uint64]:


            data = numpy.random.rand(10, 10).astype(testType)
            out = medianfilter(input_buffer=data,
                               kernel_dim=(3, 3),
                               conditionnal=False)
            self.assertTrue(out.dtype.type is testType)


class Test1DFilter(unittest.TestCase):
    """Some unit tests for the median filter"""

    def testFilter3(self):
        """Simple test of a three by three kernel median filter"""
        dataIn = numpy.arange(100, dtype=numpy.int32)

        dataOut = medianfilter(input_buffer=dataIn,
                               kernel_dim=(5),
                               conditionnal=False)
        
        self.assertTrue(dataOut[0] == 1)
        self.assertTrue(dataOut[9] == 9)
        self.assertTrue(dataOut[99] == 98)


def suite():
    test_suite = unittest.TestSuite()
    for test in [Test2DFilter, TestConditionnal2DFilter, Test2DFilterInputTypes, 
        Test1DFilter]:
        test_suite.addTest(
            unittest.defaultTestLoader.loadTestsFromTestCase(test))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
