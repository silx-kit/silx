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

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "10/02/2017"

import unittest
import numpy
import tempfile
import os
import medianfilter

import logging
_logger = logging.getLogger(__name__)

class TestFilterValues(unittest.TestCase):
    """TODO"""

    def testFilter3(self):
        dataIn = numpy.arange(100, dtype=numpy.int32)
        dataIn = dataIn.reshape((10,10))
        dataOut = numpy.zeros(dataIn.shape, dtype=numpy.int32)

        medianfilter.median_filter(dataIn, dataOut, (3, 3), conditionnal=False)
        # for now if pair number of value we are taking the higher one
        self.assertTrue(dataOut[0, 0] == 1)
        self.assertTrue(dataOut[9, 0] == 81)
        self.assertTrue(dataOut[9, 9] == 89)
        self.assertTrue(dataOut[0, 9] == 9)

        self.assertTrue(dataOut[0, 4] == 5)
        self.assertTrue(dataOut[9, 4] == 85)
        self.assertTrue(dataOut[4, 4] == 44)

    def testFilterWidthOne(self):
        dataIn = numpy.arange(100, dtype=numpy.int32)
        dataIn = dataIn.reshape((10,10))
        dataOut = numpy.zeros(dataIn.shape, dtype=numpy.int32)

        medianfilter.median_filter(dataIn, dataOut, (1, 1), conditionnal=False)

        self.assertTrue(numpy.array_equal(dataIn, dataOut))

    def testInputDataIsNotModify(self):
        dataIn = numpy.arange(100, dtype=numpy.int32)
        dataIn = dataIn.reshape((10,10))
        dataInCopy = dataIn.copy()
        dataOut = numpy.zeros(dataIn.shape, dtype=numpy.int32)

        medianfilter.median_filter(dataIn, dataOut, (3, 3), conditionnal=False)
        self.assertTrue(numpy.array_equal(dataIn, dataInCopy))

    def testThreads(self):
        dataIn = numpy.random.rand(100, 100)
        dataOut1Thr = numpy.zeros(dataIn.shape, dtype=numpy.float64)
        dataOut2Thr = numpy.zeros(dataIn.shape, dtype=numpy.float64)
        dataOut4Thr = numpy.zeros(dataIn.shape, dtype=numpy.float64)
        dataOut8Thr = numpy.zeros(dataIn.shape, dtype=numpy.float64)
        medianfilter.median_filter(dataIn, dataOut1Thr, (3, 3), conditionnal=False, nthread=1)
        medianfilter.median_filter(dataIn, dataOut2Thr, (3, 3), conditionnal=False, nthread=2)
        medianfilter.median_filter(dataIn, dataOut4Thr, (3, 3), conditionnal=False, nthread=4)
        medianfilter.median_filter(dataIn, dataOut8Thr, (3, 3), conditionnal=False, nthread=8)

        self.assertTrue(numpy.array_equal(dataOut1Thr, dataOut2Thr))
        self.assertTrue(numpy.array_equal(dataOut1Thr, dataOut4Thr))
        self.assertTrue(numpy.array_equal(dataOut1Thr, dataOut8Thr))

class TestConditionnalFilterValues(unittest.TestCase):
    """TODO"""

    def testFilter3(self):
        dataIn = numpy.arange(100, dtype=numpy.int32)
        dataIn = dataIn.reshape((10,10))
        dataOut = numpy.zeros(dataIn.shape, dtype=numpy.int32)

        medianfilter.median_filter(dataIn, dataOut, (3, 3), conditionnal=True)
        # for now if pair number of value we are taking the lower one
        self.assertTrue(dataOut[0, 0] == 1)
        self.assertTrue(dataOut[0, 1] == 1)
        self.assertTrue(numpy.array_equal(dataOut[1:8, 1:8], dataIn[1:8, 1:8]))
        self.assertTrue(dataOut[9, 9] == 89)

class TestFilterInputTypes(unittest.TestCase):
    """TODO"""

    # TODO : utiliser subtest dans silx here
    def testFloat(self):
        try:
            data = numpy.random.rand(10, 10).astype(dtype=numpy.float32)
            out = numpy.zeros(data.shape, dtype=numpy.float32)
            medianfilter.median_filter(data, out, (3, 3), conditionnal=False)
            self.assertTrue(True)
        except :
            self.assertTrue(False)

    def testDouble(self):
        try:
            data = numpy.random.rand(10, 10).astype(dtype=numpy.float64)
            out = numpy.zeros(data.shape, dtype=numpy.float64)
            medianfilter.median_filter(data, out, (3, 3), conditionnal=False)
            self.assertTrue(True)
        except:
            self.assertTrue(False)

    def testInt(self):
        try:
            data = numpy.random.rand(10, 10).astype(dtype=numpy.int32)
            out = numpy.zeros(data.shape, dtype=numpy.int32)
            medianfilter.median_filter(data, out, (3, 3), conditionnal=False)
            self.assertTrue(True)
        except:
            self.assertTrue(False)

def suite():
    test_suite = unittest.TestSuite()
    for test in [TestFilterValues, TestFilterInputTypes, TestConditionnalFilterValues]:
        test_suite.addTest(
            unittest.defaultTestLoader.loadTestsFromTestCase(test))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
