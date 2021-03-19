# coding: utf-8
# ##########################################################################
# Copyright (C) 2017-2018 European Synchrotron Radiation Facility
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
# ############################################################################
"""Tests of the median filter"""

__authors__ = ["H. Payno"]
__license__ = "MIT"
__date__ = "17/01/2018"

import unittest
import numpy
from silx.math.medianfilter import medfilt2d, medfilt1d
from silx.math.medianfilter.medianfilter import reflect, mirror
from silx.math.medianfilter.medianfilter import MODES as silx_mf_modes
from silx.utils.testutils import ParametricTestCase
try:
    import scipy
    import scipy.misc
except:
    scipy = None
else:
    import scipy.ndimage

import logging
_logger = logging.getLogger(__name__)

RANDOM_FLOAT_MAT = numpy.array([
    [0.05564293, 0.62717157, 0.75002406, 0.40555336, 0.70278975],
    [0.76532598, 0.02839148, 0.05272484, 0.65166994, 0.42161216],
    [0.23067427, 0.74219128, 0.56049024, 0.44406320, 0.28773158],
    [0.81025249, 0.20303021, 0.68382382, 0.46372299, 0.81281709],
    [0.94691602, 0.07813661, 0.81651256, 0.84220106, 0.33623165]])

RANDOM_INT_MAT = numpy.array([
    [0, 5, 2, 6, 1],
    [2, 3, 1, 7, 1],
    [9, 8, 6, 7, 8],
    [5, 6, 8, 2, 4]])


class TestMedianFilterNearest(ParametricTestCase):
    """Unit tests for the median filter in nearest mode"""

    def testFilter3_100(self):
        """Test median filter on a 10x10 matrix with a 3x3 kernel."""
        dataIn = numpy.arange(100, dtype=numpy.int32)
        dataIn = dataIn.reshape((10, 10))

        dataOut = medfilt2d(image=dataIn,
                            kernel_size=(3, 3),
                            conditional=False,
                            mode='nearest')
        self.assertTrue(dataOut[0, 0] == 1)
        self.assertTrue(dataOut[9, 0] == 90)
        self.assertTrue(dataOut[9, 9] == 98)

        self.assertTrue(dataOut[0, 9] == 9)
        self.assertTrue(dataOut[0, 4] == 5)
        self.assertTrue(dataOut[9, 4] == 93)
        self.assertTrue(dataOut[4, 4] == 44)

    def testFilter3_9(self):
        "Test median filter on a 3x3 matrix with a 3x3 kernel."
        dataIn = numpy.array([0, -1, 1,
                              12, 6, -2,
                              100, 4, 12],
                             dtype=numpy.int16)
        dataIn = dataIn.reshape((3, 3))
        dataOut = medfilt2d(image=dataIn,
                            kernel_size=(3, 3),
                            conditional=False,
                            mode='nearest')
        self.assertTrue(dataOut.shape == dataIn.shape)
        self.assertTrue(dataOut[1, 1] == 4)
        self.assertTrue(dataOut[0, 0] == 0)
        self.assertTrue(dataOut[0, 1] == 0)
        self.assertTrue(dataOut[1, 0] == 6)

    def testFilterWidthOne(self):
        """Make sure a filter of one by one give the same result as the input
        """
        dataIn = numpy.arange(100, dtype=numpy.int32)
        dataIn = dataIn.reshape((10, 10))

        dataOut = medfilt2d(image=dataIn,
                            kernel_size=(1, 1),
                            conditional=False,
                            mode='nearest')

        self.assertTrue(numpy.array_equal(dataIn, dataOut))

    def testFilter3_1d(self):
        """Test binding and result of the 1d filter"""
        self.assertTrue(numpy.array_equal(
            medfilt1d(RANDOM_INT_MAT[0], kernel_size=3, conditional=False,
                      mode='nearest'),
            [0, 2, 5, 2, 1])
        )

    def testFilter3Conditionnal(self):
        """Test that the conditional filter apply correctly in a 10x10 matrix
        with a 3x3 kernel
        """
        dataIn = numpy.arange(100, dtype=numpy.int32)
        dataIn = dataIn.reshape((10, 10))

        dataOut = medfilt2d(image=dataIn,
                            kernel_size=(3, 3),
                            conditional=True,
                            mode='nearest')
        self.assertTrue(dataOut[0, 0] == 1)
        self.assertTrue(dataOut[0, 1] == 1)
        self.assertTrue(numpy.array_equal(dataOut[1:8, 1:8], dataIn[1:8, 1:8]))
        self.assertTrue(dataOut[9, 9] == 98)

    def testFilter3_1D(self):
        """Simple test of a 3x3 median filter on a 1D array"""
        dataIn = numpy.arange(100, dtype=numpy.int32)

        dataOut = medfilt2d(image=dataIn,
                            kernel_size=(5),
                            conditional=False,
                            mode='nearest')

        self.assertTrue(dataOut[0] == 0)
        self.assertTrue(dataOut[9] == 9)
        self.assertTrue(dataOut[99] == 99)

    def testNaNs(self):
        """Test median filter on image with NaNs in nearest mode"""
        # Data with a NaN in first corner
        nan_corner = numpy.arange(100.).reshape(10, 10)
        nan_corner[0, 0] = numpy.nan
        output = medfilt2d(
            nan_corner, kernel_size=3, conditional=False, mode='nearest')
        self.assertEqual(output[0, 0], 10)
        self.assertEqual(output[0, 1], 2)
        self.assertEqual(output[1, 0], 11)
        self.assertEqual(output[1, 1], 12)

        # Data with some NaNs
        some_nans = numpy.arange(100.).reshape(10, 10)
        some_nans[0, 1] = numpy.nan
        some_nans[1, 1] = numpy.nan
        some_nans[1, 0] = numpy.nan
        output = medfilt2d(
            some_nans, kernel_size=3, conditional=False, mode='nearest')
        self.assertEqual(output[0, 0], 0)
        self.assertEqual(output[0, 1], 2)
        self.assertEqual(output[1, 0], 20)
        self.assertEqual(output[1, 1], 20)


class TestMedianFilterReflect(ParametricTestCase):
    """Unit test for the median filter in reflect mode"""

    def testArange9(self):
        """Test from a 3x3 window to RANDOM_FLOAT_MAT"""
        img = numpy.arange(9, dtype=numpy.int32)
        img = img.reshape(3, 3)
        kernel = (3, 3)
        res = medfilt2d(image=img,
                        kernel_size=kernel,
                        conditional=False,
                        mode='reflect')
        self.assertTrue(
            numpy.array_equal(res.ravel(), [1, 2, 2, 3, 4, 5, 6, 6, 7]))

    def testRandom10(self):
        """Test a (5, 3) window to a RANDOM_FLOAT_MAT"""
        kernel = (5, 3)

        thRes = numpy.array([
            [0.23067427, 0.56049024, 0.56049024, 0.4440632, 0.42161216],
            [0.23067427, 0.62717157, 0.56049024, 0.56049024, 0.46372299],
            [0.62717157, 0.62717157, 0.56049024, 0.56049024, 0.4440632],
            [0.76532598, 0.68382382, 0.56049024, 0.56049024, 0.42161216],
            [0.81025249, 0.68382382, 0.56049024, 0.68382382, 0.46372299]])

        res = medfilt2d(image=RANDOM_FLOAT_MAT,
                        kernel_size=kernel,
                        conditional=False,
                        mode='reflect')

        self.assertTrue(numpy.array_equal(thRes, res))

    def testApplyReflect1D(self):
        """Test the reflect function used for the median filter in reflect mode
        """
        # test for inside values
        self.assertTrue(reflect(2, 3) == 2)
        # test for boundaries values
        self.assertTrue(reflect(3, 3) == 2)
        self.assertTrue(reflect(4, 3) == 1)
        self.assertTrue(reflect(5, 3) == 0)
        self.assertTrue(reflect(6, 3) == 0)
        self.assertTrue(reflect(7, 3) == 1)
        self.assertTrue(reflect(-1, 3) == 0)
        self.assertTrue(reflect(-2, 3) == 1)
        self.assertTrue(reflect(-3, 3) == 2)
        self.assertTrue(reflect(-4, 3) == 2)
        self.assertTrue(reflect(-5, 3) == 1)
        self.assertTrue(reflect(-6, 3) == 0)
        self.assertTrue(reflect(-7, 3) == 0)

    def testRandom10Conditionnal(self):
        """Test the median filter in reflect mode and with the conditionnal
        option"""
        kernel = (3, 1)

        thRes = numpy.array([
            [0.05564293, 0.62717157, 0.75002406, 0.40555336, 0.70278975],
            [0.23067427, 0.62717157, 0.56049024, 0.44406320, 0.42161216],
            [0.76532598, 0.20303021, 0.56049024, 0.46372299, 0.42161216],
            [0.81025249, 0.20303021, 0.68382382, 0.46372299, 0.33623165],
            [0.94691602, 0.07813661, 0.81651256, 0.84220106, 0.33623165]])

        res = medfilt2d(image=RANDOM_FLOAT_MAT,
                        kernel_size=kernel,
                        conditional=True,
                        mode='reflect')
        self.assertTrue(numpy.array_equal(thRes, res))

    def testNaNs(self):
        """Test median filter on image with NaNs in reflect mode"""
        # Data with a NaN in first corner
        nan_corner = numpy.arange(100.).reshape(10, 10)
        nan_corner[0, 0] = numpy.nan
        output = medfilt2d(
            nan_corner, kernel_size=3, conditional=False, mode='reflect')
        self.assertEqual(output[0, 0], 10)
        self.assertEqual(output[0, 1], 2)
        self.assertEqual(output[1, 0], 11)
        self.assertEqual(output[1, 1], 12)

        # Data with some NaNs
        some_nans = numpy.arange(100.).reshape(10, 10)
        some_nans[0, 1] = numpy.nan
        some_nans[1, 1] = numpy.nan
        some_nans[1, 0] = numpy.nan
        output = medfilt2d(
            some_nans, kernel_size=3, conditional=False, mode='reflect')
        self.assertEqual(output[0, 0], 0)
        self.assertEqual(output[0, 1], 2)
        self.assertEqual(output[1, 0], 20)
        self.assertEqual(output[1, 1], 20)

    def testFilter3_1d(self):
        """Test binding and result of the 1d filter"""
        self.assertTrue(numpy.array_equal(
            medfilt1d(RANDOM_INT_MAT[0], kernel_size=5, conditional=False,
                      mode='reflect'),
            [2, 2, 2, 2, 2])
        )


class TestMedianFilterMirror(ParametricTestCase):
    """Unit test for the median filter in mirror mode
    """

    def testApplyMirror1D(self):
        """Test the reflect function used for the median filter in mirror mode
        """
        # test for inside values
        self.assertTrue(mirror(2, 3) == 2)
        # test for boundaries values
        self.assertTrue(mirror(4, 4) == 2)
        self.assertTrue(mirror(5, 4) == 1)
        self.assertTrue(mirror(6, 4) == 0)
        self.assertTrue(mirror(7, 4) == 1)
        self.assertTrue(mirror(8, 4) == 2)
        self.assertTrue(mirror(-1, 4) == 1)
        self.assertTrue(mirror(-2, 4) == 2)
        self.assertTrue(mirror(-3, 4) == 3)
        self.assertTrue(mirror(-4, 4) == 2)
        self.assertTrue(mirror(-5, 4) == 1)
        self.assertTrue(mirror(-6, 4) == 0)

    def testRandom10(self):
        """Test a (5, 3) window to a random array"""
        kernel = (3, 5)

        thRes = numpy.array([
            [0.05272484, 0.40555336, 0.42161216, 0.42161216, 0.42161216],
            [0.56049024, 0.56049024, 0.4440632, 0.4440632, 0.4440632],
            [0.56049024, 0.46372299, 0.46372299, 0.46372299, 0.46372299],
            [0.68382382, 0.56049024, 0.56049024, 0.46372299, 0.56049024],
            [0.68382382, 0.46372299, 0.68382382, 0.46372299, 0.68382382]])

        res = medfilt2d(image=RANDOM_FLOAT_MAT,
                        kernel_size=kernel,
                        conditional=False,
                        mode='mirror')

        self.assertTrue(numpy.array_equal(thRes, res))

    def testRandom10Conditionnal(self):
        """Test the median filter in reflect mode and with the conditionnal
        option"""
        kernel = (1, 3)

        thRes = numpy.array([
            [0.62717157, 0.62717157, 0.62717157, 0.70278975, 0.40555336],
            [0.02839148, 0.05272484, 0.05272484, 0.42161216, 0.65166994],
            [0.74219128, 0.56049024, 0.56049024, 0.44406320, 0.44406320],
            [0.20303021, 0.68382382, 0.46372299, 0.68382382, 0.46372299],
            [0.07813661, 0.81651256, 0.81651256, 0.81651256, 0.84220106]])

        res = medfilt2d(image=RANDOM_FLOAT_MAT,
                        kernel_size=kernel,
                        conditional=True,
                        mode='mirror')

        self.assertTrue(numpy.array_equal(thRes, res))

    def testNaNs(self):
        """Test median filter on image with NaNs in mirror mode"""
        # Data with a NaN in first corner
        nan_corner = numpy.arange(100.).reshape(10, 10)
        nan_corner[0, 0] = numpy.nan
        output = medfilt2d(
            nan_corner, kernel_size=3, conditional=False, mode='mirror')
        self.assertEqual(output[0, 0], 11)
        self.assertEqual(output[0, 1], 11)
        self.assertEqual(output[1, 0], 11)
        self.assertEqual(output[1, 1], 12)

        # Data with some NaNs
        some_nans = numpy.arange(100.).reshape(10, 10)
        some_nans[0, 1] = numpy.nan
        some_nans[1, 1] = numpy.nan
        some_nans[1, 0] = numpy.nan
        output = medfilt2d(
            some_nans, kernel_size=3, conditional=False, mode='mirror')
        self.assertEqual(output[0, 0], 0)
        self.assertEqual(output[0, 1], 12)
        self.assertEqual(output[1, 0], 21)
        self.assertEqual(output[1, 1], 20)

    def testFilter3_1d(self):
        """Test binding and result of the 1d filter"""
        self.assertTrue(numpy.array_equal(
            medfilt1d(RANDOM_INT_MAT[0], kernel_size=5, conditional=False,
                      mode='mirror'),
            [2, 5, 2, 5, 2])
        )

class TestMedianFilterShrink(ParametricTestCase):
    """Unit test for the median filter in mirror mode
    """

    def testRandom_3x3(self):
        """Test the median filter in shrink mode and with the conditionnal
        option"""
        kernel = (3, 3)

        thRes = numpy.array([
            [0.62717157, 0.62717157, 0.62717157, 0.65166994, 0.65166994],
            [0.62717157, 0.56049024, 0.56049024, 0.44406320, 0.44406320],
            [0.74219128, 0.56049024, 0.46372299, 0.46372299, 0.46372299],
            [0.74219128, 0.68382382, 0.56049024, 0.56049024, 0.46372299],
            [0.81025249, 0.81025249, 0.68382382, 0.81281709, 0.81281709]])

        res = medfilt2d(image=RANDOM_FLOAT_MAT,
                        kernel_size=kernel,
                        conditional=False,
                        mode='shrink')

        self.assertTrue(numpy.array_equal(thRes, res))

    def testBounds(self):
        """Test the median filter in shrink mode with 3 different kernels
        which should return the same result due to the large values of kernels
        used.
        """
        kernel1 = (1, 9)
        kernel2 = (1, 11)
        kernel3 = (1, 21)

        thRes = numpy.array([[2, 2, 2, 2, 2],
                             [2, 2, 2, 2, 2],
                             [8, 8, 8, 8, 8],
                             [5, 5, 5, 5, 5]])

        resK1 = medfilt2d(image=RANDOM_INT_MAT,
                          kernel_size=kernel1,
                          conditional=False,
                          mode='shrink')

        resK2 = medfilt2d(image=RANDOM_INT_MAT,
                          kernel_size=kernel2,
                          conditional=False,
                          mode='shrink')

        resK3 = medfilt2d(image=RANDOM_INT_MAT,
                          kernel_size=kernel3,
                          conditional=False,
                          mode='shrink')

        self.assertTrue(numpy.array_equal(resK1, thRes))
        self.assertTrue(numpy.array_equal(resK2, resK1))
        self.assertTrue(numpy.array_equal(resK3, resK1))

    def testRandom_3x3Conditionnal(self):
        """Test the median filter in reflect mode and with the conditionnal
        option"""
        kernel = (3, 3)

        thRes = numpy.array([
            [0.05564293, 0.62717157, 0.62717157, 0.40555336, 0.65166994],
            [0.62717157, 0.56049024, 0.05272484, 0.65166994, 0.42161216],
            [0.23067427, 0.74219128, 0.56049024, 0.44406320, 0.46372299],
            [0.81025249, 0.20303021, 0.68382382, 0.46372299, 0.81281709],
            [0.81025249, 0.81025249, 0.81651256, 0.81281709, 0.81281709]])

        res = medfilt2d(image=RANDOM_FLOAT_MAT,
                        kernel_size=kernel,
                        conditional=True,
                        mode='shrink')

        self.assertTrue(numpy.array_equal(res, thRes))

    def testRandomInt(self):
        """Test 3x3 kernel on RANDOM_INT_MAT
        """
        kernel = (3, 3)

        thRes = numpy.array([[3, 2, 5, 2, 6],
                             [5, 3, 6, 6, 7],
                             [6, 6, 6, 6, 7],
                             [8, 8, 7, 7, 7]])

        resK1 = medfilt2d(image=RANDOM_INT_MAT,
                          kernel_size=kernel,
                          conditional=False,
                          mode='shrink')

        self.assertTrue(numpy.array_equal(resK1, thRes))

    def testNaNs(self):
        """Test median filter on image with NaNs in shrink mode"""
        # Data with a NaN in first corner
        nan_corner = numpy.arange(100.).reshape(10, 10)
        nan_corner[0, 0] = numpy.nan
        output = medfilt2d(
            nan_corner, kernel_size=3, conditional=False, mode='shrink')
        self.assertEqual(output[0, 0], 10)
        self.assertEqual(output[0, 1], 10)
        self.assertEqual(output[1, 0], 11)
        self.assertEqual(output[1, 1], 12)

        # Data with some NaNs
        some_nans = numpy.arange(100.).reshape(10, 10)
        some_nans[0, 1] = numpy.nan
        some_nans[1, 1] = numpy.nan
        some_nans[1, 0] = numpy.nan
        output = medfilt2d(
            some_nans, kernel_size=3, conditional=False, mode='shrink')
        self.assertEqual(output[0, 0], 0)
        self.assertEqual(output[0, 1], 2)
        self.assertEqual(output[1, 0], 20)
        self.assertEqual(output[1, 1], 20)

    def testFilter3_1d(self):
        """Test binding and result of the 1d filter"""
        self.assertTrue(numpy.array_equal(
            medfilt1d(RANDOM_INT_MAT[0], kernel_size=3, conditional=False,
                      mode='shrink'),
            [5, 2, 5, 2, 6])
        )

class TestMedianFilterConstant(ParametricTestCase):
    """Unit test for the median filter in constant mode
    """

    def testRandom10(self):
        """Test a (5, 3) window to a random array"""
        kernel = (3, 5)

        thRes = numpy.array([
            [0., 0.02839148, 0.05564293, 0.02839148, 0.],
            [0.05272484, 0.40555336, 0.4440632, 0.42161216, 0.28773158],
            [0.05272484, 0.44406320, 0.46372299, 0.42161216, 0.28773158],
            [0.20303021, 0.46372299, 0.56049024, 0.44406320, 0.33623165],
            [0., 0.07813661, 0.33623165, 0.07813661, 0.]])

        res = medfilt2d(image=RANDOM_FLOAT_MAT,
                        kernel_size=kernel,
                        conditional=False,
                        mode='constant')

        self.assertTrue(numpy.array_equal(thRes, res))

    RANDOM_FLOAT_MAT = numpy.array([
        [0.05564293, 0.62717157, 0.75002406, 0.40555336, 0.70278975],
        [0.76532598, 0.02839148, 0.05272484, 0.65166994, 0.42161216],
        [0.23067427, 0.74219128, 0.56049024, 0.44406320, 0.28773158],
        [0.81025249, 0.20303021, 0.68382382, 0.46372299, 0.81281709],
        [0.94691602, 0.07813661, 0.81651256, 0.84220106, 0.33623165]])

    def testRandom10Conditionnal(self):
        """Test the median filter in reflect mode and with the conditionnal
        option"""
        kernel = (1, 3)

        print(RANDOM_FLOAT_MAT)

        thRes = numpy.array([
            [0.05564293, 0.62717157, 0.62717157, 0.70278975, 0.40555336],
            [0.02839148, 0.05272484, 0.05272484, 0.42161216, 0.42161216],
            [0.23067427, 0.56049024, 0.56049024, 0.44406320, 0.28773158],
            [0.20303021, 0.68382382, 0.46372299, 0.68382382, 0.46372299],
            [0.07813661, 0.81651256, 0.81651256, 0.81651256, 0.33623165]])

        res = medfilt2d(image=RANDOM_FLOAT_MAT,
                        kernel_size=kernel,
                        conditional=True,
                        mode='constant')

        self.assertTrue(numpy.array_equal(thRes, res))

    def testNaNs(self):
        """Test median filter on image with NaNs in constant mode"""
        # Data with a NaN in first corner
        nan_corner = numpy.arange(100.).reshape(10, 10)
        nan_corner[0, 0] = numpy.nan
        output = medfilt2d(nan_corner,
                           kernel_size=3,
                           conditional=False,
                           mode='constant',
                           cval=0)
        self.assertEqual(output[0, 0], 0)
        self.assertEqual(output[0, 1], 2)
        self.assertEqual(output[1, 0], 10)
        self.assertEqual(output[1, 1], 12)

        # Data with some NaNs
        some_nans = numpy.arange(100.).reshape(10, 10)
        some_nans[0, 1] = numpy.nan
        some_nans[1, 1] = numpy.nan
        some_nans[1, 0] = numpy.nan
        output = medfilt2d(some_nans,
                           kernel_size=3,
                           conditional=False,
                           mode='constant',
                           cval=0)
        self.assertEqual(output[0, 0], 0)
        self.assertEqual(output[0, 1], 0)
        self.assertEqual(output[1, 0], 0)
        self.assertEqual(output[1, 1], 20)

    def testFilter3_1d(self):
        """Test binding and result of the 1d filter"""
        self.assertTrue(numpy.array_equal(
            medfilt1d(RANDOM_INT_MAT[0], kernel_size=5, conditional=False,
                      mode='constant'),
            [0, 2, 2, 2, 1])
        )

class TestGeneralExecution(ParametricTestCase):
    """Some general test on median filter application"""

    def testTypes(self):
        """Test that all needed types have their implementation of the median
        filter
        """
        for mode in silx_mf_modes:
            for testType in [numpy.float32, numpy.float64, numpy.int16,
                             numpy.uint16, numpy.int32, numpy.int64,
                             numpy.uint64]:
                with self.subTest(mode=mode, type=testType):
                    data = (numpy.random.rand(10, 10) * 65000).astype(testType)
                    out = medfilt2d(image=data,
                                    kernel_size=(3, 3),
                                    conditional=False,
                                    mode=mode)
                    self.assertTrue(out.dtype.type is testType)

    def testInputDataIsNotModify(self):
        """Make sure input data is not modify by the median filter"""
        dataIn = numpy.arange(100, dtype=numpy.int32)
        dataIn = dataIn.reshape((10, 10))
        dataInCopy = dataIn.copy()

        for mode in silx_mf_modes:
            with self.subTest(mode=mode):
                medfilt2d(image=dataIn,
                          kernel_size=(3, 3),
                          conditional=False,
                          mode=mode)
                self.assertTrue(numpy.array_equal(dataIn, dataInCopy))

    def testAllNaNs(self):
        """Test median filter on image all NaNs"""
        all_nans = numpy.empty((10, 10), dtype=numpy.float32)
        all_nans[:] = numpy.nan

        for mode in silx_mf_modes:
            for conditional in (True, False):
                with self.subTest(mode=mode, conditional=conditional):
                    output = medfilt2d(
                        all_nans,
                        kernel_size=3,
                        conditional=conditional,
                        mode=mode,
                        cval=numpy.nan)
                    self.assertTrue(numpy.all(numpy.isnan(output)))

    def testConditionalWithNaNs(self):
        """Test that NaNs are propagated through conditional median filter"""
        for mode in silx_mf_modes:
            with self.subTest(mode=mode):
                image = numpy.ones((10, 10), dtype=numpy.float32)
                nan_mask = numpy.zeros_like(image, dtype=bool)
                nan_mask[0, 0] = True
                nan_mask[4, :] = True
                nan_mask[6, 4] = True
                image[nan_mask] = numpy.nan
                output = medfilt2d(
                    image,
                    kernel_size=3,
                    conditional=True,
                    mode=mode)
                out_isnan = numpy.isnan(output)
                self.assertTrue(numpy.all(out_isnan[nan_mask]))
                self.assertFalse(
                    numpy.any(out_isnan[numpy.logical_not(nan_mask)]))


def _getScipyAndSilxCommonModes():
    """return the mode which are comparable between silx and scipy"""
    modes = silx_mf_modes.copy()
    del modes['shrink']
    return modes


@unittest.skipUnless(scipy is not None, "scipy not available")
class TestVsScipy(ParametricTestCase):
    """Compare scipy.ndimage.median_filter vs silx.math.medianfilter
    on comparable 
    """
    def testWithArange(self):
        """Test vs scipy with different kernels on arange matrix"""
        data = numpy.arange(10000, dtype=numpy.int32)
        data = data.reshape(100, 100)

        kernels = [(3, 7), (7, 5), (1, 1), (3, 3)]
        modesToTest = _getScipyAndSilxCommonModes()
        for kernel in kernels:
            for mode in modesToTest:
                with self.subTest(kernel=kernel, mode=mode):
                    resScipy = scipy.ndimage.median_filter(input=data,
                                                           size=kernel,
                                                           mode=mode)
                    resSilx = medfilt2d(image=data,
                                        kernel_size=kernel,
                                        conditional=False,
                                        mode=mode)

                    self.assertTrue(numpy.array_equal(resScipy, resSilx))

    def testRandomMatrice(self):
        """Test vs scipy with different kernels on RANDOM_FLOAT_MAT"""
        kernels = [(3, 7), (7, 5), (1, 1), (3, 3)]
        modesToTest = _getScipyAndSilxCommonModes()
        for kernel in kernels:
            for mode in modesToTest:
                with self.subTest(kernel=kernel, mode=mode):
                    resScipy = scipy.ndimage.median_filter(input=RANDOM_FLOAT_MAT,
                                                           size=kernel,
                                                           mode=mode)

                    resSilx = medfilt2d(image=RANDOM_FLOAT_MAT,
                                        kernel_size=kernel,
                                        conditional=False,
                                        mode=mode)

                    self.assertTrue(numpy.array_equal(resScipy, resSilx))

    def testAscentOrLena(self):
        """Test vs scipy with """
        if hasattr(scipy.misc, 'ascent'):
            img = scipy.misc.ascent()
        else:
            img = scipy.misc.lena()

        kernels = [(3, 1), (3, 5), (5, 9), (9, 3)]
        modesToTest = _getScipyAndSilxCommonModes()

        for kernel in kernels:
            for mode in modesToTest:
                with self.subTest(kernel=kernel, mode=mode):
                    resScipy = scipy.ndimage.median_filter(input=img,
                                                           size=kernel,
                                                           mode=mode)

                    resSilx = medfilt2d(image=img,
                                        kernel_size=kernel,
                                        conditional=False,
                                        mode=mode)

                    self.assertTrue(numpy.array_equal(resScipy, resSilx))


def suite():
    test_suite = unittest.TestSuite()
    for test in [TestGeneralExecution,
                 TestVsScipy,
                 TestMedianFilterNearest,
                 TestMedianFilterReflect,
                 TestMedianFilterMirror,
                 TestMedianFilterShrink,
                 TestMedianFilterConstant]:
        test_suite.addTest(
            unittest.defaultTestLoader.loadTestsFromTestCase(test))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
