#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Byte-offset decompression in OpenCL
#             https://github.com/silx-kit/silx
#
#    Copyright (C) 2013-2017  European Synchrotron Radiation Facility,
#                             Grenoble, France
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

"""
Test suite for byte-offset decompression 
"""

from __future__ import division, print_function

__authors__ = ["Jérôme Kieffer"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "MIT"
__copyright__ = "2013 European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "10/11/2017"

import sys
import time
import logging
import numpy
from silx.opencl import ocl
from silx.opencl.codec import byte_offset
try:
    import fabio
except:
    fabio = None
try:
    import pyopencl
except ImportError:
    pyopencl = None
import unittest
logger = logging.getLogger(__name__)


@unittest.skipUnless(ocl and fabio and pyopencl,
                     "PyOpenCl or fabio is missing")
class TestByteOffset(unittest.TestCase):

    def _create_test_data(self, shape, nexcept, lam=200):
        """Create test (image, compressed stream) pair.

        :param shape: Shape of test image
        :param int nexcept: Number of exceptions in the image
        :param lam: Expectation of interval argument for numpy.random.poisson
        :return: (reference image array, compressed stream)
        """
        size = numpy.prod(shape)
        ref = numpy.random.poisson(lam, numpy.prod(shape))
        exception_loc = numpy.random.randint(0, size, size=nexcept)
        exception_value = numpy.random.randint(0, 1000000, size=nexcept)
        ref[exception_loc] = exception_value
        ref.shape = shape

        raw = fabio.compression.compByteOffset(ref)
        return ref, raw

    def test_decompress(self):
        """
        tests the byte offset decompression on GPU
        """
        ref, raw = self._create_test_data(shape=(2713, 2719), nexcept=2729)
        size = numpy.prod(ref.shape)

        try:
            bo = byte_offset.ByteOffset(len(raw), size, profile=True)
        except (RuntimeError, pyopencl.RuntimeError) as err:
            logger.warning(err)
            if sys.platform == "darwin":
                raise unittest.SkipTest("Byte-offset decompression is known to be buggy on MacOS-CPU")
            else:
                raise err

        t0 = time.time()
        res_cy = fabio.compression.decByteOffset(raw)
        t1 = time.time()
        res_cl = bo(raw)
        t2 = time.time()
        delta_cy = abs(ref.ravel() - res_cy).max()
        delta_cl = abs(ref.ravel() - res_cl.get()).max()
        self.assertEqual(delta_cy, 0, "Checks fabio works")
        self.assertEqual(delta_cl, 0, "Checks opencl works")

        logger.debug("Global execution time: fabio %.3fms, OpenCL: %.3fms.",
                     1000.0 * (t1 - t0),
                     1000.0 * (t2 - t1))
        bo.log_profile()

    def test_many_decompress(self, ntest=10):
        """
        tests the byte offset decompression on GPU, many images to ensure there 
        is not leaking in memory 
        """
        shape = (991, 997)
        size = numpy.prod(shape)
        ref, raw = self._create_test_data(shape=shape, nexcept=0, lam=100)

        try:
            bo = byte_offset.ByteOffset(len(raw), size, profile=False)
        except (RuntimeError, pyopencl.RuntimeError) as err:
            logger.warning(err)
            if sys.platform == "darwin":
                raise unittest.SkipTest("Byte-offset decompression is known to be buggy on MacOS-CPU")
            else:
                raise err
        t0 = time.time()
        res_cy = fabio.compression.decByteOffset(raw)
        t1 = time.time()
        res_cl = bo(raw)
        t2 = time.time()
        delta_cy = abs(ref.ravel() - res_cy).max()
        delta_cl = abs(ref.ravel() - res_cl.get()).max()
        self.assertEqual(delta_cy, 0, "Checks fabio works")
        self.assertEqual(delta_cl, 0, "Checks opencl works")
        logger.debug("Global execution time: fabio %.3fms, OpenCL: %.3fms.",
                     1000.0 * (t1 - t0),
                     1000.0 * (t2 - t1))

        for i in range(ntest):
            ref, raw = self._create_test_data(shape=shape, nexcept=2729, lam=200)

            t0 = time.time()
            res_cy = fabio.compression.decByteOffset(raw)
            t1 = time.time()
            res_cl = bo(raw)
            t2 = time.time()
            delta_cy = abs(ref.ravel() - res_cy).max()
            delta_cl = abs(ref.ravel() - res_cl.get()).max()
            self.assertEqual(delta_cy, 0, "Checks fabio works #%i" % i)
            self.assertEqual(delta_cl, 0, "Checks opencl works #%i" % i)

            logger.debug("Global execution time: fabio %.3fms, OpenCL: %.3fms.",
                         1000.0 * (t1 - t0),
                         1000.0 * (t2 - t1))

    def test_compress(self):
        """Test byte offset compression"""
        ref, raw = self._create_test_data(shape=(2713, 2719), nexcept=2729)
        size = numpy.prod(ref.shape)

        try:
            bo = byte_offset.ByteOffset(len(raw), size, profile=True)
        except (RuntimeError, pyopencl.RuntimeError) as err:
            logger.warning(err)
            if sys.platform == "darwin":
                raise unittest.SkipTest("Byte-offset decompression is known to be buggy on MacOS-CPU")
            else:
                raise err

        compressed_array = bo.encode(ref)
        compressed_stream = compressed_array.get().tostring()
        self.assertEqual(raw, compressed_stream)


def suite():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestByteOffset("test_decompress"))
    testSuite.addTest(TestByteOffset("test_many_decompress"))
    testSuite.addTest(TestByteOffset("test_compress"))
    return testSuite
