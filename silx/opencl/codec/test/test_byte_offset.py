#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Byte-offset decompression in OpenCL
#             https://github.com/silx-kit/silx
#
#    Copyright (C) 2013-2017  European Synchrotron Radiation Facility, Grenoble, France
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
__date__ = "17/10/2017"

import time
import logging
import numpy
from silx.opencl import ocl
from silx.opencl.codec import byte_offset
import fabio
import unittest
logger = logging.getLogger(__name__)


@unittest.skipUnless(ocl and fabio, "PyOpenCl or fabio is missing")
class TestByteOffset(unittest.TestCase):

    def test_decompress(self):
        """
        tests the combine (linear combination) kernel
        """
        shape = (2713, 2719)  #  (991, 997)
        size = numpy.prod(shape)
        nexcept = 2729
        ref = numpy.random.poisson(200, numpy.prod(shape))
        exception_loc = numpy.random.randint(0, size, size=nexcept)
        exception_value = numpy.random.randint(0, 1000000, size=nexcept)
        ref[exception_loc] = exception_value
        ref.shape = shape

        raw = fabio.compression.compByteOffset(ref)
        bo = byte_offset.ByteOffset(len(raw), size, profile=True)
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


def suite():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestByteOffset("test_decompress"))
#     testSuite.addTest(TestAlgebra("test_compact"))
    return testSuite
