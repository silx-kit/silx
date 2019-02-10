#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Sift implementation in Python + OpenCL
#             https://github.com/silx-kit/silx
#
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
Simple test of an addition
"""

from __future__ import division, print_function

__authors__ = ["Henri Payno, Jérôme Kieffer"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "MIT"
__copyright__ = "2013 European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "13/12/2018"

import logging
import time
import numpy

import unittest
from ..common import ocl
if ocl:
    import pyopencl
    import pyopencl.array
    from ..statistics import StatResults, Statistics
from ..utils import get_opencl_code
logger = logging.getLogger(__name__)


@unittest.skipUnless(ocl, "PyOpenCl is missing")
class TestStatistics(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.size = 1 << 20  # 1 million elements
        cls.data = numpy.random.randint(0, 65000, cls.size).astype("uint16")
        t0 = time.time()
        cls.ref = StatResults(cls.data.min(), cls.data.max(), cls.data.size,
                              cls.data.sum(), cls.data.mean(), cls.data.std() ** 2,
                              cls.data.std())
        t1 = time.time()
        cls.ref_time = t1 - t0

    @classmethod
    def tearDownClass(cls):
        cls.size = cls.ref = cls.data = cls.ref_time = None

    @classmethod
    def validate(cls, res):
        return (
            (res.min == cls.ref.min) and
            (res.max == cls.ref.max) and
            (res.cnt == cls.ref.cnt) and
            abs(res.mean - cls.ref.mean) < 0.01 and
            abs(res.std - cls.ref.std) < 0.1)

    def test_measurement(self):
        """
        tests that all devices are working properly ...
        """
        logger.info("Reference results: %s", self.ref)
        for pid, platform in enumerate(ocl.platforms):
            for did, device in enumerate(platform.devices):
                try:
                    s = Statistics(template=self.data, platformid=pid, deviceid=did)
                except Exception as err:
                    failed_init = True
                    res = StatResults(0,0,0,0,0,0,0)
                else:
                    failed_init = False
                    t0 = time.time()
                    res = s(self.data)
                    t1 = time.time()
                logger.warning("failed_init %s", failed_init)
                if failed_init or not self.validate(res):
                    logger.error("Failed on platform %s device %s", platform, device)
                    logger.error("Reference results: %s", self.ref)
                    logger.error("Faulty results: %s", res)
                    self.assertTrue(False, "Stat calculation failed on %s %s" % (platform, device))
                logger.info("Runtime on %s/%s : %.3fms x%.1f", platform, device, 1000 * (t1 - t0), self.ref_time / (t1 - t0))


def suite():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestStatistics("test_measurement"))
    return testSuite


if __name__ == '__main__':
    unittest.main(defaultTest="suite")
