#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Median filter of images + OpenCL
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
Simple test of the median filter
"""

from __future__ import division, print_function

__authors__ = ["Jérôme Kieffer"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "MIT"
__copyright__ = "2013-2017 European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "05/07/2018"


import sys
import time
import logging
import numpy
import unittest
from collections import namedtuple
try:
    import mako
except ImportError:
    mako = None
from ..common import ocl
if ocl:
    import pyopencl
    import pyopencl.array
    from .. import medfilt

logger = logging.getLogger(__name__)

Result = namedtuple("Result", ["size", "error", "sp_time", "oc_time"])

try:
    from scipy.misc import ascent
except:
    def ascent():
        """Dummy image from random data"""
        return numpy.random.random((512, 512))
try:
    from scipy.ndimage import filters
    median_filter = filters.median_filter
    HAS_SCIPY = True
except:
    HAS_SCIPY = False
    from silx.math import medfilt2d as median_filter

@unittest.skipUnless(ocl and mako, "PyOpenCl is missing")
class TestMedianFilter(unittest.TestCase):

    def setUp(self):
        if ocl is None:
            return
        self.data = ascent().astype(numpy.float32)
        self.medianfilter = medfilt.MedianFilter2D(self.data.shape, devicetype="gpu")

    def tearDown(self):
        self.data = None
        self.medianfilter = None

    def measure(self, size):
        "Common measurement of accuracy and timings"
        t0 = time.time()
        if HAS_SCIPY:
            ref = median_filter(self.data, size, mode="nearest")
        else:
            ref = median_filter(self.data, size)
        t1 = time.time()
        try:
            got = self.medianfilter.medfilt2d(self.data, size)
        except RuntimeError as msg:
            logger.error(msg)
            return
        t2 = time.time()
        delta = abs(got - ref).max()
        return Result(size, delta, t1 - t0, t2 - t1)

    @unittest.skipUnless(ocl and mako, "pyopencl is missing")
    def test_medfilt(self):
        """
        tests the median filter kernel
        """
        r = self.measure(size=11)
        if r is None:
            logger.info("test_medfilt: size: %s: skipped")
        else:
            logger.info("test_medfilt: size: %s error %s, t_ref: %.3fs, t_ocl: %.3fs" % r)
            self.assertEqual(r.error, 0, 'Results are correct')

    def benchmark(self, limit=36):
        "Run some benchmarking"
        try:
            import PyQt5
            from ...gui.matplotlib import pylab
            from ...gui.utils import update_fig
        except:
            pylab = None

            def update_fig(*ag, **kwarg):
                pass

        fig = pylab.figure()
        fig.suptitle("Median filter of an image 512x512")
        sp = fig.add_subplot(1, 1, 1)
        sp.set_title(self.medianfilter.ctx.devices[0].name)
        sp.set_xlabel("Window width & height")
        sp.set_ylabel("Execution time (s)")
        sp.set_xlim(2, limit + 1)
        sp.set_ylim(0, 4)
        data_size = []
        data_scipy = []
        data_opencl = []
        plot_sp = sp.plot(data_size, data_scipy, "-or", label="scipy")[0]
        plot_opencl = sp.plot(data_size, data_opencl, "-ob", label="opencl")[0]
        sp.legend(loc=2)
        fig.show()
        update_fig(fig)
        for s in range(3, limit, 2):
            r = self.measure(s)
            print(r)
            if r.error == 0:
                data_size.append(s)
                data_scipy.append(r.sp_time)
                data_opencl.append(r.oc_time)
                plot_sp.set_data(data_size, data_scipy)
                plot_opencl.set_data(data_size, data_opencl)
                update_fig(fig)
        fig.show()
        if sys.version_info[0] < 3:
            raw_input()
        else:
            input()


def suite():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestMedianFilter("test_medfilt"))
    return testSuite


def benchmark():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestMedianFilter("benchmark"))
    return testSuite


if __name__ == '__main__':
    unittest.main(defaultTest="suite")
