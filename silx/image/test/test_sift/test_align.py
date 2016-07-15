#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Sift implementation in Python + OpenCL
#             https://github.com/kif/sift_pyocl
#

"""
Test suite for alignment module
"""

from __future__ import division

__authors__ = ["Jérôme Kieffer"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "2013-08-28"
__license__ = """
Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

"""

import time, os, logging
import numpy
import pyopencl, pyopencl.array
import scipy, scipy.misc, scipy.ndimage
import sys
import unittest
from utilstest import UtilsTest, getLogger, ctx
import sift_pyocl as sift
from sift_pyocl.alignment import LinearAlign
logger = getLogger(__file__)
if logger.getEffectiveLevel() <= logging.INFO:
    PROFILE = True
    import pylab
else:
    PROFILE = False
PRINT_KEYPOINTS = False




class test_linalign(unittest.TestCase):
    def setUp(self):
        self.lena = scipy.misc.lena().astype(numpy.float32)
        self.shape = self.lena.shape
        self.extra = (10, 11)
#        self.img = scipy.ndimage.shift(self.lena, (7, 5))
#        self.img = scipy.ndimage.rotate(self.lena, -20, reshape=False, order=3)
#        self.img = scipy.ndimage.shift(scipy.ndimage.rotate(self.lena, 20, reshape=False, order=3), (7, 5))
        self.img = scipy.ndimage.affine_transform(self.lena, [[1.1, -0.1], [0.05, 0.9]], [7, 5])
        self.align = LinearAlign(self.lena, context=ctx)


    def test_align(self):
        """
        tests the combine (linear combination) kernel
        """
        out = self.align.align(self.img, 0, 1)
        for i in out:
            if i in ["offset","matrix"]:
                print i
                print out[i]
        self.align.log_profile()
        out = out["result"]

        if PROFILE and out is not None:
            fig = pylab.figure()
            sp0 = fig.add_subplot(221)
            im0 = sp0.imshow(self.lena)
            sp1 = fig.add_subplot(222)
            im1 = sp1.imshow(self.img)
            sp2 = fig.add_subplot(223)
            im2 = sp2.imshow(out)
            sp3 = fig.add_subplot(224)
            delta = (out - self.lena)[100:400, 100:400]
            im3 = sp3.imshow(delta)
            print({"min":delta.min(), "max:":delta.max(), "mean":delta.mean(), "std:":delta.std()})
            pylab.show()
            raw_input("enter")



def test_suite_align():
    testSuite = unittest.TestSuite()
    testSuite.addTest(test_linalign("test_align"))
    return testSuite

if __name__ == '__main__':
    mysuite = test_suite_align()
    runner = unittest.TextTestRunner()
    if not runner.run(mysuite).wasSuccessful():
        sys.exit(1)

