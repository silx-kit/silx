# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2017 European Synchrotron Radiation Facility
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
"""
Tests that the functions of tomography are valid
"""

__authors__ = ["H. Payno"]
__license__ = "MIT"
__date__ = "12/09/2017"

import unittest
import numpy
from silx.test.utils import utilstest
from silx.image import tomography

class TestTomography(unittest.TestCase):
    """

    """

    def setUp(self):
        self.sinoTrueData = numpy.load(utilstest.getfile("sino500.npz"))["data"]

    def testCalcCenterCentroid(self):
        centerTD = tomography.calc_center_centroid(self.sinoTrueData)
        self.assertTrue(numpy.isclose(centerTD, 256, rtol=0.01))

    def testCalcCenterCorr(self):
        centerTrueData = tomography.calc_center_corr(self.sinoTrueData,
                                                     fullrot=False,
                                                     props=1)
        self.assertTrue(numpy.isclose(centerTrueData, 256, rtol=0.01))


def suite():
    test_suite = unittest.TestSuite()
    for testClass in (TestTomography, ):
        test_suite.addTest(
            unittest.defaultTestLoader.loadTestsFromTestCase(testClass))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
