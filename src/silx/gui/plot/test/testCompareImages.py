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
# ###########################################################################*/
"""Tests for CompareImages widget"""

__authors__ = ["H. Payno"]
__license__ = "MIT"
__date__ = "23/07/2018"

import unittest
import numpy
import weakref

from silx.gui.utils.testutils import TestCaseQt
from silx.gui.plot.CompareImages import CompareImages


class TestCompareImages(TestCaseQt):
    """Test that CompareImages widget is working in some cases"""

    def setUp(self):
        super(TestCompareImages, self).setUp()
        self.widget = CompareImages()

    def tearDown(self):
        ref = weakref.ref(self.widget)
        self.widget = None
        self.qWaitForDestroy(ref)
        super(TestCompareImages, self).tearDown()

    def testIntensityImage(self):
        image1 = numpy.random.rand(10, 10)
        image2 = numpy.random.rand(10, 10)
        self.widget.setData(image1, image2)

    def testRgbImage(self):
        image1 = numpy.random.randint(0, 255, size=(10, 10, 3))
        image2 = numpy.random.randint(0, 255, size=(10, 10, 3))
        self.widget.setData(image1, image2)

    def testRgbaImage(self):
        image1 = numpy.random.randint(0, 255, size=(10, 10, 4))
        image2 = numpy.random.randint(0, 255, size=(10, 10, 4))
        self.widget.setData(image1, image2)

    def testVizualisations(self):
        image1 = numpy.random.rand(10, 10)
        image2 = numpy.random.rand(10, 10)
        self.widget.setData(image1, image2)
        for mode in CompareImages.VisualizationMode:
            self.widget.setVisualizationMode(mode)

    def testAlignemnt(self):
        image1 = numpy.random.rand(10, 10)
        image2 = numpy.random.rand(5, 5)
        self.widget.setData(image1, image2)
        for mode in CompareImages.AlignmentMode:
            self.widget.setAlignmentMode(mode)

    def testGetPixel(self):
        image1 = numpy.random.rand(11, 11)
        image2 = numpy.random.rand(5, 5)
        image1[5, 5] = 111.111
        image2[2, 2] = 222.222
        self.widget.setData(image1, image2)
        expectedValue = {}
        expectedValue[CompareImages.AlignmentMode.CENTER] = 222.222
        expectedValue[CompareImages.AlignmentMode.STRETCH] = 222.222
        expectedValue[CompareImages.AlignmentMode.ORIGIN] = None
        for mode in expectedValue.keys():
            self.widget.setAlignmentMode(mode)
            data = self.widget.getRawPixelData(11 / 2.0, 11 / 2.0)
            data1, data2 = data
            self.assertEqual(data1, 111.111)
            self.assertEqual(data2, expectedValue[mode])

    def testImageEmpty(self):
        self.widget.setData(image1=None, image2=None)
        self.assertTrue(self.widget.getRawPixelData(11 / 2.0, 11 / 2.0) == (None, None))

    def testSetImageSeparately(self):
        self.widget.setImage1(numpy.random.rand(10, 10))
        self.widget.setImage2(numpy.random.rand(10, 10))
        for mode in CompareImages.VisualizationMode:
            self.widget.setVisualizationMode(mode)


def suite():
    test_suite = unittest.TestSuite()
    loadTests = unittest.defaultTestLoader.loadTestsFromTestCase
    test_suite.addTest(loadTests(TestCompareImages))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
