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
"""Basic tests for Colors"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "17/01/2018"


import numpy

import unittest
from silx.utils.testutils import ParametricTestCase

from silx.gui.plot import Colors
from silx.gui.plot.Colormap import Colormap

class TestRGBA(ParametricTestCase):
    """Basic tests of rgba function"""

    def testRGBA(self):
        """"Test rgba function with accepted values"""
        tests = {  # name: (colors, expected values)
            'blue': ('blue', (0., 0., 1., 1.)),
            '#010203': ('#010203', (1. / 255., 2. / 255., 3. / 255., 1.)),
            '#01020304': ('#01020304', (1. / 255., 2. / 255., 3. / 255., 4. / 255.)),
            '3 x uint8': (numpy.array((1, 255, 0), dtype=numpy.uint8),
                          (1 / 255., 1., 0., 1.)),
            '4 x uint8': (numpy.array((1, 255, 0, 1), dtype=numpy.uint8),
                          (1 / 255., 1., 0., 1 / 255.)),
            '3 x float overflow': ((3., 0.5, 1.), (1., 0.5, 1., 1.)),
        }

        for name, test in tests.items():
            color, expected = test
            with self.subTest(msg=name):
                result = Colors.rgba(color)
                self.assertEqual(result, expected)


class TestApplyColormapToData(ParametricTestCase):
    """Tests of applyColormapToData function"""

    def testApplyColormapToData(self):
        """Simple test of applyColormapToData function"""
        colormap = Colormap(name='gray', normalization='linear',
                        vmin=0, vmax=255)

        size = 10
        expected = numpy.empty((size, 4), dtype='uint8')
        expected[:, 0] = numpy.arange(size, dtype='uint8')
        expected[:, 1] = expected[:, 0]
        expected[:, 2] = expected[:, 0]
        expected[:, 3] = 255

        for dtype in ('uint8', 'int32', 'float32', 'float64'):
            with self.subTest(dtype=dtype):
                array = numpy.arange(size, dtype=dtype)
                result = colormap.applyToData(data=array)
                self.assertTrue(numpy.all(numpy.equal(result, expected)))


def suite():
    test_suite = unittest.TestSuite()
    for testClass in (TestRGBA, TestApplyColormapToData):
        test_suite.addTest(
            unittest.defaultTestLoader.loadTestsFromTestCase(testClass))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
