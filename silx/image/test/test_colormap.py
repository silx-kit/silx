# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2018 European Synchrotron Radiation Facility
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
"""Test for colormap mapping implementation"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "16/05/2018"


import logging
import unittest
import numpy

from silx.utils.testutils import ParametricTestCase
from silx.image import colormap


_logger = logging.getLogger(__name__)


class TestColormap(ParametricTestCase):
    """Test silx.image.colormap.cmap"""

    def test(self):
        """Test all dtypes with finite data"""
        colors = numpy.zeros((256, 4), dtype=numpy.uint8)
        colors[:, 0] = numpy.arange(len(colors))
        colors[:, 3] = 255

        nan_color = (0, 0, 0, 0)
        vmin = 1.
        vmax = 7.

        # Generates (u)int and floats types
        dtypes = [e + k + i for e in '<>' for k in 'uif' for i in '1248'
                  if k != 'f' or i != '1']
        dtypes.append(numpy.dtype(numpy.longdouble).name)  # Add long double

        for normalization in ('linear', 'log', 'arcsinh', 'sqrt'):
            for dtype in dtypes:
                with self.subTest(dtype=dtype, normalization=normalization):
                    _logger.info('dtype: %s normalization: %s', dtype, normalization)
                    data = numpy.arange(-10, 10, dtype=dtype).reshape(4, 5)
                    image = colormap.cmap(
                        data, colors, vmin, vmax, normalization, nan_color)
                    self.assertEqual(image.shape, (4, 5, 4))
                    self.assertEqual(image.dtype, colors.dtype)


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestColormap))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
