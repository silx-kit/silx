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

from __future__ import division

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

    def ref_colormap(self, data, colors, vmin, vmax, normalization, nan_color):
        """Reference implementation of colormap

        :param numpy.ndarray data: Data to convert
        :param numpy.ndarray colors: Color look-up-table
        :param float vmin: Lower bound of the colormap range
        :param float vmax: Upper bound of the colormap range
        :param str normalization: Normalization to use
        :param numpy.ndarray nan_color: Color to use for NaN values
        """
        norm_functions = {'linear': lambda v: v,
                          'log': numpy.log10,
                          'arcsinh': numpy.arcsinh,
                          'sqrt': numpy.sqrt}

        norm_function = norm_functions[normalization]
        data, vmin, vmax = map(norm_function, (data, vmin, vmax))

        nb_colors = len(colors)
        scale = nb_colors / (vmax - vmin)

        # Substraction must be done in float to avoid overflow with uint
        indices = numpy.clip(scale * (data - float(vmin)), 0, nb_colors - 1)
        indices[numpy.isnan(indices)] = nb_colors  # Use an extra index for NaN
        indices = indices.astype('uint')

        # Add NaN color to array
        colors = numpy.append(colors, numpy.atleast_2d(nan_color), axis=0)

        return colors[indices]

    def test(self):
        """Test all dtypes with finite data"""
        colors = numpy.zeros((256, 4), dtype=numpy.uint8)
        colors[:, 0] = numpy.arange(len(colors))
        colors[:, 3] = 255

        nan_color = (0, 0, 0, 0)
        vmin = 1
        vmax = 10

        # Generates (u)int and floats types
        dtypes = [e + k + i for e in '<>' for k in 'uif' for i in '1248'
                  if k != 'f' or i != '1']
        dtypes.append(numpy.dtype(numpy.longdouble).name)  # Add long double

        for normalization in ('linear', 'log', 'arcsinh', 'sqrt'):
            for dtype in dtypes:
                with self.subTest(dtype=dtype, normalization=normalization):
                    _logger.info('dtype: %s normalization: %s', dtype, normalization)
                    data = numpy.arange(-5, 15, dtype=dtype).reshape(4, 5)
                    image = colormap.cmap(
                        data, colors, vmin, vmax, normalization, nan_color)

                    ref_image = self.ref_colormap(
                        data, colors, vmin, vmax, normalization, nan_color)

                    self.assertTrue(numpy.allclose(ref_image, image))
                    self.assertEqual(image.dtype, colors.dtype)
                    self.assertEqual(image.shape, (4, 5, 4))


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestColormap))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
