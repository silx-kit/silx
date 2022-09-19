# /*##########################################################################
#
# Copyright (c) 2018-2021 European Synchrotron Radiation Facility
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
import sys

import numpy

from silx.utils.testutils import ParametricTestCase
from silx.math import colormap


_logger = logging.getLogger(__name__)


class TestNormalization(ParametricTestCase):
    """Test silx.math.colormap.Normalization sub classes"""

    def _testCodec(self, normalization, rtol=1e-5):
        """Test apply/revert for normalizations"""
        test_data = (numpy.arange(1, 10, dtype=numpy.int32),
                     numpy.linspace(1., 100., 1000, dtype=numpy.float32),
                     numpy.linspace(-1., 1., 100, dtype=numpy.float32),
                     1.,
                     1)

        for index in range(len(test_data)):
            with self.subTest(normalization=normalization, data_index=index):
                data = test_data[index]
                normalized = normalization.apply(data, 1., 100.)
                result = normalization.revert(normalized, 1., 100.)

                self.assertTrue(numpy.array_equal(
                    numpy.isnan(normalized), numpy.isnan(result)))

                if isinstance(data, numpy.ndarray):
                    notNaN = numpy.logical_not(numpy.isnan(result))
                    data = data[notNaN]
                    result = result[notNaN]
                self.assertTrue(numpy.allclose(data, result, rtol=rtol))

    def testLinearNormalization(self):
        """Test for LinearNormalization"""
        normalization = colormap.LinearNormalization()
        self._testCodec(normalization)

    def testLogarithmicNormalization(self):
        """Test for LogarithmicNormalization"""
        normalization = colormap.LogarithmicNormalization()
        # relative tolerance is higher because of the log approximation
        self._testCodec(normalization, rtol=1e-3)

        # Specific extra tests
        self.assertTrue(numpy.isnan(normalization.apply(-1., 1., 100.)))
        self.assertTrue(numpy.isnan(normalization.apply(numpy.nan, 1., 100.)))
        self.assertEqual(normalization.apply(numpy.inf, 1., 100.), numpy.inf)
        self.assertEqual(normalization.apply(0, 1., 100.), - numpy.inf)

    def testArcsinhNormalization(self):
        """Test for ArcsinhNormalization"""
        self._testCodec(colormap.ArcsinhNormalization())

    def testSqrtNormalization(self):
        """Test for SqrtNormalization"""
        normalization = colormap.SqrtNormalization()
        self._testCodec(normalization)

        # Specific extra tests
        self.assertTrue(numpy.isnan(normalization.apply(-1., 0., 100.)))
        self.assertTrue(numpy.isnan(normalization.apply(numpy.nan, 0., 100.)))
        self.assertEqual(normalization.apply(numpy.inf, 0., 100.), numpy.inf)
        self.assertEqual(normalization.apply(0, 0., 100.), 0.)


class TestColormap(ParametricTestCase):
    """Test silx.math.colormap.cmap"""

    NORMALIZATIONS = (
        'linear',
        'log',
        'arcsinh',
        'sqrt',
        colormap.LinearNormalization(),
        colormap.LogarithmicNormalization(),
        colormap.GammaNormalization(2.),
        colormap.GammaNormalization(0.5))

    @staticmethod
    def ref_colormap(data, colors, vmin, vmax, normalization, nan_color):
        """Reference implementation of colormap

        :param numpy.ndarray data: Data to convert
        :param numpy.ndarray colors: Color look-up-table
        :param float vmin: Lower bound of the colormap range
        :param float vmax: Upper bound of the colormap range
        :param str normalization: Normalization to use
        :param Union[numpy.ndarray, None] nan_color: Color to use for NaN
        """
        norm_functions = {'linear': lambda v: v,
                          'log': numpy.log10,
                          'arcsinh': numpy.arcsinh,
                          'sqrt': numpy.sqrt}

        if isinstance(normalization, str):
            norm_function = norm_functions[normalization]
        else:
            def norm_function(value):
                return normalization.apply(value, vmin, vmax)

        with numpy.errstate(divide='ignore', invalid='ignore'):
            # Ignore divide by zero and invalid value encountered in log10, sqrt
            norm_data, vmin, vmax = map(norm_function, (data, vmin, vmax))

        if normalization == 'arcsinh' and sys.platform == 'win32':
            # There is a difference of behavior of numpy.arcsinh
            # between Windows and other OS for results of infinite values
            # This makes Windows behaves as Linux and MacOS
            norm_data[data == numpy.inf] = numpy.inf
            norm_data[data == -numpy.inf] = -numpy.inf

        nb_colors = len(colors)
        scale = nb_colors / (vmax - vmin)

        # Substraction must be done in float to avoid overflow with uint
        indices = numpy.clip(scale * (norm_data - float(vmin)),
                             0, nb_colors - 1)
        indices[numpy.isnan(indices)] = nb_colors  # Use an extra index for NaN
        indices = indices.astype('uint')

        # Add NaN color to array
        if nan_color is None:
            nan_color = (0,) * colors.shape[-1]
        colors = numpy.append(colors, numpy.atleast_2d(nan_color), axis=0)

        return colors[indices]

    def _test(self, data, colors, vmin, vmax, normalization, nan_color):
        """Run test of colormap against alternative implementation

        :param numpy.ndarray data: Data to convert
        :param numpy.ndarray colors: Color look-up-table
        :param float vmin: Lower bound of the colormap range
        :param float vmax: Upper bound of the colormap range
        :param str normalization: Normalization to use
        :param Union[numpy.ndarray, None] nan_color: Color to use for NaN
        """
        image = colormap.cmap(
            data, colors, vmin, vmax, normalization, nan_color)

        ref_image = self.ref_colormap(
            data, colors, vmin, vmax, normalization, nan_color)

        self.assertTrue(numpy.allclose(ref_image, image))
        self.assertEqual(image.dtype, colors.dtype)
        self.assertEqual(image.shape, data.shape + (colors.shape[-1],))

    def test(self):
        """Test all dtypes with finite data

        Test all supported types and endianness
        """
        colors = numpy.zeros((256, 4), dtype=numpy.uint8)
        colors[:, 0] = numpy.arange(len(colors))
        colors[:, 3] = 255

        # Generates (u)int and floats types
        dtypes = [e + k + i for e in '<>' for k in 'uif' for i in '1248'
                  if k != 'f' or i != '1']
        dtypes.append(numpy.dtype(numpy.longdouble).name)  # Add long double

        for normalization in self.NORMALIZATIONS:
            for dtype in dtypes:
                with self.subTest(dtype=dtype, normalization=normalization):
                    _logger.info('normalization: %s, dtype: %s',
                                 normalization, dtype)
                    data = numpy.arange(-5, 15, dtype=dtype).reshape(4, 5)

                    self._test(data, colors, 1, 10, normalization, None)

    def test_not_finite(self):
        """Test float data with not finite values"""
        colors = numpy.zeros((256, 4), dtype=numpy.uint8)
        colors[:, 0] = numpy.arange(len(colors))
        colors[:, 3] = 255

        test_data = {  # message: data
            'no finite values': (float('inf'), float('-inf'), float('nan')),
            'only NaN': (float('nan'), float('nan'), float('nan')),
            'mix finite/not finite': (float('inf'), float('-inf'), 1., float('nan')),
        }

        for normalization in self.NORMALIZATIONS:
            for msg, data in test_data.items():
                with self.subTest(msg, normalization=normalization):
                    _logger.info('normalization: %s, %s', normalization, msg)
                    data = numpy.array(data, dtype=numpy.float64)
                    self._test(data, colors, 1, 10, normalization, (0, 0, 0, 0))

    def test_errors(self):
        """Test raising exception for bad vmin, vmax, normalization parameters
        """
        colors = numpy.zeros((256, 4), dtype=numpy.uint8)
        colors[:, 0] = numpy.arange(len(colors))
        colors[:, 3] = 255

        data = numpy.arange(10, dtype=numpy.float64)

        test_params = [  # (vmin, vmax, normalization)
            (-1., 2., 'log'),
            (0., 1., 'log'),
            (1., 0., 'log'),
            (-1., 1., 'sqrt'),
            (1., -1., 'sqrt'),
        ]

        for vmin, vmax, normalization in test_params:
            with self.subTest(
                    vmin=vmin, vmax=vmax, normalization=normalization):
                _logger.info('normalization: %s, range: [%f, %f]',
                             normalization, vmin, vmax)
                with self.assertRaises(ValueError):
                    self._test(data, colors, vmin, vmax, normalization, None)


def test_apply_colormap():
    """Basic test of silx.math.colormap.apply_colormap"""
    data = numpy.arange(256)
    expected_colors = numpy.empty((256, 4), dtype=numpy.uint8)
    expected_colors[:, :3] = numpy.arange(256, dtype=numpy.uint8).reshape(256, 1)
    expected_colors[:, 3] = 255
    colors = colormap.apply_colormap(
        data,
        colormap="gray",
        norm="linear",
        autoscale="minmax",
        vmin=None,
        vmax=None,
        gamma=1.0)
    assert numpy.array_equal(colors, expected_colors)
