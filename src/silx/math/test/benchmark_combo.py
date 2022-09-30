# /*##########################################################################
# Copyright (C) 2016-2017 European Synchrotron Radiation Facility
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
"""Benchmarks of the combo module"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "17/01/2018"


import logging
import os.path
import time
import unittest

import numpy

from silx.test.utils import temp_dir
from silx.utils.testutils import ParametricTestCase

from silx.math import combo

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)


class TestBenchmarkMinMax(ParametricTestCase):
    """Benchmark of min max combo"""

    DTYPES = ('float32', 'float64',
              'int8', 'int16', 'int32', 'int64',
              'uint8', 'uint16', 'uint32', 'uint64')

    ARANGE = 'ascent', 'descent', 'random'

    EXPONENT = 3, 4, 5, 6, 7

    def test_benchmark_min_max(self):
        """Benchmark min_max without min positive.
        
        Compares with:
        
        - numpy.nanmin, numpy.nanmax and
        - numpy.argmin, numpy.argmax

        It runs bench for different types, different data size and 3
        data sets: increasing , decreasing and random data.
        """
        durations = {'min/max': [], 'argmin/max': [], 'combo': []}

        _logger.info('Benchmark against argmin/argmax and nanmin/nanmax')

        for dtype in self.DTYPES:
            for arange in self.ARANGE:
                for exponent in self.EXPONENT:
                    size = 10**exponent
                    with self.subTest(dtype=dtype, size=size, arange=arange):
                        if arange == 'ascent':
                            data = numpy.arange(0, size, 1, dtype=dtype)
                        elif arange == 'descent':
                            data = numpy.arange(size, 0, -1, dtype=dtype)
                        else:
                            if dtype in ('float32', 'float64'):
                                data = numpy.random.random(size)
                            else:
                                data = numpy.random.randint(10**6, size=size)
                            data = numpy.array(data, dtype=dtype)

                        start = time.time()
                        ref_min = numpy.nanmin(data)
                        ref_max = numpy.nanmax(data)
                        durations['min/max'].append(time.time() - start)

                        start = time.time()
                        ref_argmin = numpy.argmin(data)
                        ref_argmax = numpy.argmax(data)
                        durations['argmin/max'].append(time.time() - start)

                        start = time.time()
                        result = combo.min_max(data, min_positive=False)
                        durations['combo'].append(time.time() - start)

                        _logger.info(
                            '%s-%s-10**%d\tx%.2f argmin/max x%.2f min/max',
                            dtype, arange, exponent,
                            durations['argmin/max'][-1] / durations['combo'][-1],
                            durations['min/max'][-1] / durations['combo'][-1])

                        self.assertEqual(result.minimum, ref_min)
                        self.assertEqual(result.maximum, ref_max)
                        self.assertEqual(result.argmin, ref_argmin)
                        self.assertEqual(result.argmax, ref_argmax)

        self.show_results('min/max', durations, 'combo')

    def test_benchmark_min_pos(self):
        """Benchmark min_max wit min positive.
        
        Compares with:
        
        - numpy.nanmin(data[data > 0]); numpy.nanmin(pos); numpy.nanmax(pos)

        It runs bench for different types, different data size and 3
        data sets: increasing , decreasing and random data.
        """
        durations = {'min/max': [], 'combo': []}

        _logger.info('Benchmark against min, max, positive min')

        for dtype in self.DTYPES:
            for arange in self.ARANGE:
                for exponent in self.EXPONENT:
                    size = 10**exponent
                    with self.subTest(dtype=dtype, size=size, arange=arange):
                        if arange == 'ascent':
                            data = numpy.arange(0, size, 1, dtype=dtype)
                        elif arange == 'descent':
                            data = numpy.arange(size, 0, -1, dtype=dtype)
                        else:
                            if dtype in ('float32', 'float64'):
                                data = numpy.random.random(size)
                            else:
                                data = numpy.random.randint(10**6, size=size)
                            data = numpy.array(data, dtype=dtype)

                        start = time.time()
                        ref_min_positive = numpy.nanmin(data[data > 0])
                        ref_min = numpy.nanmin(data)
                        ref_max = numpy.nanmax(data)
                        durations['min/max'].append(time.time() - start)

                        start = time.time()
                        result = combo.min_max(data, min_positive=True)
                        durations['combo'].append(time.time() - start)

                        _logger.info(
                            '%s-%s-10**%d\tx%.2f min/minpos/max',
                            dtype, arange, exponent,
                            durations['min/max'][-1] / durations['combo'][-1])

                        self.assertEqual(result.min_positive, ref_min_positive)
                        self.assertEqual(result.minimum, ref_min)
                        self.assertEqual(result.maximum, ref_max)

        self.show_results('min/max/min positive', durations, 'combo')

    def show_results(self, title, durations, ref_key):
        try:
            from matplotlib import pyplot
        except ImportError:
            _logger.warning('matplotlib not available')
            return

        pyplot.title(title)
        pyplot.xlabel('-'.join(self.DTYPES))
        pyplot.ylabel('duration (sec)')
        for label, values in durations.items():
            pyplot.semilogy(values, label=label)
        pyplot.legend()
        pyplot.show()

        pyplot.title(title)
        pyplot.xlabel('-'.join(self.DTYPES))
        pyplot.ylabel('Duration ratio')
        ref = numpy.array(durations[ref_key])
        for label, values in durations.items():
            values = numpy.array(values)
            pyplot.plot(values/ref, label=label + ' / ' + ref_key)
        pyplot.legend()
        pyplot.show()
