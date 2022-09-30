# /*##########################################################################
# Copyright (C) 2016 European Synchrotron Radiation Facility
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
import numpy
import unittest
from silx.math.fit import filters
from silx.math.fit import functions
from silx.test.utils import add_relative_noise


class TestSmooth(unittest.TestCase):
    """
    Unit tests of smoothing functions.

    Test that the difference between a synthetic curve with 5% added random
    noise and the result of smoothing that signal is less than 5%. We compare
    the sum of all samples in each curve.
    """
    def setUp(self):
        x = numpy.arange(5000)
        # (height1, center1, fwhm1, beamfwhm...)
        slit_params = (50, 500, 200, 100,
                       50, 600, 80, 30,
                       20, 2000, 150, 150,
                       50, 2250, 110, 100,
                       40, 3000, 50, 10,
                       23, 4980, 250, 20)

        self.y1 = functions.sum_slit(x, *slit_params)
        # 5% noise
        self.y1 = add_relative_noise(self.y1, 5.)

        # (height1, center1, fwhm1...)
        step_params = (50, 500, 200,
                       50, 600, 80,
                       20, 2000, 150,
                       50, 2250, 110,
                       40, 3000, 50,
                       23, 4980, 250,)

        self.y2 = functions.sum_stepup(x, *step_params)
        # 5% noise
        self.y2 = add_relative_noise(self.y2, 5.)

        self.y3 = functions.sum_stepdown(x, *step_params)
        # 5% noise
        self.y3 = add_relative_noise(self.y3, 5.)

    def tearDown(self):
        pass

    def testSavitskyGolay(self):
        npts = 25
        for y in [self.y1, self.y2, self.y3]:
            smoothed_y = filters.savitsky_golay(y, npoints=npts)

            # we added +-5% of random noise. The difference must be much lower
            # than 5%.
            diff = abs(sum(smoothed_y) - sum(y)) / sum(y)
            self.assertLess(diff, 0.05,
                            "Difference between data with 5%% noise and " +
                            "smoothed data is > 5%% (%f %%)" % (diff * 100))

            # Try various smoothing levels
            npts += 25

    def testSmooth1d(self):
        """Test the 1D smoothing against the formula
        ys[i] = (y[i-1] + 2 * y[i] + y[i+1]) / 4   (for 1 < i < n-1)"""
        smoothed_y = filters.smooth1d(self.y1)

        for i in range(1, len(self.y1) - 1):
            self.assertAlmostEqual(4 * smoothed_y[i],
                                   self.y1[i-1] + 2 * self.y1[i] + self.y1[i+1])

    def testSmooth2d(self):
        """Test that a 2D smoothing is the same as two successive and
        orthogonal 1D smoothings"""
        x = numpy.arange(10000)

        noise = 2 * numpy.random.random(10000) - 1
        noise *= 0.05
        y = x * (1 + noise)

        y.shape = (100, 100)

        smoothed_y = filters.smooth2d(y)

        intermediate_smooth = numpy.zeros_like(y)
        expected_smooth = numpy.zeros_like(y)
        # smooth along first dimension
        for i in range(0, y.shape[0]):
            intermediate_smooth[i, :] = filters.smooth1d(y[i, :])

        # smooth along second dimension
        for j in range(0, y.shape[1]):
            expected_smooth[:, j] = filters.smooth1d(intermediate_smooth[:, j])

        for i in range(0, y.shape[0]):
            for j in range(0, y.shape[1]):
                self.assertAlmostEqual(smoothed_y[i, j],
                                       expected_smooth[i, j])
