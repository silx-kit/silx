# /*##########################################################################
# Copyright (C) 2018 European Synchrotron Radiation Facility
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
This module provides classes to calibrate data.

Classes
-------

- :class:`NoCalibration`
- :class:`LinearCalibration`
- :class:`ArrayCalibration`

"""
import numpy


class AbstractCalibration(object):
    """A calibration is a transformation to be applied to an axis (i.e. a 1D array).

    """
    def __init__(self):
        super(AbstractCalibration, self).__init__()

    def __call__(self, x):
        """Apply calibration to an axis or to a value.

        :param x: Axis (1-D array), or value"""
        raise NotImplementedError(
                "AbstractCalibration can not be used directly. " +
                "You must subclass it and implement __call__")

    def is_affine(self):
        """Returns True for an affine calibration of the form
        :math:`x  \\mapsto a + b * x`, or else False.
        """
        return False

    def get_slope(self):
        raise NotImplementedError(
                "get_slope is implemented only for affine calibrations")


class NoCalibration(AbstractCalibration):
    """No calibration :math:`x \\mapsto x`
    """
    def __init__(self):
        super(NoCalibration, self).__init__()

    def __call__(self, x):
        return x

    def is_affine(self):
        return True

    def get_slope(self):
        return 1.


class LinearCalibration(AbstractCalibration):
    """Linear calibration :math:`x \\mapsto a + b x`,
    where *a* is the y-intercept and *b* is the slope.

    :param y_intercept: y-intercept
    :param slope: Slope of the affine transformation
    """
    def __init__(self, y_intercept, slope):
        super(LinearCalibration, self).__init__()
        self.constant = y_intercept
        self.slope = slope

    def __call__(self, x):
        return self.constant + self.slope * x

    def is_affine(self):
        return True

    def get_slope(self):
        return self.slope


class ArrayCalibration(AbstractCalibration):
    """One-to-one mapping calibration, defined by an array *x'*,
    such as :math:`x \\mapsto x'`.

    This calibration can only be applied to x arrays of the same length as the
    calibration array *x'*.
    It is typically applied to an axis of indices or
    channels (:math:`0, 1, ..., n-1`).

    :param x1: Calibration array"""
    def __init__(self, x1):
        super(ArrayCalibration, self).__init__()
        if not isinstance(x1, (list, tuple)) and not hasattr(x1, "shape"):
            raise TypeError(
                    "The calibration array must be a sequence (list, dataset, array)")
        self.calibration_array = numpy.array(x1)
        self._is_affine = None

    def __call__(self, x):
        # calibrate the entire axis
        if isinstance(x, (list, tuple, numpy.ndarray)) and \
                        len(self.calibration_array) == len(x):
            return self.calibration_array
        # calibrate one value, by index
        if isinstance(x, int) and x < len(self.calibration_array):
            return self.calibration_array[x]
        raise ValueError("ArrayCalibration must be applied to array of same size "
                         "or to index.")

    def is_affine(self):
        """If all values in the calibration array are regularly spaced,
        return True."""
        if self._is_affine is None:
            delta_x = self.calibration_array[1:] - self.calibration_array[:-1]
            # use a less strict relative tolerance to account for rounding errors
            # e.g. when using float64 into float32 (see #1823)
            if not numpy.isclose(delta_x, delta_x[0], rtol=1e-4).all():
                self._is_affine = False
            else:
                self._is_affine = True
        return self._is_affine

    def get_slope(self):
        """If the calibration array is regularly spaced, return the spacing."""
        if not self.is_affine():
            raise AttributeError(
                "get_slope only makes sense for affine transformations"
            )
        return self.calibration_array[1] - self.calibration_array[0]


class FunctionCalibration(AbstractCalibration):
    """Calibration defined by a function *f*, such as :math:`x \\mapsto f(x)`*.

    :param function: Calibration function"""
    def __init__(self, function, is_affine=False):
        super(FunctionCalibration, self).__init__()
        if not hasattr(function, "__call__"):
            raise TypeError("The calibration function must be a callable")
        self.function = function
        self._is_affine = is_affine

    def __call__(self, x):
        return self.function(x)

    def is_affine(self):
        """Return True if calibration is affine.
        This is False by default, unless the object is instantiated with
        ``is_affine=True``."""
        return self._is_affine

    def get_slope(self):
        """If the calibration array is regularly spaced, return the spacing."""
        if not self.is_affine():
            raise AttributeError(
                "get_slope only makes sense for affine transformations"
            )
        # fixme: what if function is not defined at x=1 or x=2?
        return self.function(2) - self.function(1)
