# coding: utf-8
#/*##########################################################################
#
# Copyright (c) 2004-2016 European Synchrotron Radiation Facility
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
########################################################################### */
"""
"""

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "26/07/2016"


class FitTheory(object):
    """This class defines a fit theory, which consists of:

        - a model function, the actual function to be fitted
        - parameters names
        - an estimation function, that return the estimated initial parameters
          that serve as input for :func:`silx.math.fit.leastsq`
        - an optional configuration function, that can be used to modify
          configuration parameters to alter the behavior of the fit function
          and the estimation function
        - an optional derivative function, that replaces the default model
          derivative used in :func:`silx.math.fit.leastsq`
        - an optional configuration widget, that can be called for
          interactively modifying the configuration when running a fit from
          a GUI
    """
    def __init__(self, function, parameters, estimate,
                 configure=None, derivative=None,
                 config_widget=None, description=None):
        self.function = function
        """The function must have the signature ``f(x, *params)``, where ``x``
        is an array of values for the independent variable, and ``params`` are
        the parameters to be fitted.

        The number of parameters must be the same as in :attr:`parameters`, or
        a multiple of this number if the function is defined as a sum of a
        variable number of base functions and if :attr:`estimate` is designed
        to be able to estimate the number of needed base functions."""

        self.parameters = parameters
        """List of parameters names.

        This list can contain the minimum number of parameters, if the
        function takes a variable number of parameters,
        and if the estimation function is responsible for finding the number
        of required parameters """

        self.estimate = estimate
        """The estimation function must have the following signature::

            ``f(x, y, bg, yscaling)``

        Where ``x`` is an array of values for the independent variable, ``y``
        is an array of the same length as ``x`` containing the data to be
        fitted, ``bg`` is an array of background signal to be subtracted from
        ``y`` before running the fit and ``yscaling`` is a scaling factor that
        the function may multiply ``y`` values with for certain operations
        (such as searching peaks in the data)."""
        # TODO remove bg and scaling

        self.configure = configure
        """The optional configuration function must conform to the signature
        ``f(**kw) -> dict`` (i.e it must accept any named argument and
        return a dictionary).
        It can be used to modify configuration parameters to alter the
        behavior of the fit function and the estimation function."""

        self.derivative = derivative
        """The optional derivative function must conform to the signature
        ``model_deriv(xdata, parameters, index)``, where parameters is a
        sequence with the current values of the fitting parameters, index is
        the fitting parameter index for which the the derivative has to be
        provided in the supplied array of xdata points."""

        self.config_widget = config_widget
        """Optional configuration widget"""

        self.description = description
        """Optional description string for this particular fit theory."""


