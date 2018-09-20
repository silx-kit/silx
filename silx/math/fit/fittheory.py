# coding: utf-8
#/*##########################################################################
#
# Copyright (c) 2004-2018 European Synchrotron Radiation Facility
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
This module defines the :class:`FitTheory` object that is used by
:class:`silx.math.fit.FitManager` to define fit functions and background
models.
"""

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "09/08/2016"


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
    """
    def __init__(self, function, parameters,
                 estimate=None, configure=None, derivative=None,
                 description=None, pymca_legacy=False, is_background=False):
        """
        :param function function: Actual function. See documentation for
            :attr:`function`.
        :param list[str] parameters: List of parameter names for the function.
            See documentation for :attr:`parameters`.
        :param function estimate: Optional estimation function.
            See documentation for :attr:`estimate`
        :param function configure: Optional configuration function.
            See documentation for :attr:`configure`
        :param function derivative: Optional custom derivative function.
            See documentation for :attr:`derivative`
        :param str description: Optional description string.
            See documentation for :attr:`description`
        :param bool pymca_legacy: Flag to indicate that the theory is a PyMca
            legacy theory. See documentation for :attr:`pymca_legacy`
        :param bool is_background: Flag to indicate that the theory is a
            background theory. This has implications regarding the function's
            signature, as explained in the documentation for :attr:`function`.
        """
        self.function = function
        """Regular fit functions must have the signature ``f(x, *params) -> y``,
        where *x* is a 1D array of values for the independent variable,
        *params* are the parameters to be fitted and *y* is the output array
        that we want to have the best fit to a series of data points.

        Background functions used by :class:`FitManager` must have a slightly
        different signature: ``f(x, y0, *params) -> bg``, where *y0* is the
        array of original data points and *bg* is the background signal that
        we want to subtract from the data array prior to fitting the regular
        fit function.

        The number of parameters must be the same as in :attr:`parameters`, or
        a multiple of this number if the function is defined as a sum of a
        variable number of base functions and if :attr:`estimate` is designed
        to be able to estimate the number of needed base functions.
        """

        self.parameters = parameters
        """List of parameters names.

        This list can contain the minimum number of parameters, if the
        function takes a variable number of parameters,
        and if the estimation function is responsible for finding the number
        of required parameters """

        self.estimate = estimate
        """The estimation function should have the following signature::

            f(x, y) -> (estimated_param, constraints)

        Parameters:

            - ``x`` is a sequence of values for the independent variable
            - ``y`` is a sequence of the same length as ``x`` containing the
              data to be fitted

        Return values:

            - ``estimated_param`` is a sequence of estimated fit parameters to
              be used as initial values for an iterative fit.
            - ``constraints`` is a sequence of shape *(n, 3)*, where *n* is the
              number of estimated parameters, containing the constraints for each
              parameter to be fitted. See :func:`silx.math.fit.leastsq` for more
              explanations about constraints."""
        if estimate is None:
            self.estimate = self.default_estimate

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

        self.description = description
        """Optional description string for this particular fit theory."""

        self.pymca_legacy = pymca_legacy
        """This attribute can be set to *True* to indicate that the theory
        is a PyMca legacy theory.

        This tells :mod:`silx.math.fit.fitmanager` that the signature of
        the estimate function is::

            f(x, y, bg, xscaling, yscaling) -> (estimated_param, constraints)
        """

        self.is_background = is_background
        """Flag to indicate that the theory is background theory.

        A background function is an secondary function that needs to be added
        to the main fit function to better fit the original data.
        If this flag is set to *True*, modules using this theory are informed
        that :attr:`function` has the signature ``f(x, y0, *params) -> bg``,
        instead of the usual fit function signature."""

    def default_estimate(self, x=None, y=None, bg=None):
        """Default estimate function. Return an array of *ones* as the
        initial estimated parameters, and set all constraints to zero
        (FREE)"""
        estimated_parameters = [1. for _ in self.parameters]
        estimated_constraints = [[0, 0, 0] for _ in self.parameters]
        return estimated_parameters, estimated_constraints
