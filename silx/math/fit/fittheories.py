# coding: utf-8
#/*##########################################################################
#
# Copyright (c) 2004-2017 European Synchrotron Radiation Facility
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
"""This modules provides a set of fit functions and associated
estimation functions in a format that can be imported into a
:class:`silx.math.fit.FitManager` instance.

These functions are well suited for fitting multiple gaussian shaped peaks
typically found in spectroscopy data. The estimation functions are designed
to detect how many peaks are present in the data, and provide an initial
estimate for their height, their center location and their full-width
at half maximum (fwhm).

The limitation of these estimation algorithms is that only gaussians having a
similar fwhm can be detected by the peak search algorithm.
This *search fwhm* can be defined by the user, if
he knows the characteristics of his data, or can be automatically estimated
based on the fwhm of the largest peak in the data.

The source code of this module can serve as template for defining your own
fit functions.

The functions to be imported by :meth:`FitManager.loadtheories` are defined by
a dictionary :const:`THEORY`: with the following structure::

    from silx.math.fit.fittheory import FitTheory

    THEORY = {
        'theory_name_1': FitTheory(
                            description='Description of theory 1',
                            function=fitfunction1,
                            parameters=('param name 1', 'param name 2', …),
                            estimate=estimation_function1,
                            configure=configuration_function1,
                            derivative=derivative_function1),

        'theory_name_2':  FitTheory(…),
    }

.. note::

    Consider using an OrderedDict instead of a regular dictionary, when
    defining your own theory dictionary, if the order matters to you.
    This will likely be the case if you intend to load a selection of
    functions in a GUI such as :class:`silx.gui.fit.FitManager`.

Theory names can be customized (e.g. ``gauss, lorentz, splitgauss``…).

The mandatory parameters for :class:`FitTheory` are ``function`` and
``parameters``.

You can also define an ``INIT`` function that will be executed by
:meth:`FitManager.loadtheories`.

See the documentation of :class:`silx.math.fit.fittheory.FitTheory`
for more information.

Module members:
---------------
"""
import numpy
from collections import OrderedDict
import logging

from silx.math.fit import functions
from silx.math.fit.peaks import peak_search, guess_fwhm
from silx.math.fit.filters import strip, savitsky_golay
from silx.math.fit.leastsq import leastsq
from silx.math.fit.fittheory import FitTheory

_logger = logging.getLogger(__name__)

__authors__ = ["V.A. Sole", "P. Knobel"]
__license__ = "MIT"
__date__ = "15/05/2017"


DEFAULT_CONFIG = {
    'NoConstraintsFlag': False,
    'PositiveFwhmFlag': True,
    'PositiveHeightAreaFlag': True,
    'SameFwhmFlag': False,
    'QuotedPositionFlag': False,  # peak not outside data range
    'QuotedEtaFlag': False,  # force 0 < eta < 1
    # Peak detection
    'AutoScaling': False,
    'Yscaling': 1.0,
    'FwhmPoints': 8,
    'AutoFwhm': True,
    'Sensitivity': 2.5,
    'ForcePeakPresence': True,
    # Hypermet
    'HypermetTails': 15,
    'QuotedFwhmFlag': 0,
    'MaxFwhm2InputRatio': 1.5,
    'MinFwhm2InputRatio': 0.4,
    # short tail parameters
    'MinGaussArea4ShortTail': 50000.,
    'InitialShortTailAreaRatio': 0.050,
    'MaxShortTailAreaRatio': 0.100,
    'MinShortTailAreaRatio': 0.0010,
    'InitialShortTailSlopeRatio': 0.70,
    'MaxShortTailSlopeRatio': 2.00,
    'MinShortTailSlopeRatio': 0.50,
    # long tail parameters
    'MinGaussArea4LongTail': 1000.0,
    'InitialLongTailAreaRatio': 0.050,
    'MaxLongTailAreaRatio': 0.300,
    'MinLongTailAreaRatio': 0.010,
    'InitialLongTailSlopeRatio': 20.0,
    'MaxLongTailSlopeRatio': 50.0,
    'MinLongTailSlopeRatio': 5.0,
    # step tail
    'MinGaussHeight4StepTail': 5000.,
    'InitialStepTailHeightRatio': 0.002,
    'MaxStepTailHeightRatio': 0.0100,
    'MinStepTailHeightRatio': 0.0001,
    # Hypermet constraints
    #   position in range [estimated position +- estimated fwhm/2]
    'HypermetQuotedPositionFlag': True,
    'DeltaPositionFwhmUnits': 0.5,
    'SameSlopeRatioFlag': 1,
    'SameAreaRatioFlag': 1,
    # Strip bg removal
    'StripBackgroundFlag': True,
    'SmoothingFlag': True,
    'SmoothingWidth': 5,
    'StripWidth': 2,
    'StripIterations': 5000,
    'StripThresholdFactor': 1.0}
"""This dictionary defines default configuration parameters that have effects
on fit functions and estimation functions, mainly on fit constraints.
This dictionary  is accessible as attribute :attr:`FitTheories.config`,
which can be modified by configuration functions defined in
:const:`CONFIGURE`.
"""

CFREE = 0
CPOSITIVE = 1
CQUOTED = 2
CFIXED = 3
CFACTOR = 4
CDELTA = 5
CSUM = 6
CIGNORED = 7


class FitTheories(object):
    """Class wrapping functions from :class:`silx.math.fit.functions`
    and providing estimate functions for all of these fit functions."""
    def __init__(self, config=None):
        if config is None:
            self.config = DEFAULT_CONFIG
        else:
            self.config = config

    def ahypermet(self, x, *pars):
        """
        Wrapping of :func:`silx.math.fit.functions.sum_ahypermet` without
        the tail flags in the function signature.

        Depending on the value of `self.config['HypermetTails']`, one can
        activate or deactivate the various terms of the hypermet function.

        `self.config['HypermetTails']` must be an integer between 0 and 15.
        It is a set of 4 binary flags, one for activating each one of the
        hypermet terms: *gaussian function, short tail, long tail, step*.

        For example, 15 can be expressed as ``1111`` in base 2, so a flag of
        15 means all terms are active.
        """
        g_term = self.config['HypermetTails'] & 1
        st_term = (self.config['HypermetTails'] >> 1) & 1
        lt_term = (self.config['HypermetTails'] >> 2) & 1
        step_term = (self.config['HypermetTails'] >> 3) & 1
        return functions.sum_ahypermet(x, *pars,
                                       gaussian_term=g_term, st_term=st_term,
                                       lt_term=lt_term, step_term=step_term)

    def poly(self, x, *pars):
        """Order n polynomial.
        The order of the polynomial is defined by the number of
        coefficients (``*pars``).

        """
        p = numpy.poly1d(pars)
        return p(x)

    @staticmethod
    def estimate_poly(x, y, n=2):
        """Estimate polynomial coefficients for a degree n polynomial.

        """
        pcoeffs = numpy.polyfit(x, y, n)
        constraints = numpy.zeros((n + 1, 3), numpy.float)
        return pcoeffs, constraints

    def estimate_quadratic(self, x, y):
        """Estimate quadratic coefficients

        """
        return self.estimate_poly(x, y, n=2)

    def estimate_cubic(self, x, y):
        """Estimate coefficients for a degree 3 polynomial

        """
        return self.estimate_poly(x, y, n=3)

    def estimate_quartic(self, x, y):
        """Estimate coefficients for a degree 4 polynomial

        """
        return self.estimate_poly(x, y, n=4)

    def estimate_quintic(self, x, y):
        """Estimate coefficients for a degree 5 polynomial

        """
        return self.estimate_poly(x, y, n=5)

    def strip_bg(self, y):
        """Return the strip background of y, using parameters from
        :attr:`config` dictionary (*StripBackgroundFlag, StripWidth,
        StripIterations, StripThresholdFactor*)"""
        remove_strip_bg = self.config.get('StripBackgroundFlag', False)
        if remove_strip_bg:
            if self.config['SmoothingFlag']:
                y = savitsky_golay(y, self.config['SmoothingWidth'])
            strip_width = self.config['StripWidth']
            strip_niterations = self.config['StripIterations']
            strip_thr_factor = self.config['StripThresholdFactor']
            return strip(y, w=strip_width,
                         niterations=strip_niterations,
                         factor=strip_thr_factor)
        else:
            return numpy.zeros_like(y)

    def guess_yscaling(self, y):
        """Estimate scaling for y prior to peak search.
        A smoothing filter is applied to y to estimate the noise level
        (chi-squared)

        :param y: Data array
        :return: Scaling factor
        """
        # ensure y is an array
        yy = numpy.array(y, copy=False)

        # smooth
        convolution_kernel = numpy.ones(shape=(3,)) / 3.
        ysmooth = numpy.convolve(y, convolution_kernel, mode="same")

        # remove zeros
        idx_array = numpy.fabs(y) > 0.0
        yy = yy[idx_array]
        ysmooth = ysmooth[idx_array]

        # compute scaling factor
        chisq = numpy.mean((yy - ysmooth)**2 / numpy.fabs(yy))
        if chisq > 0:
            return 1. / chisq
        else:
            return 1.0

    def peak_search(self, y, fwhm, sensitivity):
        """Search for peaks in y array, after padding the array and
        multiplying its value by a scaling factor.

        :param y: 1-D data array
        :param int fwhm: Typical full width at half maximum for peaks,
            in number of points. This parameter is used for to discriminate between
            true peaks and background fluctuations.
        :param float sensitivity: Sensitivity parameter. This is a threshold factor
            for peak detection. Only peaks larger than the standard deviation
            of the noise multiplied by this sensitivity parameter are detected.
        :return: List of peak indices
        """
        # add padding
        ysearch = numpy.ones((len(y) + 2 * fwhm,), numpy.float)
        ysearch[0:fwhm] = y[0]
        ysearch[-1:-fwhm - 1:-1] = y[len(y)-1]
        ysearch[fwhm:fwhm + len(y)] = y[:]

        scaling = self.guess_yscaling(y) if self.config["AutoScaling"] else self.config["Yscaling"]

        if len(ysearch) > 1.5 * fwhm:
            peaks = peak_search(scaling * ysearch,
                                fwhm=fwhm, sensitivity=sensitivity)
            return [peak_index - fwhm for peak_index in peaks
                    if 0 <= peak_index - fwhm < len(y)]
        else:
            return []

    def estimate_height_position_fwhm(self, x, y):
        """Estimation of *Height, Position, FWHM* of peaks, for gaussian-like
        curves.

        This functions finds how many parameters are needed, based on the
        number of peaks detected. Then it estimates the fit parameters
        with a few iterations of fitting gaussian functions.

        :param x: Array of abscissa values
        :param y: Array of ordinate values (``y = f(x)``)
        :return: Tuple of estimated fit parameters and fit constraints.
            Parameters to be estimated for each peak are:
            *Height, Position, FWHM*.
            Fit constraints depend on :attr:`config`.
        """
        fittedpar = []

        bg = self.strip_bg(y)

        if self.config['AutoFwhm']:
            search_fwhm = guess_fwhm(y)
        else:
            search_fwhm = int(float(self.config['FwhmPoints']))
        search_sens = float(self.config['Sensitivity'])

        if search_fwhm < 3:
            _logger.warning("Setting peak fwhm to 3 (lower limit)")
            search_fwhm = 3
            self.config['FwhmPoints'] = 3

        if search_sens < 1:
            _logger.warning("Setting peak search sensitivity to 1. " +
                            "(lower limit to filter out noise peaks)")
            search_sens = 1
            self.config['Sensitivity'] = 1

        npoints = len(y)

        # Find indices of peaks in data array
        peaks = self.peak_search(y,
                                 fwhm=search_fwhm,
                                 sensitivity=search_sens)

        if not len(peaks):
            forcepeak = int(float(self.config.get('ForcePeakPresence', 0)))
            if forcepeak:
                delta = y - bg
                # get index of global maximum
                # (first one if several samples are equal to this value)
                peaks = [numpy.nonzero(delta == delta.max())[0][0]]

        # Find index of largest peak in peaks array
        index_largest_peak = 0
        if len(peaks) > 0:
            # estimate fwhm as 5 * sampling interval
            sig = 5 * abs(x[npoints - 1] - x[0]) / npoints
            peakpos = x[int(peaks[0])]
            if abs(peakpos) < 1.0e-16:
                peakpos = 0.0
            param = numpy.array(
                [y[int(peaks[0])] - bg[int(peaks[0])], peakpos, sig])
            height_largest_peak = param[0]
            peak_index = 1
            for i in peaks[1:]:
                param2 = numpy.array(
                    [y[int(i)] - bg[int(i)], x[int(i)], sig])
                param = numpy.concatenate((param, param2))
                if param2[0] > height_largest_peak:
                    height_largest_peak = param2[0]
                    index_largest_peak = peak_index
                peak_index += 1

            # Subtract background
            xw = x
            yw = y - bg

            cons = numpy.zeros((len(param), 3), numpy.float)

            # peak height must be positive
            cons[0:len(param):3, 0] = CPOSITIVE
            # force peaks to stay around their position
            cons[1:len(param):3, 0] = CQUOTED

            # set possible peak range to estimated peak +- guessed fwhm
            if len(xw) > search_fwhm:
                fwhmx = numpy.fabs(xw[int(search_fwhm)] - xw[0])
                cons[1:len(param):3, 1] = param[1:len(param):3] - 0.5 * fwhmx
                cons[1:len(param):3, 2] = param[1:len(param):3] + 0.5 * fwhmx
            else:
                cons[1:len(param):3, 1] = min(xw) * numpy.ones(
                                                        (param[1:len(param):3]),
                                                        numpy.float)
                cons[1:len(param):3, 2] = max(xw) * numpy.ones(
                                                        (param[1:len(param):3]),
                                                        numpy.float)

            # ensure fwhm is positive
            cons[2:len(param):3, 0] = CPOSITIVE

            # run a quick iterative fit (4 iterations) to improve
            # estimations
            fittedpar, _, _ = leastsq(functions.sum_gauss, xw, yw, param,
                                      max_iter=4, constraints=cons.tolist(),
                                      full_output=True)

        # set final constraints based on config parameters
        cons = numpy.zeros((len(fittedpar), 3), numpy.float)
        peak_index = 0
        for i in range(len(peaks)):
            # Setup height area constrains
            if not self.config['NoConstraintsFlag']:
                if self.config['PositiveHeightAreaFlag']:
                    cons[peak_index, 0] = CPOSITIVE
                    cons[peak_index, 1] = 0
                    cons[peak_index, 2] = 0
            peak_index += 1

            # Setup position constrains
            if not self.config['NoConstraintsFlag']:
                if self.config['QuotedPositionFlag']:
                    cons[peak_index, 0] = CQUOTED
                    cons[peak_index, 1] = min(x)
                    cons[peak_index, 2] = max(x)
            peak_index += 1

            # Setup positive FWHM constrains
            if not self.config['NoConstraintsFlag']:
                if self.config['PositiveFwhmFlag']:
                    cons[peak_index, 0] = CPOSITIVE
                    cons[peak_index, 1] = 0
                    cons[peak_index, 2] = 0
                if self.config['SameFwhmFlag']:
                    if i != index_largest_peak:
                        cons[peak_index, 0] = CFACTOR
                        cons[peak_index, 1] = 3 * index_largest_peak + 2
                        cons[peak_index, 2] = 1.0
            peak_index += 1

        return fittedpar, cons

    def estimate_agauss(self, x, y):
        """Estimation of *Area, Position, FWHM* of peaks, for gaussian-like
        curves.

        This functions uses :meth:`estimate_height_position_fwhm`, then
        converts the height parameters to area under the curve with the
        formula ``area = sqrt(2*pi) * height * fwhm / (2 * sqrt(2 * log(2))``

        :param x: Array of abscissa values
        :param y: Array of ordinate values (``y = f(x)``)
        :return: Tuple of estimated fit parameters and fit constraints.
            Parameters to be estimated for each peak are:
            *Area, Position, FWHM*.
            Fit constraints depend on :attr:`config`.
        """
        fittedpar, cons = self.estimate_height_position_fwhm(x, y)
        # get the number of found peaks
        npeaks = len(fittedpar) // 3
        for i in range(npeaks):
            height = fittedpar[3 * i]
            fwhm = fittedpar[3 * i + 2]
            # Replace height with area in fittedpar
            fittedpar[3 * i] = numpy.sqrt(2 * numpy.pi) * height * fwhm / (
                               2.0 * numpy.sqrt(2 * numpy.log(2)))
        return fittedpar, cons

    def estimate_alorentz(self, x, y):
        """Estimation of *Area, Position, FWHM* of peaks, for Lorentzian
        curves.

        This functions uses :meth:`estimate_height_position_fwhm`, then
        converts the height parameters to area under the curve with the
        formula ``area = height * fwhm * 0.5 * pi``

        :param x: Array of abscissa values
        :param y: Array of ordinate values (``y = f(x)``)
        :return: Tuple of estimated fit parameters and fit constraints.
            Parameters to be estimated for each peak are:
            *Area, Position, FWHM*.
            Fit constraints depend on :attr:`config`.
        """
        fittedpar, cons = self.estimate_height_position_fwhm(x, y)
        # get the number of found peaks
        npeaks = len(fittedpar) // 3
        for i in range(npeaks):
            height = fittedpar[3 * i]
            fwhm = fittedpar[3 * i + 2]
            # Replace height with area in fittedpar
            fittedpar[3 * i] = (height * fwhm * 0.5 * numpy.pi)
        return fittedpar, cons

    def estimate_splitgauss(self, x, y):
        """Estimation of *Height, Position, FWHM1, FWHM2* of peaks, for
        asymmetric gaussian-like curves.

        This functions uses :meth:`estimate_height_position_fwhm`, then
        adds a second (identical) estimation of FWHM to the fit parameters
        for each peak, and the corresponding constraint.

        :param x: Array of abscissa values
        :param y: Array of ordinate values (``y = f(x)``)
        :return: Tuple of estimated fit parameters and fit constraints.
            Parameters to be estimated for each peak are:
            *Height, Position, FWHM1, FWHM2*.
            Fit constraints depend on :attr:`config`.
        """
        fittedpar, cons = self.estimate_height_position_fwhm(x, y)
        # get the number of found peaks
        npeaks = len(fittedpar) // 3
        estimated_parameters = []
        estimated_constraints = numpy.zeros((4 * npeaks, 3), numpy.float)
        for i in range(npeaks):
            for j in range(3):
                estimated_parameters.append(fittedpar[3 * i + j])
            # fwhm2 estimate = fwhm1
            estimated_parameters.append(fittedpar[3 * i + 2])
            # height
            estimated_constraints[4 * i, 0] = cons[3 * i, 0]
            estimated_constraints[4 * i, 1] = cons[3 * i, 1]
            estimated_constraints[4 * i, 2] = cons[3 * i, 2]
            # position
            estimated_constraints[4 * i + 1, 0] = cons[3 * i + 1, 0]
            estimated_constraints[4 * i + 1, 1] = cons[3 * i + 1, 1]
            estimated_constraints[4 * i + 1, 2] = cons[3 * i + 1, 2]
            # fwhm1
            estimated_constraints[4 * i + 2, 0] = cons[3 * i + 2, 0]
            estimated_constraints[4 * i + 2, 1] = cons[3 * i + 2, 1]
            estimated_constraints[4 * i + 2, 2] = cons[3 * i + 2, 2]
            # fwhm2
            estimated_constraints[4 * i + 3, 0] = cons[3 * i + 2, 0]
            estimated_constraints[4 * i + 3, 1] = cons[3 * i + 2, 1]
            estimated_constraints[4 * i + 3, 2] = cons[3 * i + 2, 2]
            if cons[3 * i + 2, 0] == CFACTOR:
                # convert indices of related parameters
                # (this happens if SameFwhmFlag == True)
                estimated_constraints[4 * i + 2, 1] = \
                    int(cons[3 * i + 2, 1] / 3) * 4 + 2
                estimated_constraints[4 * i + 3, 1] = \
                    int(cons[3 * i + 2, 1] / 3) * 4 + 3
        return estimated_parameters, estimated_constraints

    def estimate_pvoigt(self, x, y):
        """Estimation of *Height, Position, FWHM, eta* of peaks, for
        pseudo-Voigt curves.

        Pseudo-Voigt are a sum of a gaussian curve *G(x)* and a lorentzian
        curve *L(x)* with the same height, center, fwhm parameters:
        ``y(x) = eta * G(x) + (1-eta) * L(x)``

        This functions uses :meth:`estimate_height_position_fwhm`, then
        adds a constant estimation of *eta* (0.5) to the fit parameters
        for each peak, and the corresponding constraint.

        :param x: Array of abscissa values
        :param y: Array of ordinate values (``y = f(x)``)
        :return: Tuple of estimated fit parameters and fit constraints.
            Parameters to be estimated for each peak are:
            *Height, Position, FWHM, eta*.
            Constraint for the eta parameter can be set to QUOTED (0.--1.)
            by setting :attr:`config`['QuotedEtaFlag'] to ``True``.
            If this is not the case, the constraint code is set to FREE.
        """
        fittedpar, cons = self.estimate_height_position_fwhm(x, y)
        npeaks = len(fittedpar) // 3
        newpar = []
        newcons = numpy.zeros((4 * npeaks, 3), numpy.float)
        # find out related parameters proper index
        if not self.config['NoConstraintsFlag']:
            if self.config['SameFwhmFlag']:
                j = 0
                # get the index of the free FWHM
                for i in range(npeaks):
                    if cons[3 * i + 2, 0] != 4:
                        j = i
                for i in range(npeaks):
                    if i != j:
                        cons[3 * i + 2, 1] = 4 * j + 2
        for i in range(npeaks):
            newpar.append(fittedpar[3 * i])
            newpar.append(fittedpar[3 * i + 1])
            newpar.append(fittedpar[3 * i + 2])
            newpar.append(0.5)
            # height
            newcons[4 * i, 0] = cons[3 * i, 0]
            newcons[4 * i, 1] = cons[3 * i, 1]
            newcons[4 * i, 2] = cons[3 * i, 2]
            # position
            newcons[4 * i + 1, 0] = cons[3 * i + 1, 0]
            newcons[4 * i + 1, 1] = cons[3 * i + 1, 1]
            newcons[4 * i + 1, 2] = cons[3 * i + 1, 2]
            # fwhm
            newcons[4 * i + 2, 0] = cons[3 * i + 2, 0]
            newcons[4 * i + 2, 1] = cons[3 * i + 2, 1]
            newcons[4 * i + 2, 2] = cons[3 * i + 2, 2]
            # Eta constrains
            newcons[4 * i + 3, 0] = CFREE
            newcons[4 * i + 3, 1] = 0
            newcons[4 * i + 3, 2] = 0
            if self.config['QuotedEtaFlag']:
                newcons[4 * i + 3, 0] = CQUOTED
                newcons[4 * i + 3, 1] = 0.0
                newcons[4 * i + 3, 2] = 1.0
        return newpar, newcons

    def estimate_splitpvoigt(self, x, y):
        """Estimation of *Height, Position, FWHM1, FWHM2, eta* of peaks, for
        asymmetric pseudo-Voigt curves.

        This functions uses :meth:`estimate_height_position_fwhm`, then
        adds an identical FWHM2 parameter and a constant estimation of
        *eta* (0.5) to the fit parameters for each peak, and the corresponding
        constraints.

        Constraint for the eta parameter can be set to QUOTED (0.--1.)
        by setting :attr:`config`['QuotedEtaFlag'] to ``True``.
        If this is not the case, the constraint code is set to FREE.

        :param x: Array of abscissa values
        :param y: Array of ordinate values (``y = f(x)``)
        :return: Tuple of estimated fit parameters and fit constraints.
            Parameters to be estimated for each peak are:
            *Height, Position, FWHM1, FWHM2, eta*.
        """
        fittedpar, cons = self.estimate_height_position_fwhm(x, y)
        npeaks = len(fittedpar) // 3
        newpar = []
        newcons = numpy.zeros((5 * npeaks, 3), numpy.float)
        # find out related parameters proper index
        if not self.config['NoConstraintsFlag']:
            if self.config['SameFwhmFlag']:
                j = 0
                # get the index of the free FWHM
                for i in range(npeaks):
                    if cons[3 * i + 2, 0] != 4:
                        j = i
                for i in range(npeaks):
                    if i != j:
                        cons[3 * i + 2, 1] = 4 * j + 2
        for i in range(npeaks):
            # height
            newpar.append(fittedpar[3 * i])
            # position
            newpar.append(fittedpar[3 * i + 1])
            # fwhm1
            newpar.append(fittedpar[3 * i + 2])
            # fwhm2 estimate equal to fwhm1
            newpar.append(fittedpar[3 * i + 2])
            # eta
            newpar.append(0.5)
            # constraint codes
            # ----------------
            # height
            newcons[5 * i, 0] = cons[3 * i, 0]
            # position
            newcons[5 * i + 1, 0] = cons[3 * i + 1, 0]
            # fwhm1
            newcons[5 * i + 2, 0] = cons[3 * i + 2, 0]
            # fwhm2
            newcons[5 * i + 3, 0] = cons[3 * i + 2, 0]
            # cons 1
            # ------
            newcons[5 * i, 1] = cons[3 * i, 1]
            newcons[5 * i + 1, 1] = cons[3 * i + 1, 1]
            newcons[5 * i + 2, 1] = cons[3 * i + 2, 1]
            newcons[5 * i + 3, 1] = cons[3 * i + 2, 1]
            # cons 2
            # ------
            newcons[5 * i, 2] = cons[3 * i, 2]
            newcons[5 * i + 1, 2] = cons[3 * i + 1, 2]
            newcons[5 * i + 2, 2] = cons[3 * i + 2, 2]
            newcons[5 * i + 3, 2] = cons[3 * i + 2, 2]

            if cons[3 * i + 2, 0] == CFACTOR:
                # fwhm2 connstraint depends on fwhm1
                newcons[5 * i + 3, 1] = newcons[5 * i + 2, 1] + 1
            # eta constraints
            newcons[5 * i + 4, 0] = CFREE
            newcons[5 * i + 4, 1] = 0
            newcons[5 * i + 4, 2] = 0
            if self.config['QuotedEtaFlag']:
                newcons[5 * i + 4, 0] = CQUOTED
                newcons[5 * i + 4, 1] = 0.0
                newcons[5 * i + 4, 2] = 1.0
        return newpar, newcons

    def estimate_apvoigt(self, x, y):
        """Estimation of *Area, Position, FWHM1, eta* of peaks, for
        pseudo-Voigt curves.

        This functions uses :meth:`estimate_pvoigt`, then converts the height
        parameter to area.

        :param x: Array of abscissa values
        :param y: Array of ordinate values (``y = f(x)``)
        :return: Tuple of estimated fit parameters and fit constraints.
            Parameters to be estimated for each peak are:
            *Area, Position, FWHM, eta*.
        """
        fittedpar, cons = self.estimate_pvoigt(x, y)
        npeaks = len(fittedpar) // 4
        # Assume 50% of the area is determined by the gaussian and 50% by
        # the Lorentzian.
        for i in range(npeaks):
            height = fittedpar[4 * i]
            fwhm = fittedpar[4 * i + 2]
            fittedpar[4 * i] = 0.5 * (height * fwhm * 0.5 * numpy.pi) +\
                0.5 * (height * fwhm / (2.0 * numpy.sqrt(2 * numpy.log(2)))
                       ) * numpy.sqrt(2 * numpy.pi)
        return fittedpar, cons

    def estimate_ahypermet(self, x, y):
        """Estimation of *area, position, fwhm, st_area_r, st_slope_r,
        lt_area_r, lt_slope_r, step_height_r* of peaks, for hypermet curves.

        :param x: Array of abscissa values
        :param y: Array of ordinate values (``y = f(x)``)
        :return: Tuple of estimated fit parameters and fit constraints.
            Parameters to be estimated for each peak are:
            *area, position, fwhm, st_area_r, st_slope_r,
            lt_area_r, lt_slope_r, step_height_r* .
        """
        yscaling = self.config.get('Yscaling', 1.0)
        if yscaling == 0:
            yscaling = 1.0
        fittedpar, cons = self.estimate_height_position_fwhm(x, y)
        npeaks = len(fittedpar) // 3
        newpar = []
        newcons = numpy.zeros((8 * npeaks, 3), numpy.float)
        main_peak = 0
        # find out related parameters proper index
        if not self.config['NoConstraintsFlag']:
            if self.config['SameFwhmFlag']:
                j = 0
                # get the index of the free FWHM
                for i in range(npeaks):
                    if cons[3 * i + 2, 0] != 4:
                        j = i
                for i in range(npeaks):
                    if i != j:
                        cons[3 * i + 2, 1] = 8 * j + 2
                main_peak = j
        for i in range(npeaks):
            if fittedpar[3 * i] > fittedpar[3 * main_peak]:
                main_peak = i

        for i in range(npeaks):
            height = fittedpar[3 * i]
            position = fittedpar[3 * i + 1]
            fwhm = fittedpar[3 * i + 2]
            area = (height * fwhm / (2.0 * numpy.sqrt(2 * numpy.log(2)))
                    ) * numpy.sqrt(2 * numpy.pi)
            # the gaussian parameters
            newpar.append(area)
            newpar.append(position)
            newpar.append(fwhm)
            # print "area, pos , fwhm = ",area,position,fwhm
            # Avoid zero derivatives because of not calculating contribution
            g_term = 1
            st_term = 1
            lt_term = 1
            step_term = 1
            if self.config['HypermetTails'] != 0:
                g_term = self.config['HypermetTails'] & 1
                st_term = (self.config['HypermetTails'] >> 1) & 1
                lt_term = (self.config['HypermetTails'] >> 2) & 1
                step_term = (self.config['HypermetTails'] >> 3) & 1
            if g_term == 0:
                # fix the gaussian parameters
                newcons[8 * i, 0] = CFIXED
                newcons[8 * i + 1, 0] = CFIXED
                newcons[8 * i + 2, 0] = CFIXED
            # the short tail parameters
            if ((area * yscaling) <
                self.config['MinGaussArea4ShortTail']) | \
               (st_term == 0):
                newpar.append(0.0)
                newpar.append(0.0)
                newcons[8 * i + 3, 0] = CFIXED
                newcons[8 * i + 3, 1] = 0.0
                newcons[8 * i + 3, 2] = 0.0
                newcons[8 * i + 4, 0] = CFIXED
                newcons[8 * i + 4, 1] = 0.0
                newcons[8 * i + 4, 2] = 0.0
            else:
                newpar.append(self.config['InitialShortTailAreaRatio'])
                newpar.append(self.config['InitialShortTailSlopeRatio'])
                newcons[8 * i + 3, 0] = CQUOTED
                newcons[8 * i + 3, 1] = self.config['MinShortTailAreaRatio']
                newcons[8 * i + 3, 2] = self.config['MaxShortTailAreaRatio']
                newcons[8 * i + 4, 0] = CQUOTED
                newcons[8 * i + 4, 1] = self.config['MinShortTailSlopeRatio']
                newcons[8 * i + 4, 2] = self.config['MaxShortTailSlopeRatio']
            # the long tail parameters
            if ((area * yscaling) <
                self.config['MinGaussArea4LongTail']) | \
               (lt_term == 0):
                newpar.append(0.0)
                newpar.append(0.0)
                newcons[8 * i + 5, 0] = CFIXED
                newcons[8 * i + 5, 1] = 0.0
                newcons[8 * i + 5, 2] = 0.0
                newcons[8 * i + 6, 0] = CFIXED
                newcons[8 * i + 6, 1] = 0.0
                newcons[8 * i + 6, 2] = 0.0
            else:
                newpar.append(self.config['InitialLongTailAreaRatio'])
                newpar.append(self.config['InitialLongTailSlopeRatio'])
                newcons[8 * i + 5, 0] = CQUOTED
                newcons[8 * i + 5, 1] = self.config['MinLongTailAreaRatio']
                newcons[8 * i + 5, 2] = self.config['MaxLongTailAreaRatio']
                newcons[8 * i + 6, 0] = CQUOTED
                newcons[8 * i + 6, 1] = self.config['MinLongTailSlopeRatio']
                newcons[8 * i + 6, 2] = self.config['MaxLongTailSlopeRatio']
            # the step parameters
            if ((height * yscaling) <
                self.config['MinGaussHeight4StepTail']) | \
               (step_term == 0):
                newpar.append(0.0)
                newcons[8 * i + 7, 0] = CFIXED
                newcons[8 * i + 7, 1] = 0.0
                newcons[8 * i + 7, 2] = 0.0
            else:
                newpar.append(self.config['InitialStepTailHeightRatio'])
                newcons[8 * i + 7, 0] = CQUOTED
                newcons[8 * i + 7, 1] = self.config['MinStepTailHeightRatio']
                newcons[8 * i + 7, 2] = self.config['MaxStepTailHeightRatio']
            # if self.config['NoConstraintsFlag'] == 1:
            #   newcons=numpy.zeros((8*npeaks, 3),numpy.float)
        if npeaks > 0:
            if g_term:
                if self.config['PositiveHeightAreaFlag']:
                    for i in range(npeaks):
                        newcons[8 * i, 0] = CPOSITIVE
                if self.config['PositiveFwhmFlag']:
                    for i in range(npeaks):
                        newcons[8 * i + 2, 0] = CPOSITIVE
                if self.config['SameFwhmFlag']:
                    for i in range(npeaks):
                        if i != main_peak:
                            newcons[8 * i + 2, 0] = CFACTOR
                            newcons[8 * i + 2, 1] = 8 * main_peak + 2
                            newcons[8 * i + 2, 2] = 1.0
                if self.config['HypermetQuotedPositionFlag']:
                    for i in range(npeaks):
                        delta = self.config['DeltaPositionFwhmUnits'] * fwhm
                        newcons[8 * i + 1, 0] = CQUOTED
                        newcons[8 * i + 1, 1] = newpar[8 * i + 1] - delta
                        newcons[8 * i + 1, 2] = newpar[8 * i + 1] + delta
            if self.config['SameSlopeRatioFlag']:
                for i in range(npeaks):
                    if i != main_peak:
                        newcons[8 * i + 4, 0] = CFACTOR
                        newcons[8 * i + 4, 1] = 8 * main_peak + 4
                        newcons[8 * i + 4, 2] = 1.0
                        newcons[8 * i + 6, 0] = CFACTOR
                        newcons[8 * i + 6, 1] = 8 * main_peak + 6
                        newcons[8 * i + 6, 2] = 1.0
            if self.config['SameAreaRatioFlag']:
                for i in range(npeaks):
                    if i != main_peak:
                        newcons[8 * i + 3, 0] = CFACTOR
                        newcons[8 * i + 3, 1] = 8 * main_peak + 3
                        newcons[8 * i + 3, 2] = 1.0
                        newcons[8 * i + 5, 0] = CFACTOR
                        newcons[8 * i + 5, 1] = 8 * main_peak + 5
                        newcons[8 * i + 5, 2] = 1.0
        return newpar, newcons

    def estimate_stepdown(self, x, y):
        """Estimation of parameters for stepdown curves.

        The functions estimates gaussian parameters for the derivative of
        the data, takes the largest gaussian peak and uses its estimated
        parameters to define the center of the step and its fwhm. The
        estimated amplitude returned is simply ``max(y) - min(y)``.

        :param x: Array of abscissa values
        :param y: Array of ordinate values (``y = f(x)``)
        :return: Tuple of estimated fit parameters and fit newconstraints.
            Parameters to be estimated for each stepdown are:
            *height, centroid, fwhm* .
        """
        crappyfilter = [-0.25, -0.75, 0.0, 0.75, 0.25]
        cutoff = len(crappyfilter) // 2
        y_deriv = numpy.convolve(y,
                                 crappyfilter,
                                 mode="valid")

        # make the derivative's peak have the same amplitude as the step
        if max(y_deriv) > 0:
            y_deriv = y_deriv * max(y) / max(y_deriv)

        fittedpar, newcons = self.estimate_height_position_fwhm(
                                 x[cutoff:-cutoff], y_deriv)

        data_amplitude = max(y) - min(y)

        # use parameters from largest gaussian found
        if len(fittedpar):
            npeaks = len(fittedpar) // 3
            largest_index = 0
            largest = [data_amplitude,
                       fittedpar[3 * largest_index + 1],
                       fittedpar[3 * largest_index + 2]]
            for i in range(npeaks):
                if fittedpar[3 * i] > largest[0]:
                    largest_index = i
                    largest = [data_amplitude,
                               fittedpar[3 * largest_index + 1],
                               fittedpar[3 * largest_index + 2]]
        else:
            # no peak was found
            largest = [data_amplitude,                               # height
                       x[len(x)//2],                                 # center: middle of x range
                       self.config["FwhmPoints"] * (x[1] - x[0])]    # fwhm: default value

        # Setup constrains
        newcons = numpy.zeros((3, 3), numpy.float)
        if not self.config['NoConstraintsFlag']:
                # Setup height constrains
            if self.config['PositiveHeightAreaFlag']:
                newcons[0, 0] = CPOSITIVE
                newcons[0, 1] = 0
                newcons[0, 2] = 0

            # Setup position constrains
            if self.config['QuotedPositionFlag']:
                newcons[1, 0] = CQUOTED
                newcons[1, 1] = min(x)
                newcons[1, 2] = max(x)

            # Setup positive FWHM constrains
            if self.config['PositiveFwhmFlag']:
                newcons[2, 0] = CPOSITIVE
                newcons[2, 1] = 0
                newcons[2, 2] = 0

        return largest, newcons

    def estimate_slit(self, x, y):
        """Estimation of parameters for slit curves.

        The functions estimates stepup and stepdown parameters for the largest
        steps, and uses them for calculating the center (middle between stepup
        and stepdown), the height (maximum amplitude in data), the fwhm
        (distance between the up- and down-step centers) and the beamfwhm
        (average of FWHM for up- and down-step).

        :param x: Array of abscissa values
        :param y: Array of ordinate values (``y = f(x)``)
        :return: Tuple of estimated fit parameters and fit constraints.
            Parameters to be estimated for each slit are:
            *height, position, fwhm, beamfwhm* .
        """
        largestup, cons = self.estimate_stepup(x, y)
        largestdown, cons = self.estimate_stepdown(x, y)
        fwhm = numpy.fabs(largestdown[1] - largestup[1])
        beamfwhm = 0.5 * (largestup[2] + largestdown[1])
        beamfwhm = min(beamfwhm, fwhm / 10.0)
        beamfwhm = max(beamfwhm, (max(x) - min(x)) * 3.0 / len(x))

        y_minus_bg = y - self.strip_bg(y)
        height = max(y_minus_bg)

        i1 = numpy.nonzero(y_minus_bg >= 0.5 * height)[0]
        xx = numpy.take(x, i1)
        position = (xx[0] + xx[-1]) / 2.0
        fwhm = xx[-1] - xx[0]
        largest = [height, position, fwhm, beamfwhm]
        cons = numpy.zeros((4, 3), numpy.float)
        # Setup constrains
        if not self.config['NoConstraintsFlag']:
            # Setup height constrains
            if self.config['PositiveHeightAreaFlag']:
                cons[0, 0] = CPOSITIVE
                cons[0, 1] = 0
                cons[0, 2] = 0

            # Setup position constrains
            if self.config['QuotedPositionFlag']:
                cons[1, 0] = CQUOTED
                cons[1, 1] = min(x)
                cons[1, 2] = max(x)

            # Setup positive FWHM constrains
            if self.config['PositiveFwhmFlag']:
                cons[2, 0] = CPOSITIVE
                cons[2, 1] = 0
                cons[2, 2] = 0

            # Setup positive FWHM constrains
            if self.config['PositiveFwhmFlag']:
                cons[3, 0] = CPOSITIVE
                cons[3, 1] = 0
                cons[3, 2] = 0
        return largest, cons

    def estimate_stepup(self, x, y):
        """Estimation of parameters for a single step up curve.

        The functions estimates gaussian parameters for the derivative of
        the data, takes the largest gaussian peak and uses its estimated
        parameters to define the center of the step and its fwhm. The
        estimated amplitude returned is simply ``max(y) - min(y)``.

        :param x: Array of abscissa values
        :param y: Array of ordinate values (``y = f(x)``)
        :return: Tuple of estimated fit parameters and fit constraints.
            Parameters to be estimated for each stepup are:
            *height, centroid, fwhm* .
        """
        crappyfilter = [0.25, 0.75, 0.0, -0.75, -0.25]
        cutoff = len(crappyfilter) // 2
        y_deriv = numpy.convolve(y, crappyfilter, mode="valid")
        if max(y_deriv) > 0:
            y_deriv = y_deriv * max(y) / max(y_deriv)

        fittedpar, cons = self.estimate_height_position_fwhm(
                              x[cutoff:-cutoff], y_deriv)

        # for height, use the data amplitude after removing the background
        data_amplitude = max(y) - min(y)

        # find params of the largest gaussian found
        if len(fittedpar):
            npeaks = len(fittedpar) // 3
            largest_index = 0
            largest = [data_amplitude,
                       fittedpar[3 * largest_index + 1],
                       fittedpar[3 * largest_index + 2]]
            for i in range(npeaks):
                if fittedpar[3 * i] > largest[0]:
                    largest_index = i
                    largest = [fittedpar[3 * largest_index],
                               fittedpar[3 * largest_index + 1],
                               fittedpar[3 * largest_index + 2]]
        else:
            # no peak was found
            largest = [data_amplitude,                               # height
                       x[len(x)//2],                                 # center: middle of x range
                       self.config["FwhmPoints"] * (x[1] - x[0])]    # fwhm: default value

        newcons = numpy.zeros((3, 3), numpy.float)
        # Setup constrains
        if not self.config['NoConstraintsFlag']:
                # Setup height constraints
            if self.config['PositiveHeightAreaFlag']:
                newcons[0, 0] = CPOSITIVE
                newcons[0, 1] = 0
                newcons[0, 2] = 0

            # Setup position constraints
            if self.config['QuotedPositionFlag']:
                newcons[1, 0] = CQUOTED
                newcons[1, 1] = min(x)
                newcons[1, 2] = max(x)

            # Setup positive FWHM constraints
            if self.config['PositiveFwhmFlag']:
                newcons[2, 0] = CPOSITIVE
                newcons[2, 1] = 0
                newcons[2, 2] = 0

        return largest, newcons

    def estimate_periodic_gauss(self, x, y):
        """Estimation of parameters for periodic gaussian curves:
        *number of peaks, distance between peaks, height, position of the
        first peak, fwhm*

        The functions detects all peaks, then computes the parameters the
        following way:

            - *distance*: average of distances between detected peaks
            - *height*: average height of detected peaks
            - *fwhm*: fwhm of the highest peak (in number of samples) if
                field ``'AutoFwhm'`` in :attr:`config` is ``True``, else take
                the default value (field ``'FwhmPoints'`` in :attr:`config`)

        :param x: Array of abscissa values
        :param y: Array of ordinate values (``y = f(x)``)
        :return: Tuple of estimated fit parameters and fit constraints.
        """
        yscaling = self.config.get('Yscaling', 1.0)
        if yscaling == 0:
            yscaling = 1.0

        bg = self.strip_bg(y)

        if self.config['AutoFwhm']:
            search_fwhm = guess_fwhm(y)
        else:
            search_fwhm = int(float(self.config['FwhmPoints']))
        search_sens = float(self.config['Sensitivity'])

        if search_fwhm < 3:
            search_fwhm = 3

        if search_sens < 1:
            search_sens = 1

        if len(y) > 1.5 * search_fwhm:
            peaks = peak_search(yscaling * y, fwhm=search_fwhm,
                                sensitivity=search_sens)
        else:
            peaks = []
        npeaks = len(peaks)
        if not npeaks:
            fittedpar = []
            cons = numpy.zeros((len(fittedpar), 3), numpy.float)
            return fittedpar, cons

        fittedpar = [0.0, 0.0, 0.0, 0.0, 0.0]

        # The number of peaks
        fittedpar[0] = npeaks

        # The separation between peaks in x units
        delta = 0.0
        height = 0.0
        for i in range(npeaks):
            height += y[int(peaks[i])] - bg[int(peaks[i])]
            if i != npeaks - 1:
                delta += (x[int(peaks[i + 1])] - x[int(peaks[i])])

        # delta between peaks
        if npeaks > 1:
            fittedpar[1] = delta / (npeaks - 1)

        # starting height
        fittedpar[2] = height / npeaks

        # position of the first peak
        fittedpar[3] = x[int(peaks[0])]

        # Estimate the fwhm
        fittedpar[4] = search_fwhm

        # setup constraints
        cons = numpy.zeros((5, 3), numpy.float)
        cons[0, 0] = CFIXED  # the number of gaussians
        if npeaks == 1:
            cons[1, 0] = CFIXED  # the delta between peaks
        else:
            cons[1, 0] = CFREE
        j = 2
        # Setup height area constrains
        if not self.config['NoConstraintsFlag']:
            if self.config['PositiveHeightAreaFlag']:
                # POSITIVE = 1
                cons[j, 0] = CPOSITIVE
                cons[j, 1] = 0
                cons[j, 2] = 0
        j += 1

        # Setup position constrains
        if not self.config['NoConstraintsFlag']:
            if self.config['QuotedPositionFlag']:
                # QUOTED = 2
                cons[j, 0] = CQUOTED
                cons[j, 1] = min(x)
                cons[j, 2] = max(x)
        j += 1

        # Setup positive FWHM constrains
        if not self.config['NoConstraintsFlag']:
            if self.config['PositiveFwhmFlag']:
                # POSITIVE=1
                cons[j, 0] = CPOSITIVE
                cons[j, 1] = 0
                cons[j, 2] = 0
        j += 1
        return fittedpar, cons

    def configure(self, **kw):
        """Add new / unknown keyword arguments to :attr:`config`,
        update entries in :attr:`config` if the parameter name is a existing
        key.

        :param kw: Dictionary of keyword arguments.
        :return: Configuration dictionary :attr:`config`
        """
        if not kw.keys():
            return self.config
        for key in kw.keys():
            notdone = 1
            # take care of lower / upper case problems ...
            for config_key in self.config.keys():
                if config_key.lower() == key.lower():
                    self.config[config_key] = kw[key]
                    notdone = 0
            if notdone:
                self.config[key] = kw[key]
        return self.config

fitfuns = FitTheories()

THEORY = OrderedDict((
    ('Gaussians',
        FitTheory(description='Gaussian functions',
                  function=functions.sum_gauss,
                  parameters=('Height', 'Position', 'FWHM'),
                  estimate=fitfuns.estimate_height_position_fwhm,
                  configure=fitfuns.configure)),
    ('Lorentz',
        FitTheory(description='Lorentzian functions',
                  function=functions.sum_lorentz,
                  parameters=('Height', 'Position', 'FWHM'),
                  estimate=fitfuns.estimate_height_position_fwhm,
                  configure=fitfuns.configure)),
    ('Area Gaussians',
        FitTheory(description='Gaussian functions (area)',
                  function=functions.sum_agauss,
                  parameters=('Area', 'Position', 'FWHM'),
                  estimate=fitfuns.estimate_agauss,
                  configure=fitfuns.configure)),
    ('Area Lorentz',
        FitTheory(description='Lorentzian functions (area)',
                  function=functions.sum_alorentz,
                  parameters=('Area', 'Position', 'FWHM'),
                  estimate=fitfuns.estimate_alorentz,
                  configure=fitfuns.configure)),
    ('Pseudo-Voigt Line',
        FitTheory(description='Pseudo-Voigt functions',
                  function=functions.sum_pvoigt,
                  parameters=('Height', 'Position', 'FWHM', 'Eta'),
                  estimate=fitfuns.estimate_pvoigt,
                  configure=fitfuns.configure)),
    ('Area Pseudo-Voigt',
        FitTheory(description='Pseudo-Voigt functions (area)',
                  function=functions.sum_apvoigt,
                  parameters=('Area', 'Position', 'FWHM', 'Eta'),
                  estimate=fitfuns.estimate_apvoigt,
                  configure=fitfuns.configure)),
    ('Split Gaussian',
        FitTheory(description='Asymmetric gaussian functions',
                  function=functions.sum_splitgauss,
                  parameters=('Height', 'Position', 'LowFWHM',
                              'HighFWHM'),
                  estimate=fitfuns.estimate_splitgauss,
                  configure=fitfuns.configure)),
    ('Split Lorentz',
        FitTheory(description='Asymmetric lorentzian functions',
                  function=functions.sum_splitlorentz,
                  parameters=('Height', 'Position', 'LowFWHM', 'HighFWHM'),
                  estimate=fitfuns.estimate_splitgauss,
                  configure=fitfuns.configure)),
    ('Split Pseudo-Voigt',
        FitTheory(description='Asymmetric pseudo-Voigt functions',
                  function=functions.sum_splitpvoigt,
                  parameters=('Height', 'Position', 'LowFWHM',
                              'HighFWHM', 'Eta'),
                  estimate=fitfuns.estimate_splitpvoigt,
                  configure=fitfuns.configure)),
    ('Step Down',
        FitTheory(description='Step down function',
                  function=functions.sum_stepdown,
                  parameters=('Height', 'Position', 'FWHM'),
                  estimate=fitfuns.estimate_stepdown,
                  configure=fitfuns.configure)),
    ('Step Up',
        FitTheory(description='Step up function',
                  function=functions.sum_stepup,
                  parameters=('Height', 'Position', 'FWHM'),
                  estimate=fitfuns.estimate_stepup,
                  configure=fitfuns.configure)),
    ('Slit',
        FitTheory(description='Slit function',
                  function=functions.sum_slit,
                  parameters=('Height', 'Position', 'FWHM', 'BeamFWHM'),
                  estimate=fitfuns.estimate_slit,
                  configure=fitfuns.configure)),
    ('Atan',
        FitTheory(description='Arctan step up function',
                  function=functions.atan_stepup,
                  parameters=('Height', 'Position', 'Width'),
                  estimate=fitfuns.estimate_stepup,
                  configure=fitfuns.configure)),
    ('Hypermet',
        FitTheory(description='Hypermet functions',
                  function=fitfuns.ahypermet,     # customized version of functions.sum_ahypermet
                  parameters=('G_Area', 'Position', 'FWHM', 'ST_Area',
                              'ST_Slope', 'LT_Area', 'LT_Slope', 'Step_H'),
                  estimate=fitfuns.estimate_ahypermet,
                  configure=fitfuns.configure)),
    # ('Periodic Gaussians',
    #     FitTheory(description='Periodic gaussian functions',
    #               function=functions.periodic_gauss,
    #               parameters=('N', 'Delta', 'Height', 'Position', 'FWHM'),
    #               estimate=fitfuns.estimate_periodic_gauss,
    #               configure=fitfuns.configure))
    ('Degree 2 Polynomial',
        FitTheory(description='Degree 2 polynomial'
                              '\ny = a*x^2 + b*x +c',
                  function=fitfuns.poly,
                  parameters=['a', 'b', 'c'],
                  estimate=fitfuns.estimate_quadratic)),
    ('Degree 3 Polynomial',
        FitTheory(description='Degree 3 polynomial'
                              '\ny = a*x^3 + b*x^2 + c*x + d',
                  function=fitfuns.poly,
                  parameters=['a', 'b', 'c', 'd'],
                  estimate=fitfuns.estimate_cubic)),
    ('Degree 4 Polynomial',
        FitTheory(description='Degree 4 polynomial'
                              '\ny = a*x^4 + b*x^3 + c*x^2 + d*x + e',
                  function=fitfuns.poly,
                  parameters=['a', 'b', 'c', 'd', 'e'],
                  estimate=fitfuns.estimate_quartic)),
    ('Degree 5 Polynomial',
        FitTheory(description='Degree 5 polynomial'
                              '\ny = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f',
                  function=fitfuns.poly,
                  parameters=['a', 'b', 'c', 'd', 'e', 'f'],
                  estimate=fitfuns.estimate_quintic)),
))
"""Dictionary of fit theories: fit functions and their associated estimation
function, parameters list, configuration function and description.
"""


def test(a):
    from silx.math.fit import fitmanager
    x = numpy.arange(1000).astype(numpy.float)
    p = [1500, 100., 50.0,
         1500, 700., 50.0]
    y_synthetic = functions.sum_gauss(x, *p) + 1

    fit = fitmanager.FitManager(x, y_synthetic)
    fit.addtheory('Gaussians', functions.sum_gauss, ['Height', 'Position', 'FWHM'],
                  a.estimate_height_position_fwhm)
    fit.settheory('Gaussians')
    fit.setbackground('Linear')

    fit.estimate()
    fit.runfit()

    y_fit = fit.gendata()

    print("Fit parameter names: %s" % str(fit.get_names()))
    print("Theoretical parameters: %s" % str(numpy.append([1, 0],  p)))
    print("Fitted parameters: %s" % str(fit.get_fitted_parameters()))

    try:
        from silx.gui import qt
        from silx.gui.plot import plot1D
        app = qt.QApplication([])

        # Offset of 1 to see the difference in log scale
        plot1D(x, (y_synthetic + 1, y_fit), "Input data + 1, Fit")

        app.exec_()
    except ImportError:
        _logger.warning("Unable to load qt binding, can't plot results.")


if __name__ == "__main__":
    test(fitfuns)
