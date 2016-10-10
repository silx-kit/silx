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
"""This modules provides a set of background model functions and associated
estimation functions in a format that can be imported into a
:class:`silx.math.fit.FitManager` instance.
"""
__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "10/10/2016"

import numpy
from silx.math.fit.filters import strip, snip1d,\
    smooth1d, savitsky_golay
from silx.math.fit.fittheory import FitTheory

CONFIG = {
    "SmoothStrip": False,
    "StripWidth": 2,
    "StripNIterations": 5000,
    "StripThresholdFactor": 1.0,
}


def strip_bg(y, width, niter):
    """Compute the strip bg for y"""
    if CONFIG["SmoothStrip"]:
        y = smooth1d(y)
    background = strip(y,
                       w=width,
                       niterations=niter,
                       factor=CONFIG["StripThresholdFactor"])
    return background


# def bkg_strip(self, x, *pars):
#       """
#       Internal Background based on a strip filter
#       (:meth:`silx.math.fit.filters.strip`)
#
#       Parameters are *(strip_width, n_iterations)*
#
#       A 1D smoothing is applied prior to the stripping, if configuration
#       parameter ``SmoothStrip`` in :attr:`fitconfig` is ``True``.
#
#       See http://pymca.sourceforge.net/stripbackground.html
#       """
#       if self._bkg_strip_oldpars[0] == pars[0]:
#           if self._bkg_strip_oldpars[1] == pars[1]:
#               if (len(x) == len(self._bkg_strip_oldx)) & \
#                  (len(self.ydata) == len(self._bkg_strip_oldy)):
#                   # same parameters
#                   if numpy.sum(self._bkg_strip_oldx == x) == len(x):
#                       if numpy.sum(self._bkg_strip_oldy == self.ydata) == len(self.ydata):
#                           return self._bkg_strip_oldbkg
#       self._bkg_strip_oldy = self.ydata
#       self._bkg_strip_oldx = x
#       self._bkg_strip_oldpars = pars
#       idx = numpy.nonzero((self.xdata >= x[0]) & (self.xdata <= x[-1]))[0]
#       yy = numpy.take(self.ydata, idx)
#       if self.fitconfig["SmoothStrip"]:
#           yy = smooth1d(yy)
#
#       nrx = numpy.shape(x)[0]
#       nry = numpy.shape(yy)[0]
#       if nrx == nry:
#           self._bkg_strip_oldbkg = strip(yy, pars[0], pars[1])
#           return self._bkg_strip_oldbkg
#
#       else:
#           self._bkg_strip_oldbkg = strip(numpy.take(yy, numpy.arange(0, nry, 2)),
#                                          pars[0], pars[1])
#           return self._bkg_strip_oldbkg


def estimate_linear(x, y):
    """
    Estimate the linear parameters (constant, slope) of a y signal.

    Strip peaks, then perform a linear regression.
    """
    bg = strip_bg(y,
                  width=CONFIG["StripWidth"],
                  niter=CONFIG["StripNIterations"])
    n = float(len(bg))
    Sy = numpy.sum(bg)
    Sx = float(numpy.sum(x))
    Sxx = float(numpy.sum(x * x))
    Sxy = float(numpy.sum(x * bg))

    deno = n * Sxx - (Sx * Sx)
    if deno != 0:
        bg = (Sxx * Sy - Sx * Sxy) / deno
        slope = (n * Sxy - Sx * Sy) / deno
    else:
        bg = 0.0
        slope = 0.0
    estimated_par = [bg, slope]
    # code = 0: FREE
    constraints = [[0, 0, 0], [0, 0, 0]]
    return estimated_par, constraints


def estimate_strip(x, y):
    """Estimation function for strip parameters.

    Return parameters from CONFIG dict, set constraints to FIXED."""
    estimated_par = [CONFIG["StripWidth"],
                     CONFIG["StripNIterations"]]
    constraints = numpy.zeros((len(estimated_par), 3), numpy.float)
    # code = 3: FIXED
    constraints[0][0] = 3
    constraints[1][0] = 3
    return estimated_par, constraints


def configure(**kw):
    """Update the CONFIG dict
    """
    # inspect **kw to find known keys, update them in CONFIG
    for key in CONFIG:
        if key in kw:
            CONFIG[key] = kw[key]

    return CONFIG


THEORY = {
    'No Background': FitTheory(
            description="No background function",
            function=lambda x: numpy.zeros_like(x),
            parameters=[]),
    'Constant': FitTheory(
            description='Constant background',
            function=lambda x, c: c * numpy.ones_like(x),
            parameters=['Constant', ],
            estimate=lambda x, y: ([min(y)], [(0, 0, 0)])),
    'Linear':  FitTheory(
            description="Linear background, parameters 'Constant' and 'Slope'",
            function=lambda x, a, b: a + b * x,
            parameters=['Constant', 'Slope'],
            estimate=estimate_linear,
            configure=configure),
    'Strip': FitTheory(
            description="Background based on strip filter\n" +
                        "Parameters 'StripWidth', 'StripIterations'",
            function=strip_bg,
            parameters=['StripWidth', 'StripIterations'],
            estimate=estimate_strip,
            configure=configure),
}

