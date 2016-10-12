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
    savitsky_golay
from silx.math.fit.fittheory import FitTheory

CONFIG = {
    "SmoothingFlag": False,
    "SmoothingWidth": 5,
    "AnchorsFlag": False,   # TODO
    "AnchorsList": [],      # TODO
    "StripWidth": 2,
    "StripIterations": 5000,
    "StripThresholdFactor": 1.0,
    "SnipWidth": 2,         # TODO
}


# to avoid costly computations when parameters stay the same
_BG_STRIP_OLDY = numpy.array([])
_BG_STRIP_OLDPARS = [0, 0]
_BG_STRIP_OLDBG = numpy.array([])
_BG_STRIP_OLDWIDTH = 0
_BG_STRIP_OLDFLAG = None


def strip_bg(x, y0, width, niter):
    """Compute the strip bg for y0"""
    global _BG_STRIP_OLDY
    global _BG_STRIP_OLDPARS
    global _BG_STRIP_OLDBG
    global _BG_STRIP_OLDWIDTH
    global _BG_STRIP_OLDFLAG
    # same parameters
    if _BG_STRIP_OLDPARS == [width, niter] and\
            _BG_STRIP_OLDWIDTH == CONFIG["SmoothingWidth"] and\
            _BG_STRIP_OLDFLAG == CONFIG["SmoothingFlag"]:
        # same data
        if numpy.sum(_BG_STRIP_OLDY == y0) == len(y0):
            # same result
            return _BG_STRIP_OLDBG

    _BG_STRIP_OLDY = y0
    _BG_STRIP_OLDPARS = [width, niter]
    _BG_STRIP_OLDWIDTH = CONFIG["SmoothingWidth"]
    _BG_STRIP_OLDFLAG = CONFIG["SmoothingFlag"]

    y1 = savitsky_golay(y0, CONFIG["SmoothingWidth"]) if CONFIG["SmoothingFlag"] else y0

    background = strip(y1,
                       w=width,
                       niterations=niter,
                       factor=CONFIG["StripThresholdFactor"])

    _BG_STRIP_OLDBG = background

    return background


def estimate_linear(x, y):
    """
    Estimate the linear parameters (constant, slope) of a y signal.

    Strip peaks, then perform a linear regression.
    """
    bg = strip_bg(x, y,
                  width=CONFIG["StripWidth"],
                  niter=CONFIG["StripIterations"])
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

    Return parameters from CONFIG dict, set constraints to FIXED.
    """
    estimated_par = [CONFIG["StripWidth"],
                     CONFIG["StripIterations"]]
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
            function=lambda x, y0: numpy.zeros_like(x),
            parameters=[],
            is_background=True),
    'Constant': FitTheory(
            description='Constant background',
            function=lambda x, y0, c: c * numpy.ones_like(x),
            parameters=['Constant', ],
            estimate=lambda x, y: ([min(y)], [(0, 0, 0)]),
            is_background=True),
    'Linear':  FitTheory(
            description="Linear background, parameters 'Constant' and 'Slope'",
            function=lambda x, y0, a, b: a + b * x,
            parameters=['Constant', 'Slope'],
            estimate=estimate_linear,
            configure=configure,
            is_background=True),
    'Strip': FitTheory(
            description="Background based on strip filter\n" +
                        "Parameters 'StripWidth', 'StripIterations'",
            function=strip_bg,
            parameters=['StripWidth', 'StripIterations'],
            estimate=estimate_strip,
            configure=configure,
            is_background=True),
}
