# coding: utf-8
#/*##########################################################################
#
# Copyright (c) 2004-2019 European Synchrotron Radiation Facility
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
"""This modules defines a set of background model functions and associated
estimation functions in a format that can be imported into a
:class:`silx.math.fit.FitManager` object.

A background function is a function that you want to add to a regular fit
function prior to fitting the sum of both functions. This is useful, for
instance, if you need to fit multiple gaussian peaks in an array of
measured data points when the measurement is polluted by a background signal.

The models include common background models such as a constant value or a
linear background.

It also includes background computation filters - *strip* and *snip* - that
can extract a more complex low-curvature background signal from a signal with
peaks having higher curvatures.

The source code of this module can serve as a template for defining your
own fit background theories. The minimal skeleton of such a theory definition
file is::

    from silx.math.fit.fittheory import FitTheory

    def bgfunction1(x, y0, …):
        bg_signal = …
        return bg_signal

    def estimation_function1(x, y):
        …
        estimated_params = …
        constraints = …
        return estimated_params, constraints

    THEORY = {
        'bg_theory_name1': FitTheory(
                            description='Description of theory 1',
                            function=bgfunction1,
                            parameters=('param name 1', 'param name 2', …),
                            estimate=estimation_function1,
                            configure=configuration_function1,
                            derivative=derivative_function1,
                            is_background=True),
        'theory_name_2': …,
    }
"""
__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "16/01/2017"

from collections import OrderedDict
import numpy
from silx.math.fit.filters import strip, snip1d,\
    savitsky_golay
from silx.math.fit.fittheory import FitTheory

CONFIG = {
    "SmoothingFlag": False,
    "SmoothingWidth": 5,
    "AnchorsFlag": False,
    "AnchorsList": [],
    "StripWidth": 2,
    "StripIterations": 5000,
    "StripThresholdFactor": 1.0,
    "SnipWidth": 16,
    "EstimatePolyOnStrip": True
}

# to avoid costly computations when parameters stay the same
_BG_STRIP_OLDY = numpy.array([])
_BG_STRIP_OLDPARS = [0, 0]
_BG_STRIP_OLDBG = numpy.array([])

_BG_SNIP_OLDY = numpy.array([])
_BG_SNIP_OLDWIDTH = None
_BG_SNIP_OLDBG = numpy.array([])


_BG_OLD_ANCHORS = []
_BG_OLD_ANCHORS_FLAG = None

_BG_SMOOTH_OLDWIDTH = None
_BG_SMOOTH_OLDFLAG = None


def _convert_anchors_to_indices(x):
    """Anchors stored in CONFIG["AnchorsList"] are abscissa.
    Convert then to indices (take first index where x >= anchor),
    then return the list of indices.

    :param x: Original array of abscissa
    :return: List of indices of anchors in x array.
        If CONFIG['AnchorsFlag'] is False or None, or if the list
        of indices is empty, return None.
    """
    # convert anchor X abscissa to index
    if CONFIG['AnchorsFlag'] and CONFIG['AnchorsList'] is not None:
        anchors_indices = []
        for anchor_x in CONFIG['AnchorsList']:
            if anchor_x <= x[0]:
                continue
            # take the first index where x > anchor_x
            indices = numpy.nonzero(x >= anchor_x)[0]
            if len(indices):
                anchors_indices.append(min(indices))
        if not len(anchors_indices):
            anchors_indices = None
    else:
        anchors_indices = None

    return anchors_indices


def strip_bg(x, y0, width, niter):
    """Extract and return the strip bg from y0.

    Use anchors coordinates in CONFIG["AnchorsList"] if flag
    CONFIG["AnchorsFlag"] is True. Convert anchors from x coordinate
    to array index prior to passing it to silx.math.fit.filters.strip

    :param x: Abscissa array
    :param x: Ordinate array (data values at x positions)
    :param width: strip width
    :param niter: strip niter
    """
    global _BG_STRIP_OLDY
    global _BG_STRIP_OLDPARS
    global _BG_STRIP_OLDBG
    global _BG_SMOOTH_OLDWIDTH
    global _BG_SMOOTH_OLDFLAG
    global _BG_OLD_ANCHORS
    global _BG_OLD_ANCHORS_FLAG

    parameters_changed =\
        _BG_STRIP_OLDPARS != [width, niter] or\
        _BG_SMOOTH_OLDWIDTH != CONFIG["SmoothingWidth"] or\
        _BG_SMOOTH_OLDFLAG != CONFIG["SmoothingFlag"] or\
        _BG_OLD_ANCHORS_FLAG != CONFIG["AnchorsFlag"] or\
        _BG_OLD_ANCHORS != CONFIG["AnchorsList"]

    # same parameters
    if not parameters_changed:
        # same data
        if numpy.array_equal(_BG_STRIP_OLDY, y0):
            # same result
            return _BG_STRIP_OLDBG

    _BG_STRIP_OLDY = y0
    _BG_STRIP_OLDPARS = [width, niter]
    _BG_SMOOTH_OLDWIDTH = CONFIG["SmoothingWidth"]
    _BG_SMOOTH_OLDFLAG = CONFIG["SmoothingFlag"]
    _BG_OLD_ANCHORS = CONFIG["AnchorsList"]
    _BG_OLD_ANCHORS_FLAG = CONFIG["AnchorsFlag"]

    y1 = savitsky_golay(y0, CONFIG["SmoothingWidth"]) if CONFIG["SmoothingFlag"] else y0

    anchors_indices = _convert_anchors_to_indices(x)

    background = strip(y1,
                       w=width,
                       niterations=niter,
                       factor=CONFIG["StripThresholdFactor"],
                       anchors=anchors_indices)

    _BG_STRIP_OLDBG = background

    return background


def snip_bg(x, y0, width):
    """Compute the snip bg for y0"""
    global _BG_SNIP_OLDY
    global _BG_SNIP_OLDWIDTH
    global _BG_SNIP_OLDBG
    global _BG_SMOOTH_OLDWIDTH
    global _BG_SMOOTH_OLDFLAG
    global _BG_OLD_ANCHORS
    global _BG_OLD_ANCHORS_FLAG

    parameters_changed =\
        _BG_SNIP_OLDWIDTH != width or\
        _BG_SMOOTH_OLDWIDTH != CONFIG["SmoothingWidth"] or\
        _BG_SMOOTH_OLDFLAG != CONFIG["SmoothingFlag"] or\
        _BG_OLD_ANCHORS_FLAG != CONFIG["AnchorsFlag"] or\
        _BG_OLD_ANCHORS != CONFIG["AnchorsList"]

    # same parameters
    if not parameters_changed:
        # same data
        if numpy.sum(_BG_SNIP_OLDY == y0) == len(y0):
            # same result
            return _BG_SNIP_OLDBG

    _BG_SNIP_OLDY = y0
    _BG_SNIP_OLDWIDTH = width
    _BG_SMOOTH_OLDWIDTH = CONFIG["SmoothingWidth"]
    _BG_SMOOTH_OLDFLAG = CONFIG["SmoothingFlag"]
    _BG_OLD_ANCHORS = CONFIG["AnchorsList"]
    _BG_OLD_ANCHORS_FLAG = CONFIG["AnchorsFlag"]

    y1 = savitsky_golay(y0, CONFIG["SmoothingWidth"]) if CONFIG["SmoothingFlag"] else y0

    anchors_indices = _convert_anchors_to_indices(x)

    if anchors_indices is None or not len(anchors_indices):
        anchors_indices = [0, len(y1) - 1]

    background = numpy.zeros_like(y1)
    previous_anchor = 0
    for anchor_index in anchors_indices:
        if (anchor_index > previous_anchor) and (anchor_index < len(y1)):
                background[previous_anchor:anchor_index] =\
                            snip1d(y1[previous_anchor:anchor_index],
                                   width)
                previous_anchor = anchor_index

    if previous_anchor < len(y1):
        background[previous_anchor:] = snip1d(y1[previous_anchor:],
                                              width)

    _BG_SNIP_OLDBG = background

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

    Return parameters as defined in CONFIG dict,
    set constraints to FIXED.
    """
    estimated_par = [CONFIG["StripWidth"],
                     CONFIG["StripIterations"]]
    constraints = numpy.zeros((len(estimated_par), 3), numpy.float)
    # code = 3: FIXED
    constraints[0][0] = 3
    constraints[1][0] = 3
    return estimated_par, constraints


def estimate_snip(x, y):
    """Estimation function for snip parameters.

    Return parameters as defined in CONFIG dict,
    set constraints to FIXED.
    """
    estimated_par = [CONFIG["SnipWidth"]]
    constraints = numpy.zeros((len(estimated_par), 3), numpy.float)
    # code = 3: FIXED
    constraints[0][0] = 3
    return estimated_par, constraints


def poly(x, y, *pars):
    """Order n polynomial.
    The order of the polynomial is defined by the number of
    coefficients (``*pars``).

    """
    p = numpy.poly1d(pars)
    return p(x)


def estimate_poly(x, y, deg=2):
    """Estimate polynomial coefficients.

    """
    # extract bg signal with strip, to estimate polynomial on background
    if CONFIG["EstimatePolyOnStrip"]:
        y = strip_bg(x, y,
                     CONFIG["StripWidth"],
                     CONFIG["StripIterations"])
    pcoeffs = numpy.polyfit(x, y, deg)
    cons = numpy.zeros((deg + 1, 3), numpy.float)
    return pcoeffs, cons


def estimate_quadratic_poly(x, y):
    """Estimate quadratic polynomial coefficients.
    """
    return estimate_poly(x, y, deg=2)


def estimate_cubic_poly(x, y):
    """Estimate cubic polynomial coefficients.
    """
    return estimate_poly(x, y, deg=3)


def estimate_quartic_poly(x, y):
    """Estimate degree 4 polynomial coefficients.
    """
    return estimate_poly(x, y, deg=4)


def estimate_quintic_poly(x, y):
    """Estimate degree 5 polynomial coefficients.
    """
    return estimate_poly(x, y, deg=5)


def configure(**kw):
    """Update the CONFIG dict
    """
    # inspect **kw to find known keys, update them in CONFIG
    for key in CONFIG:
        if key in kw:
            CONFIG[key] = kw[key]

    return CONFIG


THEORY = OrderedDict(
        (('No Background',
          FitTheory(
                description="No background function",
                function=lambda x, y0: numpy.zeros_like(x),
                parameters=[],
                is_background=True)),
         ('Constant',
          FitTheory(
                description='Constant background',
                function=lambda x, y0, c: c * numpy.ones_like(x),
                parameters=['Constant', ],
                estimate=lambda x, y: ([min(y)], [[0, 0, 0]]),
                is_background=True)),
         ('Linear',
          FitTheory(
                description="Linear background, parameters 'Constant' and"
                            " 'Slope'",
                function=lambda x, y0, a, b: a + b * x,
                parameters=['Constant', 'Slope'],
                estimate=estimate_linear,
                configure=configure,
                is_background=True)),
         ('Strip',
          FitTheory(
                description="Compute background using a strip filter\n"
                            "Parameters 'StripWidth', 'StripIterations'",
                function=strip_bg,
                parameters=['StripWidth', 'StripIterations'],
                estimate=estimate_strip,
                configure=configure,
                is_background=True)),
         ('Snip',
          FitTheory(
                description="Compute background using a snip filter\n"
                            "Parameter 'SnipWidth'",
                function=snip_bg,
                parameters=['SnipWidth'],
                estimate=estimate_snip,
                configure=configure,
                is_background=True)),
         ('Degree 2 Polynomial',
          FitTheory(
                description="Quadratic polynomial background, Parameters "
                            "'a', 'b' and 'c'\ny = a*x^2 + b*x +c",
                function=poly,
                parameters=['a', 'b', 'c'],
                estimate=estimate_quadratic_poly,
                configure=configure,
                is_background=True)),
         ('Degree 3 Polynomial',
          FitTheory(
                description="Cubic polynomial background, Parameters "
                            "'a', 'b', 'c' and 'd'\n"
                            "y = a*x^3 + b*x^2 + c*x + d",
                function=poly,
                parameters=['a', 'b', 'c', 'd'],
                estimate=estimate_cubic_poly,
                configure=configure,
                is_background=True)),
         ('Degree 4 Polynomial',
          FitTheory(
                description="Quartic polynomial background\n"
                            "y = a*x^4 + b*x^3 + c*x^2 + d*x + e",
                function=poly,
                parameters=['a', 'b', 'c', 'd', 'e'],
                estimate=estimate_quartic_poly,
                configure=configure,
                is_background=True)),
         ('Degree 5 Polynomial',
          FitTheory(
                description="Quaintic polynomial background\n"
                            "y = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f",
                function=poly,
                parameters=['a', 'b', 'c', 'd', 'e', 'f'],
                estimate=estimate_quintic_poly,
                configure=configure,
                is_background=True))))
