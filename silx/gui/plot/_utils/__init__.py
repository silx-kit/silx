# coding: utf-8
# /*##########################################################################
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
# ###########################################################################*/
"""Miscellaneous utility functions for the Plot"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "21/03/2017"


from .panzoom import FLOAT32_SAFE_MIN, FLOAT32_MINPOS, FLOAT32_SAFE_MAX
from .panzoom import applyZoomToPlot, applyPan


def clamp(value, min_=0., max_=1.):
    """Clip a value to a range [min, max].

    :param value: The value to clip
    :param min_: The min edge of the range
    :param max_: The max edge of the range
    :return: The clipped value
    """
    if value < min_:
        return min_
    elif value > max_:
        return max_
    else:
        return value
