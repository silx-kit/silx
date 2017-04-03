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


def clipColormapLogRange(colormap):
    """Clip colormap vmin and vmax to 1, 10 if normalization is 'log' and vmin
    or vmax <1

    :param dict colormap: the colormap for which we want to clip vmin and vmax
    """
    if colormap['normalization'] is 'log':
        if colormap['vmin'] < 1. or colormap['vmax'] < 1.:
            colormap['vmin'], colormap['vmax'] = 1., 10.

