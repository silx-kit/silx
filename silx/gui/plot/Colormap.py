# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2015-2017 European Synchrotron Radiation Facility
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
"""This module provides the Colormap object
"""

from __future__ import absolute_import

__authors__ = ["T. Vincent", "H.Payno"]

from silx.gui import qt


_COLORMAPS = {
    'gray': 0,
    'reversed gray': 1,
    'red': 2,
    'green': 3,
    'blue': 4,
    'temperature': 5
}

COLORMAPS = tuple(_COLORMAPS.keys())
"""Tuple of supported colormap names."""

NORMS = 'linear', 'log'
"""Tuple of supported normalizations."""


class Colormap(object):
    """Description of a colormap

    :param str name: Name of the colormap
    :param str norm: Normalization: 'linear' (default) or 'log'
    :param float vmin:
        Lower bound of the colormap or None for autoscale (default)
    :param float vmax:
        Upper bounds of the colormap or None for autoscale (default)
    """

    sigChanged = qt.Signal()

    def __init__(self, name, norm='linear', vmin=None, vmax=None):
        assert name in COLORMAPS
        self._name = str(name)

        assert norm in ('linear', 'log')
        self._norm = str(norm)

        self._vmin = float(vmin) if vmin is not None else None
        self._vmax = float(vmax) if vmax is not None else None

    def isAutoscale(self):
        """True if both min and max are in autoscale mode"""
        return self._vmin is None or self._vmax is None

    def getName(self):
        """Return the name of the colormap (str)"""
        return self._name

    def getNorm(self):
        """Return the normalization of the colormap (str)"""
        return self._norm

    def getVMin(self):
        """Return the lower bound of the colormap or None"""
        return self._vmin

    def getVMax(self):
        """Return the upper bounds of the colormap or None"""
        return self._vmax
