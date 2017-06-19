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
__license__ = "MIT"
__date__ = "05/12/2016"

from silx.gui import qt
import copy


_OGL_COLORMAPS = {
    'gray': 0,
    'reversed gray': 1,
    'red': 2,
    'green': 3,
    'blue': 4,
    'temperature': 5
}

OGL_COLORMAPS = tuple(_OGL_COLORMAPS.keys())
"""Tuple of supported colormap names."""

NORMS = 'linear', 'log'
"""Tuple of supported normalizations."""

NORMALIZATIONS = ('linear', 'log')
"""Tuple of managed normalizations"""


class Colormap(qt.QObject):
    """Description of a colormap

    :param str name: Name of the colormap
    :param tuple colors: optional, custom colormap.
            Nx3 or Nx4 numpy array of RGB(A) colors,
            either uint8 or float in [0, 1].
            If 'name' is None, then this array is used as the colormap.
    :param str norm: Normalization: 'linear' (default) or 'log'
    :param float vmin:
        Lower bound of the colormap or None for autoscale (default)
    :param float vmax:
        Upper bounds of the colormap or None for autoscale (default)
    """

    sigChanged = qt.Signal()

    def __init__(self, name, colors=None, norm='linear', vmin=None, vmax=None):
        qt.QObject.__init__(self)
        self._name = str(name) if name is not None else None
        self._colors = colors

        assert norm in NORMALIZATIONS
        self._norm = str(norm)

        self._vmin = float(vmin) if vmin is not None else None
        self._vmax = float(vmax) if vmax is not None else None

    def isAutoscale(self):
        """True if both min and max are in autoscale mode"""
        return self._vmin is None or self._vmax is None

    def getName(self):
        """Return the name of the colormap (str)"""
        return self._name

    def setName(self, name):
        """Set the name of the colormap and load the colors corresponding to
        the name

        :param str name: the name of the colormap (should be in ['gray',
            'reversed gray', 'temperature', 'red', 'green', 'blue', 'jet',
            'viridis', 'magma', 'inferno', 'plasma']
        """
        self._name = str(name)
        self._colors = None
        self.sigChanged.emit()

    def getColors(self):
        """
        :return tuple: the list of colors for the colormap"""
        return self._colors

    def setColors(self, colors):
        """
        Set the colors of the colormap.

        .. warning: this will set the value of name to an empty string
        """
        self._colors = colors
        self._name = ""
        self.sigChanged.emit()

    def getNorm(self):
        """Return the normalization of the colormap (str)"""
        return self._norm

    def setNorm(self, norm):
        """Set the norm ('log', 'linear')

        :param str norm: the norm to set
        """
        self._norm = str(norm)
        self.sigChanged.emit()

    def getVMin(self):
        """Return the lower bound of the colormap or None"""
        return self._vmin

    def setVMin(self, vmin):
        """Set the minimal value of the colormap

        :param float vmin: Lower bound of the colormap or None for autoscale
            (default)
            value)
        """
        self._vmin = vmin
        self.sigChanged.emit()

    def getVMax(self):
        """
        :return: the upper bounds of the colormap or None"""
        return self._vmax

    def setVMax(self, vmax):
        """Set the maximal value of the colormap

        :param float vmax: Upper bounds of the colormap or None for autoscale
            (default)
        """
        self._vmax = vmax
        self.sigChanged.emit()

    def getColorMapRange(self):
        """

        :return: the tuple vmin, vmax
        """
        return (self._vmin, self._vmax)

    def setColorMapRange(self, vmin, vmax):
        """
        Set bounds to the colormap

        :param vmin: Lower bound of the colormap or None for autoscale
            (default)
        :param vmax: Upper bounds of the colormap or None for autoscale
            (default)
        """
        self._vmin = vmin
        self._vmax = vmax
        self.sigChanged.emit()

    def __getitem__(self, item):
        attr = '_' + item
        if not hasattr(self, attr):
            raise KeyError(item)
        else:
            return getattr(self, attr)

    def getDict(self):
        """Return the equivalent colormap as a dictionnary
        (old colormap representation)
        
        :return dict: the representation of the Colormap as a dictionnary
        """
        return {
            'name': self._name,
            'colors': copy.copy(self._colors),
            'vmin': self._vmin,
            'vmax': self._vmax,
            'autoscale': self.isAutoscale(),
            'normalization': self._norm
        }

    def setFromDict(self, dic):
        """Set values to the colormap from a dictionnary
        
        :param dict dic: the colormap as a dictionnary
        """
        _name = dic['name'] if 'name' in dic else None
        _colors = dic['colors'] if 'colors' in dic else None
        _vmin = dic['vmin'] if 'vmin' in dic else None
        _vmax = dic['vmax'] if 'vmax' in dic else None
        _norm = dic['normalization'] if 'normalization' in dic else None

        if _name is None and _colors is None:
            err = 'The colormap should have a name defined or a tuple of colors'
            raise ValueError(err)
        if _norm not in NORMALIZATIONS:
            err = 'Given normalization is not recoginized (%s)' % _norm
            raise ValueError(err)

        if 'autoscale' in dic:
            if dic['autoscale'] is True:
                if _vmin is not None or _vmax is not None:
                    err = "Can't set the colormap from the dictionnary because"
                    err += " autoscale is requested but vmin and vmax are also"
                    err += " defined (!= None)"
                    raise ValueError(err)
            elif dic['autoscale'] is False:
                if _vmin is None and _vmax is None:
                    err = "Can't set the colormap from the dictionnary because"
                    err += " autoscale is not requested but vmin and vmax are"
                    err += " both set to None"
                    raise ValueError(err)
            else:
                raise ValueError('Autoscale value should be True or False')

        self._name = _name
        self._colors = _colors
        self._vmin = _vmin
        self._vmax = _vmax
        self._autoscale = True if (_vmin is None and _vmax is None ) else False
        self._norm = _norm

        self.sigChanged.emit()

    @staticmethod
    def getFromDict(dic):
        colormap = Colormap(name="")
        colormap.setFromDict(dic)
        return colormap
