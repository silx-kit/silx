# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017-2018 European Synchrotron Radiation Facility
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
"""This module provides the :class:`ImageComplexData` of the :class:`Plot`.
"""

from __future__ import absolute_import

__authors__ = ["Vincent Favre-Nicolin", "T. Vincent"]
__license__ = "MIT"
__date__ = "19/01/2018"


import logging
import numpy

from silx.third_party import enum

from ..Colormap import Colormap
from .core import ColormapMixIn, ItemChangedType
from .image import ImageBase


_logger = logging.getLogger(__name__)


# Complex colormap functions

def _phase2rgb(colormap, data):
    """Creates RGBA image with colour-coded phase.

    :param Colormap colormap: The colormap to use
    :param numpy.ndarray data: The data to convert
    :return: Array of RGBA colors
    :rtype: numpy.ndarray
    """
    if data.size == 0:
        return numpy.zeros((0, 0, 4), dtype=numpy.uint8)

    phase = numpy.angle(data)
    return colormap.applyToData(phase)


def _complex2rgbalog(phaseColormap, data, amin=0., dlogs=2, smax=None):
    """Returns RGBA colors: colour-coded phases and log10(amplitude) in alpha.

    :param Colormap phaseColormap: Colormap to use for the phase
    :param numpy.ndarray data: the complex data array to convert to RGBA
    :param float amin: the minimum value for the alpha channel
    :param float dlogs: amplitude range displayed, in log10 units
    :param float smax:
        if specified, all values above max will be displayed with an alpha=1
    """
    if data.size == 0:
        return numpy.zeros((0, 0, 4), dtype=numpy.uint8)

    rgba = _phase2rgb(phaseColormap, data)
    sabs = numpy.absolute(data)
    if smax is not None:
        sabs[sabs > smax] = smax
    a = numpy.log10(sabs + 1e-20)
    a -= a.max() - dlogs  # display dlogs orders of magnitude
    rgba[..., 3] = 255 * (amin + a / dlogs * (1 - amin) * (a > 0))
    return rgba


def _complex2rgbalin(phaseColormap, data, gamma=1.0, smax=None):
    """Returns RGBA colors: colour-coded phase and linear amplitude in alpha.

    :param Colormap phaseColormap: Colormap to use for the phase
    :param numpy.ndarray data:
    :param float gamma: Optional exponent gamma applied to the amplitude
    :param float smax:
    """
    if data.size == 0:
        return numpy.zeros((0, 0, 4), dtype=numpy.uint8)

    rgba = _phase2rgb(phaseColormap, data)
    a = numpy.absolute(data)
    if smax is not None:
        a[a > smax] = smax
    a /= a.max()
    rgba[..., 3] = 255 * a**gamma
    return rgba


class ImageComplexData(ImageBase, ColormapMixIn):
    """Specific plot item to force colormap when using complex colormap.

    This is returning the specific colormap when displaying
    colored phase + amplitude.
    """

    class Mode(enum.Enum):
        """Identify available display mode for complex"""
        ABSOLUTE = 'absolute'
        PHASE = 'phase'
        REAL = 'real'
        IMAGINARY = 'imaginary'
        AMPLITUDE_PHASE = 'amplitude_phase'
        LOG10_AMPLITUDE_PHASE = 'log10_amplitude_phase'
        SQUARE_AMPLITUDE = 'square_amplitude'

    def __init__(self):
        ImageBase.__init__(self)
        ColormapMixIn.__init__(self)
        self._data = numpy.zeros((0, 0), dtype=numpy.complex64)
        self._dataByModesCache = {}
        self._mode = self.Mode.ABSOLUTE
        self._amplitudeRangeInfo = None, 2

        # Use default from ColormapMixIn
        colormap = super(ImageComplexData, self).getColormap()

        phaseColormap = Colormap(
            name='hsv',
            vmin=-numpy.pi,
            vmax=numpy.pi)
        phaseColormap.setEditable(False)

        self._colormaps = {  # Default colormaps for all modes
            self.Mode.ABSOLUTE: colormap,
            self.Mode.PHASE: phaseColormap,
            self.Mode.REAL: colormap,
            self.Mode.IMAGINARY: colormap,
            self.Mode.AMPLITUDE_PHASE: phaseColormap,
            self.Mode.LOG10_AMPLITUDE_PHASE: phaseColormap,
            self.Mode.SQUARE_AMPLITUDE: colormap,
        }

    def _addBackendRenderer(self, backend):
        """Update backend renderer"""
        plot = self.getPlot()
        assert plot is not None
        if not self._isPlotLinear(plot):
            # Do not render with non linear scales
            return None

        mode = self.getVisualizationMode()
        if mode in (self.Mode.AMPLITUDE_PHASE,
                    self.Mode.LOG10_AMPLITUDE_PHASE):
            # For those modes, compute RGBA image here
            colormap = None
            data = self.getRgbaImageData(copy=False)
        else:
            colormap = self.getColormap()
            data = self.getData(copy=False)

        if data.size == 0:
            return None  # No data to display

        return backend.addImage(data,
                                legend=self.getLegend(),
                                origin=self.getOrigin(),
                                scale=self.getScale(),
                                z=self.getZValue(),
                                selectable=self.isSelectable(),
                                draggable=self.isDraggable(),
                                colormap=colormap,
                                alpha=self.getAlpha())


    def setVisualizationMode(self, mode):
        """Set the visualization mode to use.

        :param Mode mode:
        """
        assert isinstance(mode, self.Mode)
        assert mode in self._colormaps

        if mode != self._mode:
            self._mode = mode

            self._updated(ItemChangedType.VISUALIZATION_MODE)

            # Send data updated as value returned by getData has changed
            self._updated(ItemChangedType.DATA)

            # Update ColormapMixIn colormap
            colormap = self._colormaps[self._mode]
            if colormap is not super(ImageComplexData, self).getColormap():
                super(ImageComplexData, self).setColormap(colormap)

    def getVisualizationMode(self):
        """Returns the visualization mode in use.

        :rtype: Mode
        """
        return self._mode

    def _setAmplitudeRangeInfo(self, max_=None, delta=2):
        """Set the amplitude range to display for 'log10_amplitude_phase' mode.

        :param max_: Max of the amplitude range.
                     If None it autoscales to data max.
        :param float delta: Delta range in log10 to display
        """
        self._amplitudeRangeInfo = max_, float(delta)
        self._updated(ItemChangedType.VISUALIZATION_MODE)

    def _getAmplitudeRangeInfo(self):
        """Returns the amplitude range to use for 'log10_amplitude_phase' mode.

        :return: (max, delta), if max is None, then it autoscales to data max
        :rtype: 2-tuple"""
        return self._amplitudeRangeInfo

    def setColormap(self, colormap, mode=None):
        """Set the colormap for this specific mode.

        :param ~silx.gui.plot.Colormap.Colormap colormap: The colormap
        :param Mode mode:
            If specified, set the colormap of this specific mode.
            Default: current mode.
        """
        if mode is None:
            mode = self.getVisualizationMode()

        self._colormaps[mode] = colormap
        if mode is self.getVisualizationMode():
            super(ImageComplexData, self).setColormap(colormap)
        else:
            self._updated(ItemChangedType.COLORMAP)

    def getColormap(self, mode=None):
        """Get the colormap for the (current) mode.

        :param Mode mode:
            If specified, get the colormap of this specific mode.
            Default: current mode.
        :rtype: ~silx.gui.plot.Colormap.Colormap
        """
        if mode is None:
            mode = self.getVisualizationMode()

        return self._colormaps[mode]

    def setData(self, data, copy=True):
        """"Set the image complex data

        :param numpy.ndarray data: 2D array of complex with 2 dimensions (h, w)
        :param bool copy: True (Default) to get a copy,
                          False to use internal representation (do not modify!)
        """
        data = numpy.array(data, copy=copy)
        assert data.ndim == 2
        if not numpy.issubdtype(data.dtype, numpy.complexfloating):
            _logger.warning(
                'Image is not complex, converting it to complex to plot it.')
            data = numpy.array(data, dtype=numpy.complex64)

        self._data = data
        self._dataByModesCache = {}

        # TODO hackish data range implementation
        if self.isVisible():
            plot = self.getPlot()
            if plot is not None:
                plot._invalidateDataRange()

        self._updated(ItemChangedType.DATA)

    def getComplexData(self, copy=True):
        """Returns the image complex data

        :param bool copy: True (Default) to get a copy,
                          False to use internal representation (do not modify!)
        :rtype: numpy.ndarray of complex
        """
        return numpy.array(self._data, copy=copy)

    def getData(self, copy=True, mode=None):
        """Returns the image data corresponding to (current) mode.

        The returned data is always floats, to get the complex data, use
        :meth:`getComplexData`.

        :param bool copy: True (Default) to get a copy,
                          False to use internal representation (do not modify!)
        :param Mode mode:
            If specified, get data corresponding to the mode.
            Default: Current mode.
        :rtype: numpy.ndarray of float
        """
        if mode is None:
            mode = self.getVisualizationMode()

        if mode not in self._dataByModesCache:
            # Compute data for mode and store it in cache
            complexData = self.getComplexData(copy=False)
            if mode is self.Mode.PHASE:
                data = numpy.angle(complexData)
            elif mode is self.Mode.REAL:
                data = numpy.real(complexData)
            elif mode is self.Mode.IMAGINARY:
                data = numpy.imag(complexData)
            elif mode in (self.Mode.ABSOLUTE,
                          self.Mode.LOG10_AMPLITUDE_PHASE,
                          self.Mode.AMPLITUDE_PHASE):
                data = numpy.absolute(complexData)
            elif mode is self.Mode.SQUARE_AMPLITUDE:
                data = numpy.absolute(complexData) ** 2
            else:
                _logger.error(
                    'Unsupported conversion mode: %s, fallback to absolute',
                    str(mode))
                data = numpy.absolute(complexData)

            self._dataByModesCache[mode] = data

        return numpy.array(self._dataByModesCache[mode], copy=copy)

    def getRgbaImageData(self, copy=True, mode=None):
        """Get the displayed RGB(A) image for (current) mode

        :param bool copy: Ignored for this class
        :param Mode mode:
            If specified, get data corresponding to the mode.
            Default: Current mode.
        :rtype: numpy.ndarray of uint8 of shape (height, width, 4)
        """
        if mode is None:
            mode = self.getVisualizationMode()

        colormap = self.getColormap(mode=mode)
        if mode is self.Mode.AMPLITUDE_PHASE:
            data = self.getComplexData(copy=False)
            return _complex2rgbalin(colormap, data)
        elif mode is self.Mode.LOG10_AMPLITUDE_PHASE:
            data = self.getComplexData(copy=False)
            max_, delta = self._getAmplitudeRangeInfo()
            return _complex2rgbalog(colormap, data, dlogs=delta, smax=max_)
        else:
            data = self.getData(copy=False, mode=mode)
            return colormap.applyToData(data)
