# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017 European Synchrotron Radiation Facility
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
"""This module provides a widget to view 2D complex data.

The :class:`ComplexImageView` widget is dedicated to visualize a single 2D dataset
of complex data.
"""

from __future__ import absolute_import

__authors__ = ["Vincent Favre-Nicolin", "T. Vincent"]
__license__ = "MIT"
__date__ = "12/09/2017"


import logging
import numpy

from silx.gui import qt, icons
from .PlotWindow import Plot2D
from .Colormap import Colormap
from . import items

_logger = logging.getLogger(__name__)


# Complex colormap functions

def _phase2rgb(data):
    """Creates RGBA image with colour-coded phase.

    :param numpy.ndarray data: The data to convert
    :return: Array of RGBA colors
    :rtype: numpy.ndarray
    """
    if data.size == 0:
        return numpy.zeros((0, 0, 4), dtype=numpy.uint8)

    ph = numpy.angle(data)
    t = numpy.pi / 3
    rgba = 255 * numpy.ones(data.shape + (4,), dtype=numpy.uint8)
    rgba[..., 0] = 255 * (
        (ph < t) * (ph > -t) +
        (ph > t) * (ph < 2 * t) * (2 * t - ph) / t +
        (ph > -2 * t) * (ph < -t) * (ph + 2 * t) / t)
    rgba[..., 1] = 255 * (
        (ph > t) +
        (ph < -2 * t) * (-2 * t - ph) / t +
        (ph > 0) * (ph < t) * ph / t)
    rgba[..., 2] = 255 * (
        (ph < -t) +
        (ph > -t) * (ph < 0) * (-ph) / t +
        (ph > 2 * t) * (ph - 2 * t) / t)
    return rgba


def _complex2rgbalog(data, amin=0.5, dlogs=2, smax=None):
    """Returns RGBA colors: colour-coded phases and log10(amplitude) in alpha.

    :param numpy.ndarray data: the complex data array to convert to RGBA
    :param float amin: the minimum value for the alpha channel
    :param float dlogs: amplitude range displayed, in log10 units
    :param float smax:
        if specified, all values above max will be displayed with an alpha=1
    """
    if data.size == 0:
        return numpy.zeros((0, 0, 4), dtype=numpy.uint8)

    rgba = _phase2rgb(data)
    sabs = numpy.absolute(data)
    if smax is not None:
        sabs[sabs > smax] = smax
    a = numpy.log10(sabs + 1e-20)
    a -= a.max() - dlogs  # display dlogs orders of magnitude
    rgba[..., 3] = 255 * (amin + a / dlogs * (1 - amin) * (a > 0))
    return rgba


def _complex2rgbalin(data, gamma=1.0, smax=None):
    """Returns RGBA colors: colour-coded phase and linear amplitude in alpha.

    :param numpy.ndarray data:
    :param float gamma: Optional exponent gamma applied to the amplitude
    :param float smax:
    """
    if data.size == 0:
        return numpy.zeros((0, 0, 4), dtype=numpy.uint8)

    rgba = _phase2rgb(data)
    a = numpy.absolute(data)
    if smax is not None:
        a[a > smax] = smax
    a /= a.max()
    rgba[..., 3] = 255 * a**gamma
    return rgba


# Dedicated plot item

class _ImageComplexData(items.ImageData):
    """Specific plot item to force colormap when using complex colormap.

    This is returning the specific colormap when displaying
    colored phase + amplitude.
    """

    _COMPLEX_COLORMAP = Colormap(
        name=None,
        colors=_phase2rgb(numpy.exp(numpy.linspace(-numpy.pi, numpy.pi, 256) * 1j)),
        vmin=-numpy.pi,
        vmax=numpy.pi)
    """The phase colormap for combined visualization modes"""

    def setData(self, *args, **kwargs):
        super(_ImageComplexData, self).setData(*args, **kwargs)
        self.sigItemChanged.emit(items.ItemChangedType.COLORMAP)

    def getColormap(self):
        if self.getAlternativeImageData(copy=False) is not None:
            return self._COMPLEX_COLORMAP.copy()
        else:
            return super(_ImageComplexData, self).getColormap()


# Widgets

class _ComplexDataToolButton(qt.QToolButton):
    """QToolButton providing choices of complex data visualization modes

    :param parent: See :class:`QToolButton`
    :param plot: The :class:`ComplexImageView` to control
    """

    _MODES = [
        ('absolute', 'math-amplitude', 'Amplitude'),
        ('phase', 'math-phase', 'Phase'),
        ('real', 'math-real', 'Real part'),
        ('imaginary', 'math-imaginary', 'Imaginary part'),
        ('amplitude_phase', 'math-phase-color', 'Amplitude and Phase'),
        ('log10_amplitude_phase', 'math-phase-color-log', 'Log10(Amp.) and Phase')]

    def __init__(self, parent=None, plot=None):
        super(_ComplexDataToolButton, self).__init__(parent=parent)

        assert plot is not None
        self._plot2DComplex = plot

        menu = qt.QMenu(self)
        menu.triggered.connect(self._triggered)
        self.setMenu(menu)

        for _, icon, text in self._MODES:
            action = qt.QAction(icons.getQIcon(icon), text, self)
            action.setIconVisibleInMenu(True)
            menu.addAction(action)

        self.setPopupMode(qt.QToolButton.InstantPopup)

        self._modeChanged(self._plot2DComplex.getVisualizationMode())
        self._plot2DComplex.sigVisualizationModeChanged.connect(
            self._modeChanged)

    def _modeChanged(self, mode):
        """Handle change of visualization modes"""
        for actionMode, icon, text in self._MODES:
            if actionMode == mode:
                self.setIcon(icons.getQIcon(icon))
                self.setToolTip('Display the ' + text.lower())
                break

    def _triggered(self, action):
        """Handle triggering of menu actions"""
        actionText = action.text()

        for mode, _, text in self._MODES:
            if actionText == text:
                self._plot2DComplex.setVisualizationMode(mode)


class ComplexImageView(qt.QWidget):
    """Display an image of complex data and allow to choose the visualization.

    :param parent: See :class:`QMainWindow`
    """

    sigDataChanged = qt.Signal()
    """Signal emitted when data has changed."""

    sigVisualizationModeChanged = qt.Signal(str)
    """Signal emitted when the visualization mode has changed.

    It provides the new visualization mode.
    """

    def __init__(self, parent=None):
        super(ComplexImageView, self).__init__(parent)
        if parent is None:
            self.setWindowTitle('ComplexImageView')

        self._mode = 'absolute'
        self._data = numpy.zeros((0, 0), dtype=numpy.complex)
        self._displayedData = numpy.zeros((0, 0), dtype=numpy.float)

        self._plot2D = Plot2D(self)

        layout = qt.QHBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plot2D)
        self.setLayout(layout)

        # Create and add image to the plot
        self._plotImage = _ImageComplexData()
        self._plotImage._setLegend('__ComplexImageView__complex_image__')
        self._plotImage.setColormap(self._plot2D.getDefaultColormap().copy())
        self._plotImage.setData(self._displayedData)
        self._plot2D._add(self._plotImage)
        self._plot2D.setActiveImage(self._plotImage.getLegend())

        toolBar = qt.QToolBar('Complex', self)
        toolBar.addWidget(
            _ComplexDataToolButton(parent=self, plot=self))
        self._plot2D.insertToolBar(self._plot2D.getProfileToolbar(), toolBar)

    def getPlot(self):
        """Return the PlotWidget displaying the data"""
        return self._plot2D

    @staticmethod
    def _convertData(data, mode):
        """Convert complex data according to provided mode.

        :param numpy.ndarray data: The complex data to convert
        :param str mode: The visualization mode
        :return: The data corresponding to the mode
        :rtype: 2D numpy.ndarray of float or RGBA image
        """
        if mode == 'absolute':
            return numpy.absolute(data)
        elif mode == 'phase':
            return numpy.angle(data)
        elif mode == 'real':
            return numpy.real(data)
        elif mode == 'imaginary':
            return numpy.imag(data)
        elif mode == 'amplitude_phase':
            return _complex2rgbalin(data)
        elif mode == 'log10_amplitude_phase':
            return _complex2rgbalog(data)
        else:
            _logger.error(
                'Unsupported conversion mode: %s, fallback to absolute',
                str(mode))
            return numpy.absolute(data)

    def _updatePlot(self):
        """Update the image in the plot"""
        image = self.getDisplayedData(copy=False)
        if image.ndim == 3:  # Combined view
            absolute = numpy.absolute(self.getData(copy=False))
            self._plotImage.setData(
                absolute, alternative=image, copy=False)
        else:
            self._plotImage.setData(
                image, alternative=None, copy=False)

    def setData(self, data=None, copy=True):
        """Set the complex data to display.

        :param numpy.ndarray data: 2D complex data
        :param bool copy: True (default) to copy the data,
                          False to use provided data (do not modify!).
        """
        if data is None:
            data = numpy.zeros((0, 0), dtype=numpy.complex)
        else:
            data = numpy.array(data, copy=copy)

        assert data.ndim == 2
        if data.dtype.kind != 'c':  # Convert to complex
            data = numpy.array(data, dtype=numpy.complex)
        shape_changed = (self._data.shape != data.shape)
        self._data = data
        self._displayedData = self._convertData(
            data, self.getVisualizationMode())

        self._updatePlot()
        if shape_changed:
            self.getPlot().resetZoom()

        self.sigDataChanged.emit()

    def getData(self, copy=True):
        """Get the currently displayed complex data.

        :param bool copy: True (default) to return a copy of the data,
                          False to return internal data (do not modify!).
        :return: The complex data array.
        :rtype: numpy.ndarray of complex with 2 dimensions
        """
        return numpy.array(self._data, copy=copy)

    def getDisplayedData(self, copy=True):
        """Returns the displayed data depending on the visualization mode

        WARNING: The returned data can be a uint8 RGBA image

        :param bool copy: True (default) to return a copy of the data,
                          False to return internal data (do not modify!)
        :rtype: numpy.ndarray of float with 2 dims or RGBA image (uint8).
        """
        return numpy.array(self._displayedData, copy=copy)

    @staticmethod
    def getSupportedVisualizationModes():
        """Returns the supported visualization modes.

        Supported visualization modes are:

        - amplitude: The absolute value provided by numpy.absolute
        - phase: The phase (or argument) provided by numpy.angle
        - real: Real part
        - imaginary: Imaginary part
        - amplitude_phase: Color-coded phase with amplitude as alpha.
        - log10_amplitude_phase:
          Color-coded phase with log10(amplitude) as alpha.

        :rtype: tuple of str
        """
        return ('absolute',
                'phase',
                'real',
                'imaginary',
                'amplitude_phase',
                'log10_amplitude_phase')

    def setVisualizationMode(self, mode):
        """Set the mode of visualization of the complex data.

        See :meth:`getSupportedVisualizationModes` for the list of
        supported modes.

        :param str mode: The mode to use.
        """
        assert mode in self.getSupportedVisualizationModes()
        if mode != self._mode:
            self._mode = mode
            self._displayedData = self._convertData(
                self.getData(copy=False), mode)
            self._updatePlot()
            self.sigVisualizationModeChanged.emit(mode)

    def getVisualizationMode(self):
        """Get the current visualization mode of the complex data.

        :rtype: str
        """
        return self._mode

    # Image item proxy

    def setColormap(self, colormap):
        """Set the colormap to use for amplitude, phase, real or imaginary.

        WARNING: This colormap is not used when displaying both
        amplitude and phase.

        :param Colormap colormap: The colormap
        """
        self._plotImage.setColormap(colormap)

    def getColormap(self):
        """Returns the colormap used to display the data.

        :rtype: Colormap
        """
        # Returns internal colormap and bypass forcing colormap
        return items.ImageData.getColormap(self._plotImage)

    def getOrigin(self):
        """Returns the offset from origin at which to display the image.

        :rtype: 2-tuple of float
        """
        return self._plotImage.getOrigin()

    def setOrigin(self, origin):
        """Set the offset from origin at which to display the image.

        :param origin: (ox, oy) Offset from origin
        :type origin: float or 2-tuple of float
        """
        self._plotImage.setOrigin(origin)

    def getScale(self):
        """Returns the scale of the image in data coordinates.

        :rtype: 2-tuple of float
        """
        return self._plotImage.getScale()

    def setScale(self, scale):
        """Set the scale of the image

        :param scale: (sx, sy) Scale of the image
        :type scale: float or 2-tuple of float
        """
        self._plotImage.setScale(scale)

    # PlotWidget API proxy

    def getXAxis(self):
        """Returns the X axis

        :rtype: :class:`.items.Axis`
        """
        return self.getPlot().getXAxis()

    def getYAxis(self):
        """Returns an Y axis

        :rtype: :class:`.items.Axis`
        """
        return self.getPlot().getYAxis(axis='left')

    def getGraphTitle(self):
        """Return the plot main title as a str."""
        return self.getPlot().getGraphTitle()

    def setGraphTitle(self, title=""):
        """Set the plot main title.

        :param str title: Main title of the plot (default: '')
        """
        self.getPlot().setGraphTitle(title)
