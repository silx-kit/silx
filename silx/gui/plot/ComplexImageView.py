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

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "12/09/2017"


import logging
import numpy

from silx.gui import qt, icons
from .PlotWindow import Plot2D

_logger = logging.getLogger(__name__)


class _ComplexDataToolButton(qt.QToolButton):
    """QToolButton providing choices of complex data visualization modes

    :param parent: See :class:`QToolButton`
    :param plot: The :class:`ComplexImageView` to control
    """

    _MODES = [
        ('absolute', 'math-amplitude', 'Amplitude'),
        ('phase', 'math-phase', 'Phase'),
        ('real', 'math-real', 'Real part'),
        ('imaginary', 'math-imaginary', 'Imaginary part')]

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

        self._plotImage = self._plot2D.getImage(
            self._plot2D.addImage(self._displayedData))

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
        else:
            _logger.error(
                'Unsupported conversion mode: %s, fallback to absolute',
                str(mode))
            return numpy.absolute(data)

    def _updatePlot(self):
        """Update the image in the plot"""
        image = self.getDisplayedData(copy=False)
        self._plotImage.setData(image)

    def setData(self, data, copy=True):
        """Set the complex data to display.

        :param numpy.ndarray data: 2D complex data
        :param bool copy: True (default) to copy the data,
                          False to use provided data (do not modify!).
        """
        data = numpy.array(data, copy=copy)
        assert data.ndim == 2
        if data.dtype.kind != 'c':  # Convert to complex
            data = numpy.array(data, dtype=numpy.complex)
        self._data = data
        self._displayedData = self._convertData(
            data, self.getVisualizationMode())

        self._updatePlot()
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

        Supported visualization modes are: absolute, phase, real, imaginary.

        :rtype: tuple of str
        """
        return ('absolute',
                'phase',
                'real',
                'imaginary')

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
        return self._plotImage.getColormap()

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
