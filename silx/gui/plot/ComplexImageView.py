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
__date__ = "02/10/2017"


import logging
import numpy

from .. import qt, icons
from .PlotWindow import Plot2D
from .Colormap import Colormap
from . import items
from silx.gui.widgets.FloatEdit import FloatEdit

_logger = logging.getLogger(__name__)


_PHASE_COLORMAP = Colormap(
    name='hsv',
    vmin=-numpy.pi,
    vmax=numpy.pi)
"""Colormap to use for phase"""

# Complex colormap functions

def _phase2rgb(data):
    """Creates RGBA image with colour-coded phase.

    :param numpy.ndarray data: The data to convert
    :return: Array of RGBA colors
    :rtype: numpy.ndarray
    """
    if data.size == 0:
        return numpy.zeros((0, 0, 4), dtype=numpy.uint8)

    phase = numpy.angle(data)
    return _PHASE_COLORMAP.applyToData(phase)


def _complex2rgbalog(data, amin=0., dlogs=2, smax=None):
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

    def __init__(self):
        super(_ImageComplexData, self).__init__()
        self._readOnlyColormap = False
        self._mode = 'absolute'
        self._colormaps = {  # Default colormaps for all modes
            'absolute': Colormap(),
            'phase': _PHASE_COLORMAP.copy(),
            'real': Colormap(),
            'imaginary': Colormap(),
            'amplitude_phase': _PHASE_COLORMAP.copy(),
            'log10_amplitude_phase': _PHASE_COLORMAP.copy(),
        }

    _READ_ONLY_MODES = 'amplitude_phase', 'log10_amplitude_phase'
    """Modes that requires a read-only colormap."""

    def setVisualizationMode(self, mode):
        """Set the visualization mode to use.

        :param str mode:
        """
        mode = str(mode)
        assert mode in self._colormaps

        if mode != self._mode:
            # Save current colormap
            self._colormaps[self._mode] = self.getColormap()
            self._mode = mode

            # Set colormap for new mode
            self.setColormap(self._colormaps[mode])

    def getVisualizationMode(self):
        """Returns the visualization mode in use."""
        return self._mode

    def _isReadOnlyColormap(self):
        """Returns True if colormap should not be modified."""
        return self.getVisualizationMode() in self._READ_ONLY_MODES

    def setColormap(self, colormap):
        if not self._isReadOnlyColormap():
            super(_ImageComplexData, self).setColormap(colormap)

    def getColormap(self):
        if self._isReadOnlyColormap():
            return _PHASE_COLORMAP.copy()
        else:
            return super(_ImageComplexData, self).getColormap()


# Widgets

class _AmplitudeRangeDialog(qt.QDialog):
    """QDialog asking for the amplitude range to display."""

    sigRangeChanged = qt.Signal(tuple)
    """Signal emitted when the range has changed.

    It provides the new range as a 2-tuple: (max, delta)
    """

    def __init__(self,
                 parent=None,
                 amplitudeRange=None,
                 displayedRange=(None, 2)):
        super(_AmplitudeRangeDialog, self).__init__(parent)
        self.setWindowTitle('Set Displayed Amplitude Range')

        if amplitudeRange is not None:
            amplitudeRange = min(amplitudeRange), max(amplitudeRange)
        self._amplitudeRange = amplitudeRange
        self._defaultDisplayedRange = displayedRange

        layout = qt.QFormLayout()
        self.setLayout(layout)

        if self._amplitudeRange is not None:
            min_, max_ = self._amplitudeRange
            layout.addRow(
                qt.QLabel('Data Amplitude Range: [%g, %g]' % (min_, max_)))

        self._maxLineEdit = FloatEdit(parent=self)
        self._maxLineEdit.validator().setBottom(0.)
        self._maxLineEdit.setAlignment(qt.Qt.AlignRight)

        self._maxLineEdit.editingFinished.connect(self._rangeUpdated)
        layout.addRow('Displayed Max.:', self._maxLineEdit)

        self._autoscale = qt.QCheckBox('autoscale')
        self._autoscale.toggled.connect(self._autoscaleCheckBoxToggled)
        layout.addRow('', self._autoscale)

        self._deltaLineEdit = FloatEdit(parent=self)
        self._deltaLineEdit.validator().setBottom(1.)
        self._deltaLineEdit.setAlignment(qt.Qt.AlignRight)
        self._deltaLineEdit.editingFinished.connect(self._rangeUpdated)
        layout.addRow('Displayed delta (log10 unit):', self._deltaLineEdit)

        buttons = qt.QDialogButtonBox(self)
        buttons.addButton(qt.QDialogButtonBox.Ok)
        buttons.addButton(qt.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

        # Set dialog from default values
        self._resetDialogToDefault()

        self.rejected.connect(self._handleRejected)

    def _resetDialogToDefault(self):
        """Set Widgets of the dialog from range information
        """
        max_, delta = self._defaultDisplayedRange

        if max_ is not None:  # Not in autoscale
            displayedMax = max_
        elif self._amplitudeRange is not None:  # Autoscale with data
            displayedMax = self._amplitudeRange[1]
        else:  # Autoscale without data
            displayedMax = ''
        if displayedMax == "":
            self._maxLineEdit.setText("")
        else:
            self._maxLineEdit.setValue(displayedMax)
        self._maxLineEdit.setEnabled(max_ is not None)

        self._deltaLineEdit.setValue(delta)

        self._autoscale.setChecked(self._defaultDisplayedRange[0] is None)

    def getRangeInfo(self):
        """Returns the current range as a 2-tuple (max, delta (in log10))"""
        if self._autoscale.isChecked():
            max_ = None
        else:
            maxStr = self._maxLineEdit.text()
            max_ = self._maxLineEdit.value() if maxStr else None
        return max_, self._deltaLineEdit.value() if self._deltaLineEdit.text() else 2

    def _handleRejected(self):
        """Reset range info to default when rejected"""
        self._resetDialogToDefault()
        self._rangeUpdated()

    def _rangeUpdated(self):
        """Handle QLineEdit editing finised"""
        self.sigRangeChanged.emit(self.getRangeInfo())

    def _autoscaleCheckBoxToggled(self, checked):
        """Handle autoscale checkbox state changes"""
        if checked:  # Use default values
            if self._amplitudeRange is None:
                max_ = ''
            else:
                max_ = self._amplitudeRange[1]
            if max_ == "":
                self._maxLineEdit.setText("")
            else:
                self._maxLineEdit.setValue(max_)
        self._maxLineEdit.setEnabled(not checked)
        self._rangeUpdated()


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

    _RANGE_DIALOG_TEXT = 'Set Amplitude Range...'

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

        self._rangeDialogAction = qt.QAction(self)
        self._rangeDialogAction.setText(self._RANGE_DIALOG_TEXT)
        menu.addAction(self._rangeDialogAction)

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

        self._rangeDialogAction.setEnabled(mode == 'log10_amplitude_phase')

    def _triggered(self, action):
        """Handle triggering of menu actions"""
        actionText = action.text()

        if actionText == self._RANGE_DIALOG_TEXT:  # Show dialog
            # Get amplitude range
            data = self._plot2DComplex.getData(copy=False)

            if data.size > 0:
                absolute = numpy.absolute(data)
                dataRange = (numpy.nanmin(absolute), numpy.nanmax(absolute))
            else:
                dataRange = None

            # Show dialog
            dialog = _AmplitudeRangeDialog(
                parent=self,
                amplitudeRange=dataRange,
                displayedRange=self._plot2DComplex._getAmplitudeRangeInfo())
            dialog.sigRangeChanged.connect(self._rangeChanged)
            dialog.exec_()
            dialog.sigRangeChanged.disconnect(self._rangeChanged)

        else:  # update mode
            for mode, _, text in self._MODES:
                if actionText == text:
                    self._plot2DComplex.setVisualizationMode(mode)

    def _rangeChanged(self, range_):
        """Handle updates of range in the dialog"""
        self._plot2DComplex._setAmplitudeRangeInfo(*range_)


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
        self._amplitudeRangeInfo = None, 2
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
        self._plotImage.setData(self._displayedData)
        self._plotImage.setVisualizationMode(self._mode)
        self._plot2D._add(self._plotImage)
        self._plot2D.setActiveImage(self._plotImage.getLegend())

        toolBar = qt.QToolBar('Complex', self)
        toolBar.addWidget(
            _ComplexDataToolButton(parent=self, plot=self))

        self._plot2D.insertToolBar(self._plot2D.getProfileToolbar(), toolBar)

    def getPlot(self):
        """Return the PlotWidget displaying the data"""
        return self._plot2D

    def _convertData(self, data, mode):
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
            max_, delta = self._getAmplitudeRangeInfo()
            return _complex2rgbalog(data, dlogs=delta, smax=max_)
        else:
            _logger.error(
                'Unsupported conversion mode: %s, fallback to absolute',
                str(mode))
            return numpy.absolute(data)

    def _updatePlot(self):
        """Update the image in the plot"""

        mode = self.getVisualizationMode()

        self.getPlot().getColormapAction().setDisabled(
            mode in ('amplitude_phase', 'log10_amplitude_phase'))

        self._plotImage.setVisualizationMode(mode)

        image = self.getDisplayedData(copy=False)
        if mode in ('amplitude_phase', 'log10_amplitude_phase'):
            # Combined view
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

    def _setAmplitudeRangeInfo(self, max_=None, delta=2):
        """Set the amplitude range to display for 'log10_amplitude_phase' mode.

        :param max_: Max of the amplitude range.
                     If None it autoscales to data max.
        :param float delta: Delta range in log10 to display
        """
        self._amplitudeRangeInfo = max_, float(delta)
        mode = self.getVisualizationMode()
        if mode == 'log10_amplitude_phase':
            self._displayedData = self._convertData(
                self.getData(copy=False), mode)
            self._updatePlot()

    def _getAmplitudeRangeInfo(self):
        """Returns the amplitude range to use for 'log10_amplitude_phase' mode.

        :return: (max, delta), if max is None, then it autoscales to data max
        :rtype: 2-tuple"""
        return self._amplitudeRangeInfo

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

    def setKeepDataAspectRatio(self, flag):
        """Set whether the plot keeps data aspect ratio or not.

        :param bool flag: True to respect data aspect ratio
        """
        self.getPlot().setKeepDataAspectRatio(flag)

    def isKeepDataAspectRatio(self):
        """Returns whether the plot is keeping data aspect ratio or not."""
        return self.getPlot().isKeepDataAspectRatio()
