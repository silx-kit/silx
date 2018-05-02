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
"""This module provides a widget to view 2D complex data.

The :class:`ComplexImageView` widget is dedicated to visualize a single 2D dataset
of complex data.
"""

from __future__ import absolute_import

__authors__ = ["Vincent Favre-Nicolin", "T. Vincent"]
__license__ = "MIT"
__date__ = "24/04/2018"


import logging
import collections
import numpy

from .. import qt, icons
from .PlotWindow import Plot2D
from . import items
from .items import ImageComplexData
from silx.gui.widgets.FloatEdit import FloatEdit

_logger = logging.getLogger(__name__)


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

    _MODES = collections.OrderedDict([
        (ImageComplexData.Mode.ABSOLUTE, ('math-amplitude', 'Amplitude')),
        (ImageComplexData.Mode.SQUARE_AMPLITUDE,
         ('math-square-amplitude', 'Square amplitude')),
        (ImageComplexData.Mode.PHASE, ('math-phase', 'Phase')),
        (ImageComplexData.Mode.REAL, ('math-real', 'Real part')),
        (ImageComplexData.Mode.IMAGINARY,
         ('math-imaginary', 'Imaginary part')),
        (ImageComplexData.Mode.AMPLITUDE_PHASE,
         ('math-phase-color', 'Amplitude and Phase')),
        (ImageComplexData.Mode.LOG10_AMPLITUDE_PHASE,
         ('math-phase-color-log', 'Log10(Amp.) and Phase'))
    ])

    _RANGE_DIALOG_TEXT = 'Set Amplitude Range...'

    def __init__(self, parent=None, plot=None):
        super(_ComplexDataToolButton, self).__init__(parent=parent)

        assert plot is not None
        self._plot2DComplex = plot

        menu = qt.QMenu(self)
        menu.triggered.connect(self._triggered)
        self.setMenu(menu)

        for mode, info in self._MODES.items():
            icon, text = info
            action = qt.QAction(icons.getQIcon(icon), text, self)
            action.setData(mode)
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
        icon, text = self._MODES[mode]
        self.setIcon(icons.getQIcon(icon))
        self.setToolTip('Display the ' + text.lower())
        self._rangeDialogAction.setEnabled(mode == ImageComplexData.Mode.LOG10_AMPLITUDE_PHASE)

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
            mode = action.data()
            if isinstance(mode, ImageComplexData.Mode):
                self._plot2DComplex.setVisualizationMode(mode)

    def _rangeChanged(self, range_):
        """Handle updates of range in the dialog"""
        self._plot2DComplex._setAmplitudeRangeInfo(*range_)


class ComplexImageView(qt.QWidget):
    """Display an image of complex data and allow to choose the visualization.

    :param parent: See :class:`QMainWindow`
    """

    Mode = ImageComplexData.Mode
    """Also expose the modes inside the class"""

    sigDataChanged = qt.Signal()
    """Signal emitted when data has changed."""

    sigVisualizationModeChanged = qt.Signal(object)
    """Signal emitted when the visualization mode has changed.

    It provides the new visualization mode.
    """

    def __init__(self, parent=None):
        super(ComplexImageView, self).__init__(parent)
        if parent is None:
            self.setWindowTitle('ComplexImageView')

        self._plot2D = Plot2D(self)

        layout = qt.QHBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plot2D)
        self.setLayout(layout)

        # Create and add image to the plot
        self._plotImage = ImageComplexData()
        self._plotImage._setLegend('__ComplexImageView__complex_image__')
        self._plotImage.sigItemChanged.connect(self._itemChanged)
        self._plot2D._add(self._plotImage)
        self._plot2D.setActiveImage(self._plotImage.getLegend())

        toolBar = qt.QToolBar('Complex', self)
        toolBar.addWidget(
            _ComplexDataToolButton(parent=self, plot=self))

        self._plot2D.insertToolBar(self._plot2D.getProfileToolbar(), toolBar)

    def _itemChanged(self, event):
        """Handle item changed signal"""
        if event is items.ItemChangedType.DATA:
            self.sigDataChanged.emit()
        elif event is items.ItemChangedType.VISUALIZATION_MODE:
            mode = self.getVisualizationMode()
            self.sigVisualizationModeChanged.emit(mode)

    def getPlot(self):
        """Return the PlotWidget displaying the data"""
        return self._plot2D

    def setData(self, data=None, copy=True):
        """Set the complex data to display.

        :param numpy.ndarray data: 2D complex data
        :param bool copy: True (default) to copy the data,
                          False to use provided data (do not modify!).
        """
        if data is None:
            data = numpy.zeros((0, 0), dtype=numpy.complex)

        previousData = self._plotImage.getComplexData(copy=False)

        self._plotImage.setData(data, copy=copy)

        if previousData.shape != data.shape:
            self.getPlot().resetZoom()

    def getData(self, copy=True):
        """Get the currently displayed complex data.

        :param bool copy: True (default) to return a copy of the data,
                          False to return internal data (do not modify!).
        :return: The complex data array.
        :rtype: numpy.ndarray of complex with 2 dimensions
        """
        return self._plotImage.getComplexData(copy=copy)

    def getDisplayedData(self, copy=True):
        """Returns the displayed data depending on the visualization mode

        WARNING: The returned data can be a uint8 RGBA image

        :param bool copy: True (default) to return a copy of the data,
                          False to return internal data (do not modify!)
        :rtype: numpy.ndarray of float with 2 dims or RGBA image (uint8).
        """
        mode = self.getVisualizationMode()
        if mode in (self.Mode.AMPLITUDE_PHASE,
                    self.Mode.LOG10_AMPLITUDE_PHASE):
            return self._plotImage.getRgbaImageData(copy=copy)
        else:
            return self._plotImage.getData(copy=copy)

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
        return tuple(ImageComplexData.Mode)

    def setVisualizationMode(self, mode):
        """Set the mode of visualization of the complex data.

        See :meth:`getSupportedVisualizationModes` for the list of
        supported modes.

        :param str mode: The mode to use.
        """
        self._plotImage.setVisualizationMode(mode)

    def getVisualizationMode(self):
        """Get the current visualization mode of the complex data.

        :rtype: Mode
        """
        return self._plotImage.getVisualizationMode()

    def _setAmplitudeRangeInfo(self, max_=None, delta=2):
        """Set the amplitude range to display for 'log10_amplitude_phase' mode.

        :param max_: Max of the amplitude range.
                     If None it autoscales to data max.
        :param float delta: Delta range in log10 to display
        """
        self._plotImage._setAmplitudeRangeInfo(max_, delta)

    def _getAmplitudeRangeInfo(self):
        """Returns the amplitude range to use for 'log10_amplitude_phase' mode.

        :return: (max, delta), if max is None, then it autoscales to data max
        :rtype: 2-tuple"""
        return self._plotImage._getAmplitudeRangeInfo()

    # Image item proxy

    def setColormap(self, colormap, mode=None):
        """Set the colormap to use for amplitude, phase, real or imaginary.

        WARNING: This colormap is not used when displaying both
        amplitude and phase.

        :param ~silx.gui.colors.Colormap colormap: The colormap
        :param Mode mode: If specified, set the colormap of this specific mode
        """
        self._plotImage.setColormap(colormap, mode)

    def getColormap(self, mode=None):
        """Returns the colormap used to display the data.

        :param Mode mode: If specified, set the colormap of this specific mode
        :rtype: ~silx.gui.colors.Colormap
        """
        return self._plotImage.getColormap(mode=mode)

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
