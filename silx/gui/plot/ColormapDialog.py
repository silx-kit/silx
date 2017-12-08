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
"""A QDialog widget to set-up the colormap.

It uses a description of colormaps as dict compatible with :class:`Plot`.

To run the following sample code, a QApplication must be initialized.

Create the colormap dialog and set the colormap description and data range:

>>> from silx.gui.plot.ColormapDialog import ColormapDialog
>>> from silx.gui.plot.Colormap import Colormap

>>> dialog = ColormapDialog()

>>> dialog.setColormap(Colormap(name='red', normalization='log',
...                             vmin=1., vmax=2.))
>>> dialog.setDataRange(1., 100.)  # This scale the width of the plot area
>>> dialog.show()

Get the colormap description (compatible with :class:`Plot`) from the dialog:

>>> cmap = dialog.getColormap()
>>> cmap.getName()
'red'

It is also possible to display an histogram of the image in the dialog.
This updates the data range with the range of the bins.

>>> import numpy
>>> image = numpy.random.normal(size=512 * 512).reshape(512, -1)
>>> hist, bin_edges = numpy.histogram(image, bins=10)
>>> dialog.setHistogram(hist, bin_edges)

The updates of the colormap description are also available through the signal:
:attr:`ColormapDialog.sigColormapChanged`.
"""  # noqa

from __future__ import division

__authors__ = ["V.A. Sole", "T. Vincent"]
__license__ = "MIT"
__date__ = "02/10/2017"


import logging

import numpy

from .. import qt
from .Colormap import Colormap, preferredColormaps
from . import PlotWidget
from silx.gui.widgets.FloatEdit import FloatEdit
import weakref

_logger = logging.getLogger(__name__)


class ColormapDialog(qt.QDialog):
    """A QDialog widget to set the colormap.

    :param parent: See :class:`QDialog`
    :param str title: The QDialog title
    """

    def __init__(self, parent=None, title="Colormap Dialog"):
        qt.QDialog.__init__(self, parent)
        self.setWindowTitle(title)

        self._histogramData = None
        self._dataRange = None
        self._minMaxWasEdited = False

        # Make the GUI
        vLayout = qt.QVBoxLayout(self)

        formWidget = qt.QWidget()
        vLayout.addWidget(formWidget)
        formLayout = qt.QFormLayout(formWidget)
        formLayout.setContentsMargins(10, 10, 10, 10)
        formLayout.setSpacing(0)

        # Colormap row
        self._comboBoxColormap = qt.QComboBox()
        for cmap in preferredColormaps():
            self._comboBoxColormap.addItem(cmap.title(), cmap)
        self._comboBoxColormap.currentIndexChanged[int].connect(self._updateName)
        formLayout.addRow('Colormap:', self._comboBoxColormap)

        # Normalization row
        self._normButtonLinear = qt.QRadioButton('Linear')
        self._normButtonLinear.setChecked(True)
        self._normButtonLog = qt.QRadioButton('Log')
        self._normButtonLog.toggled.connect(self._activeLogNorm)

        normButtonGroup = qt.QButtonGroup(self)
        normButtonGroup.setExclusive(True)
        normButtonGroup.addButton(self._normButtonLinear)
        normButtonGroup.addButton(self._normButtonLog)
        normButtonGroup.buttonClicked[int].connect(self._updateLinearNorm)

        normLayout = qt.QHBoxLayout()
        normLayout.setContentsMargins(0, 0, 0, 0)
        normLayout.setSpacing(10)
        normLayout.addWidget(self._normButtonLinear)
        normLayout.addWidget(self._normButtonLog)

        formLayout.addRow('Normalization:', normLayout)

        # Range row
        self._rangeAutoscaleButton = qt.QCheckBox('Autoscale')
        self._rangeAutoscaleButton.setChecked(True)
        self._rangeAutoscaleButton.toggled.connect(self._autoscaleToggled)
        formLayout.addRow('Range:', self._rangeAutoscaleButton)

        # Min row
        self._minValue = FloatEdit(parent=self, value=1.)
        self._minValue.setEnabled(False)
        self._minValue.textEdited.connect(self._minMaxTextEdited)
        self._minValue.editingFinished.connect(self._minEditingFinished)
        formLayout.addRow('\tMin:', self._minValue)

        # Max row
        self._maxValue = FloatEdit(parent=self, value=10.)
        self._maxValue.setEnabled(False)
        self._maxValue.textEdited.connect(self._minMaxTextEdited)
        self._maxValue.editingFinished.connect(self._maxEditingFinished)
        formLayout.addRow('\tMax:', self._maxValue)

        # Add plot for histogram
        self._plotInit()
        vLayout.addWidget(self._plot)

        types = qt.QDialogButtonBox.Ok | qt.QDialogButtonBox.Cancel
        _buttons = qt.QDialogButtonBox(parent=self)
        _buttons.setStandardButtons(types)
        self.layout().addWidget(_buttons)
        _buttons.accepted.connect(self.accept)
        _buttons.rejected.connect(self.reject)

        # colormap window can not be resized
        self.setFixedSize(vLayout.minimumSize())

        # Set the colormap to default values
        self.setColormap(Colormap(name='gray', normalization='linear',
                         vmin=None, vmax=None))

    def _plotInit(self):
        """Init the plot to display the range and the values"""
        self._plot = PlotWidget()
        self._plot.setDataMargins(yMinMargin=0.125, yMaxMargin=0.125)
        self._plot.getXAxis().setLabel("Data Values")
        self._plot.getYAxis().setLabel("")
        self._plot.setInteractiveMode('select', zoomOnWheel=False)
        self._plot.setActiveCurveHandling(False)
        self._plot.setMinimumSize(qt.QSize(250, 200))
        self._plot.sigPlotSignal.connect(self._plotSlot)
        self._plot.hide()

        self._plotUpdate()

    def _plotUpdate(self, updateMarkers=True):
        """Update the plot content

        :param bool updateMarkers: True to update markers, False otherwith
        """
        dataRange = self.getDataRange()

        if dataRange is None:
            if self._plot.isVisibleTo(self):
                self._plot.setVisible(False)
                self.setFixedSize(self.layout().minimumSize())
            return

        if not self._plot.isVisibleTo(self):
            self._plot.setVisible(True)
            self.setFixedSize(self.layout().minimumSize())

        dataMin, dataMax = dataRange
        marge = (abs(dataMax) + abs(dataMin)) / 6.0
        minmd = dataMin - marge
        maxpd = dataMax + marge

        start, end = self._minValue.value(), self._maxValue.value()

        if start <= end:
            x = [minmd, start, end, maxpd]
            y = [0, 0, 1, 1]

        else:
            x = [minmd, end, start, maxpd]
            y = [1, 1, 0, 0]

        self._plot.addCurve(x, y,
                            legend="ConstrainedCurve",
                            color='black',
                            symbol='o',
                            linestyle='-',
                            resetzoom=False)

        draggable = not self._rangeAutoscaleButton.isChecked()

        if updateMarkers:
            self._plot.addXMarker(
                self._minValue.value(),
                legend='Min',
                text='Min',
                draggable=draggable,
                color='blue',
                constraint=self._plotMinMarkerConstraint)

            self._plot.addXMarker(
                self._maxValue.value(),
                legend='Max',
                text='Max',
                draggable=draggable,
                color='blue',
                constraint=self._plotMaxMarkerConstraint)

        self._plot.resetZoom()

    def _plotMinMarkerConstraint(self, x, y):
        """Constraint of the min marker"""
        return min(x, self._maxValue.value()), y

    def _plotMaxMarkerConstraint(self, x, y):
        """Constraint of the max marker"""
        return max(x, self._minValue.value()), y

    def _plotSlot(self, event):
        """Handle events from the plot"""
        if event['event'] in ('markerMoving', 'markerMoved'):
            value = float(str(event['xdata']))
            if event['label'] == 'Min':
                self._minValue.setValue(value)
            elif event['label'] == 'Max':
                self._maxValue.setValue(value)

            # This will recreate the markers while interacting...
            # It might break if marker interaction is changed
            if event['event'] == 'markerMoved':
                self._updateMinMax()
            else:
                self._plotUpdate(updateMarkers=False)

    def getHistogram(self):
        """Returns the counts and bin edges of the displayed histogram.

        :return: (hist, bin_edges)
        :rtype: 2-tuple of numpy arrays"""
        if self._histogramData is None:
            return None
        else:
            bins, counts = self._histogramData
            return numpy.array(bins, copy=True), numpy.array(counts, copy=True)

    def setHistogram(self, hist=None, bin_edges=None):
        """Set the histogram to display.

        This update the data range with the bounds of the bins.
        See :meth:`setDataRange`.

        :param hist: array-like of counts or None to hide histogram
        :param bin_edges: array-like of bins edges or None to hide histogram
        """
        if hist is None or bin_edges is None:
            self._histogramData = None
            self._plot.remove(legend='Histogram', kind='curve')
            self.setDataRange()  # Remove data range

        else:
            hist = numpy.array(hist, copy=True)
            bin_edges = numpy.array(bin_edges, copy=True)
            self._histogramData = hist, bin_edges

            # For now, draw the histogram as a curve
            # using bin centers and normalised counts
            bins_center = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            norm_hist = hist / max(hist)
            self._plot.addCurve(bins_center, norm_hist,
                                legend="Histogram",
                                color='gray',
                                symbol='',
                                linestyle='-',
                                fill=True)

            # Update the data range
            self.setDataRange(bin_edges[0], bin_edges[-1])

    def getDataRange(self):
        """Returns the data range used for the histogram area.

        :return: (dataMin, dataMax) or None if no data range is set
        :rtype: 2-tuple of float
        """
        return self._dataRange

    def setDataRange(self, min_=None, max_=None):
        """Set the range of data to use for the range of the histogram area.

        :param float min_: The min of the data or None to disable range.
        :param float max_: The max of the data or None to disable range.
        """
        if min_ is None or max_ is None:
            self._dataRange = None
            self._plotUpdate()

        else:
            min_, max_ = float(min_), float(max_)
            assert min_ <= max_
            self._dataRange = min_, max_
            if self._colormap():
                self._colormap().sigChanged.disconnect(self._applyColormap)
                self._colormap().setVRange(min_, max_)
                self._colormap().sigChanged.connect(self._applyColormap)
            if self._rangeAutoscaleButton.isChecked():
                self._minValue.setValue(min_)
                self._maxValue.setValue(max_)
                self._updateMinMax()
            else:
                self._plotUpdate()

    def getColormap(self):
        """Return the colormap description as a :class:`.Colormap`.

        """
        return self._colormap()

    def resetColormap(self):
        """
        Reset the colormap state before modification.

        ..note :: the colormap reference state is the state when set or the
                  state when validated
        """
        if self._colormap():
            self._colormap().sigChanged.disconnect(self._applyColormap)
            self._colormap()._setFromDict(self._colormapStoredState)
            self._colormap().sigChanged.connect(self._applyColormap)
            self._plotUpdate()

    def accept(self):
        self.storeCurrentState()
        qt.QDialog.accept(self)

    def storeCurrentState(self):
        """
        save the current value sof the colormap if the user want to undo is
        modifications
        """
        if self._colormap():
            self._colormapStoredState = self._colormap()._toDict()

    def reject(self):
        self.resetColormap()
        qt.QDialog.reject(self)

    def setColormap(self, colormap):
        """Set the colormap description

        If some arguments are not provided, the current values are used.

        :param str name: The name of the colormap
        :param str normalization: 'linear' or 'log'
        :param bool autoscale: Toggle colormap range autoscale
        :param float vmin: The min value, ignored if autoscale is True
        :param float vmax: The max value, ignored if autoscale is True
        """
        assert isinstance(colormap, Colormap)
        if hasattr(self, '_colormap') and self._colormap():
            self._colormap().sigChanged.disconnect(self._applyColormap)
        self._colormap = weakref.ref(colormap)
        self._colormap().sigChanged.connect(self._applyColormap)
        self.storeCurrentState()
        self._applyColormap()

    def _applyColormap(self):
        if self._colormap():
            if self._colormap().getName() is not None:
                index = self._comboBoxColormap.findData(
                    self._colormap.getName(), qt.Qt.UserRole)
                self._comboBoxColormap.setCurrentIndex(index)

            if self._colormap().getNormalization() is not None:
                assert self._colormap().getNormalization() in Colormap.NORMALIZATIONS
                self._normButtonLinear.setChecked(
                    self._colormap().getNormalization() == Colormap.LINEAR)
                self._normButtonLog.setChecked(
                    self._colormap().getNormalization() == Colormap.LOGARITHM)

            if self._colormap().getVMin() is not None:
                self._minValue.setValue(self._colormap().getVMin())

            if self._colormap().getVMax() is not None:
                self._maxValue.setValue(self._colormap().getVMax())

            self._rangeAutoscaleButton.setChecked(self._colormap().isAutoscale())
            if self._colormap().isAutoscale():
                self._minValue.setEnabled(False)
                self._maxValue.setEnabled(False)
                dataRange = self.getDataRange()
                if dataRange is not None:
                    self._minValue.setValue(dataRange[0])
                    self._maxValue.setValue(dataRange[1])
            else:
                self._minValue.setEnabled(True)
                self._maxValue.setEnabled(True)
            # Do it once for all the changes
            self._plotUpdate()

    def _updateMinMax(self):
        if self._rangeAutoscaleButton.isChecked():
            vmin = None
            vmax = None
        else:
            vmin = self._minValue.value()
            vmax = self._maxValue.value()
        if self._colormap():
            self._colormap().sigChanged.disconnect(self._applyColormap)
            self._colormap().setVRange(vmin, vmax)
            self._colormap().sigChanged.connect(self._applyColormap)
        self._plotUpdate()

    def _updateName(self):
        self._colormap().sigChanged.disconnect(self._applyColormap)
        self._colormap().setName(
            str(self._comboBoxColormap.currentText()).lower())
        if self._colormap():
            self._colormap().sigChanged.connect(self._applyColormap)
        self._plotUpdate()

    def _updateLinearNorm(self, isNormLinear):
        if self._colormap():
            self._colormap().sigChanged.disconnect(self._applyColormap)
            norm = Colormap.LINEAR if isNormLinear else Colormap.LOGARITHM
            self._colormap().setNormalization(norm)
        if self._colormap():
            self._colormap().sigChanged.connect(self._applyColormap)
        self._plotUpdate()

    def _autoscaleToggled(self, checked):
        """Handle autoscale changes by enabling/disabling min/max fields"""
        self._minValue.setEnabled(not checked)
        self._maxValue.setEnabled(not checked)
        if self._colormap():
            if checked:
                self._colormap().sigChanged.disconnect(self._applyColormap)
                self._colormap().setVRange(None, None)
                self._colormap().sigChanged.connect(self._applyColormap)
            else:
                dataRange = self.getDataRange()
                if dataRange is not None:
                    self._colormap().sigChanged.disconnect(self._applyColormap)
                    self._colormap().setVRange(dataRange[0], dataRange[1])
                    self._colormap().sigChanged.connect(self._applyColormap)
        self._plotUpdate()

    def _minMaxTextEdited(self, text):
        """Handle _minValue and _maxValue textEdited signal"""
        self._minMaxWasEdited = True

    def _minEditingFinished(self):
        """Handle _minValue editingFinished signal

        Together with :meth:`_minMaxTextEdited`, this avoids to notify
        colormap change when the min and max value where not edited.
        """
        if self._minMaxWasEdited:
            self._minMaxWasEdited = False

            # Fix start value
            if self._minValue.value() > self._maxValue.value():
                self._minValue.setValue(self._maxValue.value())
            self._updateMinMax()

    def _maxEditingFinished(self):
        """Handle _maxValue editingFinished signal

        Together with :meth:`_minMaxTextEdited`, this avoids to notify
        colormap change when the min and max value where not edited.
        """
        if self._minMaxWasEdited:
            self._minMaxWasEdited = False

            # Fix end value
            if self._minValue.value() > self._maxValue.value():
                self._maxValue.setValue(self._minValue.value())
            self._updateMinMax()

    def keyPressEvent(self, event):
        """Override key handling.

        It disables leaving the dialog when editing a text field.
        """
        if event.key() == qt.Qt.Key_Enter and (self._minValue.hasFocus() or
                                               self._maxValue.hasFocus()):
            # Bypass QDialog keyPressEvent
            # To avoid leaving the dialog when pressing enter on a text field
            super(qt.QDialog, self).keyPressEvent(event)
        else:
            # Use QDialog keyPressEvent
            super(ColormapDialog, self).keyPressEvent(event)

    def _activeLogNorm(self, isLog):
        if self._colormap():
            norm = Colormap.LOGARITHM if isLog is True else Colormap.LINEAR
            self._colormap().setNormalization(norm)
