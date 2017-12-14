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
>>> colormap = Colormap(name='red', normalization='log',
...                     vmin=1., vmax=2.)

>>> dialog.setColormap(colormap)
>>> colormap.setVRange(1., 100.)  # This scale the width of the plot area
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


class _BoundaryWidget(qt.QWidget):
    """Widget to edit a boundary of the colormap (vmin, vmax)"""
    sigValueChanged = qt.Signal(object)
    """Signal emitted when value is changed"""

    def __init__(self, parent=None, value=0.0):
        qt.QWidget.__init__(self, parent=None)
        self.setLayout(qt.QHBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self._numVal = FloatEdit(parent=self, value=value)
        self.layout().addWidget(self._numVal)
        self._autoCB = qt.QCheckBox('auto', parent=self)
        self.layout().addWidget(self._autoCB)
        self._autoCB.setChecked(False)

        self._autoCB.toggled.connect(self._numVal.setDisabled)
        self.sigValueChanged = self._autoCB.toggled
        self.textEdited = self._numVal.textEdited
        self.editingFinished = self._numVal.editingFinished

    def isAutoChecked(self):
        return self._autoCB.isChecked()

    def getValue(self):
        return None if self._autoCB.isChecked() else self._numVal.value()

    def getFiniteValue(self):
        return self._numVal.value()

    def setValue(self, value):
        self._autoCB.setChecked(value is None)
        if value is not None:
            self._numVal.setValue(value)


class ColormapDialog(qt.QDialog):
    """A QDialog widget to set the colormap.

    :param parent: See :class:`QDialog`
    :param str title: The QDialog title
    """

    def __init__(self, parent=None, title="Colormap Dialog"):
        qt.QDialog.__init__(self, parent)
        self.setWindowTitle(title)

        self._ignoreColormapChange = False
        """Used as a semaphore to avoid editing the colormap object when we are
        only attempt to display it.
        Used instead of n connect and disconnect of the sigChanged. The
        disconnection to sigChanged was also limiting when this colormapdialog
        is used in the colormapaction and associated to the activeImageChanged.
        (because the activeImageChanged is send when the colormap changed and
        the self.setcolormap is a callback)
        """

        self._histogramData = None
        self._minMaxWasEdited = False

        # Make the GUI
        vLayout = qt.QVBoxLayout(self)

        formWidget = qt.QWidget(parent=self)
        vLayout.addWidget(formWidget)
        formLayout = qt.QFormLayout(formWidget)
        formLayout.setContentsMargins(10, 10, 10, 10)
        formLayout.setSpacing(0)

        # Colormap row
        self._comboBoxColormap = qt.QComboBox(parent=formWidget)
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
        self._normButtonLinear.toggled[bool].connect(self._updateLinearNorm)

        normLayout = qt.QHBoxLayout()
        normLayout.setContentsMargins(0, 0, 0, 0)
        normLayout.setSpacing(10)
        normLayout.addWidget(self._normButtonLinear)
        normLayout.addWidget(self._normButtonLog)

        formLayout.addRow('Normalization:', normLayout)

        # Min row
        self._minValue = _BoundaryWidget(parent=self, value=1.0)
        self._minValue.textEdited.connect(self._minMaxTextEdited)
        self._minValue.editingFinished.connect(self._minEditingFinished)
        self._minValue.sigValueChanged.connect(self._updateMinMax)
        formLayout.addRow('\tMin:', self._minValue)

        # Max row
        self._maxValue = _BoundaryWidget(parent=self, value=10.0)
        self._maxValue.textEdited.connect(self._minMaxTextEdited)
        self._maxValue.sigValueChanged.connect(self._updateMinMax)
        self._maxValue.editingFinished.connect(self._maxEditingFinished)
        formLayout.addRow('\tMax:', self._maxValue)

        # Add plot for histogram
        self._plotInit()
        vLayout.addWidget(self._plot)

        # define modal buttons
        types = qt.QDialogButtonBox.Ok | qt.QDialogButtonBox.Cancel
        self._buttonsModal = qt.QDialogButtonBox(parent=self)
        self._buttonsModal.setStandardButtons(types)
        self.layout().addWidget(self._buttonsModal)
        self._buttonsModal.accepted.connect(self.accept)
        self._buttonsModal.rejected.connect(self.reject)

        # define non modal buttons
        types = qt.QDialogButtonBox.Close | qt.QDialogButtonBox.Reset
        self._buttonsNonModal = qt.QDialogButtonBox(parent=self)
        self._buttonsNonModal.setStandardButtons(types)
        self.layout().addWidget(self._buttonsNonModal)
        self._buttonsNonModal.button(qt.QDialogButtonBox.Close).clicked.connect(self.accept)
        self._buttonsNonModal.button(qt.QDialogButtonBox.Reset).clicked.connect(self.resetColormap)

        # colormap window can not be resized
        self.setFixedSize(vLayout.minimumSize())

        # Set the colormap to default values
        self.setColormap(Colormap(name='gray', normalization='linear',
                         vmin=None, vmax=None))

        self.setModal(self.isModal())

    def close(self):
        self.accept()
        qt.QDialog.close(self)

    def setModal(self, modal):
        assert type(modal) is bool
        self._buttonsNonModal.setVisible(not modal)
        self._buttonsModal.setVisible(modal)
        qt.QDialog.setModal(self, modal)

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
        if not (hasattr(self, '_colormap') and self._colormap()):
            if self._plot.isVisibleTo(self):
                self._plot.setVisible(False)
                self.setFixedSize(self.layout().minimumSize())
            return

        if not self._plot.isVisibleTo(self):
            self._plot.setVisible(True)
            self.setFixedSize(self.layout().minimumSize())

        dataMin, dataMax = self._colormap().getColormapRange()
        marge = (abs(dataMax) + abs(dataMin)) / 6.0
        minmd = dataMin - marge
        maxpd = dataMax + marge

        start, end = self._minValue.getFiniteValue(), self._maxValue.getFiniteValue()

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

        if updateMarkers:
            self._plot.addXMarker(
                self._minValue.getFiniteValue(),
                legend='Min',
                text='Min',
                draggable=self._minValue.getValue() is not None,
                color='blue',
                constraint=self._plotMinMarkerConstraint)

            self._plot.addXMarker(
                self._maxValue.getFiniteValue(),
                legend='Max',
                text='Max',
                draggable=self._maxValue.getValue() is not None,
                color='blue',
                constraint=self._plotMaxMarkerConstraint)

        self._plot.resetZoom()

    def _plotMinMarkerConstraint(self, x, y):
        """Constraint of the min marker"""
        return min(x, self._maxValue.getFiniteValue()), y

    def _plotMaxMarkerConstraint(self, x, y):
        """Constraint of the max marker"""
        return max(x, self._minValue.getFiniteValue()), y

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
            if hasattr(self, '_colormap') and self._colormap():
                self._ignoreColormapChange = True
                self._colormap().setVRange(bin_edges[0], bin_edges[-1])
                self._ignoreColormapChange = False

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
            self._ignoreColormapChange = True
            self._colormap()._setFromDict(self._colormapStoredState)
            self._ignoreColormapChange = False
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

        :param :class:`Colormap` colormap: the colormap to edit
        """
        assert isinstance(colormap, Colormap)
        if self._ignoreColormapChange is True:
            return

        if hasattr(self, '_colormap') and self._colormap():
            self._colormap().sigChanged.disconnect(self._applyColormap)
        self._colormap = weakref.ref(colormap)
        self._colormap().sigChanged.connect(self._applyColormap)
        self.storeCurrentState()
        self._applyColormap()

    def _applyColormap(self):
        if self._ignoreColormapChange is True:
            return
        if self._colormap():
            self._ignoreColormapChange = True

            if self._colormap().getName() is not None:
                index = self._comboBoxColormap.findData(
                    self._colormap.getName(), qt.Qt.UserRole)
                self._comboBoxColormap.setCurrentIndex(index)

            assert self._colormap().getNormalization() in Colormap.NORMALIZATIONS
            self._normButtonLinear.setChecked(
                self._colormap().getNormalization() == Colormap.LINEAR)
            self._normButtonLog.setChecked(
                self._colormap().getNormalization() == Colormap.LOGARITHM)

            self._minValue.setValue(self._colormap().getVMin())
            self._maxValue.setValue(self._colormap().getVMax())

            self._ignoreColormapChange = False
            self._plotUpdate()

    def _updateMinMax(self):
        if self._ignoreColormapChange is True:
            return

        vmin = self._minValue.getValue()
        vmax = self._maxValue.getValue()
        if self._colormap():
            self._colormap().setVRange(vmin, vmax)
        self._ignoreColormapChange = False
        self._plotUpdate()

    def _updateName(self):
        if self._ignoreColormapChange is True:
            return

        if self._colormap():
            self._ignoreColormapChange = True
            self._colormap().setName(
                str(self._comboBoxColormap.currentText()).lower())
            self._ignoreColormapChange = False

    def _updateLinearNorm(self, isNormLinear):
        if self._ignoreColormapChange is True:
            return

        if self._colormap():
            self._ignoreColormapChange = True
            norm = Colormap.LINEAR if isNormLinear else Colormap.LOGARITHM
            self._colormap().setNormalization(norm)
            self._ignoreColormapChange = False

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
            if self._minValue.getValue() > self._maxValue.getValue():
                self._minValue.setValue(self._maxValue.getValue())
            self._updateMinMax()

    def _maxEditingFinished(self):
        """Handle _maxValue editingFinished signal

        Together with :meth:`_minMaxTextEdited`, this avoids to notify
        colormap change when the min and max value where not edited.
        """
        if self._minMaxWasEdited:
            self._minMaxWasEdited = False

            # Fix end value
            if self._minValue.getValue() > self._maxValue.getValue():
                self._maxValue.setValue(self._minValue.getValue())
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
        if self._ignoreColormapChange is True:
            return
        if self._colormap():
            self._ignoreColormapChange = True
            norm = Colormap.LOGARITHM if isLog is True else Colormap.LINEAR
            self._colormap().setNormalization(norm)
            self._ignoreColormapChange = False
