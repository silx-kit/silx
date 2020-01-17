# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2004-2019 European Synchrotron Radiation Facility
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

>>> from silx.gui.dialog.ColormapDialog import ColormapDialog
>>> from silx.gui.colors import Colormap

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

__authors__ = ["V.A. Sole", "T. Vincent", "H. Payno"]
__license__ = "MIT"
__date__ = "27/11/2018"


import enum
import logging

import numpy

from .. import qt
from .. import utils
from ..colors import Colormap
from ..plot import PlotWidget
from ..plot.items.axis import Axis
from silx.gui.widgets.FloatEdit import FloatEdit
import weakref
from silx.math.combo import min_max
from silx.gui import icons
from silx.gui.widgets.ColormapNameComboBox import ColormapNameComboBox
from silx.math.histogram import Histogramnd

_logger = logging.getLogger(__name__)


_colormapIconPreview = {}


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
        self._autoCB.setVisible(False)

        self._autoCB.toggled.connect(self._autoToggled)
        self.sigValueChanged = self._autoCB.toggled
        self.textEdited = self._numVal.textEdited
        self.editingFinished = self._numVal.editingFinished
        self._dataValue = None

    def isAutoChecked(self):
        return self._autoCB.isChecked()

    def getValue(self):
        return None if self._autoCB.isChecked() else self._numVal.value()

    def getFiniteValue(self):
        if not self._autoCB.isChecked():
            return self._numVal.value()
        elif self._dataValue is None:
            return self._numVal.value()
        else:
            return self._dataValue

    def _autoToggled(self, enabled):
        self._numVal.setEnabled(not enabled)
        self._updateDisplayedText()

    def _updateDisplayedText(self):
        # if dataValue is finite
        if self._autoCB.isChecked() and self._dataValue is not None:
            old = self._numVal.blockSignals(True)
            self._numVal.setValue(self._dataValue)
            self._numVal.blockSignals(old)

    def setDataValue(self, dataValue):
        self._dataValue = dataValue
        self._updateDisplayedText()

    def setFiniteValue(self, value):
        assert(value is not None)
        old = self._numVal.blockSignals(True)
        self._numVal.setValue(value)
        self._numVal.blockSignals(old)

    def setValue(self, value, isAuto=False):
        self._autoCB.setChecked(isAuto or value is None)
        if value is not None:
            self._numVal.setValue(value)
        self._updateDisplayedText()


class _AutoscaleModeComboBox(qt.QComboBox):

    DATA = {
        Colormap.MINMAX: ("Min/max", "Use the data min/max"),
        Colormap.STDDEV3: ("Mean ± 3×SD", "Use the data mean ± 3 × standard deviation"),
    }

    def __init__(self, parent: qt.QWidget):
        super(_AutoscaleModeComboBox, self).__init__(parent=parent)
        self.currentIndexChanged.connect(self.__updateTooltip)
        self._init()

    def _init(self):
        for mode in Colormap.AUTOSCALE_MODES:
            label, tooltip = self.DATA.get(mode, (mode, None))
            self.addItem(label, mode)
            if tooltip is not None:
                self.setItemData(self.count() - 1, tooltip, qt.Qt.ToolTipRole)

    def setCurrentIndex(self, index):
        self.__updateTooltip(index)
        super(_AutoscaleModeComboBox, self).setCurrentIndex(index)

    def __updateTooltip(self, index):
        if index > -1:
            tooltip = self.itemData(index, qt.Qt.ToolTipRole)
        else:
            tooltip = ""
        self.setToolTip(tooltip)

    def currentMode(self):
        index = self.currentIndex()
        return self.itemData(index)

    def setCurrentMode(self, mode):
        for index in range(self.count()):
            if mode == self.itemData(index):
                self.setCurrentIndex(index)
                return
        if mode is None:
            # If None was not a value
            self.setCurrentIndex(-1)
            return
        self.addItem(mode, mode)
        self.setCurrentIndex(self.count() - 1)


class _AutoScaleButtons(qt.QWidget):

    autoRangeChanged = qt.Signal(object)

    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent=parent)
        layout = qt.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.setFocusPolicy(qt.Qt.NoFocus)

        self._minAuto = qt.QPushButton(self)
        self._minAuto.setCheckable(True)
        self._minAuto.setToolTip("Enable/disable the auto range for min")
        self._minAuto.toggled[bool].connect(self.__minToggled)
        self._minAuto.setFocusPolicy(qt.Qt.TabFocus)

        self._maxAuto = qt.QPushButton(self)
        self._maxAuto.setCheckable(True)
        self._maxAuto.setToolTip("Enable/disable the auto range for max")
        self._maxAuto.toggled[bool].connect(self.__maxToggled)
        self._maxAuto.setFocusPolicy(qt.Qt.TabFocus)

        self._bothAuto = qt.QPushButton(self)
        self._bothAuto.setText("Auto")
        self._bothAuto.setToolTip("Enable/disable the auto range for both min and max")
        self._bothAuto.setCheckable(True)
        self._bothAuto.toggled[bool].connect(self.__bothToggled)
        self._bothAuto.setFocusPolicy(qt.Qt.TabFocus)

        layout.addSpacing(1)
        layout.addWidget(self._minAuto)
        layout.addWidget(self._bothAuto)
        layout.addWidget(self._maxAuto)
        layout.addSpacing(1)

    def __bothToggled(self, checked):
        autoRange = checked, checked
        self.setAutoRange(autoRange)
        self.autoRangeChanged.emit(autoRange)

    def __minToggled(self, checked):
        autoRange = self.getAutoRange()
        self.setAutoRange(autoRange)
        self.autoRangeChanged.emit(autoRange)

    def __maxToggled(self, checked):
        autoRange = self.getAutoRange()
        self.setAutoRange(autoRange)
        self.autoRangeChanged.emit(autoRange)

    def setAutoRangeFromColormap(self, colormap):
        vRange = colormap.getVRange()
        autoRange = vRange[0] is None, vRange[1] is None
        self.setAutoRange(autoRange)

    def setAutoRange(self, autoRange):
        if autoRange[0] == autoRange[1]:
            with utils.blockSignals(self._bothAuto):
                self._bothAuto.setChecked(autoRange[0])
        else:
            with utils.blockSignals(self._bothAuto):
                self._bothAuto.setChecked(False)
        with utils.blockSignals(self._minAuto):
            self._minAuto.setChecked(autoRange[0])
        with utils.blockSignals(self._maxAuto):
            self._maxAuto.setChecked(autoRange[1])

    def getAutoRange(self):
        return self._minAuto.isChecked(), self._maxAuto.isChecked()

@enum.unique
class _DataInPlotMode(enum.Enum):
    """Enum for each mode of display of the data in the plot."""
    RANGE = 'range'
    HISTOGRAM = 'histogram'


class ColormapDialog(qt.QDialog):
    """A QDialog widget to set the colormap.

    :param parent: See :class:`QDialog`
    :param str title: The QDialog title
    """

    visibleChanged = qt.Signal(bool)
    """This event is sent when the dialog visibility change"""

    def __init__(self, parent=None, title="Colormap Dialog"):
        qt.QDialog.__init__(self, parent)
        self.setWindowTitle(title)

        self._colormap = None
        self._data = None
        self._item = None
        self._dataInPlotMode = _DataInPlotMode.RANGE

        self._ignoreColormapChange = False
        """Used as a semaphore to avoid editing the colormap object when we are
        only attempt to display it.
        Used instead of n connect and disconnect of the sigChanged. The
        disconnection to sigChanged was also limiting when this colormapdialog
        is used in the colormapaction and associated to the activeImageChanged.
        (because the activeImageChanged is send when the colormap changed and
        the self.setcolormap is a callback)
        """

        self.__displayInvalidated = False
        self._histogramData = None
        self._minMaxWasEdited = False
        self._initialRange = None

        self._dataRange = None
        """If defined 3-tuple containing information from a data:
        minimum, positive minimum, maximum"""

        self._colormapStoredState = None

        # Colormap row
        self._comboBoxColormap = ColormapNameComboBox(parent=self)
        self._comboBoxColormap.currentIndexChanged[int].connect(self._updateLut)

        # Normalization row
        self._normButtonLinear = qt.QRadioButton('Linear')
        self._normButtonLinear.setChecked(True)
        self._normButtonLog = qt.QRadioButton('Log')

        normButtonGroup = qt.QButtonGroup(self)
        normButtonGroup.setExclusive(True)
        normButtonGroup.addButton(self._normButtonLinear)
        normButtonGroup.addButton(self._normButtonLog)
        normButtonGroup.buttonClicked[qt.QAbstractButton].connect(self._updateNormalization)

        normLayout = qt.QHBoxLayout()
        normLayout.setContentsMargins(0, 0, 0, 0)
        normLayout.setSpacing(10)
        normLayout.addWidget(self._normButtonLinear)
        normLayout.addWidget(self._normButtonLog)

        autoScaleCombo = _AutoscaleModeComboBox(self)
        autoScaleCombo.currentIndexChanged.connect(self._updateAutoScaleMode)
        self._autoScaleCombo = autoScaleCombo

        # Min row

        self._minValue = _BoundaryWidget(parent=self, value=1.0)
        self._minValue.textEdited.connect(self._minMaxTextEdited)
        self._minValue.editingFinished.connect(self._minEditingFinished)
        self._minValue.sigValueChanged.connect(self._updateMinMax)

        # Max row
        self._maxValue = _BoundaryWidget(parent=self, value=10.0)
        self._maxValue.textEdited.connect(self._minMaxTextEdited)
        self._maxValue.sigValueChanged.connect(self._updateMinMax)
        self._maxValue.editingFinished.connect(self._maxEditingFinished)

        self._autoButtons = _AutoScaleButtons(self)
        self._autoButtons.autoRangeChanged.connect(self._updateAutoRange)

        rangeLayout = qt.QGridLayout()
        miniFont = qt.QFont(self.font())
        miniFont.setPixelSize(8)
        labelMin = qt.QLabel("Min", self)
        labelMin.setFont(miniFont)
        labelMin.setAlignment(qt.Qt.AlignHCenter)
        labelMax = qt.QLabel("Max", self)
        labelMax.setAlignment(qt.Qt.AlignHCenter)
        labelMax.setFont(miniFont)
        rangeLayout.addWidget(labelMin, 0, 0)
        rangeLayout.addWidget(labelMax, 0, 1)
        rangeLayout.addWidget(self._minValue, 1, 0)
        rangeLayout.addWidget(self._maxValue, 1, 1)
        rangeLayout.addWidget(self._autoButtons, 2, 0, 1, -1, qt.Qt.AlignCenter)

        # Add plot for histogram
        self._plotToolbar = qt.QToolBar(self)
        self._plotToolbar.setFloatable(False)
        self._plotToolbar.setMovable(False)
        self._plotToolbar.setIconSize(qt.QSize(8, 8))
        self._plotToolbar.setStyleSheet("QToolBar { border: 0px }")
        self._plotToolbar.setOrientation(qt.Qt.Vertical)

        group = qt.QActionGroup(self._plotToolbar)
        group.setExclusive(True)

        action = qt.QAction("Data range", self)
        action.setToolTip("Display the data range within the colormap range. A fast data processing have to be done.")
        action.setIcon(icons.getQIcon('colormap-range'))
        action.setCheckable(True)
        action.setData(_DataInPlotMode.RANGE)
        action.setChecked(action.data() == self._dataInPlotMode)
        self._plotToolbar.addAction(action)
        group.addAction(action)
        action = qt.QAction("Histogram", self)
        action.setToolTip("Display the data histogram within the colormap range. A slow data processing have to be done. ")
        action.setIcon(icons.getQIcon('colormap-histogram'))
        action.setCheckable(True)
        action.setData(_DataInPlotMode.HISTOGRAM)
        action.setChecked(action.data() == self._dataInPlotMode)
        self._plotToolbar.addAction(action)
        group.addAction(action)
        group.triggered.connect(self._displayDataInPlotModeChanged)

        self._plotBox = qt.QWidget(self)
        self._plotInit()

        plotBoxLayout = qt.QHBoxLayout()
        plotBoxLayout.setContentsMargins(0, 0, 0, 0)
        plotBoxLayout.setSpacing(2)
        plotBoxLayout.addWidget(self._plotToolbar)
        plotBoxLayout.addWidget(self._plot)
        plotBoxLayout.setSizeConstraint(qt.QLayout.SetMinimumSize)
        self._plotBox.setLayout(plotBoxLayout)

        # define modal buttons
        types = qt.QDialogButtonBox.Ok | qt.QDialogButtonBox.Cancel
        self._buttonsModal = qt.QDialogButtonBox(parent=self)
        self._buttonsModal.setStandardButtons(types)
        self._buttonsModal.accepted.connect(self.accept)
        self._buttonsModal.rejected.connect(self.reject)

        # define non modal buttons
        types = qt.QDialogButtonBox.Close | qt.QDialogButtonBox.Reset
        self._buttonsNonModal = qt.QDialogButtonBox(parent=self)
        self._buttonsNonModal.setStandardButtons(types)
        button = self._buttonsNonModal.button(qt.QDialogButtonBox.Close)
        button.clicked.connect(self.accept)
        button.setDefault(True)
        button = self._buttonsNonModal.button(qt.QDialogButtonBox.Reset)
        button.clicked.connect(self.resetColormap)

        self._buttonsModal.setFocus(qt.Qt.OtherFocusReason)
        self._buttonsNonModal.setFocus(qt.Qt.OtherFocusReason)

        # Set the colormap to default values
        self.setColormap(Colormap(name='gray', normalization='linear',
                         vmin=None, vmax=None))

        self.setModal(self.isModal())

        formLayout = qt.QFormLayout(self)
        formLayout.setContentsMargins(10, 10, 10, 10)
        formLayout.addRow('Colormap:', self._comboBoxColormap)
        formLayout.addRow('Normalization:', normLayout)
        formLayout.addRow(self._plotBox)
        formLayout.addRow(rangeLayout)
        label = qt.QLabel('Mode:', self)
        label.setToolTip("Mode for autoscale. Algorithm used to find range in auto scale.")
        formLayout.addRow(label, autoScaleCombo)
        formLayout.addRow(self._buttonsModal)
        formLayout.addRow(self._buttonsNonModal)
        formLayout.setSizeConstraint(qt.QLayout.SetMinimumSize)

        self.setFixedSize(self.sizeHint())
        self._applyColormap()

        self._plotUpdate()

    def _displayLater(self):
        self.__displayInvalidated = True

    def showEvent(self, event):
        self.visibleChanged.emit(True)
        super(ColormapDialog, self).showEvent(event)
        if self.isVisible():
            if self.__displayInvalidated:
                self._applyColormap()
                self._updateDataInPlot()
                self.__displayInvalidated = False

    def closeEvent(self, event):
        if not self.isModal():
            self.accept()
        super(ColormapDialog, self).closeEvent(event)

    def hideEvent(self, event):
        self.visibleChanged.emit(False)
        super(ColormapDialog, self).hideEvent(event)

    def close(self):
        self.accept()
        qt.QDialog.close(self)

    def setModal(self, modal):
        assert type(modal) is bool
        self._buttonsNonModal.setVisible(not modal)
        self._buttonsModal.setVisible(modal)
        qt.QDialog.setModal(self, modal)

    def exec_(self):
        wasModal = self.isModal()
        self.setModal(True)
        result = super(ColormapDialog, self).exec_()
        self.setModal(wasModal)
        return result

    def _plotInit(self):
        """Init the plot to display the range and the values"""
        self._plot = PlotWidget()
        self._plot.setDataMargins(0.125, 0.125, 0.125, 0.125)
        self._plot.getXAxis().setLabel("Data Values")
        self._plot.getYAxis().setLabel("")
        self._plot.setInteractiveMode('select', zoomOnWheel=False)
        self._plot.setActiveCurveHandling(False)
        self._plot.setMinimumSize(qt.QSize(250, 200))
        self._plot.sigPlotSignal.connect(self._plotSlot)
        palette = self.palette()
        color = palette.color(qt.QPalette.Normal, qt.QPalette.Window)
        self._plot.setBackgroundColor(color)
        self._plot.setDataBackgroundColor("white")

        lut = numpy.arange(256)
        lut.shape = 1, -1
        self._plot.addImage(lut, legend='lut')
        self._lutItem = self._plot._getItem("image", "lut")
        self._lutItem.setVisible(False)

    def sizeHint(self):
        return self.layout().minimumSize()

    def _computeView(self, dataMin, dataMax):
        """Compute the location of the view according to the bound of the data

        :rtype: Tuple(float, float)
        """
        marginRatio = 1.0 / 6.0
        scale = self._plot.getXAxis().getScale()

        if self._dataRange is not None:
            if scale == Axis.LOGARITHMIC:
                minRange = self._dataRange[1]
            else:
                minRange = self._dataRange[0]
            maxRange = self._dataRange[2]
            if minRange is not None:
                dataMin = min(dataMin, minRange)
                dataMax = max(dataMax, maxRange)

        if self._histogramData is not None:
            info = min_max(self._histogramData[1])
            if scale == Axis.LOGARITHMIC:
                minHisto = info.min_positive
            else:
                minHisto = info.minimum
            maxHisto = info.maximum
            if minHisto is not None:
                dataMin = min(dataMin, minHisto)
                dataMax = max(dataMax, maxHisto)

        if scale == Axis.LOGARITHMIC:
            epsilon = numpy.finfo(numpy.float32).eps
            if dataMin == 0:
                dataMin = epsilon
            if dataMax < dataMin:
                dataMax = dataMin + epsilon
            marge = marginRatio * abs(numpy.log10(dataMax) - numpy.log10(dataMin))
            viewMin = 10**(numpy.log10(dataMin) - marge)
            viewMax = 10**(numpy.log10(dataMax) + marge)
        else:  # scale == Axis.LINEAR:
            marge = marginRatio * abs(dataMax - dataMin)
            if marge < 0.0001:
                # Smaller that the QLineEdit precision
                marge = 0.0001
            viewMin = dataMin - marge
            viewMax = dataMax + marge

        return viewMin, viewMax

    def _getDisplayableRange(self):
        """Returns the selected min/max range to apply to the data,
        according to the used scale.

        One or both limits can be None in case it is not displayable in the
        current axes scale.

        :returns: Tuple{float, float}
        """
        scale = self._plot.getXAxis().getScale()
        def isDisplayable(pos):
            if scale == Axis.LOGARITHMIC:
                return pos > 0.0
            return True

        posMin = self._minValue.getFiniteValue()
        posMax = self._maxValue.getFiniteValue()
        if not isDisplayable(posMin):
            posMin = None
        if not isDisplayable(posMax):
            posMax = None

        return posMin, posMax

    def _plotUpdate(self, updateMarkers=True):
        """Update the plot content

        :param bool updateMarkers: True to update markers, False otherwith
        """
        colormap = self.getColormap()
        if colormap is None:
            if self._plotBox.isVisibleTo(self):
                self._plotBox.setVisible(False)
                self.setFixedSize(self.sizeHint())
            return

        if not self._plotBox.isVisibleTo(self):
            self._plotBox.setVisible(True)
            self.setFixedSize(self.sizeHint())

        if updateMarkers:
            posMin, posMax = self._getDisplayableRange()
            if posMin is not None:
                minDraggable = (self._colormap().isEditable() and
                                not self._minValue.isAutoChecked())
                color = "blue" if minDraggable else "black"
                self._plot.addXMarker(
                    posMin,
                    legend='Min',
                    text='Min',
                    draggable=minDraggable,
                    color=color,
                    constraint=self._plotMinMarkerConstraint)
            if posMax is not  None:
                maxDraggable = (self._colormap().isEditable() and
                                not self._maxValue.isAutoChecked())
                color = "blue" if maxDraggable else "black"
                self._plot.addXMarker(
                    posMax,
                    legend='Max',
                    text='Max',
                    draggable=maxDraggable,
                    color=color,
                    constraint=self._plotMaxMarkerConstraint)

            self._updateLutItem((posMin, posMax))

        self._plot.resetZoom()

    def _updateLutItem(self, vRange):
        colormap = self._colormap()
        if colormap is None:
            return

        if vRange is None:
            posMin, posMax = self._getDisplayableRange()
        else:
            posMin, posMax = vRange
        if posMin is None or posMax is None:
            self._lutItem.setVisible(False)
        else:
            colormap = colormap.copy()
            colormap.setVRange(0, 255)
            scale = (posMax - posMin) / 256
            self._lutItem.setColormap(colormap)
            self._lutItem.setOrigin((posMin, -0.1))
            self._lutItem.setScale((scale, 0.1))
            self._lutItem.setVisible(True)

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
                self._updateLutItem(None)
            elif event['label'] == 'Max':
                self._maxValue.setValue(value)
                self._updateLutItem(None)

            # This will recreate the markers while interacting...
            # It might break if marker interaction is changed
            if event['event'] == 'markerMoved':
                self._initialRange = None
                self._updateMinMax()
                self._updateLutItem(None)
            else:
                self._plotUpdate(updateMarkers=False)

    @staticmethod
    def computeDataRange(data):
        """Compute the data range as used by :meth:`setDataRange`.

        :param data: The data to process
        :rtype: List[Union[None,float]]
        """
        if data is None or len(data) == 0:
            return None, None, None

        dataRange = min_max(data, min_positive=True, finite=True)
        if dataRange.minimum is None:
            # Only non-finite data
            dataRange = None

        if dataRange is not None:
            dataRange = dataRange.minimum, dataRange.min_positive, dataRange.maximum

        if dataRange is None or len(dataRange) != 3:
            qt.QMessageBox.warning(
                None, "No Data",
                "Image data does not contain any real value")
            dataRange = 1., 1., 10.

        return dataRange

    @staticmethod
    def computeHistogram(data, scale=Axis.LINEAR):
        """Compute the data histogram as used by :meth:`setHistogram`.

        :param data: The data to process
        :rtype: Tuple(List(float),List(float)
        """
        _data = data
        if _data.ndim == 3:  # RGB(A) images
            _logger.info('Converting current image from RGB(A) to grayscale\
                in order to compute the intensity distribution')
            _data = (_data[:, :, 0] * 0.299 +
                     _data[:, :, 1] * 0.587 +
                     _data[:, :, 2] * 0.114)

        if len(_data) == 0:
            return None, None

        if scale == Axis.LOGARITHMIC:
            _data = numpy.log10(_data)
        xmin, xmax = min_max(_data, min_positive=False, finite=True)
        if xmin is None:
            return None, None

        nbins = min(256, int(numpy.sqrt(_data.size)))
        data_range = xmin, xmax

        # bad hack: get 256 bins in the case we have a B&W
        if numpy.issubdtype(_data.dtype, numpy.integer):
            if nbins > xmax - xmin:
                nbins = xmax - xmin

        nbins = max(2, nbins)
        _data = _data.ravel().astype(numpy.float32)

        histogram = Histogramnd(_data, n_bins=nbins, histo_range=data_range)
        bins = histogram.edges[0]
        if scale == Axis.LOGARITHMIC:
            bins = 10**bins
        return histogram.histo, bins

    def _getItem(self):
        if self._item is None:
            return None
        return self._item()

    def setItem(self, item):
        """Store the plot item.

        According to the state of the dialog, the item will be used to display
        the data range or the histogram of the data using :meth:`setDataRange`
        and :meth:`setHistogram`
        """
        old = self._getItem()
        if old is item:
            return

        self._data = None
        if item is None:
            self._item = None
        else:
            self._item = weakref.ref(item, self._itemAboutToFinalize)

        if self.isVisible():
            self._updateDataInPlot()
        else:
            self._displayLater()

    def _getData(self):
        if self._data is None:
            return None
        return self._data()

    def setData(self, data):
        """Store the data

        According to the state of the dialog, the data will be used to display
        the data range or the histogram of the data using :meth:`setDataRange`
        and :meth:`setHistogram`
        """
        oldData = self._getData()
        if oldData is data:
            return

        self._item = None
        if data is None:
            self._data = None
        else:
            self._data = weakref.ref(data, self._dataAboutToFinalize)

        if self.isVisible():
            self._updateDataInPlot()
        else:
            self._displayLater()

    def _getArray(self):
        item = self._getItem()
        if item is not None:
            return item.getColormappedData()
        return self._getData()

    def _setDataInPlotMode(self, mode):
        if self._dataInPlotMode == mode:
            return
        self._dataInPlotMode = mode
        self._updateDataInPlot()

    def _displayDataInPlotModeChanged(self, action):
        mode = action.data()
        self._setDataInPlotMode(mode)

    def _updateDataInPlot(self):
        data = self._getArray()
        if data is None:
            self.setDataRange()
            self.setHistogram()
            return

        if data.size == 0:
            # One or more dimensions are equal to 0
            self.setHistogram()
            self.setDataRange()
            return

        mode = self._dataInPlotMode

        if mode == _DataInPlotMode.RANGE:
            result = self.computeDataRange(data)
            self.setHistogram()
            self.setDataRange(*result)
        elif mode == _DataInPlotMode.HISTOGRAM:
            # The histogram should be done in a worker thread
            result = self.computeHistogram(data, scale=self._plot.getXAxis().getScale())
            self.setHistogram(*result)
            self.setDataRange()

    def _invalidateHistogram(self):
        """Recompute the histogram if it is displayed"""
        if self._dataInPlotMode == _DataInPlotMode.HISTOGRAM:
            self._updateDataInPlot()

    def _colormapAboutToFinalize(self, weakrefColormap):
        """Callback when the data weakref is about to be finalized."""
        if self._colormap is weakrefColormap:
            self.setColormap(None)

    def _dataAboutToFinalize(self, weakrefData):
        """Callback when the data weakref is about to be finalized."""
        if self._data is weakrefData:
            self.setData(None)

    def _itemAboutToFinalize(self, weakref):
        """Callback when the data weakref is about to be finalized."""
        if self._item is weakref:
            self.setItem(None)

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

        :param hist: array-like of counts or None to hide histogram
        :param bin_edges: array-like of bins edges or None to hide histogram
        """
        if hist is None or bin_edges is None:
            self._histogramData = None
            self._plot.remove(legend='Histogram', kind='histogram')
        else:
            hist = numpy.array(hist, copy=True)
            bin_edges = numpy.array(bin_edges, copy=True)
            self._histogramData = hist, bin_edges
            norm_hist = hist / max(hist)
            self._plot.addHistogram(norm_hist,
                                    bin_edges,
                                    legend="Histogram",
                                    color='gray',
                                    align='center',
                                    fill=True,
                                    z=1)
        self._updateMinMaxData()

    def getColormap(self):
        """Return the colormap description.

        :rtype: ~silx.gui.colors.Colormap
        """
        if self._colormap is None:
            return None
        return self._colormap()

    def resetColormap(self):
        """
        Reset the colormap state before modification.

        ..note :: the colormap reference state is the state when set or the
                  state when validated
        """
        colormap = self.getColormap()
        if colormap is not None and self._colormapStoredState is not None:
            if colormap != self._colormapStoredState:
                self._ignoreColormapChange = True
                colormap.setFromColormap(self._colormapStoredState)
                self._ignoreColormapChange = False
                self._applyColormap()

    def setDataRange(self, minimum=None, positiveMin=None, maximum=None):
        """Set the range of data to use for the range of the histogram area.

        :param float minimum: The minimum of the data
        :param float positiveMin: The positive minimum of the data
        :param float maximum: The maximum of the data
        """
        scale = self._plot.getXAxis().getScale()
        if scale == Axis.LOGARITHMIC:
            dataMin, dataMax = positiveMin, maximum
        else:
            dataMin, dataMax = minimum, maximum

        if dataMin is None or dataMax is None:
            self._dataRange = None
            self._plot.remove(legend='Range', kind='histogram')
        else:
            hist = numpy.array([1])
            bin_edges = numpy.array([dataMin, dataMax])
            self._plot.addHistogram(hist,
                                    bin_edges,
                                    legend="Range",
                                    color='gray',
                                    align='center',
                                    fill=True,
                                    z=1)
            self._dataRange = minimum, positiveMin, maximum
        self._updateMinMaxData()

    def _updateMinMaxData(self):
        """Update the min and max of the data according to the data range and
        the histogram preset."""
        colormap = self.getColormap()

        minimum = float("+inf")
        maximum = float("-inf")

        if colormap is not None:
            # find a range in the positive part of the data
            data = self._getItem()
            if data is None:
                data = self._getData()
            minimum, maximum = colormap.getColormapRange(data)
        else:
            if self._dataRange is not None:
                minimum = min(minimum, self._dataRange[0])
                maximum = max(maximum, self._dataRange[2])
            if self._histogramData is not None:
                minimum = min(minimum, self._histogramData[1][0])
                maximum = max(maximum, self._histogramData[1][-1])

        if not numpy.isfinite(minimum):
            minimum = None
        if not numpy.isfinite(maximum):
            maximum = None

        self._minValue.setDataValue(minimum)
        self._maxValue.setDataValue(maximum)
        self._plotUpdate()

    def accept(self):
        self.storeCurrentState()
        qt.QDialog.accept(self)

    def storeCurrentState(self):
        """
        save the current value sof the colormap if the user want to undo is
        modifications
        """
        colormap = self.getColormap()
        if colormap is not None:
            self._colormapStoredState = colormap.copy()
        else:
            self._colormapStoredState = None

    def reject(self):
        self.resetColormap()
        qt.QDialog.reject(self)

    def setColormap(self, colormap):
        """Set the colormap description

        :param ~silx.gui.colors.Colormap colormap: the colormap to edit
        """
        assert colormap is None or isinstance(colormap, Colormap)
        if self._ignoreColormapChange is True:
            return

        oldColormap = self.getColormap()
        if oldColormap is colormap:
            return
        if oldColormap is not None:
            oldColormap.sigChanged.disconnect(self._applyColormap)

        if colormap is not None:
            colormap.sigChanged.connect(self._applyColormap)
            colormap = weakref.ref(colormap, self._colormapAboutToFinalize)

        self._colormap = colormap
        self.storeCurrentState()
        if self.isVisible():
            self._applyColormap()
        else:
            self._updateResetButton()
            self._displayLater()

    def _updateResetButton(self):
        resetButton = self._buttonsNonModal.button(qt.QDialogButtonBox.Reset)
        rStateEnabled = False
        colormap = self.getColormap()
        if colormap is not None and colormap.isEditable():
            # can reset only in the case the colormap changed
            rStateEnabled = colormap != self._colormapStoredState
        resetButton.setEnabled(rStateEnabled)

    def _applyColormap(self):
        self._updateResetButton()
        self._updateLutItem(None)
        if self._ignoreColormapChange is True:
            return

        colormap = self.getColormap()
        if colormap is None:
            self._comboBoxColormap.setEnabled(False)
            self._normButtonLinear.setEnabled(False)
            self._normButtonLog.setEnabled(False)
            self._autoScaleCombo.setEnabled(False)
            self._minValue.setEnabled(False)
            self._maxValue.setEnabled(False)
            self._autoButtons.setEnabled(False)
        else:
            self._ignoreColormapChange = True
            self._comboBoxColormap.setCurrentLut(colormap)
            self._comboBoxColormap.setEnabled(colormap.isEditable())
            assert colormap.getNormalization() in Colormap.NORMALIZATIONS
            self._normButtonLinear.setChecked(
                colormap.getNormalization() == Colormap.LINEAR)
            self._normButtonLog.setChecked(
                colormap.getNormalization() == Colormap.LOGARITHM)
            vmin = colormap.getVMin()
            vmax = colormap.getVMax()
            self._normButtonLinear.setEnabled(colormap.isEditable())
            self._normButtonLog.setEnabled(colormap.isEditable())
            with utils.blockSignals(self._autoScaleCombo):
                self._autoScaleCombo.setCurrentMode(colormap.getAutoscaleMode())
                self._autoScaleCombo.setEnabled(colormap.isEditable())
            self._autoButtons.setEnabled(colormap.isEditable())
            self._autoButtons.setAutoRangeFromColormap(colormap)

            dataRange = colormap.getColormapRange()
            self._minValue.setValue(vmin or dataRange[0], isAuto=vmin is None)
            self._maxValue.setValue(vmax or dataRange[1], isAuto=vmax is None)
            self._minValue.setEnabled(colormap.isEditable())
            self._maxValue.setEnabled(colormap.isEditable())

            axis = self._plot.getXAxis()
            scale = axis.LINEAR if colormap.getNormalization() == Colormap.LINEAR else axis.LOGARITHMIC
            axis.setScale(scale)

            self._ignoreColormapChange = False

        self._plotUpdate()

    def _updateMinMax(self):
        if self._ignoreColormapChange is True:
            return

        vmin = self._minValue.getFiniteValue()
        vmax = self._maxValue.getFiniteValue()
        if vmax is not None and vmin is not None and vmax < vmin:
            # If only one autoscale is checked constraints are too strong
            # We have to edit a user value anyway it is not requested
            # TODO: It would be better IMO to disable the auto checkbox before
            # this case occur (valls)
            cmin = self._minValue.isAutoChecked()
            cmax = self._maxValue.isAutoChecked()
            if cmin is False:
                self._minValue.setFiniteValue(vmax)
            if cmax is False:
                self._maxValue.setFiniteValue(vmin)

        vmin = self._minValue.getValue()
        vmax = self._maxValue.getValue()
        self._ignoreColormapChange = True
        colormap = self._colormap()
        if colormap is not None:
            colormap.setVRange(vmin, vmax)
        self._ignoreColormapChange = False
        self._plotUpdate()
        self._updateResetButton()

    def _updateLut(self):
        if self._ignoreColormapChange is True:
            return

        colormap = self._colormap()
        if colormap is not None:
            self._ignoreColormapChange = True
            name = self._comboBoxColormap.getCurrentName()
            if name is not None:
                colormap.setName(name)
            else:
                lut = self._comboBoxColormap.getCurrentColors()
                colormap.setColormapLUT(lut)
            self._ignoreColormapChange = False

    def _updateAutoRange(self, autoRange):
        if self._ignoreColormapChange is True:
            return

        colormap = self.getColormap()

        data = self._getItem()
        if data is None:
            data = self._getData()
        dataRange = colormap.getColormapRange(data)

        # Final colormap range
        vmin = (dataRange[0] if not autoRange[0] else None)
        vmax = (dataRange[1] if not autoRange[1] else None)

        self._ignoreColormapChange = True
        colormap.setVRange(vmin, vmax)
        self._ignoreColormapChange = False

        self._minValue.setValue(vmin or dataRange[0], isAuto=vmin is None)
        self._maxValue.setValue(vmax or dataRange[1], isAuto=vmax is None)

        self._updateMinMaxData()

    def _updateNormalization(self, button):
        if self._ignoreColormapChange is True:
            return
        if not button.isChecked():
            return

        if button is self._normButtonLinear:
            norm = Colormap.LINEAR
            scale = Axis.LINEAR
        elif button is self._normButtonLog:
            norm = Colormap.LOGARITHM
            scale = Axis.LOGARITHMIC
        else:
            assert(False)

        colormap = self.getColormap()
        if colormap is not None:
            self._ignoreColormapChange = True
            colormap.setNormalization(norm)
            axis = self._plot.getXAxis()
            axis.setScale(scale)
            self._ignoreColormapChange = False

        self._invalidateHistogram()
        self._updateMinMaxData()

    def _updateAutoScaleMode(self):
        if self._ignoreColormapChange is True:
            return

        mode = self._autoScaleCombo.currentMode()

        colormap = self.getColormap()
        if colormap is not None:
            self._ignoreColormapChange = True
            colormap.setAutoscaleMode(mode)
            self._ignoreColormapChange = False

        self._invalidateHistogram()
        self._updateMinMaxData()

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
            if (self._maxValue.getValue() is not None and
                    self._minValue.getValue() > self._maxValue.getValue()):
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
            if (self._minValue.getValue() is not None and
                    self._minValue.getValue() > self._maxValue.getValue()):
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
