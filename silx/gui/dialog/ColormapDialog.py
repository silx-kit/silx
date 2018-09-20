# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2004-2018 European Synchrotron Radiation Facility
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
__date__ = "23/05/2018"


import logging

import numpy

from .. import qt
from ..colors import Colormap, preferredColormaps
from ..plot import PlotWidget
from silx.gui.widgets.FloatEdit import FloatEdit
import weakref
from silx.math.combo import min_max
from silx.third_party import enum
from silx.gui import icons
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


class _ColormapNameCombox(qt.QComboBox):
    def __init__(self, parent=None):
        qt.QComboBox.__init__(self, parent)
        self.__initItems()

    ORIGINAL_NAME = qt.Qt.UserRole + 1

    def __initItems(self):
        for colormapName in preferredColormaps():
            index = self.count()
            self.addItem(str.title(colormapName))
            self.setItemIcon(index, self.getIconPreview(colormapName))
            self.setItemData(index, colormapName, role=self.ORIGINAL_NAME)

    def getIconPreview(self, colormapName):
        """Return an icon preview from a LUT name.

        This icons are cached into a global structure.

        :param str colormapName: str
        :rtype: qt.QIcon
        """
        if colormapName not in _colormapIconPreview:
            icon = self.createIconPreview(colormapName)
            _colormapIconPreview[colormapName] = icon
        return _colormapIconPreview[colormapName]

    def createIconPreview(self, colormapName):
        """Create and return an icon preview from a LUT name.

        This icons are cached into a global structure.

        :param str colormapName: Name of the LUT
        :rtype: qt.QIcon
        """
        colormap = Colormap(colormapName)
        size = 32
        lut = colormap.getNColors(size)
        if lut is None or len(lut) == 0:
            return qt.QIcon()

        pixmap = qt.QPixmap(size, size)
        painter = qt.QPainter(pixmap)
        for i in range(size):
            rgb = lut[i]
            r, g, b = rgb[0], rgb[1], rgb[2]
            painter.setPen(qt.QColor(r, g, b))
            painter.drawPoint(qt.QPoint(i, 0))

        painter.drawPixmap(0, 1, size, size - 1, pixmap, 0, 0, size, 1)
        painter.end()

        return qt.QIcon(pixmap)

    def getCurrentName(self):
        return self.itemData(self.currentIndex(), self.ORIGINAL_NAME)

    def findColormap(self, name):
        return self.findData(name, role=self.ORIGINAL_NAME)

    def setCurrentName(self, name):
        index = self.findColormap(name)
        if index < 0:
            index = self.count()
            self.addItem(str.title(name))
            self.setItemIcon(index, self.getIconPreview(name))
            self.setItemData(index, name, role=self.ORIGINAL_NAME)
        self.setCurrentIndex(index)


@enum.unique
class _DataInPlotMode(enum.Enum):
    """Enum for each mode of display of the data in the plot."""
    NONE = 'none'
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

        self._histogramData = None
        self._minMaxWasEdited = False
        self._initialRange = None

        self._dataRange = None
        """If defined 3-tuple containing information from a data:
        minimum, positive minimum, maximum"""

        self._colormapStoredState = None

        # Make the GUI
        vLayout = qt.QVBoxLayout(self)

        formWidget = qt.QWidget(parent=self)
        vLayout.addWidget(formWidget)
        formLayout = qt.QFormLayout(formWidget)
        formLayout.setContentsMargins(10, 10, 10, 10)
        formLayout.setSpacing(0)

        # Colormap row
        self._comboBoxColormap = _ColormapNameCombox(parent=formWidget)
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
        self._plotToolbar = qt.QToolBar(self)
        self._plotToolbar.setFloatable(False)
        self._plotToolbar.setMovable(False)
        self._plotToolbar.setIconSize(qt.QSize(8, 8))
        self._plotToolbar.setStyleSheet("QToolBar { border: 0px }")
        self._plotToolbar.setOrientation(qt.Qt.Vertical)

        group = qt.QActionGroup(self._plotToolbar)
        group.setExclusive(True)

        action = qt.QAction("Nothing", self)
        action.setToolTip("No range nor histogram are displayed. No extra computation have to be done.")
        action.setIcon(icons.getQIcon('colormap-none'))
        action.setCheckable(True)
        action.setData(_DataInPlotMode.NONE)
        action.setChecked(action.data() == self._dataInPlotMode)
        self._plotToolbar.addAction(action)
        group.addAction(action)
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
        vLayout.addWidget(self._plotBox)

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

        # Set the colormap to default values
        self.setColormap(Colormap(name='gray', normalization='linear',
                         vmin=None, vmax=None))

        self.setModal(self.isModal())

        vLayout.setSizeConstraint(qt.QLayout.SetMinimumSize)
        self.setFixedSize(self.sizeHint())
        self._applyColormap()

    def showEvent(self, event):
        self.visibleChanged.emit(True)
        super(ColormapDialog, self).showEvent(event)

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
        self._plot.setDataMargins(yMinMargin=0.125, yMaxMargin=0.125)
        self._plot.getXAxis().setLabel("Data Values")
        self._plot.getYAxis().setLabel("")
        self._plot.setInteractiveMode('select', zoomOnWheel=False)
        self._plot.setActiveCurveHandling(False)
        self._plot.setMinimumSize(qt.QSize(250, 200))
        self._plot.sigPlotSignal.connect(self._plotSlot)

        self._plotUpdate()

    def sizeHint(self):
        return self.layout().minimumSize()

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

        minData, maxData = self._minValue.getFiniteValue(), self._maxValue.getFiniteValue()
        if minData > maxData:
            # avoid a full collapse
            minData, maxData = maxData, minData
        minimum = minData
        maximum = maxData

        if self._dataRange is not None:
            minRange = self._dataRange[0]
            maxRange = self._dataRange[2]
            minimum = min(minimum, minRange)
            maximum = max(maximum, maxRange)

        if self._histogramData is not None:
            minHisto = self._histogramData[1][0]
            maxHisto = self._histogramData[1][-1]
            minimum = min(minimum, minHisto)
            maximum = max(maximum, maxHisto)

        marge = abs(maximum - minimum) / 6.0
        if marge < 0.0001:
            # Smaller that the QLineEdit precision
            marge = 0.0001

        minView, maxView = minimum - marge, maximum + marge

        if updateMarkers:
            # Save the state in we are not moving the markers
            self._initialRange = minView, maxView
        elif self._initialRange is not None:
            minView = min(minView, self._initialRange[0])
            maxView = max(maxView, self._initialRange[1])

        x = [minView, minData, maxData, maxView]
        y = [0, 0, 1, 1]

        self._plot.addCurve(x, y,
                            legend="ConstrainedCurve",
                            color='black',
                            symbol='o',
                            linestyle='-',
                            resetzoom=False)

        if updateMarkers:
            minDraggable = (self._colormap().isEditable() and
                            not self._minValue.isAutoChecked())
            self._plot.addXMarker(
                self._minValue.getFiniteValue(),
                legend='Min',
                text='Min',
                draggable=minDraggable,
                color='blue',
                constraint=self._plotMinMarkerConstraint)

            maxDraggable = (self._colormap().isEditable() and
                            not self._maxValue.isAutoChecked())
            self._plot.addXMarker(
                self._maxValue.getFiniteValue(),
                legend='Max',
                text='Max',
                draggable=maxDraggable,
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
                self._initialRange = None
                self._updateMinMax()
            else:
                self._plotUpdate(updateMarkers=False)

    @staticmethod
    def computeDataRange(data):
        """Compute the data range as used by :meth:`setDataRange`.

        :param data: The data to process
        :rtype: Tuple(float, float, float)
        """
        if data is None or len(data) == 0:
            return None, None, None

        dataRange = min_max(data, min_positive=True, finite=True)
        if dataRange.minimum is None:
            # Only non-finite data
            dataRange = None

        if dataRange is not None:
            min_positive = dataRange.min_positive
            if min_positive is None:
                min_positive = float('nan')
            dataRange = dataRange.minimum, min_positive, dataRange.maximum

        if dataRange is None or len(dataRange) != 3:
            qt.QMessageBox.warning(
                None, "No Data",
                "Image data does not contain any real value")
            dataRange = 1., 1., 10.

        return dataRange

    @staticmethod
    def computeHistogram(data):
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

        xmin, xmax = min_max(_data, min_positive=False, finite=True)
        nbins = min(256, int(numpy.sqrt(_data.size)))
        data_range = xmin, xmax

        # bad hack: get 256 bins in the case we have a B&W
        if numpy.issubdtype(_data.dtype, numpy.integer):
            if nbins > xmax - xmin:
                nbins = xmax - xmin

        nbins = max(2, nbins)
        _data = _data.ravel().astype(numpy.float32)

        histogram = Histogramnd(_data, n_bins=nbins, histo_range=data_range)
        return histogram.histo, histogram.edges[0]

    def _getData(self):
        if self._data is None:
            return None
        return self._data()

    def setData(self, data):
        """Store the data as a weakref.

        According to the state of the dialog, the data will be used to display
        the data range or the histogram of the data using :meth:`setDataRange`
        and :meth:`setHistogram`
        """
        oldData = self._getData()
        if oldData is data:
            return

        if data is None:
            self._data = None
        else:
            self._data = weakref.ref(data, self._dataAboutToFinalize)

        self._updateDataInPlot()

    def _setDataInPlotMode(self, mode):
        if self._dataInPlotMode == mode:
            return
        self._dataInPlotMode = mode
        self._updateDataInPlot()

    def _displayDataInPlotModeChanged(self, action):
        mode = action.data()
        self._setDataInPlotMode(mode)

    def _updateDataInPlot(self):
        data = self._getData()
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

        if mode == _DataInPlotMode.NONE:
            self.setHistogram()
            self.setDataRange()
        elif mode == _DataInPlotMode.RANGE:
            result = self.computeDataRange(data)
            self.setHistogram()
            self.setDataRange(*result)
        elif mode == _DataInPlotMode.HISTOGRAM:
            # The histogram should be done in a worker thread
            result = self.computeHistogram(data)
            self.setHistogram(*result)
            self.setDataRange()

    def _colormapAboutToFinalize(self, weakrefColormap):
        """Callback when the data weakref is about to be finalized."""
        if self._colormap is weakrefColormap:
            self.setColormap(None)

    def _dataAboutToFinalize(self, weakrefData):
        """Callback when the data weakref is about to be finalized."""
        if self._data is weakrefData:
            self.setData(None)

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
                                    fill=True)
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
            if self._colormap()._toDict() != self._colormapStoredState:
                self._ignoreColormapChange = True
                colormap._setFromDict(self._colormapStoredState)
                self._ignoreColormapChange = False
                self._applyColormap()

    def setDataRange(self, minimum=None, positiveMin=None, maximum=None):
        """Set the range of data to use for the range of the histogram area.

        :param float minimum: The minimum of the data
        :param float positiveMin: The positive minimum of the data
        :param float maximum: The maximum of the data
        """
        if minimum is None or positiveMin is None or maximum is None:
            self._dataRange = None
            self._plot.remove(legend='Range', kind='histogram')
        else:
            hist = numpy.array([1])
            bin_edges = numpy.array([minimum, maximum])
            self._plot.addHistogram(hist,
                                    bin_edges,
                                    legend="Range",
                                    color='gray',
                                    align='center',
                                    fill=True)
            self._dataRange = minimum, positiveMin, maximum
        self._updateMinMaxData()

    def _updateMinMaxData(self):
        """Update the min and max of the data according to the data range and
        the histogram preset."""
        colormap = self.getColormap()

        minimum = float("+inf")
        maximum = float("-inf")

        if colormap is not None and colormap.getNormalization() == colormap.LOGARITHM:
            # find a range in the positive part of the data
            if self._dataRange is not None:
                minimum = min(minimum, self._dataRange[1])
                maximum = max(maximum, self._dataRange[2])
            if self._histogramData is not None:
                positives = list(filter(lambda x: x > 0, self._histogramData[1]))
                if len(positives) > 0:
                    minimum = min(minimum, positives[0])
                    maximum = max(maximum, positives[-1])
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
            self._colormapStoredState = colormap._toDict()
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
        self._updateResetButton()
        self._applyColormap()

    def _updateResetButton(self):
        resetButton = self._buttonsNonModal.button(qt.QDialogButtonBox.Reset)
        rStateEnabled = False
        colormap = self.getColormap()
        if colormap is not None and colormap.isEditable():
            # can reset only in the case the colormap changed
            rStateEnabled = colormap._toDict() != self._colormapStoredState
        resetButton.setEnabled(rStateEnabled)

    def _applyColormap(self):
        self._updateResetButton()
        if self._ignoreColormapChange is True:
            return

        colormap = self.getColormap()
        if colormap is None:
            self._comboBoxColormap.setEnabled(False)
            self._normButtonLinear.setEnabled(False)
            self._normButtonLog.setEnabled(False)
            self._minValue.setEnabled(False)
            self._maxValue.setEnabled(False)
        else:
            self._ignoreColormapChange = True

            if colormap.getName() is not None:
                name = colormap.getName()
                self._comboBoxColormap.setCurrentName(name)
                self._comboBoxColormap.setEnabled(self._colormap().isEditable())

            assert colormap.getNormalization() in Colormap.NORMALIZATIONS
            self._normButtonLinear.setChecked(
                colormap.getNormalization() == Colormap.LINEAR)
            self._normButtonLog.setChecked(
                colormap.getNormalization() == Colormap.LOGARITHM)
            vmin = colormap.getVMin()
            vmax = colormap.getVMax()
            dataRange = colormap.getColormapRange()
            self._normButtonLinear.setEnabled(self._colormap().isEditable())
            self._normButtonLog.setEnabled(self._colormap().isEditable())
            self._minValue.setValue(vmin or dataRange[0], isAuto=vmin is None)
            self._maxValue.setValue(vmax or dataRange[1], isAuto=vmax is None)
            self._minValue.setEnabled(self._colormap().isEditable())
            self._maxValue.setEnabled(self._colormap().isEditable())
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

    def _updateName(self):
        if self._ignoreColormapChange is True:
            return

        if self._colormap():
            self._ignoreColormapChange = True
            self._colormap().setName(
                self._comboBoxColormap.getCurrentName())
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

    def _activeLogNorm(self, isLog):
        if self._ignoreColormapChange is True:
            return
        if self._colormap():
            self._ignoreColormapChange = True
            norm = Colormap.LOGARITHM if isLog is True else Colormap.LINEAR
            self._colormap().setNormalization(norm)
            self._ignoreColormapChange = False
        self._updateMinMaxData()
