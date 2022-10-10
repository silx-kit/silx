# /*##########################################################################
#
# Copyright (c) 2004-2022 European Synchrotron Radiation Facility
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

__authors__ = ["V.A. Sole", "T. Vincent", "H. Payno"]
__license__ = "MIT"
__date__ = "08/12/2020"

import enum
import logging

import numpy

from .. import qt
from .. import utils
from ..colors import Colormap, cursorColorForColormap
from ..plot import PlotWidget
from ..plot.items.axis import Axis
from ..plot.items import BoundingRect
from silx.gui.widgets.FloatEdit import FloatEdit
import weakref
from silx.math.combo import min_max
from silx.gui.plot import items
from silx.gui import icons
from silx.gui.qt import inspect as qtinspect
from silx.gui.widgets.ColormapNameComboBox import ColormapNameComboBox
from silx.gui.widgets.FormGridLayout import FormGridLayout
from silx.math.histogram import Histogramnd
from silx.utils import deprecation
from silx.gui.plot.items.roi import RectangleROI
from silx.gui.plot.tools.roi import RegionOfInterestManager

_logger = logging.getLogger(__name__)

_colormapIconPreview = {}


class _DataRefHolder(items.Item, items.ColormapMixIn):
    """Holder for a weakref of a numpy array.

    It provides features from `ColormapMixIn`.
    """

    def __init__(self, dataRef):
        items.Item.__init__(self)
        items.ColormapMixIn.__init__(self)
        self.__dataRef = dataRef
        self._updated(items.ItemChangedType.DATA)

    def getColormappedData(self, copy=True):
        return self.__dataRef()


class _BoundaryWidget(qt.QWidget):
    """Widget to edit a boundary of the colormap (vmin or vmax)"""

    sigAutoScaleChanged = qt.Signal(object)
    """Signal emitted when the autoscale was changed

    True is sent as an argument if autoscale is set to true.
    """

    sigValueChanged = qt.Signal(object)
    """Signal emitted when value is changed

    The new value is sent as an argument.
    """

    def __init__(self, parent=None, value=0.0):
        qt.QWidget.__init__(self, parent=parent)
        self.setLayout(qt.QHBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self._numVal = FloatEdit(parent=self, value=value)

        self._iconAuto = icons.getQIcon('scale-auto')
        self._iconFixed = icons.getQIcon('scale-fixed')

        self._autoToggleAction = qt.QAction(self)
        self._autoToggleAction.setText("Auto scale")
        self._autoToggleAction.setToolTip("Toggle auto scale")
        self._autoToggleAction.setCheckable(True)
        self._autoToggleAction.setIcon(self._iconFixed)
        self._autoToggleAction.setChecked(False)
        self._autoToggleAction.toggled.connect(self._autoToggled)

        self._numVal.addAction(self._autoToggleAction, qt.QLineEdit.LeadingPosition)

        self.layout().addWidget(self._numVal)
        self._autoCB = qt.QCheckBox('auto', parent=self)
        self.layout().addWidget(self._autoCB)
        self._autoCB.setChecked(False)
        self._autoCB.setVisible(False)

        self._autoCB.toggled.connect(self._autoToggled)
        self._numVal.textEdited.connect(self.__textEdited)
        self._numVal.editingFinished.connect(self.__editingFinished)
        self.setFocusProxy(self._numVal)

        self.__textWasEdited = False
        """True if the text was edited, in order to send an event
        at the end of the user interaction"""

        self.__realValue = None
        """Store the real value set by setValue, to avoid
        rounding of the widget"""

    def __textEdited(self):
        self.__textWasEdited = True

    def __editingFinished(self):
        if self.__textWasEdited:
            value = self._numVal.value()
            self.__realValue = value
            with utils.blockSignals(self._numVal):
                # Fix the formatting
                self._numVal.setValue(self.__realValue)
            self.sigValueChanged.emit(value)
            self.__textWasEdited = False

    def isAutoChecked(self):
        return self._autoCB.isChecked()

    def getValue(self):
        """Returns the stored range. If autoscale is
        enabled, this returns None.
        """
        if self._autoCB.isChecked():
            return None
        if self.__realValue is not None:
            return self.__realValue
        return self._numVal.value()

    def _autoToggled(self, enabled):
        self._updateAutoScaleState(enabled)
        self._updateDisplayedText()
        self.sigAutoScaleChanged.emit(enabled)

    def _updateDisplayedText(self):
        self.__textWasEdited = False
        if self._autoCB.isChecked() and self.__realValue is not None:
            with utils.blockSignals(self._numVal):
                self._numVal.setValue(self.__realValue)

    def setValue(self, value, isAuto=False):
        """Set the value of the boundary.

        :param float value: A finite value for the boundary
        :param bool isAuto: If true, the finite value was automatically computed
            from the data, else it is a fixed custom value.
        """
        assert value is not None
        self._autoCB.setChecked(isAuto)
        with utils.blockSignals(self._numVal):
            if isAuto or self.__realValue != value:
                if not self.__textWasEdited:
                    self._numVal.setValue(value)
            self.__realValue = value
            self._updateAutoScaleState(isAuto)

    def _updateAutoScaleState(self, isAutoScale):
        self._numVal.setReadOnly(isAutoScale)
        palette = qt.QPalette()
        if isAutoScale:
            color = palette.color(qt.QPalette.Disabled, qt.QPalette.Base)
            icon = self._iconAuto
        else:
            color = palette.color(qt.QPalette.Normal, qt.QPalette.Base)
            icon = self._iconFixed
        palette.setColor(qt.QPalette.Base, color)
        self._numVal.setPalette(palette)
        self._autoToggleAction.setIcon(icon)


class _AutoscaleModeComboBox(qt.QComboBox):

    DATA = {
        Colormap.MINMAX: ("Min/max", "Use the data min/max"),
        Colormap.STDDEV3: ("Mean±3std", "Use the data mean ± 3 × standard deviation"),
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


class _AutoScaleButton(qt.QPushButton):

    autoRangeChanged = qt.Signal(object)

    def __init__(self, parent=None):
        qt.QPushButton.__init__(self, parent=parent)
        self.setText("Autoscale")
        self.setToolTip("Enable/disable the autoscale for both min and max")
        self.setCheckable(True)
        self.toggled[bool].connect(self.__toggled)
        self.setFocusPolicy(qt.Qt.TabFocus)

    def __toggled(self, checked):
        autoRange = checked, checked
        self.setAutoRange(autoRange)
        self.autoRangeChanged.emit(autoRange)

    def setAutoRangeFromColormap(self, colormap):
        vRange = colormap.getVRange()
        autoRange = vRange[0] is None, vRange[1] is None
        self.setAutoRange(autoRange)

    def setAutoRange(self, autoRange):
        with utils.blockSignals(self):
            self.setChecked(autoRange[0] if autoRange[0] == autoRange[1] else False)

@enum.unique
class _DataInPlotMode(enum.Enum):
    """Enum for each mode of display of the data in the plot."""
    RANGE = 'range'
    HISTOGRAM = 'histogram'


class _ColormapHistogram(qt.QWidget):
    """Display the colormap and the data as a plot."""

    sigRangeMoving = qt.Signal(object, object, object)
    """Emitted when a mouse interaction moves the location
    of the colormap range in the plot.

    This signal contains 2 elements:

    - vmin: A float value if this range was moved, else None
    - vmax: A float value if this range was moved, else None
    - gammaPos: A float value if this range was moved, else None
    """

    sigRangeMoved = qt.Signal(object, object, object)
    """Emitted when a mouse interaction stop.

    This signal contains 2 elements:

    - vmin: A float value if this range was moved, else None
    - vmax: A float value if this range was moved, else None
    - gammaPos: A float value if this range was moved, else None
    """

    def __init__(self, parent):
        qt.QWidget.__init__(self, parent=parent)
        self._dataInPlotMode = _DataInPlotMode.RANGE
        self._finiteRange = None, None
        self._initPlot()

        self._histogramData = {}
        """Histogram displayed in the plot"""

        self._dragging = False, False, False
        """True, if the min or the max handle is dragging"""

        self._dataRange = {}
        """Histogram displayed in the plot"""

        self._invalidated = False

    def paintEvent(self, event):
        if self._invalidated:
            self._updateDataInPlot()
            self._invalidated = False
        self._updateMarkerPosition()
        return super(_ColormapHistogram, self).paintEvent(event)

    def getFiniteRange(self):
        """Returns the colormap range as displayed in the plot."""
        return self._finiteRange

    def setFiniteRange(self, vRange):
        """Set the colormap range to use in the plot.

        Here there is no concept of auto. The values should
        not be None, except if there is no range or marker
        to display.
        """
        # Do not reset the limit for handle about to be dragged
        if self._dragging[0]:
            vRange = self._finiteRange[0], vRange[1]
        if self._dragging[1]:
            vRange = vRange[0], self._finiteRange[1]

        if vRange == self._finiteRange:
            return

        self._finiteRange = vRange
        self.update()

    def getColormap(self):
        return self.parent().getColormap()

    def _getNormalizedHistogram(self):
        """Return an histogram already normalized according to the colormap
        normalization.

        Returns a tuple edges, counts
        """
        norm = self._getNorm()
        histogram = self._histogramData.get(norm, None)
        if histogram is None:
            histogram = self._computeNormalizedHistogram()
            self._histogramData[norm] = histogram
        return histogram

    def _computeNormalizedHistogram(self):
        colormap = self.getColormap()
        if colormap is None:
            norm = Colormap.LINEAR
        else:
            norm = colormap.getNormalization()

        # Try to use the histogram defined in the dialog
        histo = self.parent()._getHistogram()
        if histo is not None:
            counts, edges = histo
            normalizer = Colormap(normalization=norm)._getNormalizer()
            mask = normalizer.is_valid(edges[:-1])  # Check lower bin edges only
            firstValid = numpy.argmax(mask)  # edges increases monotonically
            if firstValid == 0:  # Mask is all False or all True
                return (counts, edges) if mask[0] else (None, None)
            else:  # Clip to valid values
                return counts[firstValid:], edges[firstValid:]

        data = self.parent()._getArray()
        if data is None:
            return None, None
        dataRange = self._getNormalizedDataRange()
        if dataRange[0] is None or dataRange[1] is None:
            return None, None
        counts, edges = self.parent().computeHistogram(data, scale=norm, dataRange=dataRange)
        return counts, edges

    def _getNormalizedDataRange(self):
        """Return a data range already normalized according to the colormap
        normalization.

        Returns a tuple with min and max
        """
        norm = self._getNorm()
        dataRange = self._dataRange.get(norm, None)
        if dataRange is None:
            dataRange = self._computeNormalizedDataRange()
            self._dataRange[norm] = dataRange
        return dataRange

    def _computeNormalizedDataRange(self):
        colormap = self.getColormap()
        if colormap is None:
            norm = Colormap.LINEAR
        else:
            norm = colormap.getNormalization()

        # Try to use the one defined in the dialog
        dataRange = self.parent()._getDataRange()
        if dataRange is not None:
            if norm in (Colormap.LINEAR, Colormap.GAMMA, Colormap.ARCSINH):
                return dataRange[0], dataRange[2]
            elif norm == Colormap.LOGARITHM:
                return dataRange[1], dataRange[2]
            elif norm == Colormap.SQRT:
                return dataRange[1], dataRange[2]
            else:
                _logger.error("Undefined %s normalization", norm)

        # Try to use the histogram defined in the dialog
        histo = self.parent()._getHistogram()
        if histo is not None:
            _histo, edges = histo
            normalizer = Colormap(normalization=norm)._getNormalizer()
            edges = edges[normalizer.is_valid(edges)]
            if edges.size == 0:
                return None, None
            else:
                dataRange = min_max(edges, finite=True)
                return dataRange.minimum, dataRange.maximum

        item = self.parent()._getItem()
        if item is not None:
            # Trick to reach data range using colormap cache
            cm = Colormap()
            cm.setVRange(None, None)
            cm.setNormalization(norm)
            dataRange = item._getColormapAutoscaleRange(cm)
            return dataRange

        # If there is no item, there is no data
        return None, None

    def _getDisplayableRange(self):
        """Returns the selected min/max range to apply to the data,
        according to the used scale.

        One or both limits can be None in case it is not displayable in the
        current axes scale.

        :returns: Tuple{float, float}
        """
        scale = self._plot.getXAxis().getScale()

        def isDisplayable(pos):
            if pos is None:
                return False
            if scale == Axis.LOGARITHMIC:
                return pos > 0.0
            return True

        posMin, posMax = self.getFiniteRange()
        if not isDisplayable(posMin):
            posMin = None
        if not isDisplayable(posMax):
            posMax = None

        return posMin, posMax

    def _initPlot(self):
        """Init the plot to display the range and the values"""
        self._plot = PlotWidget(self)
        self._plot.setAxesDisplayed(False)
        self._plot.setDataMargins(0.125, 0.125, 0.01, 0.01)
        self._plot.getXAxis().setLabel("Data Values")
        self._plot.getYAxis().setLabel("")
        self._plot.setInteractiveMode('select', zoomOnWheel=False)
        self._plot.setActiveCurveHandling(False)
        self._plot.setMinimumSize(qt.QSize(250, 200))
        self._plot.sigPlotSignal.connect(self._plotEventReceived)
        palette = self.palette()
        color = palette.color(qt.QPalette.Normal, qt.QPalette.Window)
        self._plot.setBackgroundColor(color)
        self._plot.setDataBackgroundColor("white")

        lut = numpy.arange(256)
        lut.shape = 1, -1
        self._plot.addImage(lut, legend='lut')
        self._lutItem = self._plot._getItem("image", "lut")
        self._lutItem.setVisible(False)

        self._plot.addScatter(x=[], y=[], value=[], legend='lut2')
        self._lutItem2 = self._plot._getItem("scatter", "lut2")
        self._lutItem2.setVisible(False)
        self.__lutY = numpy.array([-0.05] * 256)
        self.__lutV = numpy.arange(256)

        self._bound = BoundingRect()
        self._plot.addItem(self._bound)
        self._bound.setVisible(True)

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

        plotBoxLayout = qt.QHBoxLayout()
        plotBoxLayout.setContentsMargins(0, 0, 0, 0)
        plotBoxLayout.setSpacing(2)
        plotBoxLayout.addWidget(self._plotToolbar)
        plotBoxLayout.addWidget(self._plot)
        plotBoxLayout.setSizeConstraint(qt.QLayout.SetMinimumSize)
        self.setLayout(plotBoxLayout)

    def _plotEventReceived(self, event):
        """Handle events from the plot"""
        kind = event['event']

        if kind == 'markerMoving':
            value = event['xdata']
            if event['label'] == 'Min':
                self._dragging = True, False, False
                self._finiteRange = value, self._finiteRange[1]
                self._last = value, None, None
                self._updateGammaPosition()
                self.sigRangeMoving.emit(*self._last)
            elif event['label'] == 'Max':
                self._dragging = False, True, False
                self._finiteRange = self._finiteRange[0], value
                self._last = None, value, None
                self._updateGammaPosition()
                self.sigRangeMoving.emit(*self._last)
            elif event['label'] == 'Gamma':
                self._dragging = False, False, True
                self._last = None, None, value
                self.sigRangeMoving.emit(*self._last)
            self._updateLutItem(self._finiteRange)
        elif kind == 'markerMoved':
            self.sigRangeMoved.emit(*self._last)
            self._plot.resetZoom()
            self._dragging = False, False, False
        else:
            pass

    def _updateMarkerPosition(self):
        colormap = self.getColormap()
        posMin, posMax = self._getDisplayableRange()

        if colormap is None:
            isDraggable = False
        else:
            isDraggable = colormap.isEditable()

        with utils.blockSignals(self):
            if posMin is not None and not self._dragging[0]:
                self._plot.addXMarker(
                    posMin,
                    legend='Min',
                    text='Min',
                    draggable=isDraggable,
                    color="blue",
                    constraint=self._plotMinMarkerConstraint)
            self._updateGammaPosition()
            if posMax is not None and not self._dragging[1]:
                self._plot.addXMarker(
                    posMax,
                    legend='Max',
                    text='\n\nMax',
                    draggable=isDraggable,
                    color="blue",
                    constraint=self._plotMaxMarkerConstraint)

        self._updateLutItem((posMin, posMax))
        self._plot.resetZoom()

    def _updateGammaPosition(self):
        colormap = self.getColormap()
        posMin, posMax = self._getDisplayableRange()

        if colormap is None:
            gamma = None
        else:
            if colormap.getNormalization() == Colormap.GAMMA:
                gamma = colormap.getGammaNormalizationParameter()
            else:
                gamma = None

        if gamma is not None:
            if not self._dragging[2]:
                posRange = posMax - posMin
                if posRange > 0:
                    gammaPos = posMin + posRange * 0.5**(1/gamma)
                else:
                    gammaPos = posMin
                marker = self._plot._getMarker(
                    self._plot.addXMarker(
                        gammaPos,
                        legend='Gamma',
                        text='\nGamma',
                        draggable=True,
                        color="blue",
                        constraint=self._plotGammaMarkerConstraint,
                    )
                )
                marker.setZValue(2)
        else:
            try:
                self._plot.removeMarker('Gamma')
            except Exception:
                pass

    def _updateLutItem(self, vRange):
        colormap = self.getColormap()
        if colormap is None:
            return

        if vRange is None:
            posMin, posMax = self._getDisplayableRange()
        else:
            posMin, posMax = vRange
        if posMin is None or posMax is None:
            self._lutItem.setVisible(False)
            pos = posMax if posMin is None else posMin
            if pos is not None:
                self._bound.setBounds((pos, pos, -0.1, 0))
            else:
                self._bound.setBounds((0, 0, -0.1, 0))
        else:
            norm = colormap.getNormalization()
            normColormap = colormap.copy()
            normColormap.setEditable(True)
            normColormap.setVRange(0, 255)
            normColormap.setNormalization(Colormap.LINEAR)
            if norm == Colormap.LINEAR:
                scale = (posMax - posMin) / 256
                self._lutItem.setColormap(normColormap)
                self._lutItem.setOrigin((posMin, -0.09))
                self._lutItem.setScale((scale, 0.08))
                self._lutItem.setVisible(True)
                self._lutItem2.setVisible(False)
            elif norm == Colormap.LOGARITHM:
                self._lutItem2.setVisible(False)
                self._lutItem2.setColormap(normColormap)
                xx = numpy.geomspace(posMin, posMax, 256)
                self._lutItem2.setData(x=xx,
                                       y=self.__lutY,
                                       value=self.__lutV,
                                       copy=False)
                self._lutItem2.setSymbol("|")
                self._lutItem2.setVisible(True)
                self._lutItem.setVisible(False)
            else:
                # Fallback: Display with linear axis and applied normalization
                self._lutItem2.setVisible(False)
                normColormap.setNormalization(norm)
                self._lutItem2.setColormap(normColormap)
                xx = numpy.linspace(posMin, posMax, 256, endpoint=True)
                self._lutItem2.setData(
                    x=xx,
                    y=self.__lutY,
                    value=self.__lutV,
                    copy=False)
                self._lutItem2.setSymbol("|")
                self._lutItem2.setVisible(True)
                self._lutItem.setVisible(False)

            self._bound.setBounds((posMin, posMax, -0.1, 1))

    def _plotMinMarkerConstraint(self, x, y):
        """Constraint of the min marker"""
        _vmin, vmax = self.getFiniteRange()
        if vmax is None:
            return x, y
        return min(x, vmax), y

    def _plotMaxMarkerConstraint(self, x, y):
        """Constraint of the max marker"""
        vmin, _vmax = self.getFiniteRange()
        if vmin is None:
            return x, y
        return max(x, vmin), y

    def _plotGammaMarkerConstraint(self, x, y):
        """Constraint of the gamma marker"""
        vmin, vmax = self.getFiniteRange()
        if vmin is not None:
            x = max(x, vmin)
        if vmax is not None:
            x = min(x, vmax)
        return x, y

    def _setDataInPlotMode(self, mode):
        if self._dataInPlotMode == mode:
            return
        self._dataInPlotMode = mode
        self._updateDataInPlot()

    def _displayDataInPlotModeChanged(self, action):
        mode = action.data()
        self._setDataInPlotMode(mode)

    def invalidateData(self):
        self._histogramData = {}
        self._dataRange = {}
        self._invalidated = True
        self.update()

    def _updateDataInPlot(self):
        mode = self._dataInPlotMode

        norm = self._getNorm()
        if norm == Colormap.LINEAR:
            scale = Axis.LINEAR
        elif norm == Colormap.LOGARITHM:
            scale = Axis.LOGARITHMIC
        else:
            scale = Axis.LINEAR

        axis = self._plot.getXAxis()
        axis.setScale(scale)

        if mode == _DataInPlotMode.RANGE:
            dataRange = self._getNormalizedDataRange()
            xmin, xmax = dataRange
            if xmax is None or xmin is None:
                self._plot.remove(legend='Data', kind='histogram')
            else:
                histogram = numpy.array([1])
                bin_edges = numpy.array([xmin, xmax])
                self._plot.addHistogram(histogram,
                                        bin_edges,
                                        legend="Data",
                                        color='gray',
                                        align='center',
                                        fill=True,
                                        z=1)

        elif mode == _DataInPlotMode.HISTOGRAM:
            histogram, bin_edges = self._getNormalizedHistogram()
            if histogram is None or bin_edges is None:
                self._plot.remove(legend='Data', kind='histogram')
            else:
                histogram = numpy.array(histogram, copy=True)
                bin_edges = numpy.array(bin_edges, copy=True)
                with numpy.errstate(invalid='ignore'):
                    norm_histogram = histogram / numpy.nanmax(histogram)
                self._plot.addHistogram(norm_histogram,
                                        bin_edges,
                                        legend="Data",
                                        color='gray',
                                        align='center',
                                        fill=True,
                                        z=1)
        else:
            _logger.error("Mode unsupported")

    def sizeHint(self):
        return self.layout().minimumSize()

    def updateLut(self):
        self._updateLutItem(None)

    def _getNorm(self):
        colormap = self.getColormap()
        if colormap is None:
            return Axis.LINEAR
        else:
            norm = colormap.getNormalization()
            return norm

    def updateNormalization(self):
        self._updateDataInPlot()
        self.update()


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

        self.__aboutToDelete = False
        self._colormap = None

        self._data = None
        """Weak ref to an external numpy array
        """
        self._itemHolder = None
        """Hard ref to a private item (used as holder to the data)
        This allow to reuse the item cache
        """
        self._item = None
        """Weak ref to an external item"""

        self._colormapped = None
        """Weak ref to reduce data update"""

        self._colormapChange = utils.LockReentrant()
        """Used as a semaphore to avoid editing the colormap object when we are
        only attempt to display it.
        Used instead of n connect and disconnect of the sigChanged. The
        disconnection to sigChanged was also limiting when this colormapdialog
        is used in the colormapaction and associated to the activeImageChanged.
        (because the activeImageChanged is send when the colormap changed and
        the self.setcolormap is a callback)
        """

        self.__colormapInvalidated = False
        self.__dataInvalidated = False

        self._histogramData = None

        self._dataRange = None
        """If defined 3-tuple containing information from a data:
        minimum, positive minimum, maximum"""

        self._colormapStoredState = None

        # Colormap row
        self._comboBoxColormap = ColormapNameComboBox(parent=self)
        self._comboBoxColormap.currentIndexChanged[int].connect(self._comboBoxColormapUpdated)

        # Normalization row
        self._comboBoxNormalization = qt.QComboBox(parent=self)
        normalizations = [
            ('Linear', Colormap.LINEAR),
            ('Gamma correction', Colormap.GAMMA),
            ('Arcsinh', Colormap.ARCSINH),
            ('Logarithmic', Colormap.LOGARITHM),
            ('Square root', Colormap.SQRT)]
        for name, userData in normalizations:
            try:
                icon = icons.getQIcon("colormap-norm-%s" % userData)
            except:
                icon = qt.QIcon()
            self._comboBoxNormalization.addItem(icon, name, userData)
        self._comboBoxNormalization.currentIndexChanged[int].connect(
            self._normalizationUpdated)

        self._gammaSpinBox = qt.QDoubleSpinBox(parent=self)
        self._gammaSpinBox.setEnabled(False)
        self._gammaSpinBox.setRange(0.01, 100.)
        self._gammaSpinBox.setDecimals(4)
        if hasattr(qt.QDoubleSpinBox, "setStepType"):
            # Introduced in Qt 5.12
            self._gammaSpinBox.setStepType(qt.QDoubleSpinBox.AdaptiveDecimalStepType)
        else:
            self._gammaSpinBox.setSingleStep(0.1)
        self._gammaSpinBox.valueChanged.connect(self._gammaUpdated)
        self._gammaSpinBox.setValue(2.)

        autoScaleCombo = _AutoscaleModeComboBox(self)
        autoScaleCombo.currentIndexChanged.connect(self._autoscaleModeUpdated)
        self._autoScaleCombo = autoScaleCombo

        # Min row
        self._minValue = _BoundaryWidget(parent=self, value=1.0)
        self._minValue.sigAutoScaleChanged.connect(self._minAutoscaleUpdated)
        self._minValue.sigValueChanged.connect(self._minValueUpdated)
        self._minValue.setMinimumWidth(140)

        # Max row
        self._maxValue = _BoundaryWidget(parent=self, value=10.0)
        self._maxValue.sigAutoScaleChanged.connect(self._maxAutoscaleUpdated)
        self._maxValue.sigValueChanged.connect(self._maxValueUpdated)
        self._maxValue.setMinimumWidth(140)

        self._autoButtons = _AutoScaleButton(self)
        self._autoButtons.autoRangeChanged.connect(self._autoRangeButtonsUpdated)

        rangeLayout = qt.QGridLayout()
        miniFont = qt.QFont(self.font())
        miniFont.setPixelSize(8)
        labelMin = qt.QLabel("Min", self)
        labelMin.setFont(miniFont)
        labelMin.setAlignment(qt.Qt.AlignHCenter)
        labelMax = qt.QLabel("Max", self)
        labelMax.setAlignment(qt.Qt.AlignHCenter)
        labelMax.setFont(miniFont)
        rangeLayout.addWidget(labelMin, 0, 1)
        rangeLayout.addWidget(labelMax, 0, 3)
        rangeLayout.addWidget(self._minValue, 1, 1)
        rangeLayout.addWidget(self._maxValue, 1, 3)
        rangeLayout.setColumnStretch(0, 1)
        rangeLayout.setColumnStretch(1, 2)
        rangeLayout.setColumnStretch(2, 1)
        rangeLayout.setColumnStretch(3, 2)
        rangeLayout.setColumnStretch(4, 1)

        self._histoWidget = _ColormapHistogram(self)
        self._histoWidget.sigRangeMoving.connect(self._histogramRangeMoving)
        self._histoWidget.sigRangeMoved.connect(self._histogramRangeMoved)
        self._histoWidget.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Expanding)

        # Scale to buttons
        self._visibleAreaButton = qt.QPushButton(self)
        self._visibleAreaButton.setEnabled(False)
        self._visibleAreaButton.setText("Visible Area")
        self._visibleAreaButton.clicked.connect(
            self._handleScaleToVisibleAreaClicked,
            type=qt.Qt.QueuedConnection)

        # Place-holder for selected area ROI manager
        self._roiForColormapManager = None

        self._selectedAreaButton = qt.QPushButton(self)
        self._selectedAreaButton.setCheckable(True)
        self._selectedAreaButton.setEnabled(False)
        self._selectedAreaButton.setText("Selection")
        self._selectedAreaButton.setIcon(icons.getQIcon("add-shape-rectangle"))
        self._selectedAreaButton.setCheckable(True)
        self._selectedAreaButton.toggled.connect(
            self._handleScaleToSelectionToggled,
            type=qt.Qt.QueuedConnection)

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

        layout = qt.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._visibleAreaButton)
        layout.addWidget(self._selectedAreaButton)
        layout.addStretch()
        self._scaleToAreaGroup = qt.QWidget(self)
        self._scaleToAreaGroup.setLayout(layout)
        self._scaleToAreaGroup.setVisible(False)

        layoutScale = qt.QHBoxLayout()
        layoutScale.setContentsMargins(0, 0, 0, 0)
        layoutScale.addWidget(self._autoButtons)
        layoutScale.addWidget(self._autoScaleCombo)
        layoutScale.addStretch()


        formLayout = FormGridLayout(self)
        formLayout.setContentsMargins(10, 10, 10, 10)

        formLayout.addRow('Colormap:', self._comboBoxColormap)
        formLayout.addRow('Normalization:', self._comboBoxNormalization)
        formLayout.addRow('Gamma:', self._gammaSpinBox)

        formLayout.addItem(qt.QSpacerItem(1, 1, qt.QSizePolicy.Fixed, qt.QSizePolicy.Fixed))
        formLayout.addRow(self._histoWidget)
        formLayout.addRow(rangeLayout)
        formLayout.addItem(qt.QSpacerItem(1, 1, qt.QSizePolicy.Fixed, qt.QSizePolicy.Fixed))
        formLayout.addRow('Scale:', layoutScale)
        formLayout.addRow("Fixed scale on:", self._scaleToAreaGroup)
        formLayout.addRow(self._buttonsModal)
        formLayout.addRow(self._buttonsNonModal)
        formLayout.setSizeConstraint(qt.QLayout.SetMinAndMaxSize)

        self.setTabOrder(self._comboBoxColormap, self._comboBoxNormalization)
        self.setTabOrder(self._comboBoxNormalization, self._gammaSpinBox)
        self.setTabOrder(self._gammaSpinBox, self._minValue)
        self.setTabOrder(self._minValue, self._maxValue)
        self.setTabOrder(self._maxValue, self._autoButtons)
        self.setTabOrder(self._autoButtons, self._autoScaleCombo)
        self.setTabOrder(self._autoScaleCombo, self._visibleAreaButton)
        self.setTabOrder(self._visibleAreaButton, self._selectedAreaButton)
        self.setTabOrder(self._selectedAreaButton, self._buttonsModal)
        self.setTabOrder(self._buttonsModal, self._buttonsNonModal)

        self._applyColormap()

    def _invalidateColormap(self):
        if self.isVisible():
            self._applyColormap()
        else:
            self.__colormapInvalidated = True

    def _invalidateData(self):
        if self.isVisible():
            self._updateWidgetRange()
            self._histoWidget.invalidateData()
        else:
            self.__dataInvalidated = True

    def _validate(self):
        if self.__colormapInvalidated:
            self._applyColormap()
        if self.__dataInvalidated:
            self._histoWidget.invalidateData()
        if self.__dataInvalidated or self.__colormapInvalidated:
            self._updateWidgetRange()
        self.__dataInvalidated = False
        self.__colormapInvalidated = False

    def showEvent(self, event):
        self.visibleChanged.emit(True)
        super(ColormapDialog, self).showEvent(event)
        if self.isVisible():
            self._validate()

    def closeEvent(self, event):
        if not self.isModal():
            self.accept()
        super(ColormapDialog, self).closeEvent(event)

    def hideEvent(self, event):
        if self._selectedAreaButton.isChecked():
            self._selectedAreaButton.setChecked(False)
        self.visibleChanged.emit(False)
        super(ColormapDialog, self).hideEvent(event)

    def close(self):
        if self._selectedAreaButton.isChecked():
            self._selectedAreaButton.setChecked(False)
        self.accept()
        qt.QDialog.close(self)

    def setModal(self, modal):
        assert type(modal) is bool
        self._buttonsNonModal.setVisible(not modal)
        self._buttonsModal.setVisible(modal)
        qt.QDialog.setModal(self, modal)

    def event(self, event):
        if event.type() == qt.QEvent.DeferredDelete:
            self.__aboutToDelete = True
        return super(ColormapDialog, self).event(event)

    def exec(self):
        wasModal = self.isModal()
        self.setModal(True)
        result = super(ColormapDialog, self).exec()
        if not self.__aboutToDelete:
            self.setModal(wasModal)
        return result

    def exec_(self):  # Qt5 compatibility wrapper
        return self.exec()

    def _getFiniteColormapRange(self):
        """Return a colormap range where auto ranges are fixed
        according to the available data.
        """
        colormap = self.getColormap()
        if colormap is None:
            return 1, 10

        item = self._getItem()
        if item is not None:
            return colormap.getColormapRange(item)
        # If there is not item, there is no data
        return colormap.getColormapRange(None)

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
    def computeHistogram(data, scale=Axis.LINEAR, dataRange=None):
        """Compute the data histogram as used by :meth:`setHistogram`.

        :param data: The data to process
        :param dataRange: Optional range to compute the histogram, which is a
            tuple of min, max
        :rtype: Tuple(List(float),List(float)
        """
        # For compatibility
        if scale == Axis.LOGARITHMIC:
            scale = Colormap.LOGARITHM

        if data is None:
            return None, None

        if len(data) == 0:
            return None, None

        if data.ndim == 3:  # RGB(A) images
            _logger.info('Converting current image from RGB(A) to grayscale\
                in order to compute the intensity distribution')
            data = (data[:,:, 0] * 0.299 +
                    data[:,:, 1] * 0.587 +
                    data[:,:, 2] * 0.114)

        # bad hack: get 256 continuous bins in the case we have a B&W
        normalizeData = True
        if numpy.issubdtype(data.dtype, numpy.ubyte):
            normalizeData = False
        elif numpy.issubdtype(data.dtype, numpy.integer):
            if dataRange is not None:
                xmin, xmax = dataRange
                if xmin is not None and xmax is not None:
                    normalizeData = (xmax - xmin) > 255

        if normalizeData:
            if scale == Colormap.LOGARITHM:
                with numpy.errstate(divide='ignore', invalid='ignore'):
                    data = numpy.log10(data)

        if dataRange is not None:
            xmin, xmax = dataRange
            if xmin is None:
                return None, None
            if normalizeData:
                if scale == Colormap.LOGARITHM:
                    xmin, xmax = numpy.log10(xmin), numpy.log10(xmax)
        else:
            xmin, xmax = min_max(data, min_positive=False, finite=True)

        if xmin is None:
            return None, None

        nbins = min(256, int(numpy.sqrt(data.size)))
        data_range = xmin, xmax

        # bad hack: get 256 bins in the case we have a B&W
        if numpy.issubdtype(data.dtype, numpy.integer):
            if nbins > xmax - xmin:
                nbins = int(xmax - xmin)

        nbins = max(2, nbins)
        data = data.ravel().astype(numpy.float32)

        histogram = Histogramnd(data, n_bins=nbins, histo_range=data_range)
        bins = histogram.edges[0]
        if normalizeData:
            if scale == Colormap.LOGARITHM:
                bins = 10 ** bins
        return histogram.histo, bins

    def _getItem(self):
        if self._itemHolder is not None:
            return self._itemHolder
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
            # While event from items are not supported, we can't ignore dup items
            if item is not None:
                array = item.getColormappedData(copy=False)
            else:
                array = None
            colormapped = self._colormapped
            if colormapped is not None:
                oldArray = colormapped()
            else:
                oldArray = None
            if oldArray is array:
                return

        self._data = None
        self._itemHolder = None
        try:
            if item is None:
                self._item = None
            else:
                if not isinstance(item, items.ColormapMixIn):
                    self._item = None
                    raise ValueError("Item %s is not supported" % item)
                self._item = weakref.ref(item, self._itemAboutToFinalize)
        finally:
            self._syncScaleToButtonsEnabled()
            self._dataRange = None
            self._histogramData = None
            self._invalidateData()

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
        self._syncScaleToButtonsEnabled()
        if data is None:
            self._data = None
            self._itemHolder = None
        else:
            self._data = weakref.ref(data, self._dataAboutToFinalize)
            self._itemHolder = _DataRefHolder(self._data)

        self._dataRange = None
        self._histogramData = None

        self._invalidateData()

    def _getArray(self):
        data = self._getData()
        if data is not None:
            return data
        item = self._getItem()
        if item is not None:
            colormapped = item.getColormappedData(copy=False)
            if colormapped is not None:
                self._colormapped = weakref.ref(colormapped)
            else:
                self._colormapped = None
            return colormapped
        return None

    def _colormapAboutToFinalize(self, weakrefColormap):
        """Callback when the data weakref is about to be finalized."""
        if self._colormap is weakrefColormap and qtinspect.isValid(self):
            self.setColormap(None)

    def _dataAboutToFinalize(self, weakrefData):
        """Callback when the data weakref is about to be finalized."""
        if self._data is weakrefData and qtinspect.isValid(self):
            self.setData(None)

    def _itemAboutToFinalize(self, weakref):
        """Callback when the data weakref is about to be finalized."""
        if self._item is weakref and qtinspect.isValid(self):
            self.setItem(None)

    @deprecation.deprecated(reason="It is private data", since_version="0.13")
    def getHistogram(self):
        histo = self._getHistogram()
        if histo is None:
            return None
        counts, bin_edges = histo
        return numpy.array(counts, copy=True), numpy.array(bin_edges, copy=True)

    def _getHistogram(self):
        """Returns the histogram defined by the dialog as metadata
        to describe the data in order to speed up the dialog.

        :return: (hist, bin_edges)
        :rtype: 2-tuple of numpy arrays"""
        return self._histogramData

    def setHistogram(self, hist=None, bin_edges=None):
        """Set the histogram to display.

        This update the data range with the bounds of the bins.

        :param hist: array-like of counts or None to hide histogram
        :param bin_edges: array-like of bins edges or None to hide histogram
        """
        if hist is None or bin_edges is None:
            self._histogramData = None
        else:
            self._histogramData = numpy.array(hist), numpy.array(bin_edges)

        self._invalidateData()

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
                with self._colormapChange:
                    colormap.setFromColormap(self._colormapStoredState)
                self._applyColormap()

    def _getDataRange(self):
        """Returns the data range defined by the dialog as metadata
        to describe the data in order to speed up the dialog.

        :return: (minimum, positiveMin, maximum)
        :rtype: 3-tuple of floats or None"""
        return self._dataRange

    def setDataRange(self, minimum=None, positiveMin=None, maximum=None):
        """Set the range of data to use for the range of the histogram area.

        :param float minimum: The minimum of the data
        :param float positiveMin: The positive minimum of the data
        :param float maximum: The maximum of the data
        """
        self._dataRange = minimum, positiveMin, maximum
        self._invalidateData()

    def _setColormapRange(self, xmin, xmax):
        """Set a new range to the held colormap and update the
        widget."""
        colormap = self.getColormap()
        if colormap is not None:
            with self._colormapChange:
                colormap.setVRange(xmin, xmax)
        self._updateWidgetRange()

    def setColormapRangeFromDataBounds(self, bounds):
        """Set the range of the colormap from current item and rect.

        If there is no ColormapMixIn item attached to the ColormapDialog,
        nothing is done.

        :param Union[List[float],None] bounds:
            (xmin, xmax, ymin, ymax) Rectangular region in data space
        """
        if bounds is None:
            return None  # no-op

        colormap = self.getColormap()
        if colormap is None:
            return  # no-op

        item = self._getItem()
        if not isinstance(item, items.ColormapMixIn):
            return None  # no-op

        data = item.getColormappedData(copy=False)

        xmin, xmax, ymin, ymax = bounds

        if isinstance(item, items.ImageBase):
            ox, oy = item.getOrigin()
            sx, sy = item.getScale()

            ystart = max(0, int((ymin - oy) / sy))
            ystop = max(0, int(numpy.ceil((ymax - oy) / sy)))
            xstart = max(0, int((xmin - ox) / sx))
            xstop = max(0, int(numpy.ceil((xmax - ox) / sx)))

            subset = data[ystart:ystop, xstart:xstop]

        elif isinstance(item, items.Scatter):
            x = item.getXData(copy=False)
            y = item.getYData(copy=False)
            subset = data[
                numpy.logical_and(
                    numpy.logical_and(xmin <= x, x <= xmax),
                    numpy.logical_and(ymin <= y, y <= ymax))]

        if subset.size == 0:
            return  # no-op

        vmin, vmax = colormap._computeAutoscaleRange(subset)
        self._setColormapRange(vmin, vmax)

    def _updateWidgetRange(self):
        """Update the colormap range displayed into the widget."""
        xmin, xmax = self._getFiniteColormapRange()
        colormap = self.getColormap()
        if colormap is not None:
            vRange = colormap.getVRange()
            autoMin, autoMax = (r is None for r in vRange)
        else:
            autoMin, autoMax = False, False

        with utils.blockSignals(self._minValue):
            self._minValue.setValue(xmin, autoMin)
        with utils.blockSignals(self._maxValue):
            self._maxValue.setValue(xmax, autoMax)
        with utils.blockSignals(self._histoWidget):
            self._histoWidget.setFiniteRange((xmin, xmax))
        with utils.blockSignals(self._autoButtons):
            self._autoButtons.setAutoRange((autoMin, autoMax))

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
        if self._colormapChange.locked():
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
        self._invalidateColormap()

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
        if self._colormapChange.locked():
            return

        self._syncScaleToButtonsEnabled()

        colormap = self.getColormap()
        if colormap is None:
            self._comboBoxColormap.setEnabled(False)
            self._comboBoxNormalization.setEnabled(False)
            self._gammaSpinBox.setEnabled(False)
            self._autoScaleCombo.setEnabled(False)
            self._minValue.setEnabled(False)
            self._maxValue.setEnabled(False)
            self._autoButtons.setEnabled(False)
            self._histoWidget.setVisible(False)
            self._histoWidget.setFiniteRange((None, None))
        else:
            assert colormap.getNormalization() in Colormap.NORMALIZATIONS
            with utils.blockSignals(self._comboBoxColormap):
                self._comboBoxColormap.setCurrentLut(colormap)
                self._comboBoxColormap.setEnabled(colormap.isEditable())
            with utils.blockSignals(self._comboBoxNormalization):
                index = self._comboBoxNormalization.findData(
                    colormap.getNormalization())
                if index < 0:
                    _logger.error('Unsupported normalization: %s' %
                                  colormap.getNormalization())
                else:
                    self._comboBoxNormalization.setCurrentIndex(index)
                self._comboBoxNormalization.setEnabled(colormap.isEditable())
            with utils.blockSignals(self._gammaSpinBox):
                self._gammaSpinBox.setValue(
                    colormap.getGammaNormalizationParameter())
                self._gammaSpinBox.setEnabled(
                    colormap.getNormalization() == Colormap.GAMMA and
                    colormap.isEditable())
            with utils.blockSignals(self._autoScaleCombo):
                self._autoScaleCombo.setCurrentMode(colormap.getAutoscaleMode())
                self._autoScaleCombo.setEnabled(colormap.isEditable())
            with utils.blockSignals(self._autoButtons):
                self._autoButtons.setEnabled(colormap.isEditable())
                self._autoButtons.setAutoRangeFromColormap(colormap)

            vmin, vmax = colormap.getVRange()
            if vmin is None or vmax is None:
                # Compute it only if needed
                dataRange = self._getFiniteColormapRange()
            else:
                dataRange = vmin, vmax

            with utils.blockSignals(self._minValue):
                self._minValue.setValue(vmin or dataRange[0], isAuto=vmin is None)
                self._minValue.setEnabled(colormap.isEditable())
            with utils.blockSignals(self._maxValue):
                self._maxValue.setValue(vmax or dataRange[1], isAuto=vmax is None)
                self._maxValue.setEnabled(colormap.isEditable())

            with utils.blockSignals(self._histoWidget):
                self._histoWidget.setVisible(True)
                self._histoWidget.setFiniteRange(dataRange)
                self._histoWidget.updateNormalization()

    def _comboBoxColormapUpdated(self):
        """Callback executed when the combo box with the colormap LUT
        is updated by user input.
        """
        colormap = self.getColormap()
        if colormap is not None:
            with self._colormapChange:
                name = self._comboBoxColormap.getCurrentName()
                if name is not None:
                    colormap.setName(name)
                else:
                    lut = self._comboBoxColormap.getCurrentColors()
                    colormap.setColormapLUT(lut)
        self._histoWidget.updateLut()

    def _autoRangeButtonsUpdated(self, autoRange):
        """Callback executed when the autoscale buttons widget
        is updated by user input.
        """
        dataRange = self._getFiniteColormapRange()

        # Final colormap range
        vmin = (dataRange[0] if not autoRange[0] else None)
        vmax = (dataRange[1] if not autoRange[1] else None)

        with self._colormapChange:
            colormap = self.getColormap()
            colormap.setVRange(vmin, vmax)

        with utils.blockSignals(self._minValue):
            self._minValue.setValue(vmin or dataRange[0], isAuto=vmin is None)
        with utils.blockSignals(self._maxValue):
            self._maxValue.setValue(vmax or dataRange[1], isAuto=vmax is None)

        self._updateWidgetRange()

    def _normalizationUpdated(self, index):
        """Callback executed when the normalization widget
        is updated by user input.
        """
        colormap = self.getColormap()
        if colormap is not None:
            normalization = self._comboBoxNormalization.itemData(index)
            self._gammaSpinBox.setEnabled(normalization == 'gamma')

            with self._colormapChange:
                colormap.setNormalization(normalization)
                self._histoWidget.updateNormalization()

        self._updateWidgetRange()

    def _gammaUpdated(self, value):
        """Callback used to update the gamma normalization parameter"""
        colormap = self.getColormap()
        if colormap is not None:
            colormap.setGammaNormalizationParameter(value)

    def _autoscaleModeUpdated(self):
        """Callback executed when the autoscale mode widget
        is updated by user input.
        """
        mode = self._autoScaleCombo.currentMode()

        colormap = self.getColormap()
        if colormap is not None:
            with self._colormapChange:
                colormap.setAutoscaleMode(mode)

        self._updateWidgetRange()

    def _minAutoscaleUpdated(self, autoEnabled):
        """Callback executed when the min autoscale from
        the lineedit is updated by user input"""
        colormap = self.getColormap()
        xmin, xmax = colormap.getVRange()
        if autoEnabled:
            xmin = None
        else:
            xmin, _xmax = self._getFiniteColormapRange()
        self._setColormapRange(xmin, xmax)

    def _maxAutoscaleUpdated(self, autoEnabled):
        """Callback executed when the max autoscale from
        the lineedit is updated by user input"""
        colormap = self.getColormap()
        xmin, xmax = colormap.getVRange()
        if autoEnabled:
            xmax = None
        else:
            _xmin, xmax = self._getFiniteColormapRange()
        self._setColormapRange(xmin, xmax)

    def _minValueUpdated(self, value):
        """Callback executed when the lineedit min value is
        updated by user input"""
        xmin = value
        xmax = self._maxValue.getValue()
        if xmax is not None and xmin > xmax:
            # FIXME: This should be done in the widget itself
            xmin = xmax
            with utils.blockSignals(self._minValue):
                self._minValue.setValue(xmin)
        self._setColormapRange(xmin, xmax)

    def _maxValueUpdated(self, value):
        """Callback executed when the lineedit max value is
        updated by user input"""
        xmin = self._minValue.getValue()
        xmax = value
        if xmin is not None and xmin > xmax:
            # FIXME: This should be done in the widget itself
            xmax = xmin
            with utils.blockSignals(self._maxValue):
                self._maxValue.setValue(xmax)
        self._setColormapRange(xmin, xmax)

    def _histogramRangeMoving(self, vmin, vmax, gammaPos):
        """Callback executed when for colormap range displayed in
        the histogram widget is moving.

        :param vmin: Update of the minimum range, else None
        :param vmax: Update of the maximum range, else None
        :param gammaPos: Update of the gamma location, else None
        """
        colormap = self.getColormap()
        if vmin is not None:
            with self._colormapChange:
                colormap.setVMin(vmin)
            self._minValue.setValue(vmin)
        if vmax is not None:
            with self._colormapChange:
                colormap.setVMax(vmax)
            self._maxValue.setValue(vmax)
        if gammaPos is not None:
            vmin, vmax = self._histoWidget.getFiniteRange()
            if vmax < vmin:
                gamma = 1
            elif gammaPos >= vmax:
                gamma = self._gammaSpinBox.maximum()
            elif gammaPos <= vmin:
                gamma = self._gammaSpinBox.minimum()
            else:
                gamma = numpy.clip(
                    numpy.log(0.5)/numpy.log((gammaPos - vmin) / (vmax - vmin)),
                    self._gammaSpinBox.minimum(),
                    self._gammaSpinBox.maximum(),
                )
            with self._colormapChange:
                colormap.setGammaNormalizationParameter(gamma)
            with utils.blockSignals(self._gammaSpinBox):
                self._gammaSpinBox.setValue(gamma)

    def _histogramRangeMoved(self, vmin, vmax, gammaPos):
        """Callback executed when for colormap range displayed in
        the histogram widget has finished to move
        """
        if vmin is None and vmax is None:
            return
        xmin = self._minValue.getValue()
        xmax = self._maxValue.getValue()
        if vmin is None:
            vmin = xmin
        if vmax is None:
            vmax = xmax
        self._setColormapRange(vmin, vmax)

    def _syncScaleToButtonsEnabled(self):
        """Set the state of scale to buttons according to current item and colormap"""
        colormap = self.getColormap()
        enabled = self._item is not None and colormap is not None and colormap.isEditable()
        self._scaleToAreaGroup.setVisible(enabled)
        self._visibleAreaButton.setEnabled(enabled)
        if not enabled:
            self._selectedAreaButton.setChecked(False)
        self._selectedAreaButton.setEnabled(enabled)

    def _handleScaleToVisibleAreaClicked(self):
        """Set colormap range from current item's visible area"""
        item = self._getItem()
        if item is None:
            return  # no-op

        bounds = item.getVisibleBounds()
        if bounds is None:
            return  # no-op

        self.setColormapRangeFromDataBounds(bounds)

    def _handleScaleToSelectionToggled(self, checked=False):
        """Handle toggle of scale to selected are button"""
        # Reset any previous ROI manager
        if self._roiForColormapManager is not None:
            self._roiForColormapManager.clear()
            self._roiForColormapManager.stop()
            self._roiForColormapManager = None

        if not checked:  # Reset button status
            self._selectedAreaButton.setText("Selection")
            return

        item = self._getItem()
        if item is None:
            self._selectedAreaButton.setChecked(False)
            return  # no-op

        plotWidget = item.getPlot()
        if plotWidget is None:
            self._selectedAreaButton.setChecked(False)
            return  # no-op

        self._selectedAreaButton.setText("Draw Area...")

        self._roiForColormapManager = RegionOfInterestManager(parent=plotWidget)
        cmap = self.getColormap()
        self._roiForColormapManager.setColor(
            'black' if cmap is None else cursorColorForColormap(cmap.getName()))
        self._roiForColormapManager.sigInteractiveModeFinished.connect(
            self.__roiInteractiveModeFinished)
        self._roiForColormapManager.sigInteractiveRoiFinalized.connect(self.__roiFinalized)
        self._roiForColormapManager.start(RectangleROI)

    def __roiInteractiveModeFinished(self):
        self._selectedAreaButton.setChecked(False)

    def __roiFinalized(self, roi):
        if roi is not None:
            ox, oy = roi.getOrigin()
            width, height = roi.getSize()
            self.setColormapRangeFromDataBounds((ox, ox+width, oy, oy+height))
            # clear ROI
            self._roiForColormapManager.removeRoi(roi)

    def keyPressEvent(self, event):
        """Override key handling.

        It disables leaving the dialog when editing a text field.

        But several press of Return key can be use to validate and close the
        dialog.
        """
        if event.key() in (qt.Qt.Key_Enter, qt.Qt.Key_Return):
            # Bypass QDialog keyPressEvent
            # To avoid leaving the dialog when pressing enter on a text field
            if self._minValue.hasFocus():
                nextFocus = self._maxValue
            elif self._maxValue.hasFocus():
                if self.isModal():
                    nextFocus = self._buttonsModal.button(qt.QDialogButtonBox.Apply)
                else:
                    nextFocus = self._buttonsNonModal.button(qt.QDialogButtonBox.Close)
            else:
                nextFocus = None
            if nextFocus is not None:
                nextFocus.setFocus(qt.Qt.OtherFocusReason)
        else:
            super(ColormapDialog, self).keyPressEvent(event)
