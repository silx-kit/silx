# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2004-2021 European Synchrotron Radiation Facility
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
"""
:mod:`silx.gui.plot.actions.histogram` provides actions relative to histograms
for :class:`.PlotWidget`.

The following QAction are available:

- :class:`PixelIntensitiesHistoAction`
"""

from __future__ import division

__authors__ = ["V.A. Sole", "T. Vincent", "P. Knobel"]
__date__ = "01/12/2020"
__license__ = "MIT"

import numpy
import logging
import typing
import weakref

from .PlotToolAction import PlotToolAction

from silx.math.histogram import Histogramnd
from silx.math.combo import min_max
from silx.gui import qt
from silx.gui.plot import items
from silx.gui.widgets.ElidedLabel import ElidedLabel
from silx.gui.widgets.RangeSlider import RangeSlider
from silx.utils.deprecation import deprecated

_logger = logging.getLogger(__name__)


class _ElidedLabel(ElidedLabel):
    """QLabel with a default size larger than what is displayed."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setTextInteractionFlags(qt.Qt.TextSelectableByMouse)

    def sizeHint(self):
        hint = super().sizeHint()
        nbchar = max(len(self.getText()), 12)
        width = self.fontMetrics().boundingRect('#' * nbchar).width()
        return qt.QSize(max(hint.width(), width), hint.height())


class _StatWidget(qt.QWidget):
    """Widget displaying a name and a value

    :param parent:
    :param name:
    """

    def __init__(self, parent=None, name: str=''):
        super().__init__(parent)
        layout = qt.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        keyWidget = qt.QLabel(parent=self)
        keyWidget.setText("<b>" + name.capitalize() + ":<b>")
        layout.addWidget(keyWidget)
        self.__valueWidget = _ElidedLabel(parent=self)
        self.__valueWidget.setText("-")
        self.__valueWidget.setTextInteractionFlags(
            qt.Qt.TextSelectableByMouse | qt.Qt.TextSelectableByKeyboard)
        layout.addWidget(self.__valueWidget)

    def setValue(self, value: typing.Optional[float]):
        """Set the displayed value

        :param value:
        """
        self.__valueWidget.setText(
            "-" if value is None else "{:.5g}".format(value))


class _IntEdit(qt.QLineEdit):
    """QLineEdit tuned for editing integers

    :param QWidget parent:
    """

    sigValueChanged = qt.Signal(int)
    """Signal emitted when the value has changed (on editing finished)"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(qt.Qt.AlignRight)
        validator = qt.QIntValidator()
        self.setValidator(validator)
        validator.bottomChanged.connect(self.__updateSize)
        validator.topChanged.connect(self.__updateSize)
        self.__updateSize()

        self.editingFinished.connect(self.__editingFinished)

    def __editingFinished(self):
        value = self.getValue()
        if value is not None:
            self.sigValueChanged.emit(value)

    def __updateSize(self, *args):
        """Update widget's maximum size according to bounds"""
        bottom, top = self.getRange()
        nbchar = max(len(str(bottom)), len(str(top)))
        self.setMaximumWidth(
            self.fontMetrics().boundingRect('0' * (nbchar + 1)).width()
        )

    def setRange(self, bottom: int, top: int):
        """Set the range of valid values"""
        self.validator().setRange(bottom, top)

    def getRange(self):
        """Returns the current range of valid values

        :returns: (bottom, top)
        :rtype: List[int]"""
        return self.validator().bottom(), self.validator().top()

    def setValue(self, value: int):
        """Set the current value"""
        self.setText(str(numpy.clip(value, *self.getRange())))

    def getValue(self):
        """Returns the current value or None if invalid

        :rtype: Union[int,None]
        """
        text = self.text()
        try:
            return int(text)
        except ValueError:
            return None


class HistogramWidget(qt.QWidget):
    """Widget displaying a histogram and some statistic indicators"""

    _SUPPORTED_ITEM_CLASS = items.ImageBase, items.Scatter

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle('Histogram')

        self.__itemRef = None  # weakref on the item to track

        layout = qt.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Plot
        # Lazy import to avoid circular dependencies
        from silx.gui.plot.PlotWindow import Plot1D
        self.__plot = Plot1D(self)
        layout.addWidget(self.__plot)

        self.__plot.setDataMargins(0.1, 0.1, 0.1, 0.1)
        self.__plot.getXAxis().setLabel("Value")
        self.__plot.getYAxis().setLabel("Count")
        posInfo = self.__plot.getPositionInfoWidget()
        posInfo.setSnappingMode(posInfo.SNAPPING_CURVE)

        # Histogram controls
        controlsWidget = qt.QWidget(self)
        layout.addWidget(controlsWidget)
        controlsLayout = qt.QHBoxLayout(controlsWidget)
        controlsLayout.setContentsMargins(4, 4, 4, 4)

        controlsLayout.addWidget(qt.QLabel("<b>Histogram:<b>"))
        controlsLayout.addWidget(qt.QLabel("N. bins:"))
        self.__nbinsLineEdit = _IntEdit(self)
        self.__nbinsLineEdit.setRange(2, 9999)
        self.__nbinsLineEdit.sigValueChanged.connect(self.__nBinsChanged)
        controlsLayout.addWidget(self.__nbinsLineEdit)
        self.__rangeLabel = qt.QLabel("Range:")
        controlsLayout.addWidget(self.__rangeLabel)
        self.__rangeSlider = RangeSlider(parent=self)
        self.__rangeSlider.sigValueChanged.connect(self.__rangeChanged)
        controlsLayout.addWidget(self.__rangeSlider)
        controlsLayout.addStretch(1)

        # Stats display
        statsWidget = qt.QWidget(self)
        layout.addWidget(statsWidget)
        statsLayout = qt.QHBoxLayout(statsWidget)
        statsLayout.setContentsMargins(4, 4, 4, 4)

        self.__statsWidgets = dict(
            (name, _StatWidget(parent=statsWidget, name=name))
            for name in ("min", "max", "mean", "std", "sum"))

        for widget in self.__statsWidgets.values():
            statsLayout.addWidget(widget)
        statsLayout.addStretch(1)

    def getPlotWidget(self):
        """Returns :class:`PlotWidget` use to display the histogram"""
        return self.__plot

    def resetZoom(self):
        """Reset PlotWidget zoom"""
        self.getPlotWidget().resetZoom()

    def reset(self):
        """Clear displayed information"""
        self.getPlotWidget().clear()
        self.setStatistics()

    def getItem(self) -> typing.Optional[items.Item]:
        """Returns item used to display histogram and statistics."""
        return None if self.__itemRef is None else self.__itemRef()

    def setItem(self, item: typing.Optional[items.Item]):
        """Set item from which to display histogram and statistics.

        :param item:
        """
        previous = self.getItem()
        if previous is not None:
            previous.sigItemChanged.disconnect(self.__itemChanged)

        self.__itemRef = None if item is None else weakref.ref(item)
        if item is not None:
            if isinstance(item, self._SUPPORTED_ITEM_CLASS):
                # Only listen signal for supported items
                item.sigItemChanged.connect(self.__itemChanged)
        self._updateFromItem()

    def __itemChanged(self, event):
        """Handle update of the item"""
        if event in (items.ItemChangedType.DATA, items.ItemChangedType.MASK):
            self._updateFromItem()

    def __updateHistogramFromControls(self, nbins=None, range_=None):
        """Handle udates coming from histogram control widgets"""
        if nbins is None:
            nbins = self.__nbinsLineEdit.getValue()
        if range_ is None:
            range_ = self.__rangeSlider.getValues()

        hist = self.getHistogram(copy=False)
        if hist is not None:
            count, edges = hist
            if nbins == len(count) and range_ == (edges[0], edges[-1]):
                return  # Nothing has changed

        self._updateFromItem(nbins=nbins, range_=range_)

    def __nBinsChanged(self, nbins):
        """Handle change of nbins from the int edit"""
        self.__updateHistogramFromControls(nbins=nbins)

    def __rangeChanged(self, first, second):
        """Handle change of histogram range from the range slider"""
        tooltip = "Histogram range:\n[%g, %g]" % (first, second)
        self.__rangeSlider.setToolTip(tooltip)
        self.__rangeLabel.setToolTip(tooltip)
        self.__updateHistogramFromControls(range_=(first, second))

    def _updateFromItem(self, nbins=None, range_=None):
        """Update histogram and stats from the item

        :param Union[int,None] nbins:
            Number of histogram bins. None (default) to guess it.
        :param Union[Tuple[int],None] range_:
            Histogram range. None (default) to use data range.
        """
        item = self.getItem()

        if item is None:
            self.reset()
            return

        if not isinstance(item, self._SUPPORTED_ITEM_CLASS):
            _logger.error("Unsupported item", item)
            self.reset()
            return

        # Compute histogram and stats
        array = item.getValueData(copy=False)

        if array.size == 0:
            self.reset()
            return

        xmin, xmax = min_max(array, min_positive=False, finite=True)
        if nbins is None:  # Guess nbins
            nbins = min(1024, int(numpy.sqrt(array.size)))
        data_range = (xmin, xmax) if range_ is None else range_

        # bad hack: get 256 bins in the case we have a B&W
        if numpy.issubdtype(array.dtype, numpy.integer):
            if nbins > xmax - xmin:
                nbins = xmax - xmin

        nbins = max(2, nbins)

        data = array.ravel().astype(numpy.float32)
        histogram = Histogramnd(data, n_bins=nbins, histo_range=data_range)
        if len(histogram.edges) != 1:
            _logger.error("Error while computing the histogram")
            self.reset()
            return

        self.__rangeSlider.setRange(xmin, xmax)
        self.setHistogram(histogram.histo, histogram.edges[0])
        self.resetZoom()
        self.setStatistics(
            min_=xmin,
            max_=xmax,
            mean=numpy.nanmean(array),
            std=numpy.nanstd(array),
            sum_=numpy.nansum(array))

    def setHistogram(self, histogram, edges):
        """Set displayed histogram

        :param histogram: Bin values (N)
        :param edges: Bin edges (N+1)
        """
        nbins = len(histogram)
        bottom, top = self.__nbinsLineEdit.getRange()
        self.__nbinsLineEdit.setRange(min(nbins, bottom), max(nbins, top))
        self.__nbinsLineEdit.setValue(nbins)

        self.__rangeSlider.setValues(edges[0], edges[-1])
        self.getPlotWidget().addHistogram(
            histogram=histogram,
            edges=edges,
            legend='histogram',
            fill=True,
            color='#66aad7',
            resetzoom=False)

    def getHistogram(self, copy: bool=True):
        """Returns currently displayed histogram.

        :param copy: True to get a copy,
            False to get internal representation (Do not modify!)
        :return: (histogram, edges) or None
        """
        for item in self.getPlotWidget().getItems():
            if item.getName() == 'histogram':
                return (item.getValueData(copy=copy),
                        item.getBinEdgesData(copy=copy))
        else:
            return None

    def setStatistics(self,
            min_: typing.Optional[float] = None,
            max_: typing.Optional[float] = None,
            mean: typing.Optional[float] = None,
            std: typing.Optional[float] = None,
            sum_: typing.Optional[float] = None):
        """Set displayed statistic indicators."""
        self.__statsWidgets['min'].setValue(min_)
        self.__statsWidgets['max'].setValue(max_)
        self.__statsWidgets['mean'].setValue(mean)
        self.__statsWidgets['std'].setValue(std)
        self.__statsWidgets['sum'].setValue(sum_)


class PixelIntensitiesHistoAction(PlotToolAction):
    """QAction to plot the pixels intensities diagram

    :param plot: :class:`.PlotWidget` instance on which to operate
    :param parent: See :class:`QAction`
    """

    def __init__(self, plot, parent=None):
        PlotToolAction.__init__(self,
                                plot,
                                icon='pixel-intensities',
                                text='pixels intensity',
                                tooltip='Compute image intensity distribution',
                                parent=parent)

    def _connectPlot(self, window):
        plot = self.plot
        if plot is not None:
            selection = plot.selection()
            selection.sigSelectedItemsChanged.connect(self._selectedItemsChanged)
            self._updateSelectedItem()

        PlotToolAction._connectPlot(self, window)

    def _disconnectPlot(self, window):
        plot = self.plot
        if plot is not None:
            selection = self.plot.selection()
            selection.sigSelectedItemsChanged.disconnect(self._selectedItemsChanged)

        PlotToolAction._disconnectPlot(self, window)
        self.getHistogramWidget().setItem(None)

    def _updateSelectedItem(self):
        """Synchronises selected item with plot widget."""
        plot = self.plot
        if plot is not None:
            selected = plot.selection().getSelectedItems()
            # Give priority to image over scatter
            for klass in (items.ImageBase, items.Scatter):
                for item in selected:
                    if isinstance(item, klass):
                        # Found a matching item, use it
                        self.getHistogramWidget().setItem(item)
                        return
        self.getHistogramWidget().setItem(None)

    def _selectedItemsChanged(self):
        if self._isWindowInUse():
            self._updateSelectedItem()

    @deprecated(since_version='0.15.0')
    def computeIntensityDistribution(self):
        self.getHistogramWidget()._updateFromItem()

    def getHistogramWidget(self):
        """Returns the widget displaying the histogram"""
        return self._getToolWindow()

    @deprecated(since_version='0.15.0',
                replacement='getHistogramWidget().getPlotWidget()')
    def getHistogramPlotWidget(self):
        return self._getToolWindow().getPlotWidget()

    def _createToolWindow(self):
        return HistogramWidget(self.plot, qt.Qt.Window)

    def getHistogram(self) -> typing.Optional[numpy.ndarray]:
        """Return the last computed histogram

        :return: the histogram displayed in the HistogramWidget
        """
        histogram = self.getHistogramWidget().getHistogram()
        return None if histogram is None else histogram[0]
