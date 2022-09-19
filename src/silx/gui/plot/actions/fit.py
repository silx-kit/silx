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
:mod:`silx.gui.plot.actions.fit` module provides actions relative to fit.

The following QAction are available:

- :class:`.FitAction`

.. autoclass:`.FitAction`
"""

__authors__ = ["V.A. Sole", "T. Vincent", "P. Knobel"]
__license__ = "MIT"
__date__ = "10/10/2018"

import logging
import sys
import weakref
import numpy

from .PlotToolAction import PlotToolAction
from .. import items
from ....utils.deprecation import deprecated
from silx.gui import qt
from silx.gui.plot.ItemsSelectionDialog import ItemsSelectionDialog

_logger = logging.getLogger(__name__)


def _getUniqueCurveOrHistogram(plot):
    """Returns unique :class:`Curve` or :class:`Histogram` in a `PlotWidget`.

    If there is an active curve, returns it, else return curve or histogram
    only if alone in the plot.

    :param PlotWidget plot:
    :rtype: Union[None,~silx.gui.plot.items.Curve,~silx.gui.plot.items.Histogram]
    """
    curve = plot.getActiveCurve()
    if curve is not None:
        return curve

    visibleItems = [item for item in plot.getItems() if item.isVisible()]
    histograms = [item for item in visibleItems
                  if isinstance(item, items.Histogram)]
    curves = [item for item in visibleItems
              if isinstance(item, items.Curve)]

    if len(histograms) == 1 and len(curves) == 0:
        return histograms[0]
    elif len(curves) == 1 and len(histograms) == 0:
        return curves[0]
    else:
        return None


class _FitItemSelector(qt.QObject):
    """
    :class:`PlotWidget` observer that emits signal when fit selection changes.

    Track active curve or unique curve or histogram.
    """

    sigCurrentItemChanged = qt.Signal(object)
    """Signal emitted when the item to fit has changed"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__plotWidgetRef = None
        self.__currentItem = None

    def getCurrentItem(self):
        """Return currently selected item

        :rtype: Union[Item,None]
        """
        return self.__currentItem

    def getPlotWidget(self):
        """Return currently attached :class:`PlotWidget`

        :rtype: Union[PlotWidget,None]
        """
        return None if self.__plotWidgetRef is None else self.__plotWidgetRef()

    def setPlotWidget(self, plotWidget):
        """Set the :class:`PlotWidget` for which to track changes

        :param Union[PlotWidget,None] plotWidget:
            The :class:`PlotWidget` to observe
        """
        # disconnect from previous plot
        previousPlotWidget = self.getPlotWidget()
        if previousPlotWidget is not None:
            previousPlotWidget.sigItemAdded.disconnect(
                self.__plotWidgetUpdated)
            previousPlotWidget.sigItemRemoved.disconnect(
                self.__plotWidgetUpdated)
            previousPlotWidget.sigActiveCurveChanged.disconnect(
                self.__plotWidgetUpdated)

        if plotWidget is None:
            self.__plotWidgetRef = None
            self.__setCurrentItem(None)
            return
        self.__plotWidgetRef = weakref.ref(plotWidget, self.__plotDeleted)

        # connect to new plot
        plotWidget.sigItemAdded.connect(self.__plotWidgetUpdated)
        plotWidget.sigItemRemoved.connect(self.__plotWidgetUpdated)
        plotWidget.sigActiveCurveChanged.connect(self.__plotWidgetUpdated)
        self.__plotWidgetUpdated()

    def __plotDeleted(self):
        """Handle deletion of PlotWidget"""
        self.__setCurrentItem(None)

    def __plotWidgetUpdated(self, *args, **kwargs):
        """Handle updates of PlotWidget content"""
        plotWidget = self.getPlotWidget()
        if plotWidget is None:
            return
        self.__setCurrentItem(_getUniqueCurveOrHistogram(plotWidget))

    def __setCurrentItem(self, item):
        """Handle change of current item"""
        if sys.is_finalizing():
            return

        previousItem = self.getCurrentItem()
        if item != previousItem:
            if previousItem is not None:
                previousItem.sigItemChanged.disconnect(self.__itemUpdated)

            self.__currentItem = item

            if self.__currentItem is not None:
                self.__currentItem.sigItemChanged.connect(self.__itemUpdated)
            self.sigCurrentItemChanged.emit(self.__currentItem)

    def __itemUpdated(self, event):
        """Handle change on current item"""
        if event == items.ItemChangedType.DATA:
            self.sigCurrentItemChanged.emit(self.__currentItem)


class FitAction(PlotToolAction):
    """QAction to open a :class:`FitWidget` and set its data to the
    active curve if any, or to the first curve.

    :param plot: :class:`.PlotWidget` instance on which to operate
    :param parent: See :class:`QAction`
    """

    def __init__(self, plot, parent=None):
        self.__item = None
        self.__activeCurveSynchroEnabled = False
        self.__range = 0, 1
        self.__rangeAutoUpdate = False
        self.__x, self.__y = None, None  # Data to fit
        self.__curveParams = {}  # Store curve parameters to use for fit result
        self.__legend = None

        super(FitAction, self).__init__(
            plot, icon='math-fit', text='Fit curve',
            tooltip='Open a fit dialog',
            parent=parent)

        self.__fitItemSelector = _FitItemSelector()
        self.__fitItemSelector.sigCurrentItemChanged.connect(
            self._setFittedItem)


    @property
    @deprecated(replacement='getXRange()[0]', since_version='0.13.0')
    def xmin(self):
        return self.getXRange()[0]

    @property
    @deprecated(replacement='getXRange()[1]', since_version='0.13.0')
    def xmax(self):
        return self.getXRange()[1]

    @property
    @deprecated(replacement='getXData()', since_version='0.13.0')
    def x(self):
        return self.getXData()

    @property
    @deprecated(replacement='getYData()', since_version='0.13.0')
    def y(self):
        return self.getYData()

    @property
    @deprecated(since_version='0.13.0')
    def xlabel(self):
        return self.__curveParams.get('xlabel', None)

    @property
    @deprecated(since_version='0.13.0')
    def ylabel(self):
        return self.__curveParams.get('ylabel', None)

    @property
    @deprecated(since_version='0.13.0')
    def legend(self):
        return self.__legend

    def _createToolWindow(self):
        # import done here rather than at module level to avoid circular import
        # FitWidget -> BackgroundWidget -> PlotWindow -> actions -> fit -> FitWidget
        from ...fit.FitWidget import FitWidget

        window = FitWidget(parent=self.plot)
        window.setWindowFlags(qt.Qt.Dialog)
        window.sigFitWidgetSignal.connect(self.handle_signal)
        return window

    def _connectPlot(self, window):
        if self.isXRangeUpdatedOnZoom():
            self.__setAutoXRangeEnabled(True)
        else:
            plot = self.plot
            if plot is None:
                _logger.error("No associated PlotWidget")
                return
            self._setXRange(*plot.getXAxis().getLimits())

        if self.isFittedItemUpdatedFromActiveCurve():
            self.__setFittedItemAutoUpdateEnabled(True)
        else:
            # Wait for the next iteration, else the plot is not yet initialized
            # No curve available
            qt.QTimer.singleShot(10, self._initFit)

    def _disconnectPlot(self, window):
        if self.isXRangeUpdatedOnZoom():
            self.__setAutoXRangeEnabled(False)

        if self.isFittedItemUpdatedFromActiveCurve():
            self.__setFittedItemAutoUpdateEnabled(False)

    def _initFit(self):
        plot = self.plot
        if plot is None:
            _logger.error("No associated PlotWidget")
            return

        item = _getUniqueCurveOrHistogram(plot)
        if item is None:
            # ambiguous case, we need to ask which plot item to fit
            isd = ItemsSelectionDialog(parent=plot, plot=plot)
            isd.setWindowTitle("Select item to be fitted")
            isd.setItemsSelectionMode(qt.QTableWidget.SingleSelection)
            isd.setAvailableKinds(["curve", "histogram"])
            isd.selectAllKinds()

            if not isd.exec():  # Cancel
                self._getToolWindow().setVisible(False)
            else:
                selectedItems = isd.getSelectedItems()
                item = selectedItems[0] if len(selectedItems) == 1 else None

        self._setXRange(*plot.getXAxis().getLimits())
        self._setFittedItem(item)

    def __updateFitWidget(self):
        """Update the data/range used by the FitWidget"""
        fitWidget = self._getToolWindow()

        item = self._getFittedItem()
        xdata = self.getXData(copy=False)
        ydata = self.getYData(copy=False)
        if item is None or xdata is None or ydata is None:
            fitWidget.setData(y=None)
            fitWidget.setWindowTitle("No curve selected")

        else:
            xmin, xmax = self.getXRange()
            fitWidget.setData(
                xdata, ydata, xmin=xmin, xmax=xmax)
            fitWidget.setWindowTitle(
                "Fitting " + item.getName() +
                " on x range %f-%f" % (xmin, xmax))

    # X Range management

    def getXRange(self):
        """Returns the range on the X axis on which to perform the fit."""
        return self.__range

    def _setXRange(self, xmin, xmax):
        """Set the range on which the fit is done.

        :param float xmin:
        :param float xmax:
        """
        range_ = float(xmin), float(xmax)
        if self.__range != range_:
            self.__range = range_
            self.__updateFitWidget()

    def __setAutoXRangeEnabled(self, enabled):
        """Implement the change of update mode of the X range.

        :param bool enabled:
        """
        plot = self.plot
        if plot is None:
            _logger.error("No associated PlotWidget")
            return

        if enabled:
            self._setXRange(*plot.getXAxis().getLimits())
            plot.getXAxis().sigLimitsChanged.connect(self._setXRange)
        else:
            plot.getXAxis().sigLimitsChanged.disconnect(self._setXRange)

    def setXRangeUpdatedOnZoom(self, enabled):
        """Set whether or not to update the X range on zoom change.

        :param bool enabled:
        """
        if enabled != self.__rangeAutoUpdate:
            self.__rangeAutoUpdate = enabled
            if self._getToolWindow().isVisible():
                self.__setAutoXRangeEnabled(enabled)

    def isXRangeUpdatedOnZoom(self):
        """Returns the current mode of fitted data X range update.

        :rtype: bool
        """
        return self.__rangeAutoUpdate

    # Fitted item update

    def getXData(self, copy=True):
        """Returns the X data used for the fit or None if undefined.

        :param bool copy:
            True to get a copy of the data, False to get the internal data.
        :rtype: Union[numpy.ndarray,None]
        """
        return None if self.__x is None else numpy.array(self.__x, copy=copy)

    def getYData(self, copy=True):
        """Returns the Y data used for the fit or None if undefined.

        :param bool copy:
            True to get a copy of the data, False to get the internal data.
        :rtype: Union[numpy.ndarray,None]
        """
        return None if self.__y is None else numpy.array(self.__y, copy=copy)

    def _getFittedItem(self):
        """Returns the current item used for the fit

        :rtype: Union[~silx.gui.plot.items.Curve,~silx.gui.plot.items.Histogram,None]
        """
        return self.__item

    def _setFittedItem(self, item):
        """Set the curve to use for fitting.

        :param Union[~silx.gui.plot.items.Curve,~silx.gui.plot.items.Histogram,None] item:
        """
        plot = self.plot
        if plot is None:
            _logger.error("No associated PlotWidget")

        if plot is None or item is None:
            self.__item = None
            self.__curveParams = {}
            self.__updateFitWidget()
            return

        axis = item.getYAxis() if isinstance(item, items.YAxisMixIn) else 'left'
        self.__curveParams = {
            'yaxis': axis,
            'xlabel': plot.getXAxis().getLabel(),
            'ylabel': plot.getYAxis(axis).getLabel(),
            }
        self.__legend = item.getName()

        if isinstance(item, items.Histogram):
            bin_edges = item.getBinEdgesData(copy=False)
            # take the middle coordinate between adjacent bin edges
            self.__x = (bin_edges[1:] + bin_edges[:-1]) / 2
            self.__y = item.getValueData(copy=False)
        # else take the active curve, or else the unique curve
        elif isinstance(item, items.Curve):
            self.__x = item.getXData(copy=False)
            self.__y = item.getYData(copy=False)

        self.__item  = item
        self.__updateFitWidget()

    def __setFittedItemAutoUpdateEnabled(self, enabled):
        """Implement the change of fitted item update mode

        :param bool enabled:
        """
        plot = self.plot
        if plot is None:
            _logger.error("No associated PlotWidget")
            return

        self.__fitItemSelector.setPlotWidget(self.plot if enabled else None)

    def setFittedItemUpdatedFromActiveCurve(self, enabled):
        """Toggle fitted data synchronization with plot active curve.

        :param bool enabled:
        """
        enabled = bool(enabled)
        if enabled != self.__activeCurveSynchroEnabled:
            self.__activeCurveSynchroEnabled = enabled
            if self._getToolWindow().isVisible():
                self.__setFittedItemAutoUpdateEnabled(enabled)

    def isFittedItemUpdatedFromActiveCurve(self):
        """Returns True if fitted data is synchronized with plot.

        :rtype: bool
        """
        return self.__activeCurveSynchroEnabled

    # Handle fit completed

    def handle_signal(self, ddict):
        xdata = self.getXData(copy=False)
        if xdata is None:
            _logger.error("No reference data to display fit result for")
            return

        xmin, xmax = self.getXRange()
        x_fit = xdata[xmin <= xdata]
        x_fit = x_fit[x_fit <= xmax]
        fit_legend = "Fit <%s>" % self.__legend
        fit_curve = self.plot.getCurve(fit_legend)

        if ddict["event"] == "FitFinished":
            fit_widget = self._getToolWindow()
            if fit_widget is None:
                return
            y_fit = fit_widget.fitmanager.gendata()
            if fit_curve is None:
                self.plot.addCurve(x_fit, y_fit,
                                   fit_legend,
                                   resetzoom=False,
                                   **self.__curveParams)
            else:
                fit_curve.setData(x_fit, y_fit)
                fit_curve.setVisible(True)
                fit_curve.setYAxis(self.__curveParams.get('yaxis', 'left'))

        if ddict["event"] in ["FitStarted", "FitFailed"]:
            if fit_curve is not None:
                fit_curve.setVisible(False)
