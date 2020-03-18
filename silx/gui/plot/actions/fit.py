# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2004-2020 European Synchrotron Radiation Facility
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

from __future__ import division

__authors__ = ["V.A. Sole", "T. Vincent", "P. Knobel"]
__license__ = "MIT"
__date__ = "10/10/2018"

import logging

from .PlotToolAction import PlotToolAction
from .. import items
from ....utils.deprecation import deprecated
from silx.gui import qt
from silx.gui.plot.ItemsSelectionDialog import ItemsSelectionDialog
from silx.gui.plot.items import Curve, Histogram

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

    histograms = [item for item in plot.getItems()
                  if isinstance(item, items.Histogram) and item.isVisible()]
    curves = [item for item in plot.getItems()
              if isinstance(item, items.Curve) and item.isVisible()]

    if len(histograms) == 1 and len(curves) == 0:
        return histograms[0]
    elif len(curves) == 1 and len(histograms) == 0:
        return curves[0]
    else:
        return None


class FitAction(PlotToolAction):
    """QAction to open a :class:`FitWidget` and set its data to the
    active curve if any, or to the first curve.

    :param plot: :class:`.PlotWidget` instance on which to operate
    :param parent: See :class:`QAction`
    """

    @property
    @deprecated(replacement='getXRange()[0]', since_version='0.13.0')
    def xmin(self):
        return self.getXRange()[0]

    @property
    @deprecated(replacement='getXRange()[1]', since_version='0.13.0')
    def xmax(self):
        return self.getXRange()[1]

    def __init__(self, plot, parent=None):
        self.__item = None
        self.__activeCurveSynchroEnabled = False
        self.__range = 0, 1
        self.__rangeAutoUpdate = False
        super(FitAction, self).__init__(
            plot, icon='math-fit', text='Fit curve',
            tooltip='Open a fit dialog',
            parent=parent)

    def _createToolWindow(self):
        # import done here rather than at module level to avoid circular import
        # FitWidget -> BackgroundWidget -> PlotWindow -> actions -> fit -> FitWidget
        from ...fit.FitWidget import FitWidget

        window = FitWidget(parent=self.plot)
        window.setWindowFlags(qt.Qt.Window)
        window.sigFitWidgetSignal.connect(self.handle_signal)
        return window

    def _connectPlot(self, window):
        if self.isXRangeUpdatedOnZoom():
            self.__setAutoXRangeEnabled(True)

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
            _logger.error("Associated PlotWidget not available")
            return

        item = _getUniqueCurveOrHistogram(plot)
        if item is None:
            # ambiguous case, we need to ask which plot item to fit
            isd = ItemsSelectionDialog(parent=plot, plot=plot)
            isd.setWindowTitle("Select item to be fitted")
            isd.setItemsSelectionMode(qt.QTableWidget.SingleSelection)
            isd.setAvailableKinds(["curve", "histogram"])
            isd.selectAllKinds()

            result = isd.exec_()
            if result and len(isd.getSelectedItems()) == 1:
                item = isd.getSelectedItems()[0]
            else:
                return

        self._setXRange(*plot.getXAxis().getLimits())
        self._setFittedItem(item)

    def __updateFitWidget(self):
        """Update the data/range used by the FitWidget"""
        fitWidget = self._getToolWindow()

        item = self._getFittedItem()
        if item is None:
            fitWidget.setData(y=None)
            fitWidget.setWindowTitle("- No curve selected -")

        else:
            xmin, xmax = self.getXRange()
            fitWidget.setData(
                self.x, self.y, xmin=xmin, xmax=xmax)
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
            _logger.error("Associated PlotWidget not available")
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

    def _getFittedItem(self):
        """Returns the current item used for the fit

        :rtype: Union[~silx.gui.plot.items.Curve,~silx.gui.plot.items.Histogram,None]
        """
        return self.__item

    def _setFittedItem(self, item):
        """Set the curve to use for fitting.

        :param Union[~silx.gui.plot.items.Curve,~silx.gui.plot.items.Histogram,None] item:
        """
        if item is None:
            self.__item = None
            self.__updateFitWidget()
            return

        plot = self.plot
        if plot is None:
            _logger.error("Associated PlotWidget not available")
            self.__item = None
            self.__updateFitWidget()
            return

        self.xlabel = plot.getXAxis().getLabel()
        self.ylabel = plot.getYAxis().getLabel() # TODO fit on right axis?
        self.legend = item.getName()

        if isinstance(item, Histogram):
            bin_edges = item.getBinEdgesData(copy=False)
            # take the middle coordinate between adjacent bin edges
            self.x = (bin_edges[1:] + bin_edges[:-1]) / 2
            self.y = item.getValueData(copy=False)
        # else take the active curve, or else the unique curve
        elif isinstance(item, Curve):
            self.x = item.getXData(copy=False)
            self.y = item.getYData(copy=False)

        self.__item  = item
        self.__updateFitWidget()

    def __activeCurveChanged(self, previous, current):
        """Handle change of active curve in the PlotWidget
        """
        if current is None:
            self._setFittedItem(None)
        else:
            item = self.plot.getCurve(current)
            self._setFittedItem(item)

    def __setFittedItemAutoUpdateEnabled(self, enabled):
        """Implement the change of fitted item update mode

        :param bool enabled:
        """
        plot = self.plot
        if plot is None:
            _logger.error("Associated PlotWidget not available")
            return

        if enabled:
            self._setFittedItem(plot.getActiveCurve())
            plot.sigActiveCurveChanged.connect(self.__activeCurveChanged)

        else:
            plot.sigActiveCurveChanged.disconnect(
                self.__activeCurveChanged)

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
        xmin, xmax = self.getXRange()
        x_fit = self.x[xmin <= self.x]
        x_fit = x_fit[x_fit <= xmax]
        fit_legend = "Fit <%s>" % self.legend
        fit_curve = self.plot.getCurve(fit_legend)

        if ddict["event"] == "FitFinished":
            fit_widget = self._getToolWindow()
            if fit_widget is None:
                return
            y_fit = fit_widget.fitmanager.gendata()
            if fit_curve is None:
                self.plot.addCurve(x_fit, y_fit,
                                   fit_legend,
                                   xlabel=self.xlabel, ylabel=self.ylabel,
                                   resetzoom=False)
            else:
                fit_curve.setData(x_fit, y_fit)
                fit_curve.setVisible(True)

        if ddict["event"] in ["FitStarted", "FitFailed"]:
            if fit_curve is not None:
                fit_curve.setVisible(False)
