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
    def __init__(self, plot, parent=None):
        self.__synchroEnabled = False
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
        if self.isCurveSynchronized():
            self.__setPlotSynchroEnabled(True)
        else:
            # Wait for the next iteration, else the plot is not yet initialized
            # No curve available
            qt.QTimer.singleShot(10, self._initFit)

    def _disconnectPlot(self, window):
        if self.isCurveSynchronized():
            self.__setPlotSynchroEnabled(False)

    def __setPlotSynchroEnabled(self, enabled):
        if enabled:
            currentCurve = self.plot.getActiveCurve()
            self._setCurve(currentCurve)
            self.plot.sigActiveCurveChanged.connect(self.__activeCurveChanged)
        else:
            self.plot.sigActiveCurveChanged.disconnect(
                self.__activeCurveChanged)

    def setCurveSynchronized(self, enabled):
        """Enable/Disable synchronization of fitted data with plot active curve.

        :param bool enabled:
        """
        enabled = bool(enabled)
        if enabled != self.__synchroEnabled:
            self.__synchroEnabled = enabled
            if self._getToolWindow().isVisible():
                self.__setPlotSynchroEnabled(enabled=enabled)

    def isCurveSynchronized(self):
        """Returns True if fitted data is synchronized with plot active curve.

        :rtype: bool
        """
        return self.__synchroEnabled

    def __activeCurveChanged(self, previous, current):
        """Handle change of active curve in the PlotWidget
        """
        if current is None:
            self._setCurve(None)
        else:
            item = self.plot.getCurve(current)
            self._setCurve(item)

    def _setCurve(self, item):
        """Set the curve to use for fitting.

        :param ~silx.gui.plot.items.Curve item:
        """
        fitWidget = self._getToolWindow()

        if item is None:
            fitWidget.setData(y=None)
            fitWidget.setWindowTitle("- No curve selected -")
            return

        plot = self.plot
        if plot is None:
            return

        self.xlabel = plot.getXAxis().getLabel()
        self.ylabel = plot.getYAxis().getLabel() # TODO fit on right axis?
        self.xmin, self.xmax = plot.getXAxis().getLimits()

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

        fitWidget.setData(
            self.x, self.y, xmin=self.xmin, xmax=self.xmax)
        fitWidget.setWindowTitle(
            "Fitting " + self.legend +
            " on x range %f-%f" % (self.xmin, self.xmax))

    def _initFit(self):
        plot = self.plot
        if plot is None:
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

        self._setCurve(item)

    def handle_signal(self, ddict):
        x_fit = self.x[self.xmin <= self.x]
        x_fit = x_fit[x_fit <= self.xmax]
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
