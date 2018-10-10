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

from .PlotToolAction import PlotToolAction
import logging
from silx.gui import qt
from silx.gui.plot.ItemsSelectionDialog import ItemsSelectionDialog
from silx.gui.plot.items import Curve, Histogram

_logger = logging.getLogger(__name__)


def _getUniqueCurve(plt):
    """Get a single curve from the plot.
    Get the active curve if any, else if a single curve is plotted
    get it, else return None.

    :param plt: :class:`.PlotWidget` instance on which to operate

    :return: return value of plt.getActiveCurve(), or plt.getAllCurves()[0],
        or None
    """
    curve = plt.getActiveCurve()
    if curve is not None:
        return curve

    curves = plt.getAllCurves()
    if len(curves) == 0:
        return None

    if len(curves) == 1 and len(plt._getItems(kind='histogram')) == 0:
        return curves[0]

    return None


def _getUniqueHistogram(plt):
    """Return the histogram if there is a single histogram and no curve in
    the plot. In all other cases, return None.

    :param plt: :class:`.PlotWidget` instance on which to operate
    :return: histogram or None
    """
    histograms = plt._getItems(kind='histogram')
    if len(histograms) != 1:
        return None
    if plt.getAllCurves(just_legend=True):
        return None
    return histograms[0]


class FitAction(PlotToolAction):
    """QAction to open a :class:`FitWidget` and set its data to the
    active curve if any, or to the first curve.

    :param plot: :class:`.PlotWidget` instance on which to operate
    :param parent: See :class:`QAction`
    """
    def __init__(self, plot, parent=None):
        super(FitAction, self).__init__(
            plot, icon='math-fit', text='Fit curve',
            tooltip='Open a fit dialog',
            parent=parent)
        self.fit_widget = None

    def _createToolWindow(self):
        window = qt.QMainWindow(parent=self.plot)
        # import done here rather than at module level to avoid circular import
        # FitWidget -> BackgroundWidget -> PlotWindow -> actions -> fit -> FitWidget
        from ...fit.FitWidget import FitWidget
        fit_widget = FitWidget(parent=window)
        window.setCentralWidget(fit_widget)
        fit_widget.guibuttons.DismissButton.clicked.connect(window.close)
        fit_widget.sigFitWidgetSignal.connect(self.handle_signal)
        self.fit_widget = fit_widget
        return window

    def _connectPlot(self, window):
        # Wait for the next iteration, else the plot is not yet initialized
        # No curve available
        qt.QTimer.singleShot(10, lambda: self._initFit(window))

    def _initFit(self, window):
        plot = self.plot
        self.xlabel = plot.getXAxis().getLabel()
        self.ylabel = plot.getYAxis().getLabel()
        self.xmin, self.xmax = plot.getXAxis().getLimits()

        histo = _getUniqueHistogram(self.plot)
        curve = _getUniqueCurve(self.plot)

        if histo is None and curve is None:
            # ambiguous case, we need to ask which plot item to fit
            isd = ItemsSelectionDialog(parent=plot, plot=self.plot)
            isd.setWindowTitle("Select item to be fitted")
            isd.setItemsSelectionMode(qt.QTableWidget.SingleSelection)
            isd.setAvailableKinds(["curve", "histogram"])
            isd.selectAllKinds()

            result = isd.exec_()
            if result and len(isd.getSelectedItems()) == 1:
                item = isd.getSelectedItems()[0]
            else:
                return
        elif histo is not None:
            # presence of a unique histo and no curve
            item = histo
        elif curve is not None:
            # presence of a unique or active curve
            item = curve

        self.legend = item.getLegend()

        if isinstance(item, Histogram):
            bin_edges = item.getBinEdgesData(copy=False)
            # take the middle coordinate between adjacent bin edges
            self.x = (bin_edges[1:] + bin_edges[:-1]) / 2
            self.y = item.getValueData(copy=False)
        # else take the active curve, or else the unique curve
        elif isinstance(item, Curve):
            self.x = item.getXData(copy=False)
            self.y = item.getYData(copy=False)

        self.fit_widget.setData(self.x, self.y,
                                xmin=self.xmin, xmax=self.xmax)
        window.setWindowTitle(
            "Fitting " + self.legend +
            " on x range %f-%f" % (self.xmin, self.xmax))

    def handle_signal(self, ddict):
        x_fit = self.x[self.xmin <= self.x]
        x_fit = x_fit[x_fit <= self.xmax]
        fit_legend = "Fit <%s>" % self.legend
        fit_curve = self.plot.getCurve(fit_legend)

        if ddict["event"] == "FitFinished":
            y_fit = self.fit_widget.fitmanager.gendata()
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
