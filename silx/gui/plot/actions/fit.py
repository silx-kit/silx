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
"""This module provides actions relative to fit for a :class:`.PlotWidget`.

The following QAction are available:

- :class:`FitAction`
"""

from __future__ import division

__authors__ = ["V.A. Sole", "T. Vincent", "P. Knobel"]
__license__ = "MIT"
__date__ = "24/05/2017"

from . import PlotAction
import logging
from silx.gui import qt

_logger = logging.getLogger(__name__)


def _warningMessage(informativeText='', detailedText='', parent=None):
    """Display a popup warning message."""
    msg = qt.QMessageBox(parent)
    msg.setIcon(qt.QMessageBox.Warning)
    msg.setInformativeText(informativeText)
    msg.setDetailedText(detailedText)
    msg.exec_()


def _getOneCurve(plt, mode="unique"):
    """Get a single curve from the plot.
    By default, get the active curve if any, else if a single curve is plotted
    get it, else return None and display a warning popup.

    This behavior can be adjusted by modifying the *mode* parameter: always
    return the active curve if any, but adjust the behavior in case no curve
    is active.

    :param plt: :class:`.PlotWidget` instance on which to operate
    :param mode: Parameter defining the behavior when no curve is active.
        Possible modes:
            - "none": return None (enforce curve activation)
            - "unique": return the unique curve or None if multiple curves
            - "first": return first curve
            - "last": return last curve (most recently added one)
    :return: return value of plt.getActiveCurve(), or plt.getAllCurves()[0],
        or plt.getAllCurves()[-1], or None
    """
    curve = plt.getActiveCurve()
    if curve is not None:
        return curve

    if mode is None or mode.lower() == "none":
        _warningMessage("You must activate a curve!",
                        parent=plt)
        return None

    curves = plt.getAllCurves()
    if len(curves) == 0:
        _warningMessage("No curve on this plot.",
                        parent=plt)
        return None

    if len(curves) == 1:
        return curves[0]

    if len(curves) > 1:
        if mode == "unique":
            _warningMessage("Multiple curves are plotted. " +
                            "Please activate the one you want to use.",
                            parent=plt)
            return None
        if mode.lower() == "first":
            return curves[0]
        if mode.lower() == "last":
            return curves[-1]

    raise ValueError("Illegal value for parameter 'mode'." +
                     " Allowed values: 'none', 'unique', 'first', 'last'.")


class FitAction(PlotAction):
    """QAction to open a :class:`FitWidget` and set its data to the
    active curve if any, or to the first curve.

    :param plot: :class:`.PlotWidget` instance on which to operate
    :param parent: See :class:`QAction`
    """
    def __init__(self, plot, parent=None):
        super(FitAction, self).__init__(
            plot, icon='math-fit', text='Fit curve',
            tooltip='Open a fit dialog',
            triggered=self._getFitWindow,
            checkable=False, parent=parent)
        self.fit_window = None

    def _getFitWindow(self):
        curve = _getOneCurve(self.plot)
        if curve is None:
            return
        self.xlabel = self.plot.getGraphXLabel()
        self.ylabel = self.plot.getGraphYLabel()
        self.x = curve.getXData(copy=False)
        self.y = curve.getYData(copy=False)
        self.legend = curve.getLegend()
        self.xmin, self.xmax = self.plot.getGraphXLimits()

        # open a window with a FitWidget
        if self.fit_window is None:
            self.fit_window = qt.QMainWindow()
            # import done here rather than at module level to avoid circular import
            # FitWidget -> BackgroundWidget -> PlotWindow -> actions -> fit -> FitWidget
            from ..fit.FitWidget import FitWidget
            self.fit_widget = FitWidget(parent=self.fit_window)
            self.fit_window.setCentralWidget(
                self.fit_widget)
            self.fit_widget.guibuttons.DismissButton.clicked.connect(
                self.fit_window.close)
            self.fit_widget.sigFitWidgetSignal.connect(
                self.handle_signal)
            self.fit_window.show()
        else:
            if self.fit_window.isHidden():
                self.fit_window.show()
                self.fit_widget.show()
            self.fit_window.raise_()

        self.fit_widget.setData(self.x, self.y,
                                xmin=self.xmin, xmax=self.xmax)
        self.fit_window.setWindowTitle(
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
