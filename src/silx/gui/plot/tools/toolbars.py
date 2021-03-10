# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2018-2020 European Synchrotron Radiation Facility
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
"""This module provides toolbars that work with :class:`PlotWidget`.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "01/03/2018"


from ... import qt
from .. import actions
from ..PlotWidget import PlotWidget
from .. import PlotToolButtons
from ....utils.deprecation import deprecated


class InteractiveModeToolBar(qt.QToolBar):
    """Toolbar with interactive mode actions

    :param parent: See :class:`QWidget`
    :param silx.gui.plot.PlotWidget plot: PlotWidget to control
    :param str title: Title of the toolbar.
    """

    def __init__(self, parent=None, plot=None, title='Plot Interaction'):
        super(InteractiveModeToolBar, self).__init__(title, parent)

        assert isinstance(plot, PlotWidget)

        self._zoomModeAction = actions.mode.ZoomModeAction(
            parent=self, plot=plot)
        self.addAction(self._zoomModeAction)

        self._panModeAction = actions.mode.PanModeAction(
            parent=self, plot=plot)
        self.addAction(self._panModeAction)

    def getZoomModeAction(self):
        """Returns the zoom mode QAction.

        :rtype: PlotAction
        """
        return self._zoomModeAction

    def getPanModeAction(self):
        """Returns the pan mode QAction

        :rtype: PlotAction
        """
        return self._panModeAction


class OutputToolBar(qt.QToolBar):
    """Toolbar providing icons to copy, save and print a PlotWidget

    :param parent: See :class:`QWidget`
    :param silx.gui.plot.PlotWidget plot: PlotWidget to control
    :param str title: Title of the toolbar.
    """

    def __init__(self, parent=None, plot=None, title='Plot Output'):
        super(OutputToolBar, self).__init__(title, parent)

        assert isinstance(plot, PlotWidget)

        self._copyAction = actions.io.CopyAction(parent=self, plot=plot)
        self.addAction(self._copyAction)

        self._saveAction = actions.io.SaveAction(parent=self, plot=plot)
        self.addAction(self._saveAction)

        self._printAction = actions.io.PrintAction(parent=self, plot=plot)
        self.addAction(self._printAction)

    def getCopyAction(self):
        """Returns the QAction performing copy to clipboard of the PlotWidget

        :rtype: PlotAction
        """
        return self._copyAction

    def getSaveAction(self):
        """Returns the QAction performing save to file of the PlotWidget

        :rtype: PlotAction
        """
        return self._saveAction

    def getPrintAction(self):
        """Returns the QAction performing printing of the PlotWidget

        :rtype: PlotAction
        """
        return self._printAction


class ImageToolBar(qt.QToolBar):
    """Toolbar providing PlotAction suited when displaying images

    :param parent: See :class:`QWidget`
    :param silx.gui.plot.PlotWidget plot: PlotWidget to control
    :param str title: Title of the toolbar.
    """

    def __init__(self, parent=None, plot=None, title='Image'):
        super(ImageToolBar, self).__init__(title, parent)

        assert isinstance(plot, PlotWidget)

        self._resetZoomAction = actions.control.ResetZoomAction(
            parent=self, plot=plot)
        self.addAction(self._resetZoomAction)

        self._colormapAction = actions.control.ColormapAction(
            parent=self, plot=plot)
        self.addAction(self._colormapAction)

        self._keepDataAspectRatioButton = PlotToolButtons.AspectToolButton(
            parent=self, plot=plot)
        self.addWidget(self._keepDataAspectRatioButton)

        self._yAxisInvertedButton = PlotToolButtons.YAxisOriginToolButton(
            parent=self, plot=plot)
        self.addWidget(self._yAxisInvertedButton)

    def getResetZoomAction(self):
        """Returns the QAction to reset the zoom.

        :rtype: PlotAction
        """
        return self._resetZoomAction

    def getColormapAction(self):
        """Returns the QAction to control the colormap.

        :rtype: PlotAction
        """
        return self._colormapAction

    def getKeepDataAspectRatioButton(self):
        """Returns the QToolButton controlling data aspect ratio.

        :rtype: QToolButton
        """
        return self._keepDataAspectRatioButton

    def getYAxisInvertedButton(self):
        """Returns the QToolButton controlling Y axis orientation.

        :rtype: QToolButton
        """
        return self._yAxisInvertedButton


class CurveToolBar(qt.QToolBar):
    """Toolbar providing PlotAction suited when displaying curves

    :param parent: See :class:`QWidget`
    :param silx.gui.plot.PlotWidget plot: PlotWidget to control
    :param str title: Title of the toolbar.
    """

    def __init__(self, parent=None, plot=None, title='Image'):
        super(CurveToolBar, self).__init__(title, parent)

        assert isinstance(plot, PlotWidget)

        self._resetZoomAction = actions.control.ResetZoomAction(
            parent=self, plot=plot)
        self.addAction(self._resetZoomAction)

        self._xAxisAutoScaleAction = actions.control.XAxisAutoScaleAction(
            parent=self, plot=plot)
        self.addAction(self._xAxisAutoScaleAction)

        self._yAxisAutoScaleAction = actions.control.YAxisAutoScaleAction(
            parent=self, plot=plot)
        self.addAction(self._yAxisAutoScaleAction)

        self._xAxisLogarithmicAction = actions.control.XAxisLogarithmicAction(
            parent=self, plot=plot)
        self.addAction(self._xAxisLogarithmicAction)

        self._yAxisLogarithmicAction = actions.control.YAxisLogarithmicAction(
            parent=self, plot=plot)
        self.addAction(self._yAxisLogarithmicAction)

        self._gridAction = actions.control.GridAction(
            parent=self, plot=plot)
        self.addAction(self._gridAction)

        self._curveStyleAction = actions.control.CurveStyleAction(
            parent=self, plot=plot)
        self.addAction(self._curveStyleAction)

    def getResetZoomAction(self):
        """Returns the QAction to reset the zoom.

        :rtype: PlotAction
        """
        return self._resetZoomAction

    def getXAxisAutoScaleAction(self):
        """Returns the QAction to toggle X axis autoscale.

        :rtype: PlotAction
        """
        return self._xAxisAutoScaleAction

    def getYAxisAutoScaleAction(self):
        """Returns the QAction to toggle Y axis autoscale.

        :rtype: PlotAction
        """
        return self._yAxisAutoScaleAction

    def getXAxisLogarithmicAction(self):
        """Returns the QAction to toggle X axis log/linear scale.

        :rtype: PlotAction
        """
        return self._xAxisLogarithmicAction

    def getYAxisLogarithmicAction(self):
        """Returns the QAction to toggle Y axis log/linear scale.

        :rtype: PlotAction
        """
        return self._yAxisLogarithmicAction

    def getGridAction(self):
        """Returns the action to toggle the plot grid.

        :rtype: PlotAction
        """
        return self._gridAction

    def getCurveStyleAction(self):
        """Returns the QAction to change the style of all curves.

        :rtype: PlotAction
        """
        return self._curveStyleAction


class ScatterToolBar(qt.QToolBar):
    """Toolbar providing PlotAction suited when displaying scatter plot

    :param parent: See :class:`QWidget`
    :param silx.gui.plot.PlotWidget plot: PlotWidget to control
    :param str title: Title of the toolbar.
    """

    def __init__(self, parent=None, plot=None, title='Scatter Tools'):
        super(ScatterToolBar, self).__init__(title, parent)

        assert isinstance(plot, PlotWidget)

        self._resetZoomAction = actions.control.ResetZoomAction(
            parent=self, plot=plot)
        self.addAction(self._resetZoomAction)

        self._xAxisLogarithmicAction = actions.control.XAxisLogarithmicAction(
            parent=self, plot=plot)
        self.addAction(self._xAxisLogarithmicAction)

        self._yAxisLogarithmicAction = actions.control.YAxisLogarithmicAction(
            parent=self, plot=plot)
        self.addAction(self._yAxisLogarithmicAction)

        self._keepDataAspectRatioButton = PlotToolButtons.AspectToolButton(
            parent=self, plot=plot)
        self.addWidget(self._keepDataAspectRatioButton)

        self._gridAction = actions.control.GridAction(
            parent=self, plot=plot)
        self.addAction(self._gridAction)

        self._colormapAction = actions.control.ColormapAction(
            parent=self, plot=plot)
        self.addAction(self._colormapAction)

        self._visualizationToolButton = \
            PlotToolButtons.ScatterVisualizationToolButton(parent=self, plot=plot)
        self.addWidget(self._visualizationToolButton)

    def getResetZoomAction(self):
        """Returns the QAction to reset the zoom.

        :rtype: PlotAction
        """
        return self._resetZoomAction

    def getXAxisLogarithmicAction(self):
        """Returns the QAction to toggle X axis log/linear scale.

        :rtype: PlotAction
        """
        return self._xAxisLogarithmicAction

    def getYAxisLogarithmicAction(self):
        """Returns the QAction to toggle Y axis log/linear scale.

        :rtype: PlotAction
        """
        return self._yAxisLogarithmicAction

    def getGridAction(self):
        """Returns the action to toggle the plot grid.

        :rtype: PlotAction
        """
        return self._gridAction

    def getColormapAction(self):
        """Returns the QAction to control the colormap.

        :rtype: PlotAction
        """
        return self._colormapAction

    def getKeepDataAspectRatioButton(self):
        """Returns the QToolButton controlling data aspect ratio.

        :rtype: QToolButton
        """
        return self._keepDataAspectRatioButton

    def getScatterVisualizationToolButton(self):
        """Returns the QToolButton controlling the visualization mode.

        :rtype: ScatterVisualizationToolButton
        """
        return self._visualizationToolButton

    @deprecated(replacement='getScatterVisualizationToolButton',
                since_version='0.11.0')
    def getSymbolToolButton(self):
        return self.getScatterVisualizationToolButton()
