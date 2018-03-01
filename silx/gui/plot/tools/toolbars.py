# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2018 European Synchrotron Radiation Facility
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


class InteractiveModeToolBar(qt.QToolBar):
    """Toolbar with interactive mode actions

    :param parent: See :class:`QWidget`
    :param PlotWidget plot: PlotWidget to control
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
    :param PlotWidget plot: PlotWidget to control
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

        :rtype: qt.QAction
        """
        return self._copyAction

    def getSaveAction(self):
        """Returns the QAction performing save to file of the PlotWidget

        :rtype: qt.QAction
        """
        return self._saveAction

    def getPrintAction(self):
        """Returns the QAction performing printing of the PlotWidget

        :rtype: qt.QAction
        """
        return self._printAction
