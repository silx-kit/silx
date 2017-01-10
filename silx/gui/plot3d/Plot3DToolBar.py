# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2017 European Synchrotron Radiation Facility
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
"""This module provides a toolbar with tools for a Plot3DWidget.

It provides:

- Copy
- Save
- Print
"""

from __future__ import absolute_import

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "10/01/2017"

import logging

from silx.gui import qt

from . import Plot3DActions

_logger = logging.getLogger(__name__)


class Plot3DToolBar(qt.QToolBar):
    """Toolbar providing icons to copy, save and print the OpenGL scene

    :param parent: See :class:`QWidget`
    :param str title: Title of the toolbar.
    """

    def __init__(self, parent=None, title='Plot3D'):
        super(Plot3DToolBar, self).__init__(title, parent)

        self._plot3d = None

        self._copyAction = Plot3DActions.CopyAction(parent=self)
        self.addAction(self._copyAction)

        self._saveAction = Plot3DActions.SaveAction(parent=self)
        self.addAction(self._saveAction)

        self._videoAction = Plot3DActions.VideoAction(parent=self)
        self.addAction(self._videoAction)

        self._printAction = Plot3DActions.PrintAction(parent=self)
        self.addAction(self._printAction)

    def setPlot3DWidget(self, widget):
        """Set the Plot3DWidget this toolbar is associated with

        :param Plot3DWidget widget: The widget to copy/save/print
        """
        self._plot3d = widget
        self.getCopyAction().setPlot3DWidget(widget)
        self.getSaveAction().setPlot3DWidget(widget)
        self.getVideoRecordAction().setPlot3DWidget(widget)
        self.getPrintAction().setPlot3DWidget(widget)

    def getPlot3DWidget(self):
        """Return the Plot3DWidget associated to this toolbar.

        If no widget is associated, it returns None.

        :rtype: qt.QWidget
        """
        return self._plot3d

    def getCopyAction(self):
        """Returns the QAction performing copy to clipboard of the Plot3DWidget

        :rtype: qt.QAction
        """
        return self._copyAction

    def getSaveAction(self):
        """Returns the QAction performing save to file of the Plot3DWidget

        :rtype: qt.QAction
        """
        return self._saveAction

    def getVideoRecordAction(self):
        """Returns the QAction performing record video of the Plot3DWidget

        :rtype: qt.QAction
        """
        return self._videoAction

    def getPrintAction(self):
        """Returns the QAction performing printing of the Plot3DWidget

        :rtype: qt.QAction
        """
        return self._printAction
