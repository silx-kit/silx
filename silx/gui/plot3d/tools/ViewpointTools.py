# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2015-2018 European Synchrotron Radiation Facility
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
"""This module provides a toolbar to control Plot3DWidget viewpoint."""

from __future__ import absolute_import

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "08/09/2017"


import weakref

from silx.gui import qt
from silx.gui.icons import getQIcon
from .. import actions


class ViewpointToolButton(qt.QToolButton):
    """A toolbutton with a drop-down list of ways to reset the viewpoint.

    :param parent: See :class:`QToolButton`
    """

    def __init__(self, parent=None):
        super(ViewpointToolButton, self).__init__(parent)

        self._plot3DRef = None

        menu = qt.QMenu(self)
        menu.addAction(actions.viewpoint.FrontViewpointAction(parent=self))
        menu.addAction(actions.viewpoint.BackViewpointAction(parent=self))
        menu.addAction(actions.viewpoint.TopViewpointAction(parent=self))
        menu.addAction(actions.viewpoint.BottomViewpointAction(parent=self))
        menu.addAction(actions.viewpoint.RightViewpointAction(parent=self))
        menu.addAction(actions.viewpoint.LeftViewpointAction(parent=self))
        menu.addAction(actions.viewpoint.SideViewpointAction(parent=self))

        self.setMenu(menu)
        self.setPopupMode(qt.QToolButton.InstantPopup)
        self.setIcon(getQIcon('cube'))
        self.setToolTip('Reset the viewpoint to a defined position')

    def setPlot3DWidget(self, widget):
        """Set the Plot3DWidget this toolbar is associated with

        :param ~silx.gui.plot3d.Plot3DWidget.Plot3DWidget widget:
            The widget to control
        """
        self._plot3DRef = None if widget is None else weakref.ref(widget)

        for action in self.menu().actions():
            action.setPlot3DWidget(widget)

    def getPlot3DWidget(self):
        """Return the Plot3DWidget associated to this toolbar.

        If no widget is associated, it returns None.

        :rtype: ~silx.gui.plot3d.Plot3DWidget.Plot3DWidget or None
        """
        return None if self._plot3DRef is None else self._plot3DRef()
