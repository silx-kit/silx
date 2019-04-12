# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017-2019 European Synchrotron Radiation Facility
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
"""This module provides Plot3DAction related to interaction modes.

It provides QAction to rotate or pan a Plot3DWidget
as well as toggle a picking mode.
"""

from __future__ import absolute_import, division

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "06/09/2017"


import logging

from ....utils.proxy import docstring
from ... import qt
from ...icons import getQIcon
from .Plot3DAction import Plot3DAction


_logger = logging.getLogger(__name__)


class InteractiveModeAction(Plot3DAction):
    """Base class for QAction changing interactive mode of a Plot3DWidget

    :param parent: See :class:`QAction`
    :param str interaction: The interactive mode this action controls
    :param ~silx.gui.plot3d.Plot3DWidget.Plot3DWidget plot3d:
        Plot3DWidget the action is associated with
    """

    def __init__(self, parent, interaction, plot3d=None):
        self._interaction = interaction

        super(InteractiveModeAction, self).__init__(parent, plot3d)
        self.setCheckable(True)
        self.triggered[bool].connect(self._triggered)

    def _triggered(self, checked=False):
        plot3d = self.getPlot3DWidget()
        if plot3d is None:
            _logger.error(
                'Cannot set %s interaction, no associated Plot3DWidget' %
                self._interaction)
        else:
            plot3d.setInteractiveMode(self._interaction)
            self.setChecked(True)

    @docstring(Plot3DAction)
    def setPlot3DWidget(self, widget):
        # Disconnect from previous Plot3DWidget
        plot3d = self.getPlot3DWidget()
        if plot3d is not None:
            plot3d.sigInteractiveModeChanged.disconnect(
                self._interactiveModeChanged)

        super(InteractiveModeAction, self).setPlot3DWidget(widget)

        # Connect to new Plot3DWidget
        if widget is None:
            self.setChecked(False)
        else:
            self.setChecked(widget.getInteractiveMode() == self._interaction)
            widget.sigInteractiveModeChanged.connect(
                self._interactiveModeChanged)

    def _interactiveModeChanged(self):
        plot3d = self.getPlot3DWidget()
        if plot3d is None:
            _logger.error('Received a signal while there is no widget')
        else:
            self.setChecked(plot3d.getInteractiveMode() == self._interaction)


class RotateArcballAction(InteractiveModeAction):
    """QAction to set arcball rotation interaction on a Plot3DWidget

    :param parent: See :class:`QAction`
    :param ~silx.gui.plot3d.Plot3DWidget.Plot3DWidget plot3d:
        Plot3DWidget the action is associated with
    """

    def __init__(self, parent, plot3d=None):
        super(RotateArcballAction, self).__init__(parent, 'rotate', plot3d)

        self.setIcon(getQIcon('rotate-3d'))
        self.setText('Rotate')
        self.setToolTip('Rotate the view. Press <b>Ctrl</b> to pan.')


class PanAction(InteractiveModeAction):
    """QAction to set pan interaction on a Plot3DWidget

    :param parent: See :class:`QAction`
    :param ~silx.gui.plot3d.Plot3DWidget.Plot3DWidget plot3d:
        Plot3DWidget the action is associated with
    """

    def __init__(self, parent, plot3d=None):
        super(PanAction, self).__init__(parent, 'pan', plot3d)

        self.setIcon(getQIcon('pan'))
        self.setText('Pan')
        self.setToolTip('Pan the view. Press <b>Ctrl</b> to rotate.')


class PickingModeAction(Plot3DAction):
    """QAction to toggle picking moe on a Plot3DWidget

    :param parent: See :class:`QAction`
    :param ~silx.gui.plot3d.Plot3DWidget.Plot3DWidget plot3d:
        Plot3DWidget the action is associated with
    """

    sigSceneClicked = qt.Signal(float, float)
    """Signal emitted when the scene is clicked with the left mouse button.

    This signal is only emitted when the action is checked.

    It provides the (x, y) clicked mouse position
    """

    def __init__(self, parent, plot3d=None):
        super(PickingModeAction, self).__init__(parent, plot3d)
        self.setIcon(getQIcon('pointing-hand'))
        self.setText('Picking')
        self.setToolTip('Toggle picking with left button click')
        self.setCheckable(True)
        self.triggered[bool].connect(self._triggered)

    def _triggered(self, checked=False):
        plot3d = self.getPlot3DWidget()
        if plot3d is not None:
            if checked:
                plot3d.sigSceneClicked.connect(self.sigSceneClicked)
            else:
                plot3d.sigSceneClicked.disconnect(self.sigSceneClicked)

    @docstring(Plot3DAction)
    def setPlot3DWidget(self, widget):
        # Disconnect from previous Plot3DWidget
        plot3d = self.getPlot3DWidget()
        if plot3d is not None and self.isChecked():
            plot3d.sigSceneClicked.disconnect(self.sigSceneClicked)

        super(PickingModeAction, self).setPlot3DWidget(widget)

        # Connect to new Plot3DWidget
        if widget is None:
            self.setChecked(False)
        elif self.isChecked():
            widget.sigSceneClicked.connect(self.sigSceneClicked)
