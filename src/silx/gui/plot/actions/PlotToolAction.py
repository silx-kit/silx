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
The class :class:`.PlotToolAction` help the creation of a qt.QAction associating
a tool window with a :class:`.PlotWidget`.
"""

from __future__ import division


__authors__ = ["V.A. Sole", "T. Vincent", "P. Knobel"]
__license__ = "MIT"
__date__ = "10/10/2018"


import weakref

from .PlotAction import PlotAction
from silx.gui import qt


class PlotToolAction(PlotAction):
    """Base class for QAction that maintain a tool window operating on a
    PlotWidget."""

    def __init__(self, plot, icon, text, tooltip=None,
                 triggered=None, checkable=False, parent=None):
        PlotAction.__init__(self,
                            plot=plot,
                            icon=icon,
                            text=text,
                            tooltip=tooltip,
                            triggered=self._triggered,
                            parent=parent,
                            checkable=True)
        self._previousGeometry = None
        self._toolWindow = None

    def _triggered(self, checked):
        """Update the plot of the histogram visibility status

        :param bool checked: status  of the action button
        """
        self._setToolWindowVisible(checked)

    def _setToolWindowVisible(self, visible):
        """Set the tool window visible or hidden."""
        tool = self._getToolWindow()
        if tool.isVisible() == visible:
            # Nothing to do
            return

        if visible:
            self._connectPlot(tool)
            tool.show()
            if self._previousGeometry is not None:
                # Restore the geometry
                tool.setGeometry(self._previousGeometry)
        else:
            self._disconnectPlot(tool)
            # Save the geometry
            self._previousGeometry = tool.geometry()
            tool.hide()

    def _connectPlot(self, window):
        """Called if the tool is visible and have to be updated according to
        event of the plot.

        :param qt.QWidget window: The tool window
        """
        pass

    def _disconnectPlot(self, window):
        """Called if the tool is not visible and dont have anymore to be updated
        according to event of the plot.

        :param qt.QWidget window: The tool window
        """
        pass

    def _isWindowInUse(self):
        """Returns true if the tool window is currently in use."""
        if not self.isChecked():
            return False
        return self._toolWindow is not None

    def _ownerVisibilityChanged(self, isVisible):
        """Called when the visibility of the parent of the tool window changes

        :param bool isVisible: True if the parent became visible
        """
        if self._isWindowInUse():
            self._setToolWindowVisible(isVisible)

    def eventFilter(self, qobject, event):
        """Observe when the close event is emitted then
        simply uncheck the action button

        :param qobject: the object observe
        :param event: the event received by qobject
        """
        if event.type() == qt.QEvent.Close:
            if self._toolWindow is not None:
                window = self._toolWindow()
                self._previousGeometry = window.geometry()
                window.hide()
            self.setChecked(False)

        return PlotAction.eventFilter(self, qobject, event)

    def _getToolWindow(self):
        """Returns the window containing the tool.

        It uses lazy loading to create this tool..
        """
        if self._toolWindow is None:
            window = self._createToolWindow()
            if self._previousGeometry is not None:
                window.setGeometry(self._previousGeometry)
            window.installEventFilter(self)
            plot = self.plot
            plot.sigVisibilityChanged.connect(self._ownerVisibilityChanged)
            self._toolWindow = weakref.ref(window)
        return self._toolWindow()

    def _createToolWindow(self):
        """Create the tool window managing the plot."""
        raise NotImplementedError()
