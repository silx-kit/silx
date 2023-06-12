# /*##########################################################################
#
# Copyright (c) 2004-2023 European Synchrotron Radiation Facility
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
:mod:`silx.gui.plot.actions.mode` provides a set of QAction relative to mouse
mode of a :class:`.PlotWidget`.

The following QAction are available:

- :class:`ZoomModeAction`
- :class:`PanModeAction`
"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "16/08/2017"


from silx.gui import qt

from . import PlotAction


class ZoomModeAction(PlotAction):
    """QAction controlling the zoom mode of a :class:`.PlotWidget`.

    :param plot: :class:`.PlotWidget` instance on which to operate
    :param parent: See :class:`QAction`
    """

    def __init__(self, plot, parent=None):
        super(ZoomModeAction, self).__init__(
            plot, icon='zoom', text='Zoom mode',
            tooltip='Zoom-in on mouse selection',
            triggered=self._actionTriggered,
            checkable=True, parent=parent)

        self.__menu = qt.QMenu(self.plot)
        self.__menu.addSection("Enabled axes")

        self.__xAxisAction = qt.QAction("X axis", parent=self.__menu)
        self.__yAxisAction = qt.QAction("Y left axis", parent=self.__menu)
        self.__y2AxisAction = qt.QAction("Y right axis", parent=self.__menu)

        for action in (self.__xAxisAction, self.__yAxisAction, self.__y2AxisAction):
            action.setCheckable(True)
            action.setChecked(True)
            action.triggered.connect(self._axesActionTriggered)
            self.__menu.addAction(action)

        # Listen to interaction configuration change
        self.plot.interaction().sigChanged.connect(self._interactionChanged)
        # Init the state
        self._interactionChanged()

    def isAxesMenuEnabled(self) -> bool:
        """Returns whether the axes selection menu is enabled or not (default: False)"""
        return self.menu() is self.__menu

    def setAxesMenuEnabled(self, enabled: bool):
        """Toggle the availability of the axes selection menu (default: False)"""
        if enabled == self.isAxesMenuEnabled():
            return

        self.setMenu(self.__menu if enabled else None)

        # Update associated QToolButton's popupMode if any, this is not done at least with Qt5
        parent = self.parent()
        if not isinstance(parent, qt.QToolBar):
            return
        widget = parent.widgetForAction(self)
        if not isinstance(widget, qt.QToolButton):
            return
        widget.setPopupMode(qt.QToolButton.MenuButtonPopup if enabled else qt.QToolButton.DelayedPopup)
        widget.update()

    def _interactionChanged(self):
        plot = self.plot
        if plot is None:
            return

        self.setChecked(plot.getInteractiveMode()["mode"] == "zoom")
        enabledAxes = plot.interaction().getZoomEnabledAxes()
        self.__xAxisAction.setChecked(enabledAxes.xaxis)
        self.__yAxisAction.setChecked(enabledAxes.yaxis)
        self.__y2AxisAction.setChecked(enabledAxes.y2axis)

    def _actionTriggered(self, checked=False):
        plot = self.plot
        if plot is None:
            return

        plot.setInteractiveMode('zoom', source=self)

    def _axesActionTriggered(self, checked=False):
        plot = self.plot
        if plot is None:
            return

        plot.interaction().setZoomEnabledAxes(
            self.__xAxisAction.isChecked(),
            self.__yAxisAction.isChecked(),
            self.__y2AxisAction.isChecked(),
        )


class PanModeAction(PlotAction):
    """QAction controlling the pan mode of a :class:`.PlotWidget`.

    :param plot: :class:`.PlotWidget` instance on which to operate
    :param parent: See :class:`QAction`
    """

    def __init__(self, plot, parent=None):
        super(PanModeAction, self).__init__(
            plot, icon='pan', text='Pan mode',
            tooltip='Pan the view',
            triggered=self._actionTriggered,
            checkable=True, parent=parent)
        # Listen to mode change
        self.plot.sigInteractiveModeChanged.connect(self._modeChanged)
        # Init the state
        self._modeChanged(None)

    def _modeChanged(self, source):
        modeDict = self.plot.getInteractiveMode()
        old = self.blockSignals(True)
        self.setChecked(modeDict["mode"] == "pan")
        self.blockSignals(old)

    def _actionTriggered(self, checked=False):
        plot = self.plot
        if plot is not None:
            plot.setInteractiveMode('pan', source=self)
