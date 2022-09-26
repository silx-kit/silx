# /*##########################################################################
#
# Copyright (c) 2004-2018 European Synchrotron Radiation Facility
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

from . import PlotAction
import logging

_logger = logging.getLogger(__name__)


class ZoomModeAction(PlotAction):
    """QAction controlling the zoom mode of a :class:`.PlotWidget`.

    :param plot: :class:`.PlotWidget` instance on which to operate
    :param parent: See :class:`QAction`
    """

    def __init__(self, plot, parent=None):
        super(ZoomModeAction, self).__init__(
            plot, icon='zoom', text='Zoom mode',
            tooltip='Zoom in or out',
            triggered=self._actionTriggered,
            checkable=True, parent=parent)
        # Listen to mode change
        self.plot.sigInteractiveModeChanged.connect(self._modeChanged)
        # Init the state
        self._modeChanged(None)

    def _modeChanged(self, source):
        modeDict = self.plot.getInteractiveMode()
        old = self.blockSignals(True)
        self.setChecked(modeDict["mode"] == "zoom")
        self.blockSignals(old)

    def _actionTriggered(self, checked=False):
        plot = self.plot
        if plot is not None:
            plot.setInteractiveMode('zoom', source=self)


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
