# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2004-2016 European Synchrotron Radiation Facility
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
"""Qt widget providing Plot API for 1D and 2D data.

This provides the plot API of :class:`silx.gui.plot.Plot.Plot` as a
Qt widget.
"""

__authors__ = ["V.A. Sole", "T. Vincent"]
__license__ = "MIT"
__date__ = "22/02/2016"


import logging

from . import Plot

from .. import qt


_logger = logging.getLogger(__name__)


class PlotWidget(qt.QMainWindow, Plot.Plot):
    """Qt Widget providing a 1D/2D plot.

    This widget is a QMainWindow.
    It provides Qt signals for the Plot and add supports for panning
    with arrow keys.

    :param parent: The parent of this widget or None.
    :param backend: The backend to use for the plot.
                    The default is to use matplotlib.
    :type backend: str or :class:`BackendBase.BackendBase`
    :param bool autoreplot: Toggle autoreplot mode (Default: True).
    """

    sigPlotSignal = qt.Signal(object)
    """Signal for all events of the plot.

    The signal information is provided as a dict.
    See :class:`Plot` for documentation of the content of the dict.
    """

    sigSetYAxisInverted = qt.Signal(bool)
    """Signal emitted when Y axis orientation has changed"""

    sigSetXAxisLogarithmic = qt.Signal(bool)
    """Signal emitted when X axis scale has changed"""

    sigSetYAxisLogarithmic = qt.Signal(bool)
    """Signal emitted when Y axis scale has changed"""

    sigSetXAxisAutoScale = qt.Signal(bool)
    """Signal emitted when X axis autoscale has changed"""

    sigSetYAxisAutoScale = qt.Signal(bool)
    """Signal emitted when Y axis autoscale has changed"""

    sigSetKeepDataAspectRatio = qt.Signal(bool)
    """Signal emitted when plot keep aspect ratio has changed"""

    sigSetGraphGrid = qt.Signal(str)
    """Signal emitted when plot grid has changed"""

    def __init__(self, parent=None, backend=None,
                 legends=False, callback=None, autoreplot=True, **kw):

        if kw:
            _logger.warning(
                'deprecated: __init__ extra arguments: %s', str(kw))
        if legends:
            _logger.warning('deprecated: __init__ legend argument')
        if callback:
            _logger.warning('deprecated: __init__ callback argument')

        self._panWithArrowKeys = False

        qt.QMainWindow.__init__(self, parent)
        if parent is not None:
            # behave as a widget
            self.setWindowFlags(qt.Qt.Widget)

        Plot.Plot.__init__(
            self, parent, backend=backend, autoreplot=autoreplot)

        widget = self.getWidgetHandle()
        if widget is not None:
            self.setCentralWidget(widget)
        else:
            _logger.warning("Plot backend does not support widget")

    def notify(self, event, **kwargs):
        """Override :meth:`Plot.notify` to send Qt signals."""
        eventDict = kwargs.copy()
        eventDict['event'] = event
        self.sigPlotSignal.emit(eventDict)

        if event == 'setYAxisInverted':
            self.sigSetYAxisInverted.emit(kwargs['state'])
        elif event == 'setXAxisLogarithmic':
            self.sigSetXAxisLogarithmic.emit(kwargs['state'])
        elif event == 'setYAxisLogarithmic':
            self.sigSetYAxisLogarithmic.emit(kwargs['state'])
        elif event == 'setXAxisAutoScale':
            self.sigSetXAxisAutoScale.emit(kwargs['state'])
        elif event == 'setYAxisAutoScale':
            self.sigSetYAxisAutoScale.emit(kwargs['state'])
        elif event == 'setKeepDataAspectRatio':
            self.sigSetKeepDataAspectRatio.emit(kwargs['state'])
        elif event == 'setGraphGrid':
            self.sigSetGraphGrid.emit(kwargs['which'])

        Plot.Plot.notify(self, event, **kwargs)

    # Panning with arrow keys

    def isPanWithArrowKeys(self):
        """Returns whether or not panning the graph with arrow keys is enable.

        See :meth:`setPanWithArrowKeys`.
        """
        return self._panWithArrowKeys

    def setPanWithArrowKeys(self, pan=False):
        """Enable/Disable panning the graph with arrow keys.

        This grabs the keyboard.

        :param bool pan: True to enable panning, False to disable.
        """
        self._panWithArrowKeys = bool(pan)
        if not self._panWithArrowKeys:
            self.setFocusPolicy(qt.Qt.NoFocus)
        else:
            self.setFocusPolicy(qt.Qt.StrongFocus)
            self.setFocus(qt.Qt.OtherFocusReason)

    # Dict to convert Qt arrow key code to direction str.
    _ARROWS_TO_PAN_DIRECTION = {
        qt.Qt.Key_Left: 'left',
        qt.Qt.Key_Right: 'right',
        qt.Qt.Key_Up: 'up',
        qt.Qt.Key_Down: 'down'
    }

    def keyPressEvent(self, event):
        """Key event handler handling panning on arrow keys.

        Overrides base class implementation.
        """
        key = event.key()
        if self._panWithArrowKeys and key in self._ARROWS_TO_PAN_DIRECTION:
            self.pan(self._ARROWS_TO_PAN_DIRECTION[key], factor=0.1)
        else:
            # Only call base class implementation when key is not handled.
            # See QWidget.keyPressEvent for details.
            super(PlotWidget, self).keyPressEvent(event)
