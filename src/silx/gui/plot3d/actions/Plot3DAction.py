# /*##########################################################################
#
# Copyright (c) 2016-2018 European Synchrotron Radiation Facility
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
"""Base class for QAction attached to a Plot3DWidget."""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "06/09/2017"


import logging
import weakref

from silx.gui import qt


_logger = logging.getLogger(__name__)


class Plot3DAction(qt.QAction):
    """QAction associated to a Plot3DWidget

    :param parent: See :class:`QAction`
    :param ~silx.gui.plot3d.Plot3DWidget.Plot3DWidget plot3d:
        Plot3DWidget the action is associated with
    """

    def __init__(self, parent, plot3d=None):
        super(Plot3DAction, self).__init__(parent)
        self._plot3d = None
        self.setPlot3DWidget(plot3d)

    def setPlot3DWidget(self, widget):
        """Set the Plot3DWidget this action is associated with

        :param ~silx.gui.plot3d.Plot3DWidget.Plot3DWidget widget:
            The Plot3DWidget to use
        """
        self._plot3d = None if widget is None else weakref.ref(widget)

    def getPlot3DWidget(self):
        """Return the Plot3DWidget associated to this action.

        If no widget is associated, it returns None.

        :rtype: QWidget
        """
        return None if self._plot3d is None else self._plot3d()
