# /*##########################################################################
#
# Copyright (c) 2017 European Synchrotron Radiation Facility
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
"""This module provides handling of :class:`PlotWidget` limits history.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "19/07/2017"


from .. import qt


class LimitsHistory(qt.QObject):
    """Class handling history of limits of a :class:`PlotWidget`.

    :param PlotWidget parent: The plot widget this object is bound to.
    """

    def __init__(self, parent):
        self._history = []
        super(LimitsHistory, self).__init__(parent)
        self.setParent(parent)

    def setParent(self, parent):
        """See :meth:`QObject.setParent`.

        :param PlotWidget parent: The PlotWidget this object is bound to.
        """
        self.clear()  # Clear history when changing parent
        super(LimitsHistory, self).setParent(parent)

    def push(self):
        """Append current limits to the history."""
        plot = self.parent()
        xmin, xmax = plot.getXAxis().getLimits()
        ymin, ymax = plot.getYAxis(axis='left').getLimits()
        y2min, y2max = plot.getYAxis(axis='right').getLimits()
        self._history.append((xmin, xmax, ymin, ymax, y2min, y2max))

    def pop(self):
        """Restore previously limits stored in the history.

        :return: True if limits were restored, False if history was empty.
        :rtype: bool
        """
        plot = self.parent()
        if self._history:
            limits = self._history.pop(-1)
            plot.setLimits(*limits)
            return True
        else:
            plot.resetZoom()
            return False

    def clear(self):
        """Clear stored limits states."""
        self._history = []

    def __len__(self):
        return len(self._history)
