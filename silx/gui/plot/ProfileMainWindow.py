# coding: utf-8
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
"""This module contains a QMainWindow class used to display profile plots.
"""
from silx.gui import qt


__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "21/02/2017"


class ProfileMainWindow(qt.QMainWindow):
    """QMainWindow providing 2 plot widgets specialized in
    1D and 2D plotting, with different toolbars.
    Only one of the plots is visible at any given time.
    """
    sigProfileDimensionsChanged = qt.Signal(int)
    """This signal is emitted when :meth:`setProfileDimensions` is called.
    It carries the number of dimensions for the profile data (1 or 2).
    It can be used to be notified that the profile plot widget has changed.
    """

    sigClose = qt.Signal()
    """Emitted by :meth:`closeEvent` (e.g. when the window is closed
    through the window manager's close icon)."""

    def __init__(self, parent=None):
        qt.QMainWindow.__init__(self, parent=parent)

        self.setWindowTitle('Profile window')
        # plots are created on demand, in self.setProfileDimensions()
        self._plot1D = None
        self._plot2D = None
        # by default, profile is assumed to be a 1D curve
        self._profileDimensions = 1
        self.setProfileDimensions(1)

    def setProfileDimensions(self, dimensions):
        """Set which profile plot widget (1D or 2D) is to be used

        :param int dimensions: Number of dimensions for profile data
            (1 dimension for a curve, 2  dimensions for an image)
        """
        # import here to avoid circular import
        from .PlotWindow import Plot1D, Plot2D      # noqa
        self._profileDimensions = dimensions

        if self._profileDimensions == 1:
            if self._plot2D is not None:
                self._plot2D.setParent(None)   # necessary to avoid widget destruction
            if self._plot1D is None:
                self._plot1D = Plot1D()
            self.setCentralWidget(self._plot1D)
        elif self._profileDimensions == 2:
            if self._plot1D is not None:
                self._plot1D.setParent(None)   # necessary to avoid widget destruction
            if self._plot2D is None:
                self._plot2D = Plot2D()
            self.setCentralWidget(self._plot2D)
        else:
            raise ValueError("Profile dimensions must be 1 or 2")

        self.sigProfileDimensionsChanged.emit(dimensions)

    def getPlot(self):
        """Return the profile plot widget which is currently in use.
        This can be the 2D profile plot or the 1D profile plot.
        """
        assert self._profileDimensions in [1, 2]
        if self._profileDimensions == 2:
            return self._plot2D
        else:
            return self._plot1D

    def closeEvent(self, qCloseEvent):
        self.sigClose.emit()
        qCloseEvent.accept()
