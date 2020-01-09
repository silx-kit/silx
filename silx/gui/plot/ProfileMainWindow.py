# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017-2020 European Synchrotron Radiation Facility
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

    :param qt.QWidget parent: The parent of this widget or None (default).
    :param Union[str,Class] backend: The backend to use, in:
                    'matplotlib' (default), 'mpl', 'opengl', 'gl', 'none'
                    or a :class:`BackendBase.BackendBase` class
    """

    sigProfileDimensionsChanged = qt.Signal(int)
    """This signal is emitted when :meth:`setProfileDimensions` is called.
    It carries the number of dimensions for the profile data (1 or 2).
    It can be used to be notified that the profile plot widget has changed.
    """

    sigClose = qt.Signal()
    """Emitted by :meth:`closeEvent` (e.g. when the window is closed
    through the window manager's close icon)."""

    sigProfileMethodChanged = qt.Signal(str)
    """Emitted when the method to compute the profile changed (for now can be
    sum or mean)"""

    def __init__(self, parent=None, backend=None):
        qt.QMainWindow.__init__(self, parent=parent)

        self.setWindowTitle('Profile window')
        # plots are created on demand, in self.setProfileDimensions()
        self._plot1D = None
        self._plot2D = None
        self._backend = backend
        # by default, profile is assumed to be a 1D curve
        self._profileType = None
        self.setProfileType("1D")
        self.setProfileMethod('sum')

    def setProfileType(self, profileType):
        """Set which profile plot widget (1D or 2D) is to be used

        :param str profileType: Type of profile data,
            "1D" for a curve or "2D" for an image
        """
        # import here to avoid circular import
        from .PlotWindow import Plot1D, Plot2D      # noqa
        self._profileType = profileType
        if self._profileType == "1D":
            if self._plot2D is not None:
                self._plot2D.setParent(None)   # necessary to avoid widget destruction
            if self._plot1D is None:
                self._plot1D = Plot1D(backend=self._backend)
                self._plot1D.setDataMargins(yMinMargin=0.1, yMaxMargin=0.1)
                self._plot1D.setGraphYLabel('Profile')
                self._plot1D.setGraphXLabel('')
            self.setCentralWidget(self._plot1D)
        elif self._profileType == "2D":
            if self._plot1D is not None:
                self._plot1D.setParent(None)   # necessary to avoid widget destruction
            if self._plot2D is None:
                self._plot2D = Plot2D(backend=self._backend)
            self.setCentralWidget(self._plot2D)
        else:
            raise ValueError("Profile type must be '1D' or '2D'")

        self.sigProfileDimensionsChanged.emit(profileType)

    def getPlot(self):
        """Return the profile plot widget which is currently in use.
        This can be the 2D profile plot or the 1D profile plot.
        """
        if self._profileType == "2D":
            return self._plot2D
        else:
            return self._plot1D

    def closeEvent(self, qCloseEvent):
        self.sigClose.emit()
        qCloseEvent.accept()

    def setProfileMethod(self, method):
        """

        :param str method: method to manage the 'width' in the profile
            (computing mean or sum).
        """
        assert method in ('sum', 'mean')
        self._method = method
        self.sigProfileMethodChanged.emit(self._method)
