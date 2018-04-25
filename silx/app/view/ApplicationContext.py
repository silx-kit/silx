# coding: utf-8
# /*##########################################################################
# Copyright (C) 2016-2018 European Synchrotron Radiation Facility
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
# ############################################################################*/
"""Browse a data file with a GUI"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "25/04/2018"

import weakref

from silx.gui import qt
from silx.gui.data.DataViews import DataViewHooks
from silx.gui.plot.Colormap import Colormap
from silx.gui.plot.ColormapDialog import ColormapDialog


class ApplicationContext(DataViewHooks):
    """
    Store the conmtext of the application

    It overwrites the DataViewHooks to custom the use of the DataViewer for
    the silx view application.

    - Create a single colormap shared with all the views
    - Create a single colormap dialog shared with all the views
    """

    def __init__(self, parent):
        self.__parent = weakref.ref(parent)
        self.__defaultColormap = None
        self.__defaultColormapDialog = None
        self.__settings = qt.QSettings("silx", "silx-view", parent=parent)

    def getSettings(self):
        """Returns actual application settings.

        :rtype: qt.QSettings
        """
        return self.__settings

    def restoreSettings(self):
        """Restore the settings of all the application"""
        parent = self.__parent()
        parent.restoreSettings(self.__settings)

    def saveSettings(self):
        """Save the settings of all the application"""
        parent = self.__parent()
        parent.saveSettings(self.__settings)

    def getColormap(self, view):
        """Returns a default colormap.

        Override from DataViewHooks

        :rtype: Colormap
        """
        if self.__defaultColormap is None:
            self.__defaultColormap = Colormap(name="viridis")
        return self.__defaultColormap

    def getColormapDialog(self, view):
        """Returns a shared color dialog as default for all the views.

        Override from DataViewHooks

        :rtype: ColorDialog
        """
        if self.__defaultColormapDialog is None:
            parent = self.__parent()
            if parent is None:
                return None
            dialog = ColormapDialog(parent=parent)
            dialog.setModal(False)
            self.__defaultColormapDialog = dialog
        return self.__defaultColormapDialog
