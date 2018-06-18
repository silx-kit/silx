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
__date__ = "23/05/2018"

import weakref
import logging

import silx
from silx.gui.data.DataViews import DataViewHooks
from silx.gui.colors import Colormap
from silx.gui.dialog.ColormapDialog import ColormapDialog


_logger = logging.getLogger(__name__)


class ApplicationContext(DataViewHooks):
    """
    Store the conmtext of the application

    It overwrites the DataViewHooks to custom the use of the DataViewer for
    the silx view application.

    - Create a single colormap shared with all the views
    - Create a single colormap dialog shared with all the views
    """

    def __init__(self, parent, settings=None):
        self.__parent = weakref.ref(parent)
        self.__defaultColormap = None
        self.__defaultColormapDialog = None
        self.__settings = settings
        self.__recentFiles = []

    def getSettings(self):
        """Returns actual application settings.

        :rtype: qt.QSettings
        """
        return self.__settings

    def restoreLibrarySettings(self):
        """Restore the library settings, which must be done early"""
        settings = self.__settings
        if settings is None:
            return
        settings.beginGroup("library")
        plotBackend = settings.value("plot.backend", "")
        plotImageYAxisOrientation = settings.value("plot-image.y-axis-orientation", "")
        settings.endGroup()

        if plotBackend != "":
            silx.config.DEFAULT_PLOT_BACKEND = plotBackend
        if plotImageYAxisOrientation != "":
            silx.config.DEFAULT_PLOT_IMAGE_Y_AXIS_ORIENTATION = plotImageYAxisOrientation

    def restoreSettings(self):
        """Restore the settings of all the application"""
        settings = self.__settings
        if settings is None:
            return
        parent = self.__parent()
        parent.restoreSettings(settings)

        settings.beginGroup("colormap")
        byteArray = settings.value("default", None)
        if byteArray is not None:
            try:
                colormap = Colormap()
                colormap.restoreState(byteArray)
                self.__defaultColormap = colormap
            except Exception:
                _logger.debug("Backtrace", exc_info=True)
        settings.endGroup()

        self.__recentFiles = []
        settings.beginGroup("recent-files")
        for index in range(1, 10 + 1):
            if not settings.contains("path%d" % index):
                break
            filePath = settings.value("path%d" % index)
            self.__recentFiles.append(filePath)
        settings.endGroup()

    def saveSettings(self):
        """Save the settings of all the application"""
        settings = self.__settings
        if settings is None:
            return
        parent = self.__parent()
        parent.saveSettings(settings)

        if self.__defaultColormap is not None:
            settings.beginGroup("colormap")
            settings.setValue("default", self.__defaultColormap.saveState())
            settings.endGroup()

        settings.beginGroup("library")
        settings.setValue("plot.backend", silx.config.DEFAULT_PLOT_BACKEND)
        settings.setValue("plot-image.y-axis-orientation", silx.config.DEFAULT_PLOT_IMAGE_Y_AXIS_ORIENTATION)
        settings.endGroup()

        settings.beginGroup("recent-files")
        for index in range(0, 11):
            key = "path%d" % (index + 1)
            if index < len(self.__recentFiles):
                filePath = self.__recentFiles[index]
                settings.setValue(key, filePath)
            else:
                settings.remove(key)
        settings.endGroup()

    def getRecentFiles(self):
        """Returns the list of recently opened files.

        The list is limited to the last 10 entries. The newest file path is
        in first.

        :rtype: List[str]
        """
        return self.__recentFiles

    def pushRecentFile(self, filePath):
        """Push a new recent file to the list.

        If the file is duplicated in the list, all duplications are removed
        before inserting the new filePath.

        If the list becan bigger than 10 items, oldest paths are removed.

        :param filePath: File path to push
        """
        # Remove old occurencies
        self.__recentFiles[:] = (f for f in self.__recentFiles if f != filePath)
        self.__recentFiles.insert(0, filePath)
        while len(self.__recentFiles) > 10:
            self.__recentFiles.pop()

    def clearRencentFiles(self):
        """Clear the history of the rencent files.
        """
        self.__recentFiles[:] = []

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
