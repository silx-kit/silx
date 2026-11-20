# /*##########################################################################
# Copyright (C) 2016-2023 European Synchrotron Radiation Facility
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
from collections.abc import Sequence

import silx
from silx.gui import qt
from silx.gui.data.DataViews import DataViewHooks
from silx.gui.data.NXdataWidgets import ArrayComplexImagePlot, ArrayImagePlot
from silx.gui.colors import Colormap
from silx.gui.dialog.ColormapDialog import ColormapDialog
from silx.gui.plot import Plot2D
from silx.gui.plot.ComplexImageView import ComplexImageView
from silx.gui.plot.actions.image import DynamicColormapAction

_logger = logging.getLogger(__name__)


class ApplicationContext(DataViewHooks):
    """
    Store the context of the application

    It overwrites the DataViewHooks to custom the use of the DataViewer for
    the silx view application.

    - Create a single colormap shared with all the views
    - Create a single colormap dialog shared with all the views
    - Add a dynamic colormap action (W key) to the image views.
      When the displayed data changes, it is deactivated and the colormap
      range it sets is reset.
    """

    def __init__(self, parent, settings=None):
        self.__parent = weakref.ref(parent)
        self.__defaultColormap = None
        self.__defaultColormapDialog = None
        self.__settings = settings
        self.__recentFiles = []
        self.__dynamicColormapActions = weakref.WeakSet()
        self.__colormapRangeToRestore = None
        """Default colormap range before using dynamic colormap actions"""

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
        mplTightLayout = settings.value("mpl.tight_layout", False, bool)
        plotBackend = settings.value("plot.backend", "")
        plotImageYAxisOrientation = settings.value("plot-image.y-axis-orientation", "")
        settings.endGroup()

        # Use matplotlib backend by default
        silx.config.DEFAULT_PLOT_BACKEND = (
            ("opengl", "matplotlib") if plotBackend == "opengl" else "matplotlib"
        )
        if plotImageYAxisOrientation != "":
            silx.config.DEFAULT_PLOT_IMAGE_Y_AXIS_ORIENTATION = (
                plotImageYAxisOrientation
            )
        silx.config._MPL_TIGHT_LAYOUT = mplTightLayout

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
            colormap = self.__defaultColormap
            if self.__colormapRangeToRestore is not None:
                # Do not save the range set by dynamic colormap actions
                colormap = colormap.copy()
                colormap.setVRange(*self.__colormapRangeToRestore)
            settings.beginGroup("colormap")
            settings.setValue("default", colormap.saveState())
            settings.endGroup()

        settings.beginGroup("library")
        settings.setValue("plot.backend", self.getDefaultPlotBackend())
        settings.setValue(
            "plot-image.y-axis-orientation",
            silx.config.DEFAULT_PLOT_IMAGE_Y_AXIS_ORIENTATION,
        )
        settings.setValue("mpl.tight_layout", silx.config._MPL_TIGHT_LAYOUT)
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
        """Clear the history of the rencent files."""
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

    def viewWidgetCreated(self, view, widget):
        """Add a dynamic colormap action to the image plots of the views.

        Override from DataViewHooks
        """
        if isinstance(widget, Plot2D):
            plot = widget
        elif isinstance(
            widget, (ArrayImagePlot, ArrayComplexImagePlot, ComplexImageView)
        ):
            plot = widget.getPlot()
        else:
            return

        action = DynamicColormapAction(plot, parent=plot)
        action.setShortcut(qt.QKeySequence(qt.Qt.Key_W))
        action.setShortcutContext(qt.Qt.WidgetWithChildrenShortcut)
        plot.addAction(action)  # Enable the shortcut
        plot.toolBar().addAction(action)
        action.toggled.connect(self.__dynamicColormapToggled)
        self.__dynamicColormapActions.add(action)

    def __dynamicColormapToggled(self, checked: bool):
        if checked and self.__colormapRangeToRestore is None:
            self.__colormapRangeToRestore = self.getColormap(None).getVRange()

    def dataChanged(self):
        """Notify that the displayed data has changed.

        This deactivates the dynamic colormap actions and restores the
        colormap range as it was before using them.
        """
        for action in list(self.__dynamicColormapActions):
            if action.isChecked():
                action.trigger()  # Uncheck and stop the interaction

        if self.__colormapRangeToRestore is None:
            return
        colormap = self.getColormap(None)
        if colormap.isEditable():
            colormap.setVRange(*self.__colormapRangeToRestore)
        self.__colormapRangeToRestore = None

    @staticmethod
    def getDefaultPlotBackend() -> str:
        """Returns default plot backend as a str from current config"""
        backend = silx.config.DEFAULT_PLOT_BACKEND
        if isinstance(backend, str):
            return backend
        if (
            isinstance(backend, Sequence)
            and len(backend)
            and isinstance(backend[0], str)
        ):
            return backend[0]
        return "matplotlib"  # fallback
