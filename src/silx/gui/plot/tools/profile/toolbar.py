# /*##########################################################################
#
# Copyright (c) 2018-2019 European Synchrotron Radiation Facility
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
"""This module provides tool bar helper.
"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "28/06/2018"


import logging
import weakref

from silx.gui import qt
from silx.gui.widgets.MultiModeAction import MultiModeAction
from . import manager
from .. import roi as roi_mdl
from silx.gui.plot import items


_logger = logging.getLogger(__name__)


class ProfileToolBar(qt.QToolBar):
    """Tool bar to provide profile for a plot.
    
    It is an helper class. For a dedicated application it would be better to
    use an own tool bar in order in order have more flexibility.
    """
    def __init__(self, parent=None, plot=None):
        super(ProfileToolBar, self).__init__(parent=parent)
        self.__scheme = None
        self.__manager = None
        self.__plot = weakref.ref(plot)
        self.__multiAction = None

    def getPlotWidget(self):
        """The :class:`~silx.gui.plot.PlotWidget` associated to the toolbar.

        :rtype: Union[~silx.gui.plot.PlotWidget,None]
        """
        if self.__plot is None:
            return None
        plot = self.__plot()
        if self.__plot is None:
            self.__plot = None
        return plot

    def setScheme(self, scheme):
        """Initialize the tool bar using a configuration scheme.

        It have to be done once and only once.

        :param str scheme: One of "scatter", "image", "imagestack"
        """
        assert self.__scheme is None
        self.__scheme = scheme

        plot = self.getPlotWidget()
        self.__manager = manager.ProfileManager(self, plot)

        if scheme == "image":
            self.__manager.setItemType(image=True)
            self.__manager.setActiveItemTracking(True)

            multiAction = MultiModeAction(self)
            self.addAction(multiAction)
            for action in self.__manager.createImageActions(self):
                multiAction.addAction(action)
            self.__multiAction = multiAction

            cleanAction = self.__manager.createClearAction(self)
            self.addAction(cleanAction)
            editorAction = self.__manager.createEditorAction(self)
            self.addAction(editorAction)

            plot.sigActiveImageChanged.connect(self._activeImageChanged)
            self._activeImageChanged()

        elif scheme == "scatter":
            self.__manager.setItemType(scatter=True)
            self.__manager.setActiveItemTracking(True)

            multiAction = MultiModeAction(self)
            self.addAction(multiAction)
            for action in self.__manager.createScatterActions(self):
                multiAction.addAction(action)
            for action in self.__manager.createScatterSliceActions(self):
                multiAction.addAction(action)
            self.__multiAction = multiAction

            cleanAction = self.__manager.createClearAction(self)
            self.addAction(cleanAction)
            editorAction = self.__manager.createEditorAction(self)
            self.addAction(editorAction)

            plot.sigActiveScatterChanged.connect(self._activeScatterChanged)
            self._activeScatterChanged()

        elif scheme == "imagestack":
            self.__manager.setItemType(image=True)
            self.__manager.setActiveItemTracking(True)

            multiAction = MultiModeAction(self)
            self.addAction(multiAction)
            for action in self.__manager.createImageStackActions(self):
                multiAction.addAction(action)
            self.__multiAction = multiAction

            cleanAction = self.__manager.createClearAction(self)
            self.addAction(cleanAction)
            editorAction = self.__manager.createEditorAction(self)
            self.addAction(editorAction)

            plot.sigActiveImageChanged.connect(self._activeImageChanged)
            self._activeImageChanged()

        else:
            raise ValueError("Toolbar scheme %s unsupported" % scheme)

    def _setRoiActionEnabled(self, itemKind, enabled):
        for action in self.__multiAction.getMenu().actions():
            if not isinstance(action, roi_mdl.CreateRoiModeAction):
                continue
            roiClass = action.getRoiClass()
            if issubclass(itemKind, roiClass.ITEM_KIND):
                action.setEnabled(enabled)

    def _activeImageChanged(self, previous=None, legend=None):
        """Handle active image change to toggle actions"""
        if legend is None:
            self._setRoiActionEnabled(items.ImageStack, False)
            self._setRoiActionEnabled(items.ImageBase, False)
        else:
            plot = self.getPlotWidget()
            image = plot.getActiveImage()
            # Disable for empty image
            enabled = image.getData(copy=False).size > 0
            self._setRoiActionEnabled(type(image), enabled)

    def _activeScatterChanged(self, previous=None, legend=None):
        """Handle active scatter change to toggle actions"""
        if legend is None:
            self._setRoiActionEnabled(items.Scatter, False)
        else:
            plot = self.getPlotWidget()
            scatter = plot.getActiveScatter()
            # Disable for empty image
            enabled = scatter.getValueData(copy=False).size > 0
            self._setRoiActionEnabled(type(scatter), enabled)
