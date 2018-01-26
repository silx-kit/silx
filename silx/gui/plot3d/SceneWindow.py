# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017-2018 European Synchrotron Radiation Facility
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
"""This module provides a QMainWindow with a 3D SceneWidget and toolbars.
"""

from __future__ import absolute_import

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "29/11/2017"


from silx.gui import qt

from .SceneWidget import SceneWidget
from .tools import OutputToolBar, InteractiveModeToolBar, ViewpointToolBar
from silx.gui.plot3d.tools.GroupPropertiesWidget import GroupPropertiesWidget

from .ParamTreeView import ParamTreeView

# Imported here for convenience
from . import items  # noqa


__all__ = ['items', 'SceneWidget', 'SceneWindow']


class SceneWindow(qt.QMainWindow):
    """OpenGL 3D scene widget with toolbars."""

    def __init__(self, parent=None):
        super(SceneWindow, self).__init__(parent)
        if parent is not None:
            # behave as a widget
            self.setWindowFlags(qt.Qt.Widget)

        self._sceneWidget = SceneWidget()
        self.setCentralWidget(self._sceneWidget)

        self._interactiveModeToolBar = InteractiveModeToolBar(parent=self)
        self._viewpointToolBar = ViewpointToolBar(parent=self)
        self._outputToolBar = OutputToolBar(parent=self)

        for toolbar in (self._interactiveModeToolBar,
                        self._viewpointToolBar,
                        self._outputToolBar):
            toolbar.setPlot3DWidget(self._sceneWidget)
            self.addToolBar(toolbar)
            self.addActions(toolbar.actions())

        self._paramTreeView = ParamTreeView()
        self._paramTreeView.setModel(self._sceneWidget.model())

        selectionModel = self._paramTreeView.selectionModel()
        self._sceneWidget.selection()._setSyncSelectionModel(
            selectionModel)

        paramDock = qt.QDockWidget()
        paramDock.setWindowTitle('Object parameters')
        paramDock.setWidget(self._paramTreeView)
        self.addDockWidget(qt.Qt.RightDockWidgetArea, paramDock)

        self._sceneGroupResetWidget = GroupPropertiesWidget()
        self._sceneGroupResetWidget.setGroup(
            self._sceneWidget.getSceneGroup())

        resetDock = qt.QDockWidget()
        resetDock.setWindowTitle('Global parameters')
        resetDock.setWidget(self._sceneGroupResetWidget)
        self.addDockWidget(qt.Qt.RightDockWidgetArea, resetDock)
        self.tabifyDockWidget(paramDock, resetDock)

        paramDock.raise_()

    def getSceneWidget(self):
        """Returns the SceneWidget of this window.

        :rtype: ~silx.gui.plot3d.SceneWidget.SceneWidget
        """
        return self._sceneWidget

    def getParamTreeView(self):
        """Returns the :class:`ParamTreeView` of this window.

        :rtype: ParamTreeView
        """
        return self._paramTreeView

    def getInteractiveModeToolBar(self):
        """Returns the interactive mode toolbar.

        :rtype: InteractiveModeToolBar
        """
        return self._interactiveModeToolBar

    def getViewpointToolBar(self):
        """Returns the viewpoint toolbar.

        :rtype: ViewpointToolBar
        """
        return self._viewpointToolBar

    def getOutputToolBar(self):
        """Returns the output toolbar.

        :rtype: OutputToolBar
        """
        return self._outputToolBar
