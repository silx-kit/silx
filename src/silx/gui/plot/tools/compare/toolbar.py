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
from typing import List, Optional

from silx.gui import qt
from silx.gui import icons
from .core import AlignmentMode
from .core import VisualizationMode
from .core import sift


_logger = logging.getLogger(__name__)


class AlignmentModeToolButton(qt.QToolButton):
    """ToolButton to select a AlignmentMode"""

    sigSelected = qt.Signal(AlignmentMode)

    def __init__(self, parent=None):
        super(AlignmentModeToolButton, self).__init__(parent=parent)

        menu = qt.QMenu(self)
        self.setMenu(menu)

        self.__group = qt.QActionGroup(self)
        self.__group.setExclusive(True)
        self.__group.triggered.connect(self.__selectionChanged)

        icon = icons.getQIcon("compare-align-origin")
        action = qt.QAction(icon, "Align images on their upper-left pixel", self)
        action.setProperty("enum", AlignmentMode.ORIGIN)
        action.setIconVisibleInMenu(True)
        action.setCheckable(True)
        self.__originAlignAction = action
        menu.addAction(action)
        self.__group.addAction(action)

        icon = icons.getQIcon("compare-align-center")
        action = qt.QAction(icon, "Center images", self)
        action.setProperty("enum", AlignmentMode.CENTER)
        action.setIconVisibleInMenu(True)
        action.setCheckable(True)
        self.__centerAlignAction = action
        menu.addAction(action)
        self.__group.addAction(action)

        icon = icons.getQIcon("compare-align-stretch")
        action = qt.QAction(icon, "Stretch the second image on the first one", self)
        action.setProperty("enum", AlignmentMode.STRETCH)
        action.setIconVisibleInMenu(True)
        action.setCheckable(True)
        self.__stretchAlignAction = action
        menu.addAction(action)
        self.__group.addAction(action)

        icon = icons.getQIcon("compare-align-auto")
        action = qt.QAction(icon, "Auto-alignment of the second image", self)
        action.setProperty("enum", AlignmentMode.AUTO)
        action.setIconVisibleInMenu(True)
        action.setCheckable(True)
        self.__autoAlignAction = action
        menu.addAction(action)
        if sift is None:
            action.setEnabled(False)
            action.setToolTip("Sift module is not available")
        self.__group.addAction(action)

    def getActionFromMode(self, mode: AlignmentMode) -> Optional[qt.QAction]:
        """Returns an action from it's mode"""
        for action in self.__group.actions():
            actionMode = action.property("enum")
            if mode == actionMode:
                return action
        return None

    def setVisibleModes(self, modes: List[AlignmentMode]):
        """Make visible only a set of modes.

        The order does not matter.
        """
        modes = set(modes)
        for action in self.__group.actions():
            mode = action.property("enum")
            action.setVisible(mode in modes)

    def __selectionChanged(self, selectedAction: qt.QAction):
        """Called when user requesting changes of the alignment mode."""
        self.__updateMenu()
        mode = self.getSelected()
        self.sigSelected.emit(mode)

    def __updateMenu(self):
        """Update the state of the action containing alignment menu."""
        selectedAction = self.__group.checkedAction()
        if selectedAction is not None:
            self.setText(selectedAction.text())
            self.setIcon(selectedAction.icon())
            self.setToolTip(selectedAction.toolTip())
        else:
            self.setText("")
            self.setIcon(qt.QIcon())
            self.setToolTip("")

    def getSelected(self) -> AlignmentMode:
        action = self.__group.checkedAction()
        if action is None:
            return None
        return action.property("enum")

    def setSelected(self, mode: AlignmentMode):
        action = self.getActionFromMode(mode)
        old = self.__group.blockSignals(True)
        if action is not None:
            # Check this action
            action.setChecked(True)
        else:
            action = self.__group.checkedAction()
            if action is not None:
                # Uncheck this action
                action.setChecked(False)
        self.__updateMenu()
        self.__group.blockSignals(old)


class VisualizationModeToolButton(qt.QToolButton):
    """ToolButton to select a VisualisationMode"""

    sigSelected = qt.Signal(VisualizationMode)

    def __init__(self, parent=None):
        super(VisualizationModeToolButton, self).__init__(parent=parent)

        menu = qt.QMenu(self)
        self.setMenu(menu)

        self.__group = qt.QActionGroup(self)
        self.__group.setExclusive(True)
        self.__group.triggered.connect(self.__selectionChanged)

        icon = icons.getQIcon("compare-mode-a")
        action = qt.QAction(icon, "Display the first image only", self)
        action.setIconVisibleInMenu(True)
        action.setCheckable(True)
        action.setShortcut(qt.QKeySequence(qt.Qt.Key_A))
        action.setProperty("enum", VisualizationMode.ONLY_A)
        menu.addAction(action)
        self.__aModeAction = action
        self.__group.addAction(action)

        icon = icons.getQIcon("compare-mode-b")
        action = qt.QAction(icon, "Display the second image only", self)
        action.setIconVisibleInMenu(True)
        action.setCheckable(True)
        action.setShortcut(qt.QKeySequence(qt.Qt.Key_B))
        action.setProperty("enum", VisualizationMode.ONLY_B)
        menu.addAction(action)
        self.__bModeAction = action
        self.__group.addAction(action)

        icon = icons.getQIcon("compare-mode-vline")
        action = qt.QAction(icon, "Vertical compare mode", self)
        action.setIconVisibleInMenu(True)
        action.setCheckable(True)
        action.setShortcut(qt.QKeySequence(qt.Qt.Key_V))
        action.setProperty("enum", VisualizationMode.VERTICAL_LINE)
        menu.addAction(action)
        self.__vlineModeAction = action
        self.__group.addAction(action)

        icon = icons.getQIcon("compare-mode-hline")
        action = qt.QAction(icon, "Horizontal compare mode", self)
        action.setIconVisibleInMenu(True)
        action.setCheckable(True)
        action.setShortcut(qt.QKeySequence(qt.Qt.Key_H))
        action.setProperty("enum", VisualizationMode.HORIZONTAL_LINE)
        menu.addAction(action)
        self.__hlineModeAction = action
        self.__group.addAction(action)

        icon = icons.getQIcon("compare-mode-rb-channel")
        action = qt.QAction(icon, "Blue/red compare mode (additive mode)", self)
        action.setIconVisibleInMenu(True)
        action.setCheckable(True)
        action.setShortcut(qt.QKeySequence(qt.Qt.Key_C))
        action.setProperty("enum", VisualizationMode.COMPOSITE_RED_BLUE_GRAY)
        menu.addAction(action)
        self.__brChannelModeAction = action
        self.__group.addAction(action)

        icon = icons.getQIcon("compare-mode-rbneg-channel")
        action = qt.QAction(icon, "Yellow/cyan compare mode (subtractive mode)", self)
        action.setIconVisibleInMenu(True)
        action.setCheckable(True)
        action.setShortcut(qt.QKeySequence(qt.Qt.Key_Y))
        action.setProperty("enum", VisualizationMode.COMPOSITE_RED_BLUE_GRAY_NEG)
        menu.addAction(action)
        self.__ycChannelModeAction = action
        self.__group.addAction(action)

        icon = icons.getQIcon("compare-mode-a-minus-b")
        action = qt.QAction(icon, "Raw A minus B compare mode", self)
        action.setIconVisibleInMenu(True)
        action.setCheckable(True)
        action.setShortcut(qt.QKeySequence(qt.Qt.Key_W))
        action.setProperty("enum", VisualizationMode.COMPOSITE_A_MINUS_B)
        menu.addAction(action)
        self.__ycChannelModeAction = action
        self.__group.addAction(action)

    def getActionFromMode(self, mode: VisualizationMode) -> Optional[qt.QAction]:
        """Returns an action from it's mode"""
        for action in self.__group.actions():
            actionMode = action.property("enum")
            if mode == actionMode:
                return action
        return None

    def setVisibleModes(self, modes: List[VisualizationMode]):
        """Make visible only a set of modes.

        The order does not matter.
        """
        modes = set(modes)
        for action in self.__group.actions():
            mode = action.property("enum")
            action.setVisible(mode in modes)

    def __selectionChanged(self, selectedAction: qt.QAction):
        """Called when user requesting changes of the visualization mode."""
        self.__updateMenu()
        mode = self.getSelected()
        self.sigSelected.emit(mode)

    def __updateMenu(self):
        """Update the state of the action containing visualization menu."""
        selectedAction = self.__group.checkedAction()
        if selectedAction is not None:
            self.setText(selectedAction.text())
            self.setIcon(selectedAction.icon())
            self.setToolTip(selectedAction.toolTip())
        else:
            self.setText("")
            self.setIcon(qt.QIcon())
            self.setToolTip("")

    def getSelected(self) -> VisualizationMode:
        action = self.__group.checkedAction()
        if action is None:
            return None
        return action.property("enum")

    def setSelected(self, mode: VisualizationMode):
        action = self.getActionFromMode(mode)
        old = self.__group.blockSignals(True)
        if action is not None:
            # Check this action
            action.setChecked(True)
        else:
            action = self.__group.checkedAction()
            if action is not None:
                # Uncheck this action
                action.setChecked(False)
        self.__updateMenu()
        self.__group.blockSignals(old)


class CompareImagesToolBar(qt.QToolBar):
    """ToolBar containing specific tools to custom the configuration of a
    :class:`CompareImages` widget

    Use :meth:`setCompareWidget` to connect this toolbar to a specific
    :class:`CompareImages` widget.

    :param Union[qt.QWidget,None] parent: Parent of this widget.
    """

    def __init__(self, parent=None):
        qt.QToolBar.__init__(self, parent)
        self.setWindowTitle("Compare images")

        self.__compareWidget = None

        self.__visualizationToolButton = VisualizationModeToolButton(self)
        self.__visualizationToolButton.setPopupMode(qt.QToolButton.InstantPopup)
        self.__visualizationToolButton.sigSelected.connect(self.__visualizationChanged)
        self.addWidget(self.__visualizationToolButton)

        self.__alignmentToolButton = AlignmentModeToolButton(self)
        self.__alignmentToolButton.setPopupMode(qt.QToolButton.InstantPopup)
        self.__alignmentToolButton.sigSelected.connect(self.__alignmentChanged)
        self.addWidget(self.__alignmentToolButton)

        icon = icons.getQIcon("compare-keypoints")
        action = qt.QAction(icon, "Display/hide alignment keypoints", self)
        action.setCheckable(True)
        action.triggered.connect(self.__keypointVisibilityChanged)
        self.addAction(action)
        self.__displayKeypoints = action

    def __visualizationChanged(self, mode: VisualizationMode):
        widget = self.getCompareWidget()
        if widget is not None:
            widget.setVisualizationMode(mode)

    def __alignmentChanged(self, mode: AlignmentMode):
        widget = self.getCompareWidget()
        if widget is not None:
            widget.setAlignmentMode(mode)

    def setCompareWidget(self, widget):
        """
        Connect this tool bar to a specific :class:`CompareImages` widget.

        :param Union[None,CompareImages] widget: The widget to connect with.
        """
        compareWidget = self.getCompareWidget()
        if compareWidget is not None:
            compareWidget.sigConfigurationChanged.disconnect(
                self.__updateSelectedActions
            )
        compareWidget = widget
        self.setEnabled(compareWidget is not None)
        if compareWidget is None:
            self.__compareWidget = None
        else:
            self.__compareWidget = weakref.ref(compareWidget)
        if compareWidget is not None:
            widget.sigConfigurationChanged.connect(self.__updateSelectedActions)
        self.__updateSelectedActions()

    def getCompareWidget(self):
        """Returns the connected widget.

        :rtype: CompareImages
        """
        if self.__compareWidget is None:
            return None
        else:
            return self.__compareWidget()

    def __updateSelectedActions(self):
        """
        Update the state of this tool bar according to the state of the
        connected :class:`CompareImages` widget.
        """
        widget = self.getCompareWidget()
        if widget is None:
            return
        self.__visualizationToolButton.setSelected(widget.getVisualizationMode())
        self.__alignmentToolButton.setSelected(widget.getAlignmentMode())
        self.__displayKeypoints.setChecked(widget.getKeypointsVisible())

    def __keypointVisibilityChanged(self):
        """Called when action managing keypoints visibility changes"""
        widget = self.getCompareWidget()
        if widget is not None:
            keypointsVisible = self.__displayKeypoints.isChecked()
            widget.setKeypointsVisible(keypointsVisible)
