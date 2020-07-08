# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2018-2020 European Synchrotron Radiation Facility
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
"""This module provides editors which are used to custom profile ROI properties.
"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "28/06/2018"

import logging

from silx.gui import qt

from silx.gui.utils import blockSignals
from silx.gui.plot.PlotToolButtons import ProfileOptionToolButton
from silx.gui.plot.PlotToolButtons import ProfileToolButton
from . import rois
from . import core


_logger = logging.getLogger(__name__)


class _NoProfileRoiEditor(qt.QWidget):

    sigDataCommited = qt.Signal()

    def setEditorData(self, roi):
        pass

    def setRoiData(self, roi):
        pass


class _DefaultImageProfileRoiEditor(qt.QWidget):

    sigDataCommited = qt.Signal()

    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent=parent)
        layout = qt.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._initLayout(layout)

    def _initLayout(self, layout):
        self._lineWidth = qt.QSpinBox(self)
        self._lineWidth.setRange(1, 1000)
        self._lineWidth.setValue(1)
        self._lineWidth.valueChanged[int].connect(self._widgetChanged)

        self._methodsButton = ProfileOptionToolButton(parent=self, plot=None)
        self._methodsButton.sigMethodChanged.connect(self._widgetChanged)

        label = qt.QLabel('W:')
        label.setToolTip("Line width in pixels")
        layout.addWidget(label)
        layout.addWidget(self._lineWidth)
        layout.addWidget(self._methodsButton)

    def _widgetChanged(self, value=None):
        self.commitData()

    def commitData(self):
        self.sigDataCommited.emit()

    def setEditorData(self, roi):
        with blockSignals(self._lineWidth):
            self._lineWidth.setValue(roi.getProfileLineWidth())
        with blockSignals(self._methodsButton):
            method = roi.getProfileMethod()
            self._methodsButton.setMethod(method)

    def setRoiData(self, roi):
        lineWidth = self._lineWidth.value()
        roi.setProfileLineWidth(lineWidth)
        method = self._methodsButton.getMethod()
        roi.setProfileMethod(method)


class _DefaultImageStackProfileRoiEditor(_DefaultImageProfileRoiEditor):

    def _initLayout(self, layout):
        super(_DefaultImageStackProfileRoiEditor, self)._initLayout(layout)
        self._profileDim = ProfileToolButton(parent=self, plot=None)
        self._profileDim.sigDimensionChanged.connect(self._widgetChanged)
        layout.addWidget(self._profileDim)

    def setEditorData(self, roi):
        super(_DefaultImageStackProfileRoiEditor, self).setEditorData(roi)
        with blockSignals(self._profileDim):
            kind = roi.getProfileType()
            dim = {"1D": 1, "2D": 2}[kind]
            self._profileDim.setDimension(dim)

    def setRoiData(self, roi):
        super(_DefaultImageStackProfileRoiEditor, self).setRoiData(roi)
        dim = self._profileDim.getDimension()
        kind = {1: "1D", 2: "2D"}[dim]
        roi.setProfileType(kind)


class _DefaultScatterProfileRoiEditor(qt.QWidget):

    sigDataCommited = qt.Signal()

    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent=parent)

        self._nPoints = qt.QSpinBox(self)
        self._nPoints.setRange(1, 9999)
        self._nPoints.setValue(1024)
        self._nPoints.valueChanged[int].connect(self.__widgetChanged)

        layout = qt.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        label = qt.QLabel('Samples:')
        label.setToolTip("Number of sample points of the profile")
        layout.addWidget(label)
        layout.addWidget(self._nPoints)

    def __widgetChanged(self, value=None):
        self.commitData()

    def commitData(self):
        self.sigDataCommited.emit()

    def setEditorData(self, roi):
        with blockSignals(self._nPoints):
            self._nPoints.setValue(roi.getNPoints())

    def setRoiData(self, roi):
        nPoints = self._nPoints.value()
        roi.setNPoints(nPoints)


class ProfileRoiEditorAction(qt.QWidgetAction):
    """
    Action displaying GUI to edit the selected ROI.

    :param qt.QWidget parent: Parent widget
    """
    def __init__(self, parent=None):
        super(ProfileRoiEditorAction, self).__init__(parent)
        self.__roiManager = None
        self.__roi = None
        self.__inhibiteReentance = None

    def createWidget(self, parent):
        """Inherit the method to create a new editor"""
        widget = qt.QWidget(parent)
        layout = qt.QHBoxLayout(widget)
        if isinstance(parent, qt.QMenu):
            margins = layout.contentsMargins()
            layout.setContentsMargins(margins.left(), 0, margins.right(), 0)
        else:
            layout.setContentsMargins(0, 0, 0, 0)

        editorClass = self.getEditorClass(self.__roi)
        editor = editorClass(parent)
        editor.setEditorData(self.__roi)
        self.__setEditor(widget, editor)
        return widget

    def deleteWidget(self, widget):
        """Inherit the method to delete an editor"""
        self.__setEditor(widget, None)
        return qt.QWidgetAction.deleteWidget(self, widget)

    def _getEditor(self, widget):
        """Returns the editor contained in the widget holder"""
        layout = widget.layout()
        if layout.count() == 0:
            return None
        return layout.itemAt(0).widget()

    def setRoiManager(self, roiManager):
        """
        Connect this action to a ROI manager.

        :param RegionOfInterestManager roiManager: A ROI manager
        """
        if self.__roiManager is roiManager:
            return
        if self.__roiManager is not None:
            self.__roiManager.sigCurrentRoiChanged.disconnect(self.__currentRoiChanged)
        self.__roiManager = roiManager
        if self.__roiManager is not None:
            self.__roiManager.sigCurrentRoiChanged.connect(self.__currentRoiChanged)
            self.__currentRoiChanged(roiManager.getCurrentRoi())

    def __currentRoiChanged(self, roi):
        """Handle changes of the selected ROI"""
        if roi is not None and not isinstance(roi, core.ProfileRoiMixIn):
            return
        self.setProfileRoi(roi)

    def setProfileRoi(self, roi):
        """Set a profile ROI to edit.

        :param ProfileRoiMixIn roi: A profile ROI
        """
        if self.__roi is roi:
            return
        if self.__roi is not None:
            self.__roi.sigProfilePropertyChanged.disconnect(self.__roiPropertyChanged)
        self.__roi = roi
        if self.__roi is not None:
            self.__roi.sigProfilePropertyChanged.connect(self.__roiPropertyChanged)
        self._updateWidgets()

    def getRoiProfile(self):
        """Returns the edited profile ROI.

        :rtype: ProfileRoiMixIn
        """
        return self.__roi

    def __roiPropertyChanged(self):
        """Handle changes on the property defining the ROI.
        """
        self._updateWidgetValues()

    def __setEditor(self, widget, editor):
        """Set the editor to display.

        :param qt.QWidget editor: The editor to display
        """
        previousEditor = self._getEditor(widget)
        if previousEditor is editor:
            return
        layout = widget.layout()
        if previousEditor is not None:
            previousEditor.sigDataCommited.disconnect(self._editorDataCommited)
            layout.removeWidget(previousEditor)
            previousEditor.deleteLater()
        if editor is not None:
            editor.sigDataCommited.connect(self._editorDataCommited)
            layout.addWidget(editor)

    def getEditorClass(self, roi):
        """Returns the editor class to use according to the ROI."""
        if roi is None:
            editorClass = _NoProfileRoiEditor
        elif isinstance(roi, (rois._DefaultImageStackProfileRoiMixIn,
                              rois.ProfileImageStackCrossROI)):
            # Must be done before the default image ROI
            # Cause ImageStack ROIs inherit from Image ROIs
            editorClass = _DefaultImageStackProfileRoiEditor
        elif isinstance(roi, (rois._DefaultImageProfileRoiMixIn,
                              rois.ProfileImageCrossROI)):
            editorClass = _DefaultImageProfileRoiEditor
        elif isinstance(roi, (rois._DefaultScatterProfileRoiMixIn,
                              rois.ProfileScatterCrossROI)):
            editorClass = _DefaultScatterProfileRoiEditor
        else:
            # Unsupported
            editorClass = _NoProfileRoiEditor
        return editorClass

    def _updateWidgets(self):
        """Update the kind of editor to display, according to the selected
        profile ROI."""
        parent = self.parent()
        editorClass = self.getEditorClass(self.__roi)
        for widget in self.createdWidgets():
            editor = editorClass(parent)
            editor.setEditorData(self.__roi)
            self.__setEditor(widget, editor)

    def _updateWidgetValues(self):
        """Update the content of the displayed editor, according to the
        selected profile ROI."""
        for widget in self.createdWidgets():
            editor = self._getEditor(widget)
            if self.__inhibiteReentance is editor:
                continue
            editor.setEditorData(self.__roi)

    def _editorDataCommited(self):
        """Handle changes from the editor."""
        editor = self.sender()
        if self.__roi is not None:
            self.__inhibiteReentance = editor
            editor.setRoiData(self.__roi)
            self.__inhibiteReentance = None
