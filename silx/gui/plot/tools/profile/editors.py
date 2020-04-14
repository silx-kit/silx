# coding: utf-8
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
"""This module provides editors which are used to custom profile ROI properties.
"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "28/06/2018"

import logging

from silx.gui import qt

from silx.gui.utils import blockSignals
from silx.gui.plot.PlotToolButtons import ProfileOptionToolButton
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

        self._lineWidth = qt.QSpinBox(self)
        self._lineWidth.setRange(1, 1000)
        self._lineWidth.setValue(1)
        self._lineWidth.valueChanged[int].connect(self.__widgetChanged)

        self._methodsButton = ProfileOptionToolButton(parent=self, plot=None)
        self._methodsButton.sigMethodChanged.connect(self.__widgetChanged)

        layout = qt.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        label = qt.QLabel('W:')
        label.setToolTip("Line width in pixels")
        layout.addWidget(label)
        layout.addWidget(self._lineWidth)
        layout.addWidget(self._methodsButton)

    def __widgetChanged(self, value=None):
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
        label = qt.QLabel('W:')
        label.setToolTip("Line width in pixels")
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


class ProfileRoiEditAction(qt.QWidgetAction):

    def __init__(self, parent=None):
        super(ProfileRoiEditAction, self).__init__(parent)
        self.__widget = qt.QWidget(parent)
        layout = qt.QHBoxLayout(self.__widget)
        layout.setContentsMargins(0, 0, 0, 0)
        self.__editor = None
        self.__setEditor(_NoProfileRoiEditor(parent))
        self.setDefaultWidget(self.__widget)
        self.__roiManager = None
        self.__roi = None
        self.__inhibiteReentance = False

    def setRoiManager(self, roiManager):
        if self.__roiManager is roiManager:
            return
        if self.__roiManager is not None:
            self.__roiManager.sigRoiSelected.disconnect(self.__selectedRoiChanged)
        self.__roiManager = roiManager
        if self.__roiManager is not None:
            self.__roiManager.sigRoiSelected.connect(self.__selectedRoiChanged)

    def __selectedRoiChanged(self, roi):
        if roi is not None and not isinstance(roi, core.ProfileRoiMixIn):
            return
        self.setProfileRoi(roi)

    def setProfileRoi(self, roi):
        if self.__roi is roi:
            return
        if self.__roi is not None:
            self.__roi.sigPropertyChanged.disconnect(self.__roiPropertyChanged)
        self.__roi = roi
        if self.__roi is not None:
            self.__roi.sigPropertyChanged.connect(self.__roiPropertyChanged)
        self._updateWidget()

    def __roiPropertyChanged(self):
        if self.__inhibiteReentance:
            return
        self._updateWidgetValues()

    def __setEditor(self, editor):
        layout = self.__widget.layout()
        if self.__editor is editor:
            return
        if self.__editor is not None:
            self.__editor.sigDataCommited.disconnect(self.__editorDataCommited)
            layout.removeWidget(self.__editor)
            self.__editor.deleteLater()
        self.__editor = editor
        if self.__editor is not None:
            self.__editor.sigDataCommited.connect(self.__editorDataCommited)
            layout.addWidget(self.__editor)

    def _updateWidget(self):
        parent = self.parent()
        if self.__roi is None:
            editor = _NoProfileRoiEditor(parent)
        elif isinstance(self.__roi, (rois._DefaultImageProfileRoiMixIn,
                                     rois.ProfileImageCrossROI)):
            editor = _DefaultImageProfileRoiEditor(parent)
        elif isinstance(self.__roi, (rois._DefaultScatterProfileRoiMixIn,
                                     rois.ProfileScatterCrossROI)):
            editor = _DefaultScatterProfileRoiEditor(parent)
        else:
            # Unsupported
            editor = _NoProfileRoiEditor(parent)
        editor.setEditorData(self.__roi)
        self.__setEditor(editor)

    def _updateWidgetValues(self):
        self.__editor.setEditorData(self.__roi)

    def __editorDataCommited(self):
        if self.__roi is not None:
            self.__inhibiteReentance = True
            self.__editor.setRoiData(self.__roi)
            self.__inhibiteReentance = False
