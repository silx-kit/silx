# /*##########################################################################
#
# Copyright (c) 2018-2023 European Synchrotron Radiation Facility
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
"""This module provides ROI interaction for :class:`~silx.gui.plot.PlotWidget`.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "28/06/2018"


import enum
import logging
import time
import weakref
import functools
from typing import Optional

import numpy

from ... import qt, icons
from ...utils import blockSignals
from ...utils import LockReentrant
from .. import PlotWidget
from ..items import roi as roi_items
from ..items import ItemChangedType
from ..items.roi import RegionOfInterest

from ...colors import rgba


logger = logging.getLogger(__name__)


class CreateRoiModeAction(qt.QAction):
    """
    This action is a plot mode which allows to create new ROIs using a ROI
    manager.

    A ROI is created using a specific `roiClass`. `initRoi` and `finalizeRoi`
    can be inherited to custom the ROI initialization.

    :param class roiClass: The ROI class which will be created by this action.
    :param qt.QObject parent: The action parent
    :param RegionOfInterestManager roiManager: The ROI manager
    """

    def __init__(self, parent, roiManager, roiClass):
        assert roiManager is not None
        assert roiClass is not None
        qt.QAction.__init__(self, parent=parent)
        self._roiManager = weakref.ref(roiManager)
        self._roiClass = roiClass
        self._singleShot = False
        self._initAction()
        self.triggered[bool].connect(self._actionTriggered)

    def _initAction(self):
        """Default initialization of the action"""
        roiClass = self._roiClass

        name = None
        iconName = None
        if hasattr(roiClass, "NAME"):
            name = roiClass.NAME
        if hasattr(roiClass, "ICON"):
            iconName = roiClass.ICON

        if iconName is None:
            iconName = "add-shape-unknown"
        if name is None:
            name = roiClass.__name__
        text = "Add %s" % name
        self.setIcon(icons.getQIcon(iconName))
        self.setText(text)
        self.setCheckable(True)
        self.setToolTip(text)

    def getRoiClass(self):
        """Return the ROI class used by this action to create ROIs"""
        return self._roiClass

    def getRoiManager(self):
        return self._roiManager()

    def setSingleShot(self, singleShot):
        """Set it to True to deactivate the action after the first creation
        of a ROI.

        :param bool singleShot: New single short state
        """
        self._singleShot = singleShot

    def getSingleShot(self):
        """If True, after the first creation of a ROI with this mode,
        the mode is deactivated.

        :rtype: bool
        """
        return self._singleShot

    def _actionTriggered(self, checked):
        """Handle mode actions being checked by the user

        :param bool checked:
        :param str kind: Corresponding shape kind
        """
        roiManager = self.getRoiManager()
        if roiManager is None:
            return

        if checked:
            roiManager.start(self._roiClass, self)
            self.__interactiveModeStarted(roiManager)
        else:
            source = roiManager.getInteractionSource()
            if source is self:
                roiManager.stop()

    def __interactiveModeStarted(self, roiManager):
        roiManager.sigInteractiveRoiCreated.connect(self.initRoi)
        roiManager.sigInteractiveRoiFinalized.connect(self.__finalizeRoi)
        roiManager.sigInteractiveModeFinished.connect(self.__interactiveModeFinished)

    def __interactiveModeFinished(self):
        roiManager = self.getRoiManager()
        if roiManager is not None:
            roiManager.sigInteractiveRoiCreated.disconnect(self.initRoi)
            roiManager.sigInteractiveRoiFinalized.disconnect(self.__finalizeRoi)
            roiManager.sigInteractiveModeFinished.disconnect(
                self.__interactiveModeFinished
            )
        self.setChecked(False)

    def initRoi(self, roi):
        """Inherit it to custom the new ROI at it's creation during the
        interaction."""
        pass

    def __finalizeRoi(self, roi):
        self.finalizeRoi(roi)
        if self._singleShot:
            roiManager = self.getRoiManager()
            if roiManager is not None:
                roiManager.stop()

    def finalizeRoi(self, roi):
        """Inherit it to custom the new ROI after it's creation when the
        interaction is finalized."""
        pass


class RoiModeSelector(qt.QWidget):
    def __init__(self, parent=None):
        super(RoiModeSelector, self).__init__(parent=parent)
        self.__roi = None
        self.__reentrant = LockReentrant()

        layout = qt.QHBoxLayout(self)
        if isinstance(parent, qt.QMenu):
            margins = layout.contentsMargins()
            layout.setContentsMargins(margins.left(), 0, margins.right(), 0)
        else:
            layout.setContentsMargins(0, 0, 0, 0)

        self._label = qt.QLabel(self)
        self._label.setText("Mode:")
        self._label.setToolTip("Select a specific interaction to edit the ROI")
        self._combo = qt.QComboBox(self)
        self._combo.currentIndexChanged.connect(self._modeSelected)
        layout.addWidget(self._label)
        layout.addWidget(self._combo)
        self._updateAvailableModes()

    def getRoi(self):
        """Returns the edited ROI.

        :rtype: roi_items.RegionOfInterest
        """
        return self.__roi

    def setRoi(self, roi):
        """Returns the edited ROI.

        :rtype: roi_items.RegionOfInterest
        """
        if self.__roi is roi:
            return
        if not isinstance(roi, roi_items.InteractionModeMixIn):
            self.__roi = None
            self._updateAvailableModes()
            return

        if self.__roi is not None:
            self.__roi.sigInteractionModeChanged.disconnect(self._modeChanged)
        self.__roi = roi
        if self.__roi is not None:
            self.__roi.sigInteractionModeChanged.connect(self._modeChanged)
        self._updateAvailableModes()

    def isEmpty(self):
        return not self._label.isVisibleTo(self)

    def _updateAvailableModes(self):
        roi = self.getRoi()
        if isinstance(roi, roi_items.InteractionModeMixIn):
            modes = roi.availableInteractionModes()
        else:
            modes = []
        if len(modes) <= 1:
            self._label.setVisible(False)
            self._combo.setVisible(False)
        else:
            self._label.setVisible(True)
            self._combo.setVisible(True)
            with blockSignals(self._combo):
                self._combo.clear()
                for im, m in enumerate(modes):
                    self._combo.addItem(m.label, m)
                    self._combo.setItemData(im, m.description, qt.Qt.ToolTipRole)
                mode = roi.getInteractionMode()
                self._modeChanged(mode)
                index = modes.index(mode)
                self._combo.setCurrentIndex(index)

    def _modeChanged(self, mode):
        """Triggered when the ROI interaction mode was changed externally"""
        if self.__reentrant.locked():
            # This event was initialised by the widget
            return
        roi = self.__roi
        modes = roi.availableInteractionModes()
        index = modes.index(mode)
        with blockSignals(self._combo):
            self._combo.setCurrentIndex(index)

    def _modeSelected(self):
        """Triggered when the ROI interaction mode was selected in the widget"""
        index = self._combo.currentIndex()
        if index == -1:
            return
        roi = self.getRoi()
        if roi is not None:
            mode = self._combo.itemData(index, qt.Qt.UserRole)
            with self.__reentrant:
                roi.setInteractionMode(mode)


class RoiModeSelectorAction(qt.QWidgetAction):
    """Display the selected mode of a ROI and allow to change it"""

    def __init__(self, parent=None):
        super(RoiModeSelectorAction, self).__init__(parent)
        self.__roiManager = None

    def createWidget(self, parent):
        """Inherit the method to create a new widget"""
        widget = RoiModeSelector(parent)
        manager = self.__roiManager
        if manager is not None:
            roi = manager.getCurrentRoi()
            widget.setRoi(roi)
            self.setVisible(not widget.isEmpty())
        return widget

    def deleteWidget(self, widget):
        """Inherit the method to delete a widget"""
        widget.setRoi(None)
        return qt.QWidgetAction.deleteWidget(self, widget)

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
        self.setRoi(roi)

    def setRoi(self, roi):
        """Set a profile ROI to edit.

        :param ProfileRoiMixIn roi: A profile ROI
        """
        widget = None
        for widget in self.createdWidgets():
            widget.setRoi(roi)
        if widget is not None:
            self.setVisible(not widget.isEmpty())


class RegionOfInterestManager(qt.QObject):
    """Class handling ROI interaction on a PlotWidget.

    It supports the multiple ROIs: points, rectangles, polygons,
    lines, horizontal and vertical lines.

    See ``plotInteractiveImageROI.py`` sample code (:ref:`sample-code`).

    :param silx.gui.plot.PlotWidget parent:
        The plot widget in which to control the ROIs.
    """

    sigRoiAdded = qt.Signal(roi_items.RegionOfInterest)
    """Signal emitted when a new ROI has been added.

    It provides the newly add :class:`RegionOfInterest` object.
    """

    sigRoiAboutToBeRemoved = qt.Signal(roi_items.RegionOfInterest)
    """Signal emitted just before a ROI is removed.

    It provides the :class:`RegionOfInterest` object that is about to be removed.
    """

    sigRoiChanged = qt.Signal()
    """Signal emitted whenever the ROIs have changed."""

    sigCurrentRoiChanged = qt.Signal(object)
    """Signal emitted whenever a ROI is selected."""

    sigInteractiveModeStarted = qt.Signal(object)
    """Signal emitted when switching to ROI drawing interactive mode.

    It provides the class of the ROI which will be created by the interactive
    mode.
    """

    sigInteractiveRoiCreated = qt.Signal(object)
    """Signal emitted when a ROI is created during the interaction.
    The interaction is still incomplete and can be aborted.

    It provides the ROI object which was just been created.
    """

    sigInteractiveRoiFinalized = qt.Signal(object)
    """Signal emitted when a ROI creation is complet.

    It provides the ROI object which was just been created.
    """

    sigInteractiveModeFinished = qt.Signal()
    """Signal emitted when leaving interactive ROI drawing mode.
    """

    ROI_CLASSES = (
        roi_items.PointROI,
        roi_items.CrossROI,
        roi_items.RectangleROI,
        roi_items.CircleROI,
        roi_items.EllipseROI,
        roi_items.PolygonROI,
        roi_items.LineROI,
        roi_items.HorizontalLineROI,
        roi_items.VerticalLineROI,
        roi_items.ArcROI,
        roi_items.HorizontalRangeROI,
        roi_items.BandROI,
    )

    def __init__(self, parent):
        assert isinstance(parent, PlotWidget)
        super(RegionOfInterestManager, self).__init__(parent)
        self._rois = []  # List of ROIs
        self._drawnROI = None  # New ROI being currently drawn

        self._roiClass = None
        self._source = None
        self._lastHoveredMarkerLabel = None
        self._color = rgba("red")

        self._label = "__RegionOfInterestManager__%d" % id(self)

        self._currentRoi = None
        """Hold currently selected ROI"""

        self._eventLoop = None

        self._modeActions = {}

        parent.sigPlotSignal.connect(self._plotSignals)

        parent.sigInteractiveModeChanged.connect(self._plotInteractiveModeChanged)

        parent.sigItemRemoved.connect(self._itemRemoved)

        parent._sigDefaultContextMenu.connect(self._feedContextMenu)

    @classmethod
    def getSupportedRoiClasses(cls):
        """Returns the default available ROI classes

        :rtype: List[class]
        """
        return tuple(cls.ROI_CLASSES)

    # Associated QActions

    def getInteractionModeAction(self, roiClass):
        """Returns the QAction corresponding to a kind of ROI

        The QAction allows to enable the corresponding drawing
        interactive mode.

        :param class roiClass: The ROI class which will be created by this action.
        :rtype: QAction
        :raise ValueError: If kind is not supported
        """
        if not issubclass(roiClass, roi_items.RegionOfInterest):
            raise ValueError("Unsupported ROI class %s" % roiClass)

        action = self._modeActions.get(roiClass, None)
        if action is None:  # Lazy-loading
            action = CreateRoiModeAction(self, self, roiClass)
            self._modeActions[roiClass] = action
        return action

    # PlotWidget eventFilter and listeners

    def _plotInteractiveModeChanged(self, source):
        """Handle change of interactive mode in the plot"""
        if source is not self:
            self.__roiInteractiveModeEnded()

    def _getRoiFromItem(self, item):
        """Returns the ROI which own this item, else None
        if this manager do not have knowledge of this ROI."""
        for roi in self._rois:
            if isinstance(roi, roi_items.RegionOfInterest):
                for child in roi.getItems():
                    if child is item:
                        return roi
        return None

    def _itemRemoved(self, item):
        """Called after an item was removed from the plot."""
        if not hasattr(item, "_roiGroup"):
            # Early break to avoid to use _getRoiFromItem
            # And to avoid reentrant signal when the ROI remove the item itself
            return
        roi = self._getRoiFromItem(item)
        if roi is not None:
            self.removeRoi(roi)

    # Handle ROI interaction

    def _handleInteraction(self, event):
        """Handle mouse interaction for ROI addition"""
        roiClass = self.getCurrentInteractionModeRoiClass()
        if roiClass is None:
            return  # Should not happen

        kind = roiClass.getFirstInteractionShape()
        if kind == "point":
            if event["event"] == "mouseClicked" and event["button"] == "left":
                points = numpy.array([(event["x"], event["y"])], dtype=numpy.float64)
                # Not an interactive creation
                roi = self._createInteractiveRoi(roiClass, points=points)
                roi.creationFinalized()
                self.sigInteractiveRoiFinalized.emit(roi)
        else:  # other shapes
            if (
                event["event"] in ("drawingProgress", "drawingFinished")
                and event["parameters"]["label"] == self._label
            ):
                points = numpy.array(
                    (event["xdata"], event["ydata"]), dtype=numpy.float64
                ).T

                if self._drawnROI is None:  # Create new ROI
                    # NOTE: Set something before createRoi, so isDrawing is True
                    self._drawnROI = object()
                    self._drawnROI = self._createInteractiveRoi(roiClass, points=points)
                else:
                    self._drawnROI.setFirstShapePoints(points)

                if event["event"] == "drawingFinished":
                    if kind == "polygon" and len(points) > 1:
                        self._drawnROI.setFirstShapePoints(points[:-1])
                    roi = self._drawnROI
                    self._drawnROI = None  # Stop drawing
                    roi.creationFinalized()
                    self.sigInteractiveRoiFinalized.emit(roi)

    # RegionOfInterest selection

    def __getRoiFromMarker(self, marker):
        """Returns a ROI from a marker, else None"""
        # This should be speed up
        for roi in self._rois:
            if isinstance(roi, roi_items.HandleBasedROI):
                for m in roi.getHandles():
                    if m is marker:
                        return roi
            else:
                for m in roi.getItems():
                    if m is marker:
                        return roi
        return None

    def setCurrentRoi(self, roi: Optional[RegionOfInterest]):
        """Set the currently selected ROI, and emit a signal.

        :param Union[RegionOfInterest,None] roi: The ROI to select
        """
        if self._currentRoi is roi:
            return
        if roi is not None:
            # Note: Fixed range to avoid infinite loops
            for _ in range(10):
                target = roi.getFocusProxy()
                if target is None:
                    break
                roi = target
            else:
                raise RuntimeError("Max selection proxy depth (10) reached.")

        if self._currentRoi is not None:
            self._currentRoi.setHighlighted(False)
        self._currentRoi = roi
        if self._currentRoi is not None:
            self._currentRoi.setHighlighted(True)
        self.sigCurrentRoiChanged.emit(roi)

    def getCurrentRoi(self) -> Optional[RegionOfInterest]:
        """Returns the currently selected ROI, else None."""
        return self._currentRoi

    def _plotSignals(self, event):
        """Handle mouse interaction for ROI addition"""
        clicked = False
        roi = None
        if event["event"] in ("markerClicked", "markerMoving"):
            plot = self.parent()
            legend = event["label"]
            marker = plot._getMarker(legend=legend)
            roi = self.__getRoiFromMarker(marker)
        elif event["event"] == "mouseClicked" and event["button"] == "left":
            # Marker click is only for dnd
            # This also can click on a marker
            clicked = True
            plot = self.parent()
            marker = plot._getMarkerAt(event["xpixel"], event["ypixel"])
            roi = self.__getRoiFromMarker(marker)
        elif event["event"] == "hover":
            self._lastHoveredMarkerLabel = event["label"]
        else:
            return

        if roi not in self._rois:
            # The ROI is not own by this manager
            return

        if roi is not None:
            currentRoi = self.getCurrentRoi()
            if currentRoi is roi:
                if clicked:
                    self.__updateMode(roi)
            elif roi.isSelectable():
                self.setCurrentRoi(roi)
        else:
            self.setCurrentRoi(None)

    def __updateMode(self, roi: RegionOfInterest):
        if isinstance(roi, roi_items.InteractionModeMixIn):
            available = roi.availableInteractionModes()
            mode = roi.getInteractionMode()
            imode = available.index(mode)
            mode = available[(imode + 1) % len(available)]
            roi.setInteractionMode(mode)

    def _feedContextMenu(self, menu: qt.QMenu):
        """Called when the default plot context menu is about to be displayed"""
        roi = self.getCurrentRoi()
        if roi is not None:
            if roi.isEditable():
                if self._isMouseHoverRoi(roi):
                    roiMenu = self._createMenuForRoi(menu, roi)
                    menu.addMenu(roiMenu)

    def _isMouseHoverRoi(self, roi: RegionOfInterest) -> bool:
        """Check that the mouse hovers this roi"""
        plot = self.parent()

        if self._lastHoveredMarkerLabel is not None:
            marker = plot._getMarker(self._lastHoveredMarkerLabel)
            if marker is not None:
                r = self.__getRoiFromMarker(marker)
                if roi is r:
                    return True

        # Filter by data position
        # FIXME: It would be better to use GUI coords for it
        pos = plot.getWidgetHandle().mapFromGlobal(qt.QCursor.pos())
        data = plot.pixelToData(pos.x(), pos.y())
        return roi.contains(data)

    def _createMenuForRoi(self, parent: qt.QWidget, roi: RegionOfInterest) -> qt.QMenu:
        """Create a QMenu for the given RegionOfInterest"""
        roiMenu = qt.QMenu(parent)
        roiMenu.setTitle(roi.getName())

        if isinstance(roi, roi_items.InteractionModeMixIn):
            interactionMenu = roi.createMenuForInteractionMode(roiMenu)
            roiMenu.addMenu(interactionMenu)

        removeAction = qt.QAction(roiMenu)
        removeAction.setText("Remove")
        callback = functools.partial(self.removeRoi, roi)
        removeAction.triggered.connect(callback)
        roiMenu.addAction(removeAction)

        roi.populateContextMenu(roiMenu)

        return roiMenu

    # RegionOfInterest API

    def getRois(self):
        """Returns the list of ROIs.

        It returns an empty tuple if there is currently no ROI.

        :return: Tuple of arrays of objects describing the ROIs
        :rtype: List[RegionOfInterest]
        """
        return tuple(self._rois)

    def clear(self):
        """Reset current ROIs

        :return: True if ROIs were reset.
        :rtype: bool
        """
        if self.getRois():  # Something to reset
            for roi in self._rois:
                roi.sigRegionChanged.disconnect(self._regionOfInterestChanged)
                roi.setParent(None)
            self._rois = []
            self._roisUpdated()
            return True

        else:
            return False

    def _regionOfInterestChanged(self, event=None):
        """Handle ROI object changed"""
        self.sigRoiChanged.emit()

    def _createInteractiveRoi(self, roiClass, points, label=None, index=None):
        """Create a new ROI with interactive creation.

        :param class roiClass: The class of the ROI to create
        :param numpy.ndarray points: The first shape used to create the ROI
        :param str label: The label to display along with the ROI.
        :param int index: The position where to insert the ROI.
            By default it is appended to the end of the list.
        :return: The created ROI object
        :rtype: roi_items.RegionOfInterest
        :raise RuntimeError: When ROI cannot be added because the maximum
           number of ROIs has been reached.
        """
        roi = roiClass(parent=None)
        if label is not None:
            roi.setName(str(label))
        roi.creationStarted()
        roi.setFirstShapePoints(points)

        self.addRoi(roi, index)
        if roi.isSelectable():
            self.setCurrentRoi(roi)
        self.sigInteractiveRoiCreated.emit(roi)
        return roi

    def containsRoi(self, roi):
        """Returns true if the ROI is part of this manager.

        :param roi_items.RegionOfInterest roi: The ROI to add
        :rtype: bool
        """
        return roi in self._rois

    def addRoi(self, roi, index=None, useManagerColor=True):
        """Add the ROI to the list of ROIs.

        :param roi_items.RegionOfInterest roi: The ROI to add
        :param int index: The position where to insert the ROI,
            By default it is appended to the end of the list of ROIs
        :param bool useManagerColor:
            Whether to set the ROI color to the default one of the manager or not.
            (Default: True).
        :raise RuntimeError: When ROI cannot be added because the maximum
           number of ROIs has been reached.
        """
        plot = self.parent()
        if plot is None:
            raise RuntimeError("Cannot add ROI: PlotWidget no more available")

        roi.setParent(self)

        if useManagerColor:
            roi.setColor(self.getColor())

        roi.sigRegionChanged.connect(self._regionOfInterestChanged)
        roi.sigItemChanged.connect(self._regionOfInterestChanged)

        if index is None:
            self._rois.append(roi)
        else:
            self._rois.insert(index, roi)
        self.sigRoiAdded.emit(roi)
        self._roisUpdated()

    def removeRoi(self, roi):
        """Remove a ROI from the list of ROIs.

        :param roi_items.RegionOfInterest roi: The ROI to remove
        :raise ValueError: When ROI does not belong to this object
        """
        if not (
            isinstance(roi, roi_items.RegionOfInterest)
            and roi.parent() is self
            and roi in self._rois
        ):
            raise ValueError("RegionOfInterest does not belong to this instance")

        roi.sigAboutToBeRemoved.emit()
        self.sigRoiAboutToBeRemoved.emit(roi)

        if roi is self._currentRoi:
            self.setCurrentRoi(None)

        mustRestart = False
        if roi is self._drawnROI:
            self._drawnROI = None
            mustRestart = True
        self._rois.remove(roi)
        roi.sigRegionChanged.disconnect(self._regionOfInterestChanged)
        roi.sigItemChanged.disconnect(self._regionOfInterestChanged)
        roi.setParent(None)
        self._roisUpdated()

        if mustRestart:
            self._restart()

    def _roisUpdated(self):
        """Handle update of the ROI list"""
        self.sigRoiChanged.emit()

    # RegionOfInterest parameters

    def getColor(self):
        """Return the default color of created ROIs

        :rtype: QColor
        """
        return qt.QColor.fromRgbF(*self._color)

    def setColor(self, color):
        """Set the default color to use when creating ROIs.

        Existing ROIs are not affected.

        :param color: The color to use for displaying ROIs as
           either a color name, a QColor, a list of uint8 or float in [0, 1].
        """
        self._color = rgba(color)

    # Control ROI

    def getCurrentInteractionModeRoiClass(self):
        """Returns the current ROI class used by the interactive drawing mode.

        Returns None if the ROI manager is not in an interactive mode.

        :rtype: Union[class,None]
        """
        return self._roiClass

    def getInteractionSource(self):
        """Returns the object which have requested the ROI creation.

        Returns None if the ROI manager is not in an interactive mode.

        :rtype: Union[object,None]
        """
        return self._source

    def isStarted(self):
        """Returns True if an interactive ROI drawing mode is active.

        :rtype: bool
        """
        return self._roiClass is not None

    def isDrawing(self):
        """Returns True if an interactive ROI is drawing.

        :rtype: bool
        """
        return self._drawnROI is not None

    def start(self, roiClass, source=None):
        """Start an interactive ROI drawing mode.

        :param class roiClass: The ROI class to create. It have to inherite from
            `roi_items.RegionOfInterest`.
        :param object source: SOurce of the ROI interaction.
        :return: True if interactive ROI drawing was started, False otherwise
        :rtype: bool
        :raise ValueError: If roiClass is not supported
        """
        self.stop()

        if not issubclass(roiClass, roi_items.RegionOfInterest):
            raise ValueError("Unsupported ROI class %s" % roiClass)

        plot = self.parent()
        if plot is None:
            return False

        self._roiClass = roiClass
        self._source = source

        self._restart()

        plot.sigPlotSignal.connect(self._handleInteraction)

        self.sigInteractiveModeStarted.emit(roiClass)

        return True

    def _restart(self):
        """Restart the plot interaction without changing the
        source or the ROI class.
        """
        roiClass = self._roiClass
        plot = self.parent()
        firstInteractionShapeKind = roiClass.getFirstInteractionShape()

        if firstInteractionShapeKind == "point":
            plot.setInteractiveMode(mode="select", source=self)
        else:
            if roiClass.showFirstInteractionShape():
                color = rgba(self.getColor())
            else:
                color = None
            plot.setInteractiveMode(
                mode="draw",
                source=self,
                shape=firstInteractionShapeKind,
                color=color,
                label=self._label,
            )

    def __roiInteractiveModeEnded(self):
        """Handle end of ROI draw interactive mode"""
        if self.isStarted():
            self._roiClass = None
            self._source = None

            if self._drawnROI is not None:
                # Cancel ROI create
                roi = self._drawnROI
                self._drawnROI = None
                self.removeRoi(roi)

            plot = self.parent()
            if plot is not None:
                plot.sigPlotSignal.disconnect(self._handleInteraction)

            self.sigInteractiveModeFinished.emit()

    def stop(self):
        """Stop interactive ROI drawing mode.

        :return: True if an interactive ROI drawing mode was actually stopped
        :rtype: bool
        """
        if not self.isStarted():
            return False

        plot = self.parent()
        if plot is not None:
            # This leads to call __roiInteractiveModeEnded through
            # interactive mode changed signal
            plot.resetInteractiveMode()
        else:  # Fallback
            self.__roiInteractiveModeEnded()

        return True

    def exec(self, roiClass):
        """Block until :meth:`quit` is called.

        :param class kind: The class of the ROI which have to be created.
            See `silx.gui.plot.items.roi`.
        :return: The list of ROIs
        :rtype: tuple
        """
        self.start(roiClass)

        plot = self.parent()
        plot.show()
        plot.raise_()

        self._eventLoop = qt.QEventLoop()
        self._eventLoop.exec()
        self._eventLoop = None

        self.stop()

        rois = self.getRois()
        self.clear()
        return rois

    def exec_(self, roiClass):  # Qt5-like compatibility
        return self.exec(roiClass)

    def quit(self):
        """Stop a blocking :meth:`exec` and call :meth:`stop`"""
        if self._eventLoop is not None:
            self._eventLoop.quit()
            self._eventLoop = None
        self.stop()


class InteractiveRegionOfInterestManager(RegionOfInterestManager):
    """RegionOfInterestManager with features for use from interpreter.

    It is meant to be used through the :meth:`exec`.
    It provides some messages to display in a status bar and
    different modes to end blocking calls to :meth:`exec`.

    :param parent: See QObject
    """

    sigMessageChanged = qt.Signal(str)
    """Signal emitted when a new message should be displayed to the user

    It provides the message as a str.
    """

    def __init__(self, parent):
        super(InteractiveRegionOfInterestManager, self).__init__(parent)
        self._maxROI = None
        self.__timeoutEndTime = None
        self.__message = ""
        self.__validationMode = self.ValidationMode.ENTER
        self.__execClass = None

        self.sigRoiAdded.connect(self.__added)
        self.sigRoiAboutToBeRemoved.connect(self.__aboutToBeRemoved)
        self.sigInteractiveModeStarted.connect(self.__started)
        self.sigInteractiveModeFinished.connect(self.__finished)

    # Max ROI

    def getMaxRois(self):
        """Returns the maximum number of ROIs or None if no limit.

        :rtype: Union[int,None]
        """
        return self._maxROI

    def setMaxRois(self, max_):
        """Set the maximum number of ROIs.

        :param Union[int,None] max_: The max limit or None for no limit.
        :raise ValueError: If there is more ROIs than max value
        """
        if max_ is not None:
            max_ = int(max_)
            if max_ <= 0:
                raise ValueError("Max limit must be strictly positive")

            if len(self.getRois()) > max_:
                raise ValueError("Cannot set max limit: Already too many ROIs")

        self._maxROI = max_

    def isMaxRois(self):
        """Returns True if the maximum number of ROIs is reached.

        :rtype: bool
        """
        max_ = self.getMaxRois()
        return max_ is not None and len(self.getRois()) >= max_

    # Validation mode

    @enum.unique
    class ValidationMode(enum.Enum):
        """Mode of validation to leave blocking :meth:`exec`"""

        AUTO = "auto"
        """Automatically ends the interactive mode once
        the user terminates the last ROI shape."""

        ENTER = "enter"
        """Ends the interactive mode when the *Enter* key is pressed."""

        AUTO_ENTER = "auto_enter"
        """Ends the interactive mode when reaching max ROIs or
        when the *Enter* key is pressed.
        """

        NONE = "none"
        """Do not provide the user a way to end the interactive mode.

        The end of :meth:`exec` is done through :meth:`quit` or timeout.
        """

    def getValidationMode(self):
        """Returns the interactive mode validation in use.

        :rtype: ValidationMode
        """
        return self.__validationMode

    def setValidationMode(self, mode):
        """Set the way to perform interactive mode validation.

        See :class:`ValidationMode` enumeration for the supported
        validation modes.

        :param ValidationMode mode: The interactive mode validation to use.
        """
        assert isinstance(mode, self.ValidationMode)
        if mode != self.__validationMode:
            self.__validationMode = mode

        if self.isExec():
            if self.isMaxRois() and self.getValidationMode() in (
                self.ValidationMode.AUTO,
                self.ValidationMode.AUTO_ENTER,
            ):
                self.quit()

            self.__updateMessage()

    def eventFilter(self, obj, event):
        if event.type() == qt.QEvent.Hide:
            self.quit()

        if event.type() == qt.QEvent.KeyPress:
            key = event.key()
            if key in (
                qt.Qt.Key_Return,
                qt.Qt.Key_Enter,
            ) and self.getValidationMode() in (
                self.ValidationMode.ENTER,
                self.ValidationMode.AUTO_ENTER,
            ):
                # Stop on return key pressed
                self.quit()
                return True  # Stop further handling of this keys

            if key in (qt.Qt.Key_Delete, qt.Qt.Key_Backspace) or (
                key == qt.Qt.Key_Z and event.modifiers() & qt.Qt.ControlModifier
            ):
                rois = self.getRois()
                if rois:  # Something to undo
                    self.removeRoi(rois[-1])
                    # Stop further handling of keys if something was undone
                    return True

        return super(InteractiveRegionOfInterestManager, self).eventFilter(obj, event)

    # Message API

    def getMessage(self):
        """Returns the current status message.

        This message is meant to be displayed in a status bar.

        :rtype: str
        """
        if self.__timeoutEndTime is None:
            return self.__message
        else:
            remaining = self.__timeoutEndTime - time.time()
            return self.__message + (" - %d seconds remaining" % max(1, int(remaining)))

    # Listen to ROI updates

    def __added(self, *args, **kwargs):
        """Handle new ROI added"""
        max_ = self.getMaxRois()
        if max_ is not None:
            # When reaching max number of ROIs, redo last one
            while len(self.getRois()) > max_:
                self.removeRoi(self.getRois()[-2])

        self.__updateMessage()
        if self.isMaxRois() and self.getValidationMode() in (
            self.ValidationMode.AUTO,
            self.ValidationMode.AUTO_ENTER,
        ):
            self.quit()

    def __aboutToBeRemoved(self, *args, **kwargs):
        """Handle removal of a ROI"""
        # RegionOfInterest not removed yet
        self.__updateMessage(nbrois=len(self.getRois()) - 1)

    def __started(self, roiKind):
        """Handle interactive mode started"""
        self.__updateMessage()

    def __finished(self):
        """Handle interactive mode finished"""
        self.__updateMessage()

    def __updateMessage(self, nbrois=None):
        """Update message"""
        if not self.isExec():
            message = "Done"

        elif not self.isStarted():
            message = "Use %s ROI edition mode" % self.__execClass

        else:
            if nbrois is None:
                nbrois = len(self.getRois())

            name = self.__execClass._getShortName()

            max_ = self.getMaxRois()
            if max_ is None:
                message = "Select %ss (%d selected)" % (name, nbrois)

            elif max_ <= 1:
                message = "Select a %s" % name
            else:
                message = "Select %d/%d %ss" % (nbrois, max_, name)

            if (
                self.getValidationMode() == self.ValidationMode.ENTER
                and self.isMaxRois()
            ):
                message += " - Press Enter to confirm"

        if message != self.__message:
            self.__message = message
            # Use getMessage to add timeout message
            self.sigMessageChanged.emit(self.getMessage())

    # Handle blocking call

    def __timeoutUpdate(self):
        """Handle update of timeout"""
        if (
            self.__timeoutEndTime is not None
            and (self.__timeoutEndTime - time.time()) > 0
        ):
            self.sigMessageChanged.emit(self.getMessage())
        else:  # Stop interactive mode and message timer
            timer = self.sender()
            if timer is not None:
                timer.stop()
            self.__timeoutEndTime = None
            self.quit()

    def isExec(self):
        """Returns True if :meth:`exec` is currently running.

        :rtype: bool"""
        return self.__execClass is not None

    def exec(self, roiClass, timeout=0):
        """Block until ROI selection is done or timeout is elapsed.

        :meth:`quit` also ends this blocking call.

        :param class roiClass: The class of the ROI which have to be created.
            See `silx.gui.plot.items.roi`.
        :param int timeout: Maximum duration in seconds to block.
            Default: No timeout
        :return: The list of ROIs
        :rtype: List[RegionOfInterest]
        """
        plot = self.parent()
        if plot is None:
            return

        self.__execClass = roiClass

        plot.installEventFilter(self)

        if timeout > 0:
            self.__timeoutEndTime = time.time() + timeout
            timer = qt.QTimer(self)
            timer.timeout.connect(self.__timeoutUpdate)
            timer.start(1000)

            rois = super(InteractiveRegionOfInterestManager, self).exec(roiClass)

            timer.stop()
            self.__timeoutEndTime = None

        else:
            rois = super(InteractiveRegionOfInterestManager, self).exec(roiClass)

        plot.removeEventFilter(self)

        self.__execClass = None
        self.__updateMessage()

        return rois

    def exec_(self, roiClass, timeout=0):  # Qt5-like compatibility
        return self.exec(roiClass, timeout)


class _DeleteRegionOfInterestToolButton(qt.QToolButton):
    """Tool button deleting a ROI object

    :param parent: See QWidget
    :param RegionOfInterest roi: The ROI to delete
    """

    def __init__(self, parent, roi):
        super(_DeleteRegionOfInterestToolButton, self).__init__(parent)
        self.setIcon(icons.getQIcon("remove"))
        self.setToolTip("Remove this ROI")
        self.__roiRef = roi if roi is None else weakref.ref(roi)
        self.clicked.connect(self.__clicked)

    def __clicked(self, checked):
        """Handle button clicked"""
        roi = None if self.__roiRef is None else self.__roiRef()
        if roi is not None:
            manager = roi.parent()
            if manager is not None:
                manager.removeRoi(roi)
                self.__roiRef = None


class RegionOfInterestTableWidget(qt.QTableWidget):
    """Widget displaying the ROIs of a :class:`RegionOfInterestManager`"""

    # Columns indices of the different displayed information
    (
        _LABEL_VISIBLE_COL,
        _EDITABLE_COL,
        _KIND_COL,
        _COORDINATES_COL,
        _DELETE_COL,
    ) = range(5)

    def __init__(self, parent=None):
        super(RegionOfInterestTableWidget, self).__init__(parent)
        self._roiManagerRef = None

        headers = ["Label", "Edit", "Kind", "Coordinates", ""]
        self.setColumnCount(len(headers))
        self.setHorizontalHeaderLabels(headers)

        horizontalHeader = self.horizontalHeader()
        horizontalHeader.setDefaultAlignment(qt.Qt.AlignLeft)

        horizontalHeader.setSectionResizeMode(0, qt.QHeaderView.Interactive)
        horizontalHeader.setSectionResizeMode(1, qt.QHeaderView.ResizeToContents)
        horizontalHeader.setSectionResizeMode(2, qt.QHeaderView.ResizeToContents)
        horizontalHeader.setSectionResizeMode(3, qt.QHeaderView.Stretch)
        horizontalHeader.setSectionResizeMode(4, qt.QHeaderView.ResizeToContents)

        verticalHeader = self.verticalHeader()
        verticalHeader.setVisible(False)

        self.setSelectionMode(qt.QAbstractItemView.NoSelection)
        self.setFocusPolicy(qt.Qt.NoFocus)

        self.itemChanged.connect(self.__itemChanged)

    def __itemChanged(self, item):
        """Handle QTableWidget item updates"""
        column = item.column()
        roi = item.data(qt.Qt.UserRole)
        if roi is None:
            return

        if column == 0:
            # First collect information from item, then update ROI
            # Otherwise, this causes issues
            checked = item.checkState() == qt.Qt.Checked
            text = item.text()
            roi.setVisible(checked)
            roi.setName(text)
        elif column == 1:
            roi.setEditable(item.checkState() == qt.Qt.Checked)
        elif column in (2, 3, 4):
            pass  # TODO
        else:
            logger.error("Unhandled column %d", column)

    def setRegionOfInterestManager(self, manager):
        """Set the :class:`RegionOfInterestManager` object to sync with

        :param RegionOfInterestManager manager:
        """
        assert manager is None or isinstance(manager, RegionOfInterestManager)

        previousManager = self.getRegionOfInterestManager()

        if previousManager is not None:
            previousManager.sigRoiAdded.disconnect(self.__roiAdded)
            previousManager.sigRoiAboutToBeRemoved.disconnect(
                self.__roiAboutToBeRemoved
            )
            for roi in previousManager.getRois():
                self.__disconnectRoi(roi)

        self.setRowCount(0)

        self._roiManagerRef = weakref.ref(manager)

        self._sync()

        if manager is not None:
            for roi in manager.getRois():
                self.__connectRoi(roi)
            manager.sigRoiAdded.connect(self.__roiAdded)
            manager.sigRoiAboutToBeRemoved.connect(self.__roiAboutToBeRemoved)

    def _getReadableRoiDescription(self, roi):
        """Returns modelisation of a ROI as a readable sequence of values.

        :rtype: str
        """
        text = str(roi)
        try:
            # Extract the params from syntax "CLASSNAME(PARAMS)"
            elements = text.split("(", 1)
            if len(elements) != 2:
                return text
            result = elements[1]
            result = result.strip()
            if not result.endswith(")"):
                return text
            result = result[0:-1]
            # Capitalize each words
            result = result.title()
            return result
        except Exception:
            logger.debug("Backtrace", exc_info=True)
        return text

    def __connectRoi(self, roi: RegionOfInterest):
        """Start listening ROI signals"""
        roi.sigItemChanged.connect(self.__roiItemChanged)
        roi.sigRegionChanged.connect(self.__roiRegionChanged)

    def __disconnectRoi(self, roi: RegionOfInterest):
        """Stop listening ROI signals"""
        roi.sigItemChanged.disconnect(self.__roiItemChanged)
        roi.sigRegionChanged.disconnect(self.__roiRegionChanged)

    def __getRoiRow(self, roi: RegionOfInterest) -> int:
        """Returns row index of given region of interest

        :raises ValueError: If region of interest is not in the list
        """
        manager = self.getRegionOfInterestManager()
        if manager is None:
            return
        return manager.getRois().index(roi)

    def __roiAdded(self, roi: RegionOfInterest):
        """Handle new ROI added to the manager"""
        self.__connectRoi(roi)
        self._sync()

    def __roiAboutToBeRemoved(self, roi: RegionOfInterest):
        """Handle removing a ROI from the manager"""
        self.__disconnectRoi(roi)
        self.removeRow(self.__getRoiRow(roi))

    def __roiItemChanged(self, event: ItemChangedType):
        """Handle ROI sigItemChanged events"""
        roi = self.sender()
        if roi is None:
            return

        try:
            row = self.__getRoiRow(roi)
        except ValueError:
            return

        if event == ItemChangedType.VISIBLE:
            item = self.item(row, self._LABEL_VISIBLE_COL)
            item.setCheckState(qt.Qt.Checked if roi.isVisible() else qt.Qt.Unchecked)
            return

        if event == ItemChangedType.NAME:
            item = self.item(row, self._LABEL_VISIBLE_COL)
            item.setText(roi.getName())
            return

        if event == ItemChangedType.EDITABLE:
            item = self.item(row, self._EDITABLE_COL)
            item.setCheckState(qt.Qt.Checked if roi.isEditable() else qt.Qt.Unchecked)
            return

    def __roiRegionChanged(self):
        """Handle change of ROI coordinates"""
        roi = self.sender()
        if roi is None:
            return

        item = self.item(self.__getRoiRow(roi), self._COORDINATES_COL)
        if item is None:
            return

        text = self._getReadableRoiDescription(roi)
        item.setText(text)

    def _sync(self):
        """Update widget content according to ROI manger"""
        manager = self.getRegionOfInterestManager()

        if manager is None:
            self.setRowCount(0)
            return

        rois = manager.getRois()

        self.setRowCount(len(rois))
        for index, roi in enumerate(rois):
            baseFlags = qt.Qt.ItemIsSelectable | qt.Qt.ItemIsEnabled

            # Label and visible
            item = qt.QTableWidgetItem()
            item.setFlags(baseFlags | qt.Qt.ItemIsEditable | qt.Qt.ItemIsUserCheckable)
            item.setData(qt.Qt.UserRole, roi)
            item.setText(roi.getName())
            item.setCheckState(qt.Qt.Checked if roi.isVisible() else qt.Qt.Unchecked)
            self.setItem(index, self._LABEL_VISIBLE_COL, item)

            # Editable
            item = qt.QTableWidgetItem()
            item.setFlags(baseFlags | qt.Qt.ItemIsUserCheckable)
            item.setData(qt.Qt.UserRole, roi)
            item.setCheckState(qt.Qt.Checked if roi.isEditable() else qt.Qt.Unchecked)
            self.setItem(index, self._EDITABLE_COL, item)
            item.setTextAlignment(qt.Qt.AlignCenter)
            item.setText(None)

            # Kind
            label = roi._getShortName()
            if label is None:
                # Default value if kind is not overrided
                label = roi.__class__.__name__
            item = qt.QTableWidgetItem(label.capitalize())
            item.setFlags(baseFlags)
            self.setItem(index, self._KIND_COL, item)

            # Coordinates
            item = qt.QTableWidgetItem()
            item.setFlags(baseFlags)
            text = self._getReadableRoiDescription(roi)
            item.setText(text)
            self.setItem(index, self._COORDINATES_COL, item)

            # Delete
            widget = qt.QWidget(self)
            delBtn = _DeleteRegionOfInterestToolButton(widget, roi)
            layout = qt.QHBoxLayout()
            layout.setContentsMargins(2, 2, 2, 2)
            layout.setSpacing(0)
            widget.setLayout(layout)
            layout.addStretch(1)
            layout.addWidget(delBtn)
            layout.addStretch(1)
            self.setCellWidget(index, self._DELETE_COL, widget)

    def getRegionOfInterestManager(self):
        """Returns the :class:`RegionOfInterestManager` this widget supervise.

        It returns None if not sync with an :class:`RegionOfInterestManager`.

        :rtype: RegionOfInterestManager
        """
        return None if self._roiManagerRef is None else self._roiManagerRef()
