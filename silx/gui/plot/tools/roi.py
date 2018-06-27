# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2018 European Synchrotron Radiation Facility
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

This API is not mature and will probably change in the future.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "27/06/2018"


import collections
import functools
import logging
import time
import weakref

import numpy

from ....third_party import enum
from ....utils.weakref import WeakMethodProxy
from ... import qt, icons
from .. import PlotWidget
from ..items import roi
roi_items = roi  # Allow to use variable named 'roi' in the code

from ...colors import rgba


logger = logging.getLogger(__name__)


class RegionOfInterestManager(qt.QObject):
    """Class handling ROI interaction on a PlotWidget.

    It supports the multiple ROIs: points, rectangles, polygons,
    lines, horizontal and vertical lines.

    See ``plotInteractiveImageROI.py`` sample code (:ref:`sample-code`).

    :param silx.gui.plot.PlotWidget parent:
        The plot widget in which to control the ROIs.
    """

    sigRegionOfInterestAdded = qt.Signal(roi_items.RegionOfInterest)
    """Signal emitted when a new ROI has been added.

    It provides the newly add :class:`RegionOfInterest` object.
    """

    sigRegionOfInterestAboutToBeRemoved = qt.Signal(roi_items.RegionOfInterest)
    """Signal emitted just before a ROI is removed.

    It provides the :class:`RegionOfInterest` object that is about to be removed.
    """

    sigRegionOfInterestChanged = qt.Signal()
    """Signal emitted whenever the ROIs have changed.

    It provides the list of ROIs.
    """

    sigInteractiveModeStarted = qt.Signal(str)
    """Signal emitted when switching to ROI drawing interactive mode.

    It provides the kind of shape of the active interactive mode.
    """

    sigInteractiveModeFinished = qt.Signal()
    """Signal emitted when leaving and interactive ROI drawing.

    It provides the list of ROIs.
    """

    _MODE_ACTIONS_PARAMS = collections.OrderedDict()
    # Interactive mode: (icon name, text)
    _MODE_ACTIONS_PARAMS['point'] = roi_items.PointROI, 'add-shape-point', 'Add point markers'
    _MODE_ACTIONS_PARAMS['rectangle'] = roi_items.RectangleROI, 'add-shape-rectangle', 'Add rectangle ROI'
    _MODE_ACTIONS_PARAMS['polygon'] = roi_items.PolygonROI, 'add-shape-polygon', 'Add polygon ROI'
    _MODE_ACTIONS_PARAMS['line'] = roi_items.LineROI, 'add-shape-diagonal', 'Add line ROI'
    _MODE_ACTIONS_PARAMS['hline'] = roi_items.HorizontalLineROI, 'add-shape-horizontal', 'Add horizontal line ROI'
    _MODE_ACTIONS_PARAMS['vline'] = roi_items.VerticalLineROI, 'add-shape-vertical', 'Add vertical line ROI'
    _MODE_ACTIONS_PARAMS['arc'] = roi_items.ArcROI, 'add-shape-arc', 'Add arc ROI'

    def __init__(self, parent):
        assert isinstance(parent, PlotWidget)
        super(RegionOfInterestManager, self).__init__(parent)
        self._rois = []  # List of ROIs
        self._drawnROI = None  # New ROI being currently drawn

        self._shapeKind = None
        self._color = rgba('red')

        self._label = "__RegionOfInterestManager__%d" % id(self)

        self._eventLoop = None

        self._modeActions = {}

        parent.sigInteractiveModeChanged.connect(
            self._plotInteractiveModeChanged)

    @classmethod
    def getSupportedRegionOfInterestKinds(cls):
        """Returns available ROI kinds

        :rtype: List[str]
        """
        return tuple(cls._MODE_ACTIONS_PARAMS.keys())

    # Associated QActions

    def getInteractionModeAction(self, kind):
        """Returns the QAction corresponding to a kind of ROI

        The QAction allows to enable the corresponding drawing
        interactive mode.

        :param str kind: Kind of ROI
        :rtype: QAction
        :raise ValueError: If kind is not supported
        """
        if kind not in self.getSupportedRegionOfInterestKinds():
            raise ValueError('Unsupported kind %s' % kind)

        action = self._modeActions.get(kind, None)
        if action is None:  # Lazy-loading
            _roiClass, iconName, text = self._MODE_ACTIONS_PARAMS[kind]
            action = qt.QAction(self)
            action.setIcon(icons.getQIcon(iconName))
            action.setText(text)
            action.setCheckable(True)
            action.setChecked(self.getRegionOfInterestKind() == kind)
            action.setToolTip(text)

            action.triggered[bool].connect(functools.partial(
                WeakMethodProxy(self._modeActionTriggered), kind=kind))
            self._modeActions[kind] = action
        return action

    def _modeActionTriggered(self, checked, kind):
        """Handle mode actions being checked by the user

        :param bool checked:
        :param str kind: Corresponding shape kind
        """
        if checked:
            self.start(kind)
        else:  # Keep action checked
            action = self.sender()
            action.setChecked(True)

    def _updateModeActions(self):
        """Check/Uncheck action corresponding to current mode"""
        for kind, action in self._modeActions.items():
            action.setChecked(kind == self.getRegionOfInterestKind())

    # PlotWidget eventFilter and listeners

    def _plotInteractiveModeChanged(self, source):
        """Handle change of interactive mode in the plot"""
        if source is not self:
            self.__roiInteractiveModeEnded()

        else:  # Check the corresponding action
            self._updateModeActions()

    # Handle ROI interaction

    def _handleInteraction(self, event):
        """Handle mouse interaction for ROI addition"""
        kind = self.getRegionOfInterestKind()
        if kind is None:
            return  # Should not happen

        if kind == 'point':
            if event['event'] == 'mouseClicked' and event['button'] == 'left':
                points = numpy.array([(event['x'], event['y'])],
                                     dtype=numpy.float64)
                self.createRegionOfInterest(kind=kind, points=points)

        else:  # other shapes
            if (event['event'] in ('drawingProgress', 'drawingFinished') and
                    event['parameters']['label'] == self._label):
                points = numpy.array((event['xdata'], event['ydata']),
                                     dtype=numpy.float64).T

                if self._drawnROI is None:  # Create new ROI
                    self._drawnROI = self.createRegionOfInterest(
                        kind=kind, points=points)
                else:
                    self._drawnROI.setFirstShapePoints(points)

                if event['event'] == 'drawingFinished':
                    if kind == 'polygon' and len(points) > 1:
                        self._drawnROI.setFirstShapePoints(points[:-1])
                    self._drawnROI = None  # Stop drawing

    # RegionOfInterest API

    def getRegionOfInterests(self):
        """Returns the list of ROIs.

        It returns an empty tuple if there is currently no ROI.

        :return: Tuple of arrays of objects describing the ROIs
        :rtype: List[RegionOfInterest]
        """
        return tuple(self._rois)

    def clearRegionOfInterests(self):
        """Reset current ROIs

        :return: True if ROIs were reset.
        :rtype: bool
        """
        if self.getRegionOfInterests():  # Something to reset
            for roi in self._rois:
                roi.sigRegionChanged.disconnect(
                    self._regionOfInterestPointsChanged)
                roi.setParent(None)
            self._rois = []
            self._roisUpdated()
            return True

        else:
            return False

    def _regionOfInterestPointsChanged(self):
        """Handle ROI object points changed"""
        self.sigRegionOfInterestChanged.emit()

    def createRegionOfInterest(self, kind, points, label='', index=None):
        """Create a new ROI and add it to list of ROIs.

        :param str kind: The kind of ROI to add
        :param numpy.ndarray points: The first shape used to create the ROI
        :param str label: The label to display along with the ROI.
        :param int index: The position where to insert the ROI.
            By default it is appended to the end of the list.
        :return: The created ROI object
        :rtype: roi_items.RegionOfInterest
        :raise RuntimeError: When ROI cannot be added because the maximum
           number of ROIs has been reached.
        """
        roiClass = self._MODE_ACTIONS_PARAMS[kind][0]
        roi = roiClass(parent=None)
        roi.setLabel(str(label))
        roi.setFirstShapePoints(points)

        self.addRegionOfInterest(roi, index)
        return roi

    def addRegionOfInterest(self, roi, index=None, useManagerColor=True):
        """Add the ROI to the list of ROIs.

        :param roi_items.RegionOfInterest roi: The ROI to add
        :param int index: The position where to insert the ROI,
            By default it is appended to the end of the list of ROIs
        :raise RuntimeError: When ROI cannot be added because the maximum
           number of ROIs has been reached.
        """
        plot = self.parent()
        if plot is None:
            raise RuntimeError(
                'Cannot add ROI: PlotWidget no more available')

        roi.setParent(self)

        if useManagerColor:
            roi.setColor(self.getColor())

        roi.sigRegionChanged.connect(self._regionOfInterestPointsChanged)

        if index is None:
            self._rois.append(roi)
        else:
            self._rois.insert(index, roi)
        self.sigRegionOfInterestAdded.emit(roi)
        self._roisUpdated()

    def removeRegionOfInterest(self, roi):
        """Remove a ROI from the list of ROIs.

        :param roi_items.RegionOfInterest roi: The ROI to remove
        :raise ValueError: When ROI does not belong to this object
        """
        if not (isinstance(roi, roi_items.RegionOfInterest) and
                roi.parent() is self and
                roi in self._rois):
            raise ValueError(
                'RegionOfInterest does not belong to this instance')

        self.sigRegionOfInterestAboutToBeRemoved.emit(roi)

        self._rois.remove(roi)
        roi.sigRegionChanged.disconnect(self._regionOfInterestPointsChanged)
        roi.setParent(None)
        self._roisUpdated()

    def _roisUpdated(self):
        """Handle update of the ROI list"""
        self.sigRegionOfInterestChanged.emit()

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

    def getRegionOfInterestKind(self):
        """Returns the current interactive ROI drawing mode or None.

        :rtype: Union[str,None]
        """
        return self._shapeKind

    def isStarted(self):
        """Returns True  if an interactive ROI drawing mode is active.

        :rtype: bool
        """
        return self._shapeKind is not None

    def start(self, kind):
        """Start an interactive ROI drawing mode.

        :param str kind: The kind of ROI shape in:
           'point', 'rectangle', 'line', 'polygon', 'hline', 'vline'
        :return: True if interactive ROI drawing was started, False otherwise
        :rtype: bool
        :raise ValueError: If kind is not supported
        """
        self.stop()

        plot = self.parent()
        if plot is None:
            return False

        if kind not in self.getSupportedRegionOfInterestKinds():
            raise ValueError('Unsupported kind %s' % kind)
        self._shapeKind = kind
        # TODO: The ROI should be created here
        roiClass = self._MODE_ACTIONS_PARAMS[kind][0]
        firstInteractionShapeKind = roiClass.getFirstInteractionShape()

        if self._shapeKind == 'point':
            plot.setInteractiveMode(mode='select', source=self)
        else:
            if roiClass.showFirstInteractionShape():
                color = rgba(self.getColor())
            else:
                color = None
            plot.setInteractiveMode(mode='select-draw',
                                    source=self,
                                    shape=firstInteractionShapeKind,
                                    color=color,
                                    label=self._label)

        plot.sigPlotSignal.connect(self._handleInteraction)

        self.sigInteractiveModeStarted.emit(kind)

        return True

    def __roiInteractiveModeEnded(self):
        """Handle end of ROI draw interactive mode"""
        if self.isStarted():
            self._shapeKind = None

            if self._drawnROI is not None:
                # Cancel ROI create
                self.removeRegionOfInterest(self._drawnROI)
                self._drawnROI = None

            plot = self.parent()
            if plot is not None:
                plot.sigPlotSignal.disconnect(self._handleInteraction)

            self._updateModeActions()

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
            plot.setInteractiveMode(mode='zoom', source=None)
        else:  # Fallback
            self.__roiInteractiveModeEnded()

        return True

    def exec_(self, kind):
        """Block until :meth:`quit` is called.

        :param str kind: The kind of ROI shape in:
           'point', 'rectangle', 'line', 'polygon', 'hline', 'vline'
        :return: The list of ROIs
        :rtype: tuple
        """
        self.start(kind=kind)

        plot = self.parent()
        plot.show()
        plot.raise_()

        self._eventLoop = qt.QEventLoop()
        self._eventLoop.exec_()
        self._eventLoop = None

        self.stop()

        rois = self.getRegionOfInterests()
        self.clearRegionOfInterests()
        return rois

    def quit(self):
        """Stop a blocking :meth:`exec_` and call :meth:`stop`"""
        if self._eventLoop is not None:
            self._eventLoop.quit()
            self._eventLoop = None
        self.stop()


class InteractiveRegionOfInterestManager(RegionOfInterestManager):
    """RegionOfInterestManager with features for use from interpreter.

    It is meant to be used through the :meth:`exec_`.
    It provides some messages to display in a status bar and
    different modes to end blocking calls to :meth:`exec_`.

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
        self.__message = ''
        self.__validationMode = self.ValidationMode.ENTER
        self.__execKind = None

        self.sigRegionOfInterestAdded.connect(self.__added)
        self.sigRegionOfInterestAboutToBeRemoved.connect(self.__aboutToBeRemoved)
        self.sigInteractiveModeStarted.connect(self.__started)
        self.sigInteractiveModeFinished.connect(self.__finished)

    # Max ROI

    def getMaxRegionOfInterests(self):
        """Returns the maximum number of ROIs or None if no limit.

        :rtype: Union[int,None]
        """
        return self._maxROI

    def setMaxRegionOfInterests(self, max_):
        """Set the maximum number of ROIs.

        :param Union[int,None] max_: The max limit or None for no limit.
        :raise ValueError: If there is more ROIs than max value
        """
        if max_ is not None:
            max_ = int(max_)
            if max_ <= 0:
                raise ValueError('Max limit must be strictly positive')

            if len(self.getRegionOfInterests()) > max_:
                raise ValueError(
                    'Cannot set max limit: Already too many ROIs')

        self._maxROI = max_

    def isMaxRegionOfInterests(self):
        """Returns True if the maximum number of ROIs is reached.

        :rtype: bool
        """
        max_ = self.getMaxRegionOfInterests()
        return max_ is not None and len(self.getRegionOfInterests()) >= max_

    # Validation mode

    @enum.unique
    class ValidationMode(enum.Enum):
        """Mode of validation to leave blocking :meth:`exec_`"""

        AUTO = 'auto'
        """Automatically ends the interactive mode once
        the user terminates the last ROI shape."""

        ENTER = 'enter'
        """Ends the interactive mode when the *Enter* key is pressed."""

        AUTO_ENTER = 'auto_enter'
        """Ends the interactive mode when reaching max ROIs or
        when the *Enter* key is pressed.
        """

        NONE = 'none'
        """Do not provide the user a way to end the interactive mode.

        The end of :meth:`exec_` is done through :meth:`quit` or timeout.
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
            if (self.isMaxRegionOfInterests() and self.getValidationMode() in
                    (self.ValidationMode.AUTO,
                     self.ValidationMode.AUTO_ENTER)):
                self.quit()

            self.__updateMessage()

    def eventFilter(self, obj, event):
        if event.type() == qt.QEvent.Hide:
            self.quit()

        if event.type() == qt.QEvent.KeyPress:
            key = event.key()
            if (key in (qt.Qt.Key_Return, qt.Qt.Key_Enter) and
                    self.getValidationMode() in (
                        self.ValidationMode.ENTER,
                        self.ValidationMode.AUTO_ENTER)):
                # Stop on return key pressed
                self.quit()
                return True  # Stop further handling of this keys

            if (key in (qt.Qt.Key_Delete, qt.Qt.Key_Backspace) or (
                    key == qt.Qt.Key_Z and
                    event.modifiers() & qt.Qt.ControlModifier)):
                rois = self.getRegionOfInterests()
                if rois:  # Something to undo
                    self.removeRegionOfInterest(rois[-1])
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
            return self.__message + (' - %d seconds remaining' %
                                     max(1, int(remaining)))

    # Listen to ROI updates

    def __added(self, *args, **kwargs):
        """Handle new ROI added"""
        max_ = self.getMaxRegionOfInterests()
        if max_ is not None:
            # When reaching max number of ROIs, redo last one
            while len(self.getRegionOfInterests()) > max_:
                self.removeRegionOfInterest(self.getRegionOfInterests()[-2])

        self.__updateMessage()
        if (self.isMaxRegionOfInterests() and
                self.getValidationMode() in (self.ValidationMode.AUTO,
                                             self.ValidationMode.AUTO_ENTER)):
            self.quit()

    def __aboutToBeRemoved(self, *args, **kwargs):
        """Handle removal of a ROI"""
        # RegionOfInterest not removed yet
        self.__updateMessage(nbrois=len(self.getRegionOfInterests()) - 1)

    def __started(self, roiKind):
        """Handle interactive mode started"""
        self.__updateMessage()

    def __finished(self):
        """Handle interactive mode finished"""
        self.__updateMessage()

    def __updateMessage(self, nbrois=None):
        """Update message"""
        if not self.isExec():
            message = 'Done'

        elif not self.isStarted():
            message = 'Use %s ROI edition mode' % self.__execKind

        else:
            if nbrois is None:
                nbrois = len(self.getRegionOfInterests())

            kind = self.__execKind
            max_ = self.getMaxRegionOfInterests()

            if max_ is None:
                message = 'Select %ss (%d selected)' % (kind, nbrois)

            elif max_ <= 1:
                message = 'Select a %s' % kind
            else:
                message = 'Select %d/%d %ss' % (nbrois, max_, kind)

            if (self.getValidationMode() == self.ValidationMode.ENTER and
                    self.isMaxRegionOfInterests()):
                message += ' - Press Enter to confirm'

        if message != self.__message:
            self.__message = message
            # Use getMessage to add timeout message
            self.sigMessageChanged.emit(self.getMessage())

    # Handle blocking call

    def __timeoutUpdate(self):
        """Handle update of timeout"""
        if (self.__timeoutEndTime is not None and
                (self.__timeoutEndTime - time.time()) > 0):
                self.sigMessageChanged.emit(self.getMessage())
        else:  # Stop interactive mode and message timer
            timer = self.sender()
            if timer is not None:
                timer.stop()
            self.__timeoutEndTime = None
            self.quit()

    def isExec(self):
        """Returns True if :meth:`exec_` is currently running.

        :rtype: bool"""
        return self.__execKind is not None

    def exec_(self, kind, timeout=0):
        """Block until ROI selection is done or timeout is elapsed.

        :meth:`quit` also ends this blocking call.

        :param str kind: The kind of ROI shape in:
           'point', 'rectangle', 'line', 'polygon', 'hline', 'vline'
        :param int timeout: Maximum duration in seconds to block.
            Default: No timeout
        :return: The list of ROIs
        :rtype: List[RegionOfInterest]
        """
        plot = self.parent()
        if plot is None:
            return

        self.__execKind = kind

        plot.installEventFilter(self)

        if timeout > 0:
            self.__timeoutEndTime = time.time() + timeout
            timer = qt.QTimer(self)
            timer.timeout.connect(self.__timeoutUpdate)
            timer.start(1000)

            rois = super(InteractiveRegionOfInterestManager, self).exec_(kind)

            timer.stop()
            self.__timeoutEndTime = None

        else:
            rois = super(InteractiveRegionOfInterestManager, self).exec_(kind)

        plot.removeEventFilter(self)

        self.__execKind = None
        self.__updateMessage()

        return rois


class _DeleteRegionOfInterestToolButton(qt.QToolButton):
    """Tool button deleting a ROI object

    :param parent: See QWidget
    :param RegionOfInterest roi: The ROI to delete
    """

    def __init__(self, parent, roi):
        super(_DeleteRegionOfInterestToolButton, self).__init__(parent)
        self.setIcon(icons.getQIcon('remove'))
        self.setToolTip("Remove this ROI")
        self.__roiRef = roi if roi is None else weakref.ref(roi)
        self.clicked.connect(self.__clicked)

    def __clicked(self, checked):
        """Handle button clicked"""
        roi = None if self.__roiRef is None else self.__roiRef()
        if roi is not None:
            manager = roi.parent()
            if manager is not None:
                manager.removeRegionOfInterest(roi)
                self.__roiRef = None


class RegionOfInterestTableWidget(qt.QTableWidget):
    """Widget displaying the ROIs of a :class:`RegionOfInterestManager`"""

    def __init__(self, parent=None):
        super(RegionOfInterestTableWidget, self).__init__(parent)
        self._roiManagerRef = None

        self.setColumnCount(5)
        self.setHorizontalHeaderLabels(
            ['Label', 'Edit', 'Kind', 'Coordinates', ''])

        horizontalHeader = self.horizontalHeader()
        horizontalHeader.setDefaultAlignment(qt.Qt.AlignLeft)
        if hasattr(horizontalHeader, 'setResizeMode'):  # Qt 4
            setSectionResizeMode = horizontalHeader.setResizeMode
        else:  # Qt5
            setSectionResizeMode = horizontalHeader.setSectionResizeMode

        setSectionResizeMode(0, qt.QHeaderView.Interactive)
        setSectionResizeMode(1, qt.QHeaderView.ResizeToContents)
        setSectionResizeMode(2, qt.QHeaderView.ResizeToContents)
        setSectionResizeMode(3, qt.QHeaderView.Stretch)
        setSectionResizeMode(4, qt.QHeaderView.ResizeToContents)

        verticalHeader = self.verticalHeader()
        verticalHeader.setVisible(False)

        self.setSelectionMode(qt.QAbstractItemView.NoSelection)
        self.setFocusPolicy(qt.Qt.NoFocus)

        self.itemChanged.connect(self.__itemChanged)

    @staticmethod
    def __itemChanged(item):
        """Handle item updates"""
        column = item.column()
        roi = item.data(qt.Qt.UserRole)
        if column == 0:
            roi.setLabel(item.text())
        elif column == 1:
            roi.setEditable(
                item.checkState() == qt.Qt.Checked)
        elif column in (2, 3, 4):
            pass  # TODO
        else:
            logger.error('Unhandled column %d', column)

    def setRegionOfInterestManager(self, manager):
        """Set the :class:`RegionOfInterestManager` object to sync with

        :param RegionOfInterestManager manager:
        """
        assert manager is None or isinstance(manager, RegionOfInterestManager)

        previousManager = self.getRegionOfInterestManager()

        if previousManager is not None:
            previousManager.sigRegionOfInterestChanged.disconnect(self._sync)
        self.setRowCount(0)

        self._roiManagerRef = weakref.ref(manager)

        self._sync()

        if manager is not None:
            manager.sigRegionOfInterestChanged.connect(self._sync)

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

    def _sync(self):
        """Update widget content according to ROI manger"""
        manager = self.getRegionOfInterestManager()

        if manager is None:
            self.setRowCount(0)
            return

        rois = manager.getRegionOfInterests()

        self.setRowCount(len(rois))
        for index, roi in enumerate(rois):
            baseFlags = qt.Qt.ItemIsSelectable | qt.Qt.ItemIsEnabled

            # Label
            label = roi.getLabel()
            item = qt.QTableWidgetItem(label)
            item.setFlags(baseFlags | qt.Qt.ItemIsEditable)
            item.setData(qt.Qt.UserRole, roi)
            self.setItem(index, 0, item)

            # Editable
            item = qt.QTableWidgetItem()
            item.setFlags(baseFlags | qt.Qt.ItemIsUserCheckable)
            item.setData(qt.Qt.UserRole, roi)
            item.setCheckState(
                qt.Qt.Checked if roi.isEditable() else qt.Qt.Unchecked)
            self.setItem(index, 1, item)
            item.setTextAlignment(qt.Qt.AlignCenter)
            item.setText(None)

            # Kind
            label = roi._getKind()
            if label is None:
                # Default value if kind is not overrided
                label = roi.__class__.__name__
            item = qt.QTableWidgetItem(label.capitalize())
            item.setFlags(baseFlags)
            self.setItem(index, 2, item)

            item = qt.QTableWidgetItem()
            item.setFlags(baseFlags)

            # Coordinates
            text = self._getReadableRoiDescription(roi)
            item.setText(text)
            self.setItem(index, 3, item)

            # Delete
            delBtn = _DeleteRegionOfInterestToolButton(None, roi)
            widget = qt.QWidget(self)
            layout = qt.QHBoxLayout()
            layout.setContentsMargins(2, 2, 2, 2)
            layout.setSpacing(0)
            widget.setLayout(layout)
            layout.addStretch(1)
            layout.addWidget(delBtn)
            layout.addStretch(1)
            self.setCellWidget(index, 4, widget)

    def getRegionOfInterestManager(self):
        """Returns the :class:`RegionOfInterestManager` this widget supervise.

        It returns None if not sync with an :class:`RegionOfInterestManager`.

        :rtype: RegionOfInterestManager
        """
        return None if self._roiManagerRef is None else self._roiManagerRef()
