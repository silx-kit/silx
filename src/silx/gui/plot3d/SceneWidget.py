# /*##########################################################################
#
# Copyright (c) 2017-2019 European Synchrotron Radiation Facility
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
"""This module provides a widget to view data sets in 3D."""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "24/04/2018"

import enum
import weakref

import numpy

from .. import qt
from ..colors import rgba

from .Plot3DWidget import Plot3DWidget
from . import items
from .items.core import RootGroupWithAxesItem
from .scene import interaction
from ._model import SceneModel, visitQAbstractItemModel
from ._model.items import Item3DRow

__all__ = ['items', 'SceneWidget']


class _SceneSelectionHighlightManager(object):
    """Class controlling the highlight of the selection in a SceneWidget

    :param ~silx.gui.plot3d.SceneWidget.SceneSelection:
    """

    def __init__(self, selection):
        assert isinstance(selection, SceneSelection)
        self._sceneWidget = weakref.ref(selection.parent())

        self._enabled = True
        self._previousBBoxState = None

        self.__selectItem(selection.getCurrentItem())
        selection.sigCurrentChanged.connect(self.__currentChanged)

    def isEnabled(self):
        """Returns True if highlight of selection in enabled.

        :rtype: bool
        """
        return self._enabled

    def setEnabled(self, enabled=True):
        """Activate/deactivate selection highlighting

        :param bool enabled: True (default) to enable selection highlighting
        """
        enabled = bool(enabled)
        if enabled != self._enabled:
            self._enabled = enabled

            sceneWidget = self.getSceneWidget()
            if sceneWidget is not None:
                selection = sceneWidget.selection()
                current = selection.getCurrentItem()

                if enabled:
                    self.__selectItem(current)
                    selection.sigCurrentChanged.connect(self.__currentChanged)

                else:  # disabled
                    self.__unselectItem(current)
                    selection.sigCurrentChanged.disconnect(
                        self.__currentChanged)

    def getSceneWidget(self):
        """Returns the SceneWidget this class controls highlight for.

        :rtype: ~silx.gui.plot3d.SceneWidget.SceneWidget
        """
        return self._sceneWidget()

    def __selectItem(self, current):
        """Highlight given item.

         :param ~silx.gui.plot3d.items.Item3D current: New current or None
        """
        if current is None:
            return

        sceneWidget = self.getSceneWidget()
        if sceneWidget is None:
            return

        if isinstance(current, items.DataItem3D):
            self._previousBBoxState = current.isBoundingBoxVisible()
            current.setBoundingBoxVisible(True)
        current._setForegroundColor(sceneWidget.getHighlightColor())
        current.sigItemChanged.connect(self.__selectedChanged)

    def __unselectItem(self, current):
        """Remove highlight of given item.

        :param ~silx.gui.plot3d.items.Item3D current:
            Currently highlighted item
        """
        if current is None:
            return

        sceneWidget = self.getSceneWidget()
        if sceneWidget is None:
            return

        # Restore bbox visibility and color
        current.sigItemChanged.disconnect(self.__selectedChanged)
        if (self._previousBBoxState is not None and
                isinstance(current, items.DataItem3D)):
            current.setBoundingBoxVisible(self._previousBBoxState)
        current._setForegroundColor(sceneWidget.getForegroundColor())

    def __currentChanged(self, current, previous):
        """Handle change of current item in the selection

        :param ~silx.gui.plot3d.items.Item3D current: New current or None
        :param ~silx.gui.plot3d.items.Item3D previous: Previous current or None
        """
        self.__unselectItem(previous)
        self.__selectItem(current)

    def __selectedChanged(self, event):
        """Handle updates of selected item bbox.

        If bbox gets changed while selected, do not restore state.

        :param event:
        """
        if event == items.Item3DChangedType.BOUNDING_BOX_VISIBLE:
            self._previousBBoxState = None


@enum.unique
class HighlightMode(enum.Enum):
    """:class:`SceneSelection` highlight modes"""

    NONE = 'noHighlight'
    """Do not highlight selected item"""

    BOUNDING_BOX = 'boundingBox'
    """Highlight selected item bounding box"""


class SceneSelection(qt.QObject):
    """Object managing a :class:`SceneWidget` selection

    :param SceneWidget parent:
    """

    NO_SELECTION = 0
    """Flag for no item selected"""

    sigCurrentChanged = qt.Signal(object, object)
    """This signal is emitted whenever the current item changes.

    It provides the current and previous items.
    Either of those can be :attr:`NO_SELECTION`.
    """

    def __init__(self, parent=None):
        super(SceneSelection, self).__init__(parent)
        self.__current = None  # Store weakref to current item
        self.__selectionModel = None  # Store sync selection model
        self.__syncInProgress = False  # True during model synchronization

        self.__highlightManager = _SceneSelectionHighlightManager(self)

    def getHighlightMode(self):
        """Returns current selection highlight mode.

        Either NONE or BOUNDING_BOX.

        :rtype: HighlightMode
        """
        if self.__highlightManager.isEnabled():
            return HighlightMode.BOUNDING_BOX
        else:
            return HighlightMode.NONE

    def setHighlightMode(self, mode):
        """Set selection highlighting mode

        :param HighlightMode mode: The mode to use
        """
        assert isinstance(mode, HighlightMode)
        self.__highlightManager.setEnabled(mode == HighlightMode.BOUNDING_BOX)

    def getCurrentItem(self):
        """Returns the current item in the scene or None.

        :rtype: Union[~silx.gui.plot3d.items.Item3D, None]
        """
        return None if self.__current is None else self.__current()

    def setCurrentItem(self, item):
        """Set the current item in the scene.

        :param Union[Item3D, None] item:
            The new item to select or None to clear the selection.
        :raise ValueError: If the item is not the widget's scene
        """
        previous = self.getCurrentItem()
        if item is previous:
            return  # Fast path, nothing to do

        if previous is not None:
            previous.sigItemChanged.disconnect(self.__currentChanged)

        if item is None:
            self.__current = None

        elif isinstance(item, items.Item3D):
            parent = self.parent()
            assert isinstance(parent, SceneWidget)

            sceneGroup = parent.getSceneGroup()
            if item is sceneGroup or item.root() is sceneGroup:
                item.sigItemChanged.connect(self.__currentChanged)
                self.__current = weakref.ref(item)
            else:
                raise ValueError(
                    'Item is not in this SceneWidget: %s' % str(item))

        else:
            raise ValueError(
                'Not an Item3D: %s' % str(item))

        current = self.getCurrentItem()
        self.sigCurrentChanged.emit(current, previous)
        self.__updateSelectionModel()

    def __currentChanged(self, event):
        """Handle updates of the selected item"""
        if event == items.Item3DChangedType.ROOT_ITEM:
            item = self.sender()

            parent = self.parent()
            assert isinstance(parent, SceneWidget)

            if item.root() != parent.getSceneGroup():
                self.setCurrentItem(None)

    # Synchronization with QItemSelectionModel

    def _getSyncSelectionModel(self):
        """Returns the QItemSelectionModel this selection is synchronized with.

        :rtype: Union[QItemSelectionModel, None]
        """
        return self.__selectionModel

    def _setSyncSelectionModel(self, selectionModel):
        """Synchronizes this selection object with a selection model.

        :param Union[QItemSelectionModel, None] selectionModel:
        :raise ValueError: If the selection model does not correspond
                           to the same :class:`SceneWidget`
        """
        if (not isinstance(selectionModel, qt.QItemSelectionModel) or
                not isinstance(selectionModel.model(), SceneModel) or
                selectionModel.model().sceneWidget() is not self.parent()):
            raise ValueError("Expecting a QItemSelectionModel "
                             "attached to the same SceneWidget")

        # Disconnect from previous selection model
        previousSelectionModel = self._getSyncSelectionModel()
        if previousSelectionModel is not None:
            previousSelectionModel.selectionChanged.disconnect(
                self.__selectionModelSelectionChanged)

        self.__selectionModel = selectionModel

        if selectionModel is not None:
            # Connect to new selection model
            selectionModel.selectionChanged.connect(
                self.__selectionModelSelectionChanged)
            self.__updateSelectionModel()

    def __selectionModelSelectionChanged(self, selected, deselected):
        """Handle QItemSelectionModel selection updates.

        :param QItemSelection selected:
        :param QItemSelection deselected:
        """
        if self.__syncInProgress:
            return

        indices = selected.indexes()
        if not indices:
            item = None

        else:  # Select the first selected item
            index = indices[0]
            itemRow = index.internalPointer()
            if isinstance(itemRow, Item3DRow):
                item = itemRow.item()
            else:
                item = None

        self.setCurrentItem(item)

    def __updateSelectionModel(self):
        """Sync selection model when current item has been updated"""
        selectionModel = self._getSyncSelectionModel()
        if selectionModel is None:
            return

        currentItem = self.getCurrentItem()

        if currentItem is None:
            selectionModel.clear()

        else:
            # visit the model to find selectable index corresponding to item
            model = selectionModel.model()
            for index in visitQAbstractItemModel(model):
                itemRow = index.internalPointer()
                if (isinstance(itemRow, Item3DRow) and
                        itemRow.item() is currentItem and
                        index.flags() & qt.Qt.ItemIsSelectable):
                    # This is the item we are looking for: select it in the model
                    self.__syncInProgress = True
                    selectionModel.select(
                        index, qt.QItemSelectionModel.Clear |
                               qt.QItemSelectionModel.Select |
                               qt.QItemSelectionModel.Current)
                    self.__syncInProgress = False
                    break


class SceneWidget(Plot3DWidget):
    """Widget displaying data sets in 3D"""

    def __init__(self, parent=None):
        super(SceneWidget, self).__init__(parent)
        self._model = None  # Store lazy-loaded model
        self._selection = None  # Store lazy-loaded SceneSelection
        self._items = []

        self._textColor = 1., 1., 1., 1.
        self._foregroundColor = 1., 1., 1., 1.
        self._highlightColor = 0.7, 0.7, 0., 1.

        self._sceneGroup = RootGroupWithAxesItem(parent=self)
        self._sceneGroup.setLabel('Data')

        self.viewport.scene.children.append(
            self._sceneGroup._getScenePrimitive())

    def model(self):
        """Returns the model corresponding the scene of this widget

        :rtype: SceneModel
        """
        if self._model is None:
            # Lazy-loading of the model
            self._model = SceneModel(parent=self)
        return self._model

    def selection(self):
        """Returns the object managing selection in the scene

        :rtype: SceneSelection
        """
        if self._selection is None:
            # Lazy-loading of the SceneSelection
            self._selection = SceneSelection(parent=self)
        return self._selection

    def getSceneGroup(self):
        """Returns the root group of the scene

        :rtype: GroupItem
        """
        return self._sceneGroup

    def pickItems(self, x, y, condition=None):
        """Iterator over picked items in the scene at given position.

        Each picked item yield a
        :class:`~silx.gui.plot3d.items._pick.PickingResult` object
        holding the picking information.

        It traverses the scene tree in a left-to-right top-down way.

        :param int x: X widget coordinate
        :param int y: Y widget coordinate
        :param callable condition: Optional test called for each item
            checking whether to process it or not.
        """
        if not self.isValid() or not self.isVisible():
            return  # Empty iterator

        devicePixelRatio = self.getDevicePixelRatio()
        for result in self.getSceneGroup().pickItems(
            x * devicePixelRatio, y * devicePixelRatio, condition):
            yield result

    # Interactive modes

    def _handleSelectionChanged(self, current, previous):
        """Handle change of selection to update interactive mode"""
        if self.getInteractiveMode() == 'panSelectedPlane':
            if isinstance(current, items.PlaneMixIn):
                # Update pan plane to use new selected plane
                self.setInteractiveMode('panSelectedPlane')

            else:  # Switch to rotate scene if new selection is not a plane
                self.setInteractiveMode('rotate')

    def setInteractiveMode(self, mode):
        """Set the interactive mode.

        'panSelectedPlane' mode set plane panning if a plane is selected,
        otherwise it fall backs to 'rotate'.

        :param str mode:
            The interactive mode: 'rotate', 'pan', 'panSelectedPlane' or None
        """
        if self.getInteractiveMode() == 'panSelectedPlane':
            self.selection().sigCurrentChanged.disconnect(
                self._handleSelectionChanged)

        if mode == 'panSelectedPlane':
            selected = self.selection().getCurrentItem()

            if isinstance(selected, items.PlaneMixIn):
                mode = interaction.PanPlaneZoomOnWheelControl(
                    self.viewport,
                    selected._getPlane(),
                    mode='position',
                    orbitAroundCenter=False,
                    scaleTransform=self._sceneScale)

                self.selection().sigCurrentChanged.connect(
                    self._handleSelectionChanged)

            else:  # No selected plane, fallback to rotate scene
                mode = 'rotate'

        super(SceneWidget, self).setInteractiveMode(mode)

    def getInteractiveMode(self):
        """Returns the interactive mode in use.

        :rtype: str
        """
        if isinstance(self.eventHandler, interaction.PanPlaneZoomOnWheelControl):
            return 'panSelectedPlane'
        else:
            return super(SceneWidget, self).getInteractiveMode()

    # Add/remove items

    def addVolume(self, data, copy=True, index=None):
        """Add 3D data volume of scalar or complex to :class:`SceneWidget` content.

        Dataset order is zyx (i.e., first dimension is z).

        :param data: 3D array of complex with shape at least (2, 2, 2)
        :type data: numpy.ndarray[Union[numpy.complex64,numpy.float32]]
        :param bool copy:
            True (default) to make a copy,
            False to avoid copy (DO NOT MODIFY data afterwards)
        :param int index: The index at which to place the item.
                          By default it is appended to the end of the list.
        :return: The newly created 3D volume item
        :rtype: Union[ScalarField3D,ComplexField3D]

        """
        if data is not None:
            data = numpy.array(data, copy=False)

        if numpy.iscomplexobj(data):
            volume = items.ComplexField3D()
        else:
            volume = items.ScalarField3D()
        volume.setData(data, copy=copy)
        self.addItem(volume, index)
        return volume

    def add3DScalarField(self, data, copy=True, index=None):
        # TODO deprecate in the future
        return self.addVolume(data, copy=copy, index=index)

    def add3DScatter(self, x, y, z, value, copy=True, index=None):
        """Add 3D scatter data to :class:`SceneWidget` content.

        :param numpy.ndarray x: Array of X coordinates (single value not accepted)
        :param y: Points Y coordinate (array-like or single value)
        :param z: Points Z coordinate (array-like or single value)
        :param value: Points values (array-like or single value)
        :param bool copy:
            True (default) to copy the data,
            False to use provided data (do not modify!)
        :param int index: The index at which to place the item.
                          By default it is appended to the end of the list.
        :return: The newly created 3D scatter item
        :rtype: ~silx.gui.plot3d.items.scatter.Scatter3D
        """
        scatter3d = items.Scatter3D()
        scatter3d.setData(x=x, y=y, z=z, value=value, copy=copy)
        self.addItem(scatter3d, index)
        return scatter3d

    def add2DScatter(self, x, y, value, copy=True, index=None):
        """Add 2D scatter data to :class:`SceneWidget` content.

        Provided arrays must have the same length.

        :param numpy.ndarray x: X coordinates (array-like)
        :param numpy.ndarray y: Y coordinates (array-like)
        :param value: Points value: array-like or single scalar
        :param bool copy: True (default) to copy the data,
                          False to use as is (do not modify!).
        :param int index: The index at which to place the item.
                          By default it is appended to the end of the list.
        :return: The newly created 2D scatter item
        :rtype: ~silx.gui.plot3d.items.scatter.Scatter2D
        """
        scatter2d = items.Scatter2D()
        scatter2d.setData(x=x, y=y, value=value, copy=copy)
        self.addItem(scatter2d, index)
        return scatter2d

    def addImage(self, data, copy=True, index=None):
        """Add a 2D data or RGB(A) image to :class:`SceneWidget` content.

        2D data is casted to float32.
        RGBA supported formats are: float32 in [0, 1] and uint8.

        :param numpy.ndarray data: Image as a 2D data array or
            RGBA image as a 3D array (height, width, channels)
        :param bool copy: True (default) to copy the data,
                          False to use as is (do not modify!).
        :param int index: The index at which to place the item.
                          By default it is appended to the end of the list.
        :return: The newly created image item
        :rtype: ~silx.gui.plot3d.items.image.ImageData or ~silx.gui.plot3d.items.image.ImageRgba
        :raise ValueError: For arrays of unsupported dimensions
        """
        data = numpy.array(data, copy=False)
        if data.ndim == 2:
            image = items.ImageData()
        elif data.ndim == 3:
            image = items.ImageRgba()
        else:
            raise ValueError("Unsupported array dimensions: %d" % data.ndim)
        image.setData(data, copy=copy)
        self.addItem(image, index)
        return image

    def addItem(self, item, index=None):
        """Add an item to :class:`SceneWidget` content

        :param Item3D item: The item  to add
        :param int index: The index at which to place the item.
                          By default it is appended to the end of the list.
        :raise ValueError: If the item is already in the :class:`SceneWidget`.
        """
        return self.getSceneGroup().addItem(item, index)

    def removeItem(self, item):
        """Remove an item from :class:`SceneWidget` content.

        :param Item3D item: The item to remove from the scene
        :raises ValueError: If the item does not belong to the group
        """
        return self.getSceneGroup().removeItem(item)

    def getItems(self):
        """Returns the list of :class:`SceneWidget` items.

        Only items in the top-level group are returned.

        :rtype: tuple
        """
        return self.getSceneGroup().getItems()

    def clearItems(self):
        """Remove all item from :class:`SceneWidget`."""
        return self.getSceneGroup().clearItems()

    # Colors

    def getTextColor(self):
        """Return color used for text

        :rtype: QColor"""
        return qt.QColor.fromRgbF(*self._textColor)

    def setTextColor(self, color):
        """Set the text color.

        :param color: RGB color: name, #RRGGBB or RGB values
        :type color:
            QColor, str or array-like of 3 or 4 float in [0., 1.] or uint8
        """
        color = rgba(color)
        if color != self._textColor:
            self._textColor = color

            # Update text color
            # TODO make entry point in Item3D for this
            bbox = self._sceneGroup._getScenePrimitive()
            bbox.tickColor = color

            self.sigStyleChanged.emit('textColor')

    def getForegroundColor(self):
        """Return color used for bounding box

        :rtype: QColor
        """
        return qt.QColor.fromRgbF(*self._foregroundColor)

    def setForegroundColor(self, color):
        """Set the foreground color.

        :param color: RGB color: name, #RRGGBB or RGB values
        :type color:
            QColor, str or array-like of 3 or 4 float in [0., 1.] or uint8
        """
        color = rgba(color)
        if color != self._foregroundColor:
            self._foregroundColor = color

            # Update scene items
            selected = self.selection().getCurrentItem()
            for item in self.getSceneGroup().visit(included=True):
                if item is not selected:
                    item._setForegroundColor(color)

            self.sigStyleChanged.emit('foregroundColor')

    def getHighlightColor(self):
        """Return color used for highlighted item bounding box

        :rtype: QColor
        """
        return qt.QColor.fromRgbF(*self._highlightColor)

    def setHighlightColor(self, color):
        """Set highlighted item color.

        :param color: RGB color: name, #RRGGBB or RGB values
        :type color:
            QColor, str or array-like of 3 or 4 float in [0., 1.] or uint8
        """
        color = rgba(color)
        if color != self._highlightColor:
            self._highlightColor = color

            selected = self.selection().getCurrentItem()
            if selected is not None:
                selected._setForegroundColor(color)

            self.sigStyleChanged.emit('highlightColor')
