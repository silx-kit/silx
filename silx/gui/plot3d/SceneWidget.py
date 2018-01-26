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
"""This module provides a widget to view data sets in 3D."""

from __future__ import absolute_import

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "26/10/2017"

import numpy
import weakref

from .. import qt
from ..plot.Colors import rgba

from .Plot3DWidget import Plot3DWidget
from . import items
from ._model import SceneModel, visitQAbstractItemModel
from ._model.items import Item3DRow


__all__ = ['items', 'SceneWidget']


class SceneSelection(qt.QObject):
    """Object managing a :class:`SceneWidget` selection

    :param QObject parent:
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
        if current is not previous:
            self.sigCurrentChanged.emit(current, previous)
            self.__updateSelectionModel()

    def __currentChanged(self, event):
        """Handle updates of the selected item"""
        if event == items.Item3DChangedType.ROOT_ITEM:
            item = self.sender()
            if item.root() != self.getSceneGroup():
                self.setSelectedItem(None)

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

        self._sceneGroup = items.GroupWithAxesItem(parent=self)
        self._sceneGroup.setLabel('Data')

        self.viewport.scene.children.append(self._sceneGroup._getScenePrimitive())

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

    # Add/remove items

    def add3DScalarField(self, data, copy=True, index=None):
        """Add 3D scalar data volume to :class:`SceneWidget` content.

        Dataset order is zyx (i.e., first dimension is z).

        :param data: 3D array
        :type data: 3D numpy.ndarray of float32 with shape at least (2, 2, 2)
        :param bool copy:
            True (default) to make a copy,
            False to avoid copy (DO NOT MODIFY data afterwards)
        :param int index: The index at which to place the item.
                          By default it is appended to the end of the list.
        :return: The newly created scalar volume item
        :rtype: items.ScalarField3D
        """
        volume = items.ScalarField3D()
        volume.setData(data, copy=copy)
        self.addItem(volume, index)
        return volume

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
        :rtype: items.Scatter3D
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
        :rtype: items.Scatter2D
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
        :rtype: items.ImageData or items.ImageRgba
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
        return self.getSceneGroup().clear()

    # Colors

    def _updateColors(self):
        """Update item depending on foreground/highlight color"""
        bbox = self._sceneGroup._getScenePrimitive()  # TODO move in group
        bbox.tickColor = self._textColor
        bbox.color = self._foregroundColor

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
            self._updateColors()
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
            self._updateColors()

            # Update scene items
            for item in self.getSceneGroup().visit():
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
            self._updateColors()
            self.sigStyleChanged.emit('highlightColor')
