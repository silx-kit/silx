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

from .. import qt
from ..plot.Colors import rgba

from .Plot3DWidget import Plot3DWidget
from .scene import axes
from . import items
from ._model import SceneModel


__all__ = ['items', 'SceneWidget']


class SceneWidget(Plot3DWidget):
    """Widget displaying data sets in 3D"""

    def __init__(self, parent=None):
        super(SceneWidget, self).__init__(parent)
        self._model = None
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
