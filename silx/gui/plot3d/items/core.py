# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017 European Synchrotron Radiation Facility
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
"""This module provides the base class for items of the :class:`.SceneView`.
"""

from __future__ import absolute_import

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "15/11/2017"

import numpy

from silx.third_party import enum

from ... import qt
from ...plot.items import ItemChangedType
from .. import scene
from ..scene import transform


@enum.unique
class Item3DChangedType(enum.Enum):
    """Type of modification provided by :attr:`Item3D.sigItemChanged` signal."""

    INTERPOLATION = 'interpolationChanged'
    """Item3D image interpolation changed flag."""

    TRANSFORM = 'transformChanged'
    """Item3D transform changed flag."""

    VISUALIZATION_MODE = 'visualizationModeChanged'
    """Item3D visualization mode changed flag."""

    HEIGHT_MAP = 'heightMapChanged'
    """Item3D height map changed flag."""

    ISO_LEVEL = 'isoLevelChanged'
    """Isosurface level changed flag."""


class Item3D(qt.QObject):
    """Base class representing an item in the scene.

    :param parent: The View widget this item belongs to.
    :param primitive: An optional primitive to use as scene primitive
    """

    sigItemChanged = qt.Signal(object)
    """Signal emitted when an item's property has changed.

    It provides a flag describing which property of the item has changed.
    See :class:`ItemChangedType` and :class:`Item3DChangedType`
    for flags description.
    """

    def __init__(self, parent, primitive=None):
        super(Item3D, self).__init__(parent)

        if primitive is None:
            primitive = scene.Group()

        self._primitive = primitive

    def _getScenePrimitive(self):
        """Return the group containing the item rendering"""
        return self._primitive

    def _updated(self, event=None):
        """Handle MixIn class updates.

        :param event: The event to send to :attr:`sigItemChanged` signal.
        """
        if event is not None:
            self.sigItemChanged.emit(event)

    # Visibility

    def isVisible(self):
        """Returns True if item is visible, else False

        :rtype: bool
        """
        return self._getScenePrimitive().visible

    def setVisible(self, visible=True):
        """Set the visibility of the item in the scene.

        :param bool visible: True (default) to show the item, False to hide
        """
        visible = bool(visible)
        primitive = self._getScenePrimitive()
        if visible != primitive.visible:
            primitive.visible = visible
            self._updated(ItemChangedType.VISIBLE)


# TODO add anchor (i.e. center of rotation)
# TODO add bounding box visible + color
class DataItem3D(Item3D):
    """Base class representing a data item in the scene.

    :param parent: The View widget this item belongs to.
    :param primitive: An optional primitive to use as scene primitive
    """

    def __init__(self, parent, primitive=None):
        Item3D.__init__(self, parent=parent, primitive=primitive)

        # Transformations
        self._rotate = transform.Rotate()
        self._translate = transform.Translate()
        self._matrix = transform.Matrix()
        self._scale = transform.Scale()

        self._getScenePrimitive().transforms = [
            self._translate, self._rotate, self._matrix, self._scale]

    # Transformations

    def setScale(self, sx=1., sy=1., sz=1.):
        """Set the scale of the item in the scene.

        :param float sx: Scale factor along the X axis
        :param float sy: Scale factor along the Y axis
        :param float sz: Scale factor along the Z axis
        """
        scale = numpy.array((sx, sy, sz), dtype=numpy.float32)
        if not numpy.all(numpy.equal(scale, self.getScale())):
            self._scale.scale = scale
            self._updated(Item3DChangedType.TRANSFORM)

    def getScale(self):
        """Returns the scales provided by :meth:`setScale`.

        :rtype: numpy.ndarray
        """
        return self._scale.scale

    def setTranslation(self, x=0., y=0., z=0.):
        """Set the translation of the origin of the item in the scene.

        :param float x: Offset of the data origin on the X axis
        :param float y: Offset of the data origin on the Y axis
        :param float z: Offset of the data origin on the Z axis
        """
        translation = numpy.array((x, y, z), dtype=numpy.float32)
        if not numpy.all(numpy.equal(translation, self.getTranslation())):
            self._translate.translation = translation
            self._updated(Item3DChangedType.TRANSFORM)

    def getTranslation(self):
        """Returns the offset set by :meth:`setTranslation`.

        :rtype: numpy.ndarray
        """
        return self._translate.translation

    def setRotation(self, angle=0., axis=(0., 0., 1.)):  # TODO add center of rotation
        """Set the rotation of the item in the scene

        :param float angle: The rotation angle in degrees.
        :param axis: The (x, y, z) coordinates of the rotation axis.
        """
        axis = numpy.array(axis, dtype=numpy.float32)
        assert axis.ndim == 1
        assert axis.size == 3
        if (self._rotate.angle != angle or
                not numpy.all(numpy.equal(axis, self._rotate.axis))):
            self._rotate.setAngleAxis(angle, axis)
            self._updated(Item3DChangedType.TRANSFORM)

    def getRotation(self):
        """Returns the rotation set by :meth:`setRotation`.

        :return: (angle, axis)
        :rtype: tuple
        """
        return self._rotate.angle, self._rotate.axis

    def setMatrix(self, matrix=None):
        """Set the transform matrix

        :param numpy.ndarray matrix: 3x3 transform matrix
        """
        matrix4x4 = numpy.identity(4, dtype=numpy.float32)

        if matrix is not None:
            matrix = numpy.array(matrix, dtype=numpy.float32)
            assert matrix.shape in ((3, 3), (4, 4))
            matrix4x4[:matrix.shape[0], :matrix.shape[1]] = matrix

        if not numpy.all(numpy.equal(matrix4x4, self._matrix.getMatrix())):
            self._matrix.setMatrix(matrix4x4)
            self._updated(Item3DChangedType.TRANSFORM)

    def getMatrix(self):
        """Returns the matrix set by :meth:`setMatrix`

        :return: 4x4 matrix
        :rtype: numpy.ndarray"""
        return self._matrix.getMatrix(copy=True)


class GroupItem(DataItem3D):
    """Group of items sharing a common transform."""

    sigItemAdded = qt.Signal(object)
    """Signal emitted when a new item is added to the group.

    The newly added item is provided by this signal
    """

    sigItemRemoved = qt.Signal(object)
    """Signal emitted when an item is removed from the group.

    The removed item is provided by this signal.
    """

    def __init__(self, parent=None):
        """Base class representing a group of items in the scene.

        :param parent: The View widget this item belongs to.
        """
        DataItem3D.__init__(self, parent=parent)
        self._items = []

    def addItem(self, item, index=None):
        """Append an item to the group

        :param Item3D item: The item  to add
        :param int index: The index at which to place the item.
                          By default it is appended to the end of the list.
        :raise ValueError: If the item is already in the group.
        """
        assert isinstance(item, Item3D)
        assert item.parent() in (None, self)

        if item in self.getItems():
            raise ValueError("Item3D already in group: %s" % item)

        item.setParent(self)
        if index is None:
            self._getScenePrimitive().children.append(
                item._getScenePrimitive())
            self._items.append(item)
        else:
            self._getScenePrimitive().children.insert(
                index, item._getScenePrimitive())
            self._items.insert(index, item)
        self.sigItemAdded.emit(item)

    def getItems(self):
        """Returns the list of items currently present in the group.

        :rtype: tuple
        """
        return tuple(self._items)

    def removeItem(self, item):
        """Remove an item from the scene.

        :param Item3D item: The item to remove from the scene
        :raises ValueError: If the item does not belong to the group
        """
        if item not in self.getItems():
            raise ValueError("Item3D not in group: %s" % str(item))

        self._getScenePrimitive().children.remove(item._getScenePrimitive())
        self._items.remove(item)
        item.setParent(None)
        self.sigItemRemoved.emit(item)

    def clear(self):
        """Remove all item from the group."""
        for item in self.getItems():
            self.removeItem(item)
