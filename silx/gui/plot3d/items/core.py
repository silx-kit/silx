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
"""This module provides the base class for items of the :class:`.SceneWidget`.
"""

from __future__ import absolute_import

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "15/11/2017"

from collections import defaultdict

import numpy

from silx.third_party import enum, six

from ... import qt
from ...plot.items import ItemChangedType
from .. import scene
from ..scene import axes, primitives, transform


@enum.unique
class Item3DChangedType(enum.Enum):
    """Type of modification provided by :attr:`Item3D.sigItemChanged` signal."""

    INTERPOLATION = 'interpolationChanged'
    """Item3D image interpolation changed flag."""

    TRANSFORM = 'transformChanged'
    """Item3D transform changed flag."""

    HEIGHT_MAP = 'heightMapChanged'
    """Item3D height map changed flag."""

    ISO_LEVEL = 'isoLevelChanged'
    """Isosurface level changed flag."""

    LABEL = 'labelChanged'
    """Item's label changed flag."""

    BOUNDING_BOX_VISIBLE = 'boundingBoxVisibleChanged'
    """Item's bounding box visibility changed"""

    ROOT_ITEM = 'rootItemChanged'
    """Item's root changed flag."""


class Item3D(qt.QObject):
    """Base class representing an item in the scene.

    :param parent: The View widget this item belongs to.
    :param primitive: An optional primitive to use as scene primitive
    """

    _LABEL_INDICES = defaultdict(int)
    """Store per class label indices"""

    sigItemChanged = qt.Signal(object)
    """Signal emitted when an item's property has changed.

    It provides a flag describing which property of the item has changed.
    See :class:`ItemChangedType` and :class:`Item3DChangedType`
    for flags description.
    """

    def __init__(self, parent, primitive=None):
        qt.QObject.__init__(self, parent)

        if primitive is None:
            primitive = scene.Group()

        self._primitive = primitive

        self.__syncForegroundColor()

        labelIndex = self._LABEL_INDICES[self.__class__]
        self._label = six.text_type(self.__class__.__name__)
        if labelIndex != 0:
            self._label += u' %d' % labelIndex
        self._LABEL_INDICES[self.__class__] += 1

        if isinstance(parent, Item3D):
            parent.sigItemChanged.connect(self.__parentItemChanged)

    def setParent(self, parent):
        """Override set parent to handle root item change"""
        previousParent = self.parent()
        if isinstance(previousParent, Item3D):
            previousParent.sigItemChanged.disconnect(self.__parentItemChanged)

        super(Item3D, self).setParent(parent)

        if isinstance(parent, Item3D):
            parent.sigItemChanged.connect(self.__parentItemChanged)

        self._updated(Item3DChangedType.ROOT_ITEM)

    def __parentItemChanged(self, event):
        """Handle updates of the parent if it is an Item3D

        :param Item3DChangedType event:
        """
        if event == Item3DChangedType.ROOT_ITEM:
            self._updated(Item3DChangedType.ROOT_ITEM)

    def root(self):
        """Returns the root of the scene this item belongs to.

        The root is the up-most Item3D in the scene tree hierarchy.

        :rtype: Union[Item3D, None]
        """
        root = None
        ancestor = self.parent()
        while isinstance(ancestor, Item3D):
            root = ancestor
            ancestor = ancestor.parent()

        return root

    def _getScenePrimitive(self):
        """Return the group containing the item rendering"""
        return self._primitive

    def _updated(self, event=None):
        """Handle MixIn class updates.

        :param event: The event to send to :attr:`sigItemChanged` signal.
        """
        if event == Item3DChangedType.ROOT_ITEM:
            self.__syncForegroundColor()

        if event is not None:
            self.sigItemChanged.emit(event)

    # Label

    def getLabel(self):
        """Returns the label associated to this item.

        :rtype: str
        """
        return self._label

    def setLabel(self, label):
        """Set the label associated to this item.

        :param str label:
        """
        label = six.text_type(label)
        if label != self._label:
            self._label = label
            self._updated(Item3DChangedType.LABEL)

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

    # Foreground color

    def _setForegroundColor(self, color):
        """Set the foreground color of the item.

        The default implementation does nothing, override it in subclass.

        :param color: RGBA color
        :type color: tuple of 4 float in [0., 1.]
        """
        if hasattr(super(Item3D, self), '_setForegroundColor'):
            super(Item3D, self)._setForegroundColor(color)

    def __syncForegroundColor(self):
        """Retrieve foreground color from parent and update this item"""
        # Look-up for SceneWidget to get its foreground color
        root = self.root()
        if root is not None:
            widget = root.parent()
            if isinstance(widget, qt.QWidget):
                self._setForegroundColor(
                    widget.getForegroundColor().getRgbF())


class DataItem3D(Item3D):
    """Base class representing a data item with transform in the scene.

    :param parent: The View widget this item belongs to.
    :param Union[GroupBBox, None] group:
        The scene group to use for rendering
    """

    def __init__(self, parent, group=None):
        if group is None:
            group = primitives.GroupBBox()

            # Set-up bounding box
            group.boxVisible = False
            group.axesVisible = False
        else:
            assert isinstance(group, primitives.GroupBBox)

        Item3D.__init__(self, parent=parent, primitive=group)

        # Transformations
        self._translate = transform.Translate()
        self._rotateForwardTranslation = transform.Translate()
        self._rotate = transform.Rotate()
        self._rotateBackwardTranslation = transform.Translate()
        self._translateFromRotationCenter = transform.Translate()
        self._matrix = transform.Matrix()
        self._scale = transform.Scale()
        # Group transforms to do to data before rotation
        # This is useful to handle rotation center relative to bbox
        self._transformObjectToRotate = transform.TransformList(
            [self._matrix, self._scale])
        self._transformObjectToRotate.addListener(self._updateRotationCenter)

        self._rotationCenter = 0., 0., 0.

        self._getScenePrimitive().transforms = [
            self._translate,
            self._rotateForwardTranslation,
            self._rotate,
            self._rotateBackwardTranslation,
            self._transformObjectToRotate]

    def _updated(self, event=None):
        """Handle MixIn class updates.

        :param event: The event to send to :attr:`sigItemChanged` signal.
        """
        if event == ItemChangedType.DATA:
            self._updateRotationCenter()
        super(DataItem3D, self)._updated(event)

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

    _ROTATION_CENTER_TAGS = 'lower', 'center', 'upper'

    def _updateRotationCenter(self, *args, **kwargs):
        """Update rotation center relative to bounding box"""
        center = []
        for index, position in enumerate(self.getRotationCenter()):
            # Patch position relative to bounding box
            if position in self._ROTATION_CENTER_TAGS:
                bounds = self._getScenePrimitive().bounds(
                    transformed=False, dataBounds=True)
                bounds = self._transformObjectToRotate.transformBounds(bounds)

                if bounds is None:
                    position = 0.
                elif position == 'lower':
                    position = bounds[0, index]
                elif position == 'center':
                    position = 0.5 * (bounds[0, index] + bounds[1, index])
                elif position == 'upper':
                    position = bounds[1, index]

            center.append(position)

        if not numpy.all(numpy.equal(
                center, self._rotateForwardTranslation.translation)):
            self._rotateForwardTranslation.translation = center
            self._rotateBackwardTranslation.translation = \
                - self._rotateForwardTranslation.translation
            self._updated(Item3DChangedType.TRANSFORM)

    def setRotationCenter(self, x=0., y=0., z=0.):
         """Set the center of rotation of the item.

         Position of the rotation center is either a float
         for an absolute position or one of the following
         string to define a position relative to the item's bounding box:
         'lower', 'center', 'upper'

         :param x: rotation center position on the X axis
         :rtype: float or str
         :param y: rotation center position on the Y axis
         :rtype: float or str
         :param z: rotation center position on the Z axis
         :rtype: float or str
         """
         center = []
         for position in (x, y, z):
             if isinstance(position, six.string_types):
                 assert position in self._ROTATION_CENTER_TAGS
             else:
                 position = float(position)
             center.append(position)
         center = tuple(center)

         if center != self._rotationCenter:
             self._rotationCenter = center
             self._updateRotationCenter()

    def getRotationCenter(self):
        """Returns the rotation center set by :meth:`setRotationCenter`.

        :rtype: 3-tuple of float or str
        """
        return self._rotationCenter

    def setRotation(self, angle=0., axis=(0., 0., 1.)):
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
        :rtype: 2-tuple (float, numpy.ndarray)
        """
        return self._rotate.angle, self._rotate.axis

    def setMatrix(self, matrix=None):
        """Set the transform matrix

        :param numpy.ndarray matrix: 3x3 transform matrix
        """
        matrix4x4 = numpy.identity(4, dtype=numpy.float32)

        if matrix is not None:
            matrix = numpy.array(matrix, dtype=numpy.float32)
            assert matrix.shape == (3, 3)
            matrix4x4[:3, :3] = matrix

        if not numpy.all(numpy.equal(matrix4x4, self._matrix.getMatrix())):
            self._matrix.setMatrix(matrix4x4)
            self._updated(Item3DChangedType.TRANSFORM)

    def getMatrix(self):
        """Returns the matrix set by :meth:`setMatrix`

        :return: 3x3 matrix
        :rtype: numpy.ndarray"""
        return self._matrix.getMatrix(copy=True)[:3, :3]

    # Bounding box

    def _setForegroundColor(self, color):
        """Set the color of the bounding box

        :param color: RGBA color as 4 floats in [0, 1]
        """
        self._getScenePrimitive().color = color
        super(DataItem3D, self)._setForegroundColor(color)

    def isBoundingBoxVisible(self):
        """Returns item's bounding box visibility.

        :rtype: bool
        """
        return self._getScenePrimitive().boxVisible

    def setBoundingBoxVisible(self, visible):
        """Set item's bounding box visibility.

        :param bool visible:
            True to show the bounding box, False (default) to hide it
        """
        visible = bool(visible)
        primitive = self._getScenePrimitive()
        if visible != primitive.boxVisible:
            primitive.boxVisible = visible
            self._updated(Item3DChangedType.BOUNDING_BOX_VISIBLE)


class _BaseGroupItem(DataItem3D):
    """Base class for group of items sharing a common transform."""

    sigItemAdded = qt.Signal(object)
    """Signal emitted when a new item is added to the group.

    The newly added item is provided by this signal
    """

    sigItemRemoved = qt.Signal(object)
    """Signal emitted when an item is removed from the group.

    The removed item is provided by this signal.
    """

    def __init__(self, parent=None, group=None):
        """Base class representing a group of items in the scene.

        :param parent: The View widget this item belongs to.
        :param Union[GroupBBox, None] group:
            The scene group to use for rendering
        """
        DataItem3D.__init__(self, parent=parent, group=group)
        self._items = []

    def addItem(self, item, index=None):
        """Add an item to the group

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

    def clearItems(self):
        """Remove all item from the group."""
        for item in self.getItems():
            self.removeItem(item)

    def visit(self, included=True):
        """Generator visiting the group content.

        It traverses the group sub-tree in a top-down left-to-right way.

        :param bool included: True (default) to include self in visit
        """
        if included:
            yield self
        for child in self.getItems():
            yield child
            if hasattr(child, 'visit'):
                for item in child.visit(included=False):
                    yield item


class GroupItem(_BaseGroupItem):
    """Group of items sharing a common transform."""

    def __init__(self, parent=None):
        super(GroupItem, self).__init__(parent=parent)


class GroupWithAxesItem(_BaseGroupItem):
    """
    Group of items sharing a common transform surrounded with labelled axes.
    """

    def __init__(self, parent=None):
        """Class representing a group of items in the scene with labelled axes.

        :param parent: The View widget this item belongs to.
        """
        super(GroupWithAxesItem, self).__init__(parent=parent,
                                                group=axes.LabelledAxes())

    # Axes labels

    def setAxesLabels(self, xlabel=None, ylabel=None, zlabel=None):
        """Set the text labels of the axes.

        :param str xlabel: Label of the X axis, None to leave unchanged.
        :param str ylabel: Label of the Y axis, None to leave unchanged.
        :param str zlabel: Label of the Z axis, None to leave unchanged.
        """
        labelledAxes = self._getScenePrimitive()
        if xlabel is not None:
            labelledAxes.xlabel = xlabel

        if ylabel is not None:
            labelledAxes.ylabel = ylabel

        if zlabel is not None:
            labelledAxes.zlabel = zlabel

    class _Labels(tuple):
        """Return type of :meth:`getAxesLabels`"""

        def getXLabel(self):
            """Label of the X axis (str)"""
            return self[0]

        def getYLabel(self):
            """Label of the Y axis (str)"""
            return self[1]

        def getZLabel(self):
            """Label of the Z axis (str)"""
            return self[2]

    def getAxesLabels(self):
        """Returns the text labels of the axes

        >>> group = GroupWithAxesItem()
        >>> group.setAxesLabels(xlabel='X')

        You can get the labels either as a 3-tuple:

        >>> xlabel, ylabel, zlabel = group.getAxesLabels()

        Or as an object with methods getXLabel, getYLabel and getZLabel:

        >>> labels = group.getAxesLabels()
        >>> labels.getXLabel()
        ... 'X'

        :return: object describing the labels
        """
        labelledAxes = self._getScenePrimitive()
        return self._Labels((labelledAxes.xlabel,
                             labelledAxes.ylabel,
                             labelledAxes.zlabel))
