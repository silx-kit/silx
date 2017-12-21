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
"""
This package implements the SceneWidget model
"""

from __future__ import absolute_import

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "05/12/2017"


import collections
import functools
import weakref

import numpy

from silx.third_party import six

from ...utils.weakref import WeakMethodProxy
from .._utils import convertArrayToQImage
from ..plot.Colormap import preferredColormaps
from .. import qt
from . import items


class BaseRowNode(qt.QObject):
    """Base class for node of the tree model representing a row.

    The root node parent MUST be set to the QAbstractItemModel it belongs to.
    By default item is enabled.

    :param children: Iterable of BaseRowNode to start with (not signaled)
    """

    def __init__(self, children=()):
        super(BaseRowNode, self).__init__()
        self.__children = []
        for child in children:
            print('add child', child)
            assert isinstance(child, BaseRowNode)
            child.setParent(self)
            self.__children.append(child)
        self.__flags = collections.defaultdict(lambda: qt.Qt.ItemIsEnabled)

    def model(self):
        """Return the model this node belongs to.

        :rtype: QAbstractItemModel
        """
        parent = self.parent()
        if isinstance(parent, BaseRowNode):
            return self.parent().model()
        else:
            assert isinstance(parent, qt.QAbstractItemModel)
            return parent

    def index(self, column=0):
        """Return corresponding index in the model.

        :param int column: The column to make the index for
        :rtype: QModelIndex
        """
        parent = self.parent()
        model = self.model()

        if parent is model:  # Root node
            return qt.QModelIndex()
        else:
            parentIndex = parent.index()
            row = parent.children().index(self)
            return model.index(row, column, parentIndex)

    def columnCount(self):
        """Returns number of columns (default: 2)

        :rtype: int
        """
        return 2

    def children(self):
        """Returns the list of children nodes

        :rtype: tuple of Node
        """
        return tuple(self.__children)

    def rowCount(self):
        """Returns number of rows

        :rtype: int
        """
        return len(self.children())

    def addRowNode(self, node, row=None):
        """Add a node to the children

        :param Node node: The node to add
        :param int index: The index at which to insert it or
                          None to append
        """
        if row is None:
            row = self.rowCount()
        assert row <= self.rowCount()

        parent = self.index()
        model = self.model()
        model.beginInsertRows(parent, row, row)

        self.__children.insert(row, node)
        node.setParent(self)

        model.endInsertRows()

    def removeRowNode(self, rowOrNode):
        """Remove a node from the children list.

        It removes either a node or a row index.

        :param rowOrNode: Node or row number to remove
        """
        if isinstance(rowOrNode, Node):
            row = self.__children.index(rowOrNode)
        else:
            row = rowOrNode
        assert row < self.rowCount()

        model = self.model()
        parent = self.index()
        model.beginRemoveRows(parent, row, row)

        node = self.__children.pop(row)
        node.setParent(None)

        model.endRemoveRows()

    def data(self, column, role):
        """Returns data for given column and role

        :param int column: Column index for this row
        :param int role: The role to get
        :return: Corresponding data (Default: None)
        """
        return None

    def setData(self, column, value, role):
        """Set data for given column and role

        :param int column: Column index for this row
        :param value: The data to set
        :param int role: The role to set
        :return: True on success, False on failure
        :rtype: bool
        """
        return False

    def setFlags(self, flags, column=None):
        """Set the static flags to return.

        Default is ItemIsEnabled for all columns.

        :param int column: The column for which to set the flags
        :param flags: Item flags
        """
        if column is None:
            self.__flags = collections.defaultdict(lambda : flags)
        else:
            self.__flags[column] = flags

    def flags(self, column):
        """Returns flags for given column

        :rtype: int"""
        return self.__flags[column]


class StaticRowNode(BaseRowNode):
    """Row with static data.

    :param tuple display: List of data for DisplayRole for each column
    :param dict roles: Optional mapping of roles to list of data.
    :param children: Iterable of BaseRowNode to start with (not signaled)
    """

    def __init__(self, display=('', None), roles=None, children=()):
        super(StaticRowNode, self).__init__(children)
        self._dataByRoles = {} if roles is None else roles
        self._dataByRoles[qt.Qt.DisplayRole] = display

        self.setFlags(qt.Qt.ItemIsEnabled)

    def data(self, column, role):
        if role in self._dataByRoles:
            dataRow = self._dataByRoles[role]
            if column < len(dataRow):
                return dataRow[column]
        return super(StaticRowNode, self).data(column, role)

    def columnCount(self):
        return len(self._dataByRoles[qt.Qt.DisplayRole])


class ProxyRowNode(BaseRowNode):
    """Provides a node to proxy a data accessible through functions.

    Warning: Only weak reference are kept on fget and fset.

    :param str name: The name of this node
    :param callable fget: A callable returning the data
    :param callable fset:
        An optional callable setting the data with data as a single argument.
    :param notify:
        An optional signal emitted when data has changed.
    """

    def __init__(self, name='',
                 fget=None, fset=None, notify=None,
                 toModelData=None, fromModelData=None,
                 editorHint=None):

        super(ProxyRowNode, self).__init__()
        self.__name = name
        self.__editorHint = editorHint

        assert fget is not None
        self._fget = WeakMethodProxy(fget)
        self._fset = WeakMethodProxy(fset) if fset is not None else None
        if fset is not None:
            self.setFlags(qt.Qt.ItemIsEditable, 1)
        self._toModelData = toModelData
        self._fromModelData = fromModelData

        if notify is not None:
            notify.connect(self._notified)  # TODO support sigItemChanged flags

    def _notified(self, *args, **kwargs):
        index = self.index(column=1)
        model = self.model()
        model.dataChanged.emit(index, index)

    def data(self, column, role):
        if column == 0:
            if role == qt.Qt.DisplayRole:
                return self.__name

        elif column == 1:
            if role == qt.Qt.UserRole:  # EditorHint
                return self.__editorHint
            elif role in (qt.Qt.DisplayRole, qt.Qt.EditRole):
                data = self._fget()
                if self._toModelData is not None:
                    return self._toModelData(data)

        return super(ProxyRowNode, self).data(column, role)

    def setData(self, column, value, role):
        if role == qt.Qt.EditRole and self._fset is not None:
            if self._fromModelData is not None:
                value = self._fromModelData(value)
            self._fset(value)
            return True

        return super(ProxyRowNode, self).setData(column, value, role)


class ColorProxyRowNode(ProxyRowNode):

    def data(self, column, role):
        if column == 1:  # Show color as decoration, not text
            if role == qt.Qt.DisplayRole:
                return None
            if role == qt.Qt.DecorationRole:
                role = qt.Qt.DisplayRole
        return super(ProxyRowNode, self).data(column, role)


class Node(qt.QObject):
    """Base class for tree model nodes.

    The base node provides a static data and has no children.

    :param parent: The parent Node in the tree model
    :param str name: The name of the node (available in column 0).
    :param data: The value of this node (available in column 1).
    :param editorHint: For EditorHint role
    """

    def __init__(self, parent, name='', data=None, editorHint=None):
        super(Node, self).__init__(parent)
        self.__name = name
        self.__data = data
        self._editorHint = editorHint  # TODO support callable
        self.__children = ()

    def __repr__(self):
        return ('<' + self.__class__.__name__ +
                '("' + self.name() + '") at ' +
                hex(id(self)) + '>')

    def model(self):
        """Returns the model this node belongs to.

        :rtype: QAbstractItemModel
        """
        parent = self.parent()
        if isinstance(parent, Node):
            return self.parent().model()
        else:
            assert isinstance(parent, qt.QAbstractItemModel)
            return parent

    def index(self, column=0):
        """Return corresponding index in the model.

        :param int column: The column to make the index for
        :rtype: QModelIndex
        """
        parent = self.parent()
        model = self.model()

        if parent is model:  # Root node
            return qt.QModelIndex()
        else:
            parentIndex = parent.index()
            row = parent.getChildren().index(self)
            return model.index(row, column, parentIndex)

    def setChildren(self, children):
        """Set the children of this node.

        :param iterable children: List of children node (default no children)
        """
        self.__children = tuple(children)

    def getChildren(self):
        """Returns the children of this node.

        :rtype: tuple of Node or Item3D
        """
        return self.__children

    def name(self):
        """Returns the name associated with this node.

        :rtype: str
        """
        return self.__name

    def supportedRoles(self):
        """Returns supported model roles.

        :rtype: tuple
        """
        return qt.Qt.DisplayRole, qt.Qt.EditRole, qt.Qt.UserRole

    def data(self, role):
        """Returns the data associated with this node.

        :param role: The model role
        """
        if role == qt.Qt.UserRole:
            return self._editorHint
        else:
            return self._data(role)

    def _data(self, role):
        """Override in subclass to return edit/display role data"""
        return self.__data

    def setData(self, data, role):
        """Set the data of this node.

        This is not implemented in the base class.

        :param data: The data to set
        :param role: The role to set
        :return: True if successfully set, False otherwise
        :rtype: bool
        """
        return False

    def isEditable(self):
        """Returns True if the data of this node is editable.

        The base class always returns False.
        If the returned value is False, :meth:`setData` should no be called.

        :rtype: bool
        """
        return False

    def isEnabled(self):
        """Returns True is this node is enabled

        The base class always returns True.

        :rtype: bool
        """
        return True

    def isCheckable(self):
        """Returns True if the name of this node is checkable.

        The base class always returns False.

        :rtype: bool
        """
        return False


# TODO remove async?
class ProxyNode(Node):
    """Provides a node to proxy a data accessible through functions.

    Warning: Only weak reference are kept on fget and fset.

    :param Node parent: The parent node in the tree hierarchy
    :param str name: The name of this node
    :param callable fget: A callable returning the data
    :param callable fset:
        An optional callable setting the data with data as a single argument.
    :param notify:
        An optional signal emitted when data has changed.
    :param bool async: True to set the data asynchronously (Default: False).
    """

    _asyncSignal = qt.Signal(object)
    """Signal used internally to set data asynchronously.

    There is an issue with mutable data...
    """

    def __init__(self, parent, name='',
                 fget=None, fset=None, notify=None,
                 async=False, toModelData=None, fromModelData=None, editorHint=None):

        super(ProxyNode, self).__init__(parent, name=name, editorHint=editorHint)
        assert fget is not None
        self._fget = WeakMethodProxy(fget)
        self._fset = WeakMethodProxy(fset) if fset is not None else None
        self._toModelData = toModelData
        self._fromModelData = fromModelData

        if notify is not None:
            notify.connect(self._notified)  # TODO support sigItemChanged flags

        self._async = bool(async)
        if self._async:
            self._asyncSignal.connect(self._setData, qt.Qt.QueuedConnection)

    def _notified(self, *args, **kwargs):
        index = self.index(column=1)
        model = self.model()
        model.dataChanged.emit(index, index)

    def _data(self, role):
        data = self._fget()
        if self._toModelData is not None:
            data = self._toModelData(data)
        return data

    def _setData(self, data):
        if self._fromModelData is not None:
            data = self._fromModelData(data)
        self._fset(data)

    def setData(self, data, role):
        if not self.isEditable() or role != qt.Qt.EditRole:
            return False

        if not self._async:
            self._setData(data)
        else:
            self._asyncSignal.emit(data)
        return True

    def isEditable(self):
        return self._fset is not None


class _DirectionalLightProxy(qt.QObject):
    """Proxy to handle directional light with angles rather than vector.
    """

    sigAzimuthAngleChanged = qt.Signal()
    """Signal sent when the azimuth angle has changed."""

    sigAltitudeAngleChanged = qt.Signal()
    """Signal sent when altitude angle has changed."""

    def __init__(self, light):
        super(_DirectionalLightProxy, self).__init__()
        self._light = light
        light.addListener(self._directionUpdated)
        self._azimuth = 0.
        self._altitude = 0.

    def getAzimuthAngle(self):
        """Returns the signed angle in the horizontal plane.

         Unit: degrees.
        The 0 angle corresponds to the axis perpendicular to the screen.

        :rtype: float
        """
        return self._azimuth

    def getAltitudeAngle(self):
        """Returns the signed vertical angle from the horizontal plane.

        Unit: degrees.
        Range: [-90, +90]

        :rtype: float
        """
        return self._altitude

    def setAzimuthAngle(self, angle):
        """Set the horizontal angle.

        :param float angle: Angle from -z axis in zx plane in degrees.
        """
        if angle != self._azimuth:
            self._azimuth = angle
            self._updateLight()
            self.sigAzimuthAngleChanged.emit()

    def setAltitudeAngle(self, angle):
        """Set the horizontal angle.

        :param float angle: Angle from -z axis in zy plane in degrees.
        """
        if angle != self._altitude:
            self._altitude = angle
            self._updateLight()
            self.sigAltitudeAngleChanged.emit()

    def _directionUpdated(self, *args, **kwargs):
        """Handle light direction update in the scene"""
        # Invert direction to manipulate the 'source' pointing to
        # the center of the viewport
        x, y, z = - self._light.direction

        # Horizontal plane is plane xz
        azimuth = numpy.degrees(numpy.arctan2(x, z))
        altitude = numpy.degrees(numpy.pi/2. - numpy.arccos(y))

        if (abs(azimuth - self.getAzimuthAngle()) > 0.01 and
                abs(abs(altitude) - 90.) >= 0.001):  # Do not update when at zenith
            self.setAzimuthAngle(azimuth)

        if abs(altitude - self.getAltitudeAngle()) > 0.01:
            self.setAltitudeAngle(altitude)

    def _updateLight(self):
        """Update light direction in the scene"""
        azimuth = numpy.radians(self._azimuth)
        delta = numpy.pi/2. - numpy.radians(self._altitude)
        z = - numpy.sin(delta) * numpy.cos(azimuth)
        x = - numpy.sin(delta) * numpy.sin(azimuth)
        y = - numpy.cos(delta)
        self._light.direction = x, y, z


class DirectionalLightNode(Node):
    """Node for :class:`SceneWidget` light direction setting."""

    def __init__(self, parent, light):
        super(DirectionalLightNode, self).__init__(
            parent, name='Light Direction')

        self._lightProxy = _DirectionalLightProxy(light)

        azimuthNode = ProxyNode(
            self,
            name='Azimuth',
            fget=self._lightProxy.getAzimuthAngle,
            fset=self._lightProxy.setAzimuthAngle,
            notify=self._lightProxy.sigAzimuthAngleChanged,
            editorHint=(-90, 90))

        altitudeNode = ProxyNode(
            self,
            name='Altitude',
            fget=self._lightProxy.getAltitudeAngle,
            fset=self._lightProxy.setAltitudeAngle,
            notify=self._lightProxy.sigAltitudeAngleChanged,
            editorHint=(-90, 90))

        self.setChildren((azimuthNode, altitudeNode))


class ColorProxyNode(ProxyNode):
    def supportedRoles(self):
        return super(ColorProxyNode, self).supportedRoles() + (qt.Qt.DecorationRole,)


class Style(Node):
    """Node for :class:`SceneWidget` style settings."""
    def __init__(self, parent, sceneWidget):
        super(Style, self).__init__(
            parent, name='Style')

        bgColor = ColorProxyNode(
            self,
            name='Background',
            fget=sceneWidget.getBackgroundColor,
            fset=sceneWidget.setBackgroundColor)

        fgColor = ColorProxyNode(
            self,
            name='Foreground',
            fget=sceneWidget.getForegroundColor,
            fset=sceneWidget.setForegroundColor)

        highlightColor = ColorProxyNode(
            self,
            name='Highlight',
            fget=sceneWidget.getHighlightColor,
            fset=sceneWidget.setHighlightColor)

        boundingBox = ProxyNode(
            self,
            name='Bounding Box',
            fget=sceneWidget.isBoundingBoxVisible,
            fset=sceneWidget.setBoundingBoxVisible)

        axesIndicator = ProxyNode(
            self,
            name='Axes Indicator',
            fget=sceneWidget.isOrientationIndicatorVisible,
            fset=sceneWidget.setOrientationIndicatorVisible)

        lightDirection = DirectionalLightNode(
            self,
            sceneWidget.viewport.light)

        self.setChildren((bgColor,
                          fgColor,
                          highlightColor,
                          boundingBox,
                          axesIndicator,
                          lightDirection))


class Item3DNode(Node):

    def __init__(self, parent, item, name=None):
        if name is None:
            name = item.__class__.__name__
        super(Item3DNode, self).__init__(parent, name=name, data=None)

        self._item = weakref.ref(item)
        item.sigItemChanged.connect(self._itemChanged)

    def _itemChanged(self, event):
        if event == items.ItemChangedType.VISIBLE:
            index = self.index(column=1)
            model = self.model()
            model.dataChanged.emit(index, index)

    def item(self):
        return self._item()

    def _data(self, role):
        if role == qt.Qt.CheckStateRole:
            item = self.item()
            if item is not None and item.isVisible():
                return qt.Qt.Checked
            else:
                return qt.Qt.Unchecked
        else:
            return super(Item3DNode, self)._data(role)

    def setData(self, data, role):
        if role == qt.Qt.CheckStateRole:
            item = self.item()
            if item is not None:
                item.setVisible(data == qt.Qt.Checked)
                return True
            else:
                return False
        return super(Item3DNode, self).setData(data, role)

    def isEditable(self):
        return True

    def isCheckable(self):
        return True


class AngleNode(ProxyNode):
    def __init__(self, *args, **kwargs):
        super(AngleNode, self).__init__(*args, **kwargs)

    def _data(self, role):
        data = super(AngleNode, self)._data(role)
        return (u'%g°' % data) if role == qt.Qt.DisplayRole else data


class DataItem3DNode(Item3DNode):

    _ROTATION_CENTER_OPTIONS = 'Origin', 'Lower', 'Center', 'Upper'

    def __init__(self, parent, item):
        super(DataItem3DNode, self).__init__(
            parent, item, item.getLabel())

        self._transform = Node(self, name='Transforms')

        self._rotate = Node(self._transform, name='Rotation')
        self._rotateCenter = Node(self._rotate, name='Center')
        # Here to keep a reference
        self._xCenterToModelData = functools.partial(
            self._centerToModelData, index=0)
        self._xSetCenter = functools.partial(self._setCenter, index=0)
        # Here to keep a reference
        self._yCenterToModelData = functools.partial(
            self._centerToModelData, index=1)
        self._ySetCenter = functools.partial(self._setCenter, index=1)
        # Here to keep a reference
        self._zCenterToModelData = functools.partial(
            self._centerToModelData, index=2)
        self._zSetCenter = functools.partial(self._setCenter, index=2)

        self._rotateCenter.setChildren((
            ProxyNode(self._rotateCenter,
                      name='X axis',
                      fget=item.getRotationCenter,
                      fset=self._xSetCenter,
                      notify=item.sigItemChanged,
                      toModelData=self._xCenterToModelData,
                      editorHint=self._ROTATION_CENTER_OPTIONS),
            ProxyNode(self._rotateCenter,
                      name='Y axis',
                      fget=item.getRotationCenter,
                      fset=self._ySetCenter,
                      notify=item.sigItemChanged,
                      toModelData=self._yCenterToModelData,
                      editorHint=self._ROTATION_CENTER_OPTIONS),
            ProxyNode(self._rotateCenter,
                      name='Z axis',
                      fget=item.getRotationCenter,
                      fset=self._zSetCenter,
                      notify=item.sigItemChanged,
                      toModelData=self._zCenterToModelData,
                      editorHint=self._ROTATION_CENTER_OPTIONS),
        ))
        self._rotate.setChildren((
            AngleNode(self._rotate,
                      name='Angle',
                      fget=item.getRotation,
                      fset=self._setAngle,
                      notify=item.sigItemChanged,  # TODO
                      toModelData=lambda data: data[0]),
            ProxyNode(self._rotate,
                      name='Axis',
                      fget=item.getRotation,
                      fset=self._setAxis,
                      notify=item.sigItemChanged,  # TODO
                      toModelData=lambda data: qt.QVector3D(*data[1])),
            self._rotateCenter
        ))
        self._transform.setChildren((
            ProxyNode(self._transform,
                      name='Translation',
                      fget=item.getTranslation,
                      fset=self._setTranslation,
                      notify=item.sigItemChanged,  # TODO
                      toModelData=lambda data: qt.QVector3D(*data)),
            self._rotate,
            ProxyNode(self._transform,
                      name='Scale',
                      fget=item.getScale,
                      fset=self._setScale,
                      notify=item.sigItemChanged,  # TODO
                      toModelData=lambda data: qt.QVector3D(*data)),
        ))
        self.setChildren(())

    @staticmethod
    def _centerToModelData(center, index):
        value = center[index]
        if isinstance(value, six.string_types):
            return value.title()
        elif value == 0.:
            return 'Origin'
        else:
            return six.text_type(value)

    def _setCenter(self, value, index):
        item = self.item()
        if item is not None:
            if value == 'Origin':
                value = 0.
            elif value not in self._ROTATION_CENTER_OPTIONS:
                value = float(value)
            else:
                value = value.lower()

            center = list(item.getRotationCenter())
            center[index] = value
            item.setRotationCenter(*center)

    def _setAngle(self, angle):
        item = self.item()
        if item is not None:
            _, axis = item.getRotation()
            item.setRotation(angle, axis)

    def _setAxis(self, axis):
        item = self.item()
        if item is not None:
            angle, _ = item.getRotation()
            item.setRotation(angle, (axis.x(), axis.y(), axis.z()))

    def _setTranslation(self, translation):
        item = self.item()
        if item is not None:
            item.setTranslation(translation.x(), translation.y(), translation.z())

    def _setScale(self, scale):
        item = self.item()
        if item is not None:
            item.setScale(scale.x(), scale.y(), scale.z())

    def setChildren(self, children):
        super(DataItem3DNode, self).setChildren(
            (self._transform,) + tuple(children))


class InterpolationNode(ProxyNode):
    def __init__(self, parent, item):
         super(InterpolationNode, self).__init__(
             parent,
             name='Interpolation',
             fget=item.getInterpolation,
             fset=item.setInterpolation,
             notify=item.sigItemChanged,  # TODO
             editorHint=item.INTERPOLATION_MODES)

class ImageRgbaNode(DataItem3DNode):
     def __init__(self, parent, item):
         super(ImageRgbaNode, self).__init__(parent, item)
         self._interpolation = InterpolationNode(self, item)
         self.setChildren((self._interpolation,))


class _RangeProxyNode(ProxyNode):
    def __init__(self, *args, **kwargs):
        super(_RangeProxyNode, self).__init__(*args, **kwargs)

    def _notified(self, *args, **kwargs):
        topLeft = self.index(column=0)
        bottomRight = self.index(column=1)
        model = self.model()
        model.dataChanged.emit(topLeft, bottomRight)

    def isEnabled(self):
        parent = self.parent()
        item = parent.item()
        if item is not None:
            colormap = item.getColormap()
            return not colormap.isAutoscale()
        return False


class ColormapNode(Node):

    _sigColormapChanged = qt.Signal()

    def __init__(self, parent, item):
        super(ColormapNode, self).__init__(parent, name='Colormap')
        self._item = weakref.ref(item)
        item.sigItemChanged.connect(self._itemChanged)

        self._colormap = item.getColormap()
        self._colormap.sigChanged.connect(self._sigColormapChanged)

        self._colormapImage = None
        self._dataRange = None

        self._name = ProxyNode(
            self,
            name='Name',
            fget=self._getName,
            fset=self._setName,
            notify=self._sigColormapChanged,
            editorHint=preferredColormaps())
        self._normalization = ProxyNode(
            self,
            name='Normalization',
            fget=self._getNormalization,
            fset=self._setNormalization,
            notify=self._sigColormapChanged,
            editorHint=self._colormap.NORMALIZATIONS)
        self._autoscale = ProxyNode(
            self,
            name='Autoscale',
            fget=self._isAutoscale,
            fset=self._setAutoscale,
            notify=self._sigColormapChanged)
        self._vmin = _RangeProxyNode(
            self,
            name='Min.',
            fget=self._getVMin,
            fset=self._setVMin,
            notify=self._sigColormapChanged)
        self._vmax = _RangeProxyNode(
            self,
            name='Max.',
            fget=self._getVMax,
            fset=self._setVMax,
            notify=self._sigColormapChanged)

        self.setChildren((self._name, self._normalization,
                          self._autoscale, self._vmin, self._vmax))

        self._sigColormapChanged.connect(self._updateColormapImage)

    _getName = lambda self: self._colormap.getName()
    _setName = lambda self, name: self._colormap.setName(name)

    _getNormalization = lambda self: self._colormap.getNormalization()
    _setNormalization = lambda self, normalization: self._colormap.setNormalization(normalization)

    _isAutoscale = lambda self: self._colormap.isAutoscale()

    def _updateColormapImage(self, *args, **kwargs):
        if self._colormapImage is not None:
            self._colormapImage = None
            index = self.index(column=1)
            self.model().dataChanged.emit(index, index)

    def _data(self, role):
        if self._colormapImage is None:
            image = numpy.zeros((16, 130, 3), dtype=numpy.uint8)
            image[1:-1, 1:-1] = self._colormap.getNColors(image.shape[1] - 2)[:, :3]
            self._colormapImage = convertArrayToQImage(image)
        return self._colormapImage

    def supportedRoles(self):
        return super(ColormapNode, self).supportedRoles() + (qt.Qt.DecorationRole,)

    def _getDataRange(self):
        if self._dataRange is None:
            item = self.item()
            if item is not None:
                if hasattr(item, 'getDataRange'):
                    data = item.getDataRange()
                else:
                    data = item.getData(copy=False)
                self._dataRange = item.getColormap().getColormapRange(data)
            else:  # Fallback
                self._dataRange = 1, 100
        return self._dataRange

    def _getVMin(self):
        min_ = self._colormap.getVMin()
        if min_ is None:
            min_ = self._getDataRange()[0]
        return min_

    def _setVMin(self, min_):
        max_ = self._colormap.getVMax()
        if max_ is not None and min_ > max_:
            min_, max_ = max_, min_
        self._colormap.setVRange(min_, max_)

    def _getVMax(self):
        max_ = self._colormap.getVMax()
        if max_ is None:
            max_ = self._getDataRange()[1]
        return max_

    def _setVMax(self, max_):
        min_ = self._colormap.getVMin()
        if min_ is not None and min_ > max_:
            min_, max_ = max_, min_
        self._colormap.setVRange(min_, max_)

    def _setAutoscale(self, autoscale):
        item = self.item()
        if item is not None:
            colormap = item.getColormap()
            if autoscale:
                vmin, vmax = None, None
            else:
                vmin, vmax = self._getDataRange()
            colormap.setVRange(vmin, vmax)

    def item(self):
        return self._item()

    def _itemChanged(self, event):
        if event == items.ItemChangedType.COLORMAP:
            self._sigColormapChanged.emit()
            self._colormap.sigChanged.disconnect(self._sigColormapChanged)
            item = self.item()
            if item is not None:
                colormap = item.getColormap()
                colormap.sigChanged.connect(self._sigColormapChanged)

        elif event == items.ItemChangedType.DATA:
            self._dataRange = None
            self._sigColormapChanged.emit()


class ImageDataNode(DataItem3DNode):
     def __init__(self, parent, item):
         super(ImageDataNode, self).__init__(parent, item)
         self._colormap = ColormapNode(self, item)
         self._interpolation = InterpolationNode(self, item)
         self.setChildren((self._colormap, self._interpolation))


class SymbolNode(ProxyNode):
    def __init__(self, parent, item):
        names = [item.getSymbolName(s) for s in item.getSupportedSymbols()]
        super(SymbolNode, self).__init__(
             parent,
             name='Marker',
             fget=item.getSymbolName,
             fset=item.setSymbol,
             notify=item.sigItemChanged,  # TODO
             editorHint=names)


class SymbolSizeNode(ProxyNode):
    def __init__(self, parent, item):
         super(SymbolSizeNode, self).__init__(
             parent,
             name='Marker size',
             fget=item.getSymbolSize,
             fset=item.setSymbolSize,
             notify=item.sigItemChanged,  # TODO
             editorHint=(1, 50))  # TODO link with OpenGL max point size


class Scatter3DNode(DataItem3DNode):
     def __init__(self, parent, item):
         super(Scatter3DNode, self).__init__(parent, item)
         self._symbol = SymbolNode(self, item)
         self._symbolSize = SymbolSizeNode(self, item)
         self._colormap = ColormapNode(self, item)
         self.setChildren((self._symbol, self._symbolSize, self._colormap))


class Scatter2DNode(DataItem3DNode):
     def __init__(self, parent, item):
         super(Scatter2DNode, self).__init__(parent, item)
         self._symbol = SymbolNode(self, item)
         self._symbolSize = SymbolSizeNode(self, item)
         self._colormap = ColormapNode(self, item)

         self._visualization = ProxyNode(
             self,
             name='Mode',
             fget=item.getVisualization,
             fset=item.setVisualization,
             notify=item.sigItemChanged,  # TODO
             editorHint=[m.title() for m in item.supportedVisualizations()],
             toModelData=lambda data: data.title(),
             fromModelData=lambda data: data.lower())

         self._heightMap = ProxyNode(
             self,
             name='Height map',
             fget=item.isHeightMap,
             fset=item.setHeightMap,
             notify=item.sigItemChanged)  # TODO

         self._lineWidth = ProxyNode(
             self,
             name='Line width',
             fget=item.getLineWidth,
             fset=item.setLineWidth,
             notify=item.sigItemChanged,  # TODO
             editorHint=(1, 10))  # TODO link with OpenGL max line width

         # TODO enable/disable symbol, symbol size, linewidth

         self.setChildren((self._visualization, self._heightMap,
                           self._colormap,
                           self._symbol, self._symbolSize,
                           self._lineWidth))


class PlaneNode(ProxyNode):
    def __init__(self, parent, item):
        super(PlaneNode, self).__init__(
            parent,
            name='Equation',
            fget=item.getParameters,
            fset=item.setParameters,
            notify=item.sigItemChanged,  # TODO
            toModelData=lambda data: qt.QVector4D(*data),
            fromModelData=lambda data: (data.x(), data.y(), data.z(), data.w()))
        self._item = weakref.ref(item)

    def _data(self, role):
        if role == qt.Qt.DisplayRole:
            item = self._item()
            if item is not None:
                params = item.getParameters()
                return ('%gx %+gy %+gz %+g = 0' %
                        (params[0], params[1], params[2], params[3]))
        return super(PlaneNode, self)._data(role)


class ClipPlaneNode(Item3DNode):

     def __init__(self, parent, item):
         super(ClipPlaneNode, self).__init__(parent, item)
         self._plane = PlaneNode(self, item)
         self.setChildren((self._plane,))


class CutPlaneNode(Item3DNode):

     def __init__(self, parent, item):
         super(CutPlaneNode, self).__init__(parent, item)
         self._plane = PlaneNode(self, item)
         self._colormap = ColormapNode(self, item)
         self._interpolation = InterpolationNode(self, item)
         self.setChildren((self._plane, self._colormap, self._interpolation))


class IsoSurfaceNode(Item3DNode):

    def __init__(self, parent, item):
        super(IsoSurfaceNode, self).__init__(parent, item)
        self._level = ProxyNode(
            self,
            name='Level',
            fget=item.getLevel,
            fset=item.setLevel,
            notify=item.sigItemChanged)  # TODO
        self._color = ColorProxyNode(
            self,
            name='Color',
            fget=item.getColor,
            fset=item.setColor,
            notify=item.sigItemChanged)  # TODO
        self.setChildren((self._level, self._color))

        # TODO level slider, opacity slider


class AddRemoveIso(Node):

    def __init__(self, parent, item):
        super(AddRemoveIso, self).__init__(
            parent, name='', editorHint='add_remove_iso')
        self.item = weakref.ref(item)

    def isEditable(self):
        return True


class IsoSurfacesNode(Node):

    def __init__(self, parent, item):
        super(IsoSurfacesNode, self).__init__(parent, name='Isosurfaces')
        self._item = weakref.ref(item)

        isosurfaces = []
        for iso in item.getIsosurfaces():
            isosurfaces.append(IsoSurfaceNode(self, iso))
        isosurfaces.append(AddRemoveIso(self, item))
        self.setChildren(isosurfaces)

        item.sigIsosurfaceAdded.connect(self._isosurfaceAdded)
        item.sigIsosurfaceRemoved.connect(self._isosurfaceRemoved)

    def item(self):
        return self._item()

    # TODO merge with GroupItemNode implementation
    def _isosurfaceAdded(self, iso):
        item = self.item()
        if item is None:
            return

        parent = self.index()
        row = item.getIsosurfaces().index(iso)
        model = self.model()

        model.beginInsertRows(parent, row, row)

        # Update content node children
        children = list(self.getChildren())

        children.insert(row, IsoSurfaceNode(self, iso))
        self.setChildren(children)

        model.endInsertRows()

    def _isosurfaceRemoved(self, iso):
        item = self.item()
        if item is None:
            return

        parent = self.index()
        # Find item
        for row, node in enumerate(self.getChildren()):
            if node.item() is iso:
                break  # Got it
        else:
            raise RuntimeError("Model does not correspond to scene content")
        model = self.model()
        model.beginRemoveRows(parent, row, row)

         # Update content node children
        children = list(self.getChildren())
        children.pop(row)
        self.setChildren(children)

        model.endRemoveRows()


class ScalarField3DNode(DataItem3DNode):

      def __init__(self, parent, item):
          super(ScalarField3DNode, self).__init__(parent, item)
          self._cutPlane = CutPlaneNode(self, item.getCutPlanes()[0])
          self._isosurfaces = IsoSurfacesNode(self, item)
          self.setChildren((self._cutPlane, self._isosurfaces))


# TODO merge with items to have a simpler implementation?
def nodeFromItem(parent, item):
    """Create :class:`Node` corresponding to item

    :param Node parent: The parent of the new node
    :param Item3D item: The item fow which to create the node
    :rtype: Node
    """
    if isinstance(item, items.GroupItem):
        return GroupItemNode(parent, item)
    elif isinstance(item, items.ImageRgba):
        return ImageRgbaNode(parent, item)
    elif isinstance(item, items.ImageData):
        return ImageDataNode(parent, item)
    elif isinstance(item, items.Scatter3D):
        return Scatter3DNode(parent, item)
    elif isinstance(item, items.Scatter2D):
        return Scatter2DNode(parent, item)
    elif isinstance(item, items.ClipPlane):
        return ClipPlaneNode(parent, item)
    elif isinstance(item, items.ScalarField3D):
        return ScalarField3DNode(parent, item)
    elif isinstance(item, items.DataItem3D):
        return DataItem3DNode(parent, item)
    elif isinstance(item, items.Item3D):
        return Item3DNode(parent, item)
    else:
        raise RuntimeError("Cannot create node from item")


# TODO improve management of children (move it in content node?)
class GroupItemNode(DataItem3DNode):

    def __init__(self, parent, group):
        super(GroupItemNode, self).__init__(parent, group)
        group.sigItemAdded.connect(self._itemAdded)
        group.sigItemRemoved.connect(self._itemRemoved)

        self._content = Node(self, name='Content')
        self._createContentChildren()
        self.setChildren((self._content,))

    def _createContentChildren(self):
        children = []

        group = self.item()
        if group is not None:
            for item in group.getItems():
                children.append(nodeFromItem(self._content, item))

        self._content.setChildren(children)

    def _itemAdded(self, item):
        group = self.item()
        if group is None:
            return

        parent = self._content.index()
        row = group.getItems().index(item)
        model = self.model()

        model.beginInsertRows(parent, row, row)

        # Update content node children
        children = list(self._content.getChildren())

        children.insert(row, nodeFromItem(self._content, item))
        self._content.setChildren(children)

        model.endInsertRows()

    def _itemRemoved(self, item):
        group = self.item()
        if group is None:
            return

        parent = self._content.index()
        # Find item
        for row, node in enumerate(self._content.getChildren()):
            if node.item() is item:
                break  # Got it
        else:
            raise RuntimeError("Model does not correspond to scene content")
        model = self.model()
        model.beginRemoveRows(parent, row, row)

         # Update content node children
        children = list(self._content.getChildren())
        children.pop(row)
        self._content.setChildren(children)

        model.endRemoveRows()


class Style(Node):
    """Node for :class:`SceneWidget` style settings."""
    def __init__(self, parent, sceneWidget):
        super(Style, self).__init__(
            parent, name='Style')

        bgColor = ColorProxyNode(
            self,
            name='Background',
            fget=sceneWidget.getBackgroundColor,
            fset=sceneWidget.setBackgroundColor)

        fgColor = ColorProxyNode(
            self,
            name='Foreground',
            fget=sceneWidget.getForegroundColor,
            fset=sceneWidget.setForegroundColor)

        highlightColor = ColorProxyNode(
            self,
            name='Highlight',
            fget=sceneWidget.getHighlightColor,
            fset=sceneWidget.setHighlightColor)

        boundingBox = ProxyNode(
            self,
            name='Bounding Box',
            fget=sceneWidget.isBoundingBoxVisible,
            fset=sceneWidget.setBoundingBoxVisible)

        axesIndicator = ProxyNode(
            self,
            name='Axes Indicator',
            fget=sceneWidget.isOrientationIndicatorVisible,
            fset=sceneWidget.setOrientationIndicatorVisible)

        lightDirection = DirectionalLightNode(
            self,
            sceneWidget.viewport.light)

        self.setChildren((bgColor,
                          fgColor,
                          highlightColor,
                          boundingBox,
                          axesIndicator,
                          lightDirection))


class Root(BaseRowNode):
    """Root node of :class:`SceneWidget` parameters.

    It has two children:
    - Style
    - Scene group
    """

    def __init__(self, model, sceneWidget):
        style = StaticRowNode(('Style', None))

        super(Root, self).__init__(children=[style])
        self.setParent(model)  # Needed for root node
        self._sceneWidget = weakref.ref(sceneWidget)

        bgColor = ColorProxyRowNode(
            name='Background',
            fget=sceneWidget.getBackgroundColor,
            fset=sceneWidget.setBackgroundColor)

        style.addRowNode(bgColor)

        fgColor = ColorProxyRowNode(
            name='Foreground',
            fget=sceneWidget.getForegroundColor,
            fset=sceneWidget.setForegroundColor)
        style.addRowNode(fgColor)

    def children(self):
        sceneWidget = self._sceneWidget()
        if sceneWidget is None:
            return ()
        else:
            return super(Root, self).children()


class SceneModel(qt.QAbstractItemModel):
    """Model of a :class:`SceneWidget`.

    :param SceneWidget parent: The SceneWidget this model represents.
    """

    def __init__(self, parent):
        self._root = None
        super(SceneModel, self).__init__(parent)
        self._sceneWidget = weakref.ref(parent)
        self._root = Root(self, parent)

    def sceneWidget(self):
        """Returns the :class:`SceneWidget` this model represents.

        In case the widget has already been deleted, it returns None

        :rtype: SceneWidget
        """
        return self._sceneWidget()

    def _itemFromIndex(self, index):
        """Returns the corresponding :class:`Node` or :class:`Item3D`.

        :param QModelIndex index:
        :rtype: Node or Item3D
        """
        return index.internalPointer() if index.isValid() else self._root

    def index(self, row, column, parent=qt.QModelIndex()):
        if column >= self.columnCount(parent) or row >= self.rowCount(parent):
            return qt.QModelIndex()

        item = self._itemFromIndex(parent)
        return self.createIndex(row, column, item.children()[row])

    def parent(self, index):
        if not index.isValid():
            return qt.QModelIndex()

        item = self._itemFromIndex(index)
        parent = item.parent()

        ancestor = parent.parent()

        if ancestor is not self:  # root node
            children = ancestor.children()
            row = children.index(parent)
            return self.createIndex(row, 0, parent)

        return qt.QModelIndex()

    def rowCount(self, parent=qt.QModelIndex()):
        item = self._itemFromIndex(parent)
        return item.rowCount()

    def columnCount(self, parent=qt.QModelIndex()):
        item = self._itemFromIndex(parent)
        return item.columnCount()

    def data(self, index, role=qt.Qt.DisplayRole):
        item = self._itemFromIndex(index)
        column = index.column()
        return item.data(column, role)

    def setData(self, index, value, role=qt.Qt.EditRole):
        item = self._itemFromIndex(index)
        column = index.column()
        if item.setData(column, value, role):
            self.dataChanged.emit(index, index)
            return True
        return False

    def flags(self, index):
        item = self._itemFromIndex(index)
        column = index.column()
        return item.flags(column)

    def headerData(self, section, orientation, role=qt.Qt.DisplayRole):
        if orientation == qt.Qt.Horizontal and role == qt.Qt.DisplayRole:
            return 'Item' if section == 0 else 'Value'
        else:
            return None
