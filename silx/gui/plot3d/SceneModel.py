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

from __future__ import absolute_import, division

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
from .. import qt, icons
from . import items
from .items.volume import Isosurface


class BaseRow(qt.QObject):
    """Base class for rows of the tree model.

    The root node parent MUST be set to the QAbstractItemModel it belongs to.
    By default item is enabled.

    :param children: Iterable of BaseRow to start with (not signaled)
    """

    def __init__(self, children=()):
        super(BaseRow, self).__init__()
        self.__children = []
        for row in children:
            assert isinstance(row, BaseRow)
            row.setParent(self)
            self.__children.append(row)
        self.__flags = collections.defaultdict(lambda: qt.Qt.ItemIsEnabled)

    def model(self):
        """Return the model this node belongs to or None if not in a model.

        :rtype: Union[QAbstractItemModel, None]
        """
        parent = self.parent()
        if isinstance(parent, BaseRow):
            return self.parent().model()
        elif parent is None:
            return None
        else:
            assert isinstance(parent, qt.QAbstractItemModel)
            return parent

    def index(self, column=0):
        """Return corresponding index in the model or None if not in a model.

        :param int column: The column to make the index for
        :rtype: Union[QModelIndex, None]
        """
        parent = self.parent()
        model = self.model()

        if model is None:  # Not in a model
            return None
        elif parent is model:  # Root node
            return qt.QModelIndex()
        else:
            index = parent.index()
            row = parent.children().index(self)
            return model.index(row, column, index)

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
        return len(self.__children)

    def addRow(self, row, index=None):
        """Add a node to the children

        :param BaseRow row: The node to add
        :param int index: The index at which to insert it or
                          None to append
        """
        if index is None:
            index = self.rowCount()
        assert index <= self.rowCount()

        model = self.model()

        if model is not None:
            parent = self.index()
            model.beginInsertRows(parent, index, index)

        self.__children.insert(index, row)
        row.setParent(self)

        if model is not None:
            model.endInsertRows()

    def removeRow(self, row):
        """Remove a row from the children list.

        It removes either a node or a row index.

        :param row: BaseRow object or index of row to remove
        :type row: Union[BaseRow, int]
        """
        if isinstance(row, BaseRow):
            row = self.__children.index(row)
        else:
            row = int(row)
        assert row < self.rowCount()

        model = self.model()

        if model is not None:
            index = self.index()
            model.beginRemoveRows(index, row, row)

        node = self.__children.pop(row)
        node.setParent(None)

        if model is not None:
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
            self.__flags = collections.defaultdict(lambda: flags)
        else:
            self.__flags[column] = flags

    def flags(self, column):
        """Returns flags for given column

        :rtype: int
        """
        return self.__flags[column]


class StaticRow(BaseRow):
    """Row with static data.

    :param tuple display: List of data for DisplayRole for each column
    :param dict roles: Optional mapping of roles to list of data.
    :param children: Iterable of BaseRow to start with (not signaled)
    """

    def __init__(self, display=('', None), roles=None, children=()):
        super(StaticRow, self).__init__(children)
        self._dataByRoles = {} if roles is None else roles
        self._dataByRoles[qt.Qt.DisplayRole] = display

    def data(self, column, role):
        if role in self._dataByRoles:
            data = self._dataByRoles[role]
            if column < len(data):
                return data[column]
        return super(StaticRow, self).data(column, role)

    def columnCount(self):
        return len(self._dataByRoles[qt.Qt.DisplayRole])


class ProxyRow(BaseRow):
    """Provides a node to proxy a data accessible through functions.

    Warning: Only weak reference are kept on fget and fset.

    :param str name: The name of this node
    :param callable fget: A callable returning the data
    :param callable fset:
        An optional callable setting the data with data as a single argument.
    :param notify:
        An optional signal emitted when data has changed.
    :param callable toModelData:
        An optional callable to convert from fget
        callable to data returned by the model.
    :param callable fromModelData:
        An optional callable converting data provided to the model to
        data for fset.
    :param editorHint: Data to provide as UserRole for editor selection/setup
    """

    def __init__(self,
                 name='',
                 fget=None,
                 fset=None,
                 notify=None,
                 toModelData=None,
                 fromModelData=None,
                 editorHint=None):

        super(ProxyRow, self).__init__()
        self.__name = name
        self.__editorHint = editorHint

        assert fget is not None
        self._fget = WeakMethodProxy(fget)
        self._fset = WeakMethodProxy(fset) if fset is not None else None
        if fset is not None:
            self.setFlags(self.flags(1) | qt.Qt.ItemIsEditable, 1)
        self._toModelData = toModelData
        self._fromModelData = fromModelData

        if notify is not None:
            notify.connect(self._notified)  # TODO support sigItemChanged flags

    def _notified(self, *args, **kwargs):
        """Send update to the model upon signal notifications"""
        index = self.index(column=1)
        model = self.model()
        if model is not None:
            model.dataChanged.emit(index, index)

    def data(self, column, role):
        if column == 0:
            if role == qt.Qt.DisplayRole:
                return self.__name

        elif column == 1:
            if role == qt.Qt.UserRole:  # EditorHint
                return self.__editorHint
            elif role == qt.Qt.DisplayRole or (role == qt.Qt.EditRole and
                                               self._fset is not None):
                data = self._fget()
                if self._toModelData is not None:
                    data = self._toModelData(data)
                return data

        return super(ProxyRow, self).data(column, role)

    def setData(self, column, value, role):
        if role == qt.Qt.EditRole and self._fset is not None:
            if self._fromModelData is not None:
                value = self._fromModelData(value)
            self._fset(value)
            return True

        return super(ProxyRow, self).setData(column, value, role)


class ColorProxyRow(ProxyRow):
    """Provides a proxy to a QColor property.

    The color is returned through the decorative role.

    See :class:`ProxyRow`
    """

    def data(self, column, role):
        if column == 1:  # Show color as decoration, not text
            if role == qt.Qt.DisplayRole:
                return None
            if role == qt.Qt.DecorationRole:
                role = qt.Qt.DisplayRole
        return super(ColorProxyRow, self).data(column, role)


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


class Settings(StaticRow):
    """Subtree for :class:`SceneWidget` style parameters.

    :param SceneWidget sceneWidget: The widget to control
    """

    def __init__(self, sceneWidget):
        background = ColorProxyRow(
            name='Background',
            fget=sceneWidget.getBackgroundColor,
            fset=sceneWidget.setBackgroundColor)

        foreground = ColorProxyRow(
            name='Foreground',
            fget=sceneWidget.getForegroundColor,
            fset=sceneWidget.setForegroundColor)

        highlight = ColorProxyRow(
            name='Highlight',
            fget=sceneWidget.getHighlightColor,
            fset=sceneWidget.setHighlightColor)

        boundingBox = ProxyRow(
            name='Bounding Box',
            fget=sceneWidget.isBoundingBoxVisible,
            fset=sceneWidget.setBoundingBoxVisible)

        axesIndicator = ProxyRow(
            name='Axes Indicator',
            fget=sceneWidget.isOrientationIndicatorVisible,
            fset=sceneWidget.setOrientationIndicatorVisible)

        # Light direction

        self._lightProxy = _DirectionalLightProxy(sceneWidget.viewport.light)

        azimuthNode = ProxyRow(
            name='Azimuth',
            fget=self._lightProxy.getAzimuthAngle,
            fset=self._lightProxy.setAzimuthAngle,
            notify=self._lightProxy.sigAzimuthAngleChanged,
            editorHint=(-90, 90))

        altitudeNode = ProxyRow(
            name='Altitude',
            fget=self._lightProxy.getAltitudeAngle,
            fset=self._lightProxy.setAltitudeAngle,
            notify=self._lightProxy.sigAltitudeAngleChanged,
            editorHint=(-90, 90))

        lightDirection = StaticRow(('Light Direction', None),
                                   children=(azimuthNode, altitudeNode))

        # Settings row
        children = (background, foreground, highlight,
                    boundingBox, axesIndicator, lightDirection)
        super(Settings, self).__init__(('Settings', None), children=children)


class Item3DRow(StaticRow):
    """Represents an :class:`Item3D` with checkable visibility

    :param Item3D item: The scene item to represent.
    :param str name: The optional name of the item
    """

    def __init__(self, item, name=None):
        if name is None:
            name = item.getLabel()
        super(Item3DRow, self).__init__((name, None))

        self.setFlags(
            self.flags(0) | qt.Qt.ItemIsUserCheckable,
            0)

        self._item = weakref.ref(item)
        item.sigItemChanged.connect(self._itemChanged)

    def _itemChanged(self, event):
        """Handle visibility change"""
        if event == items.ItemChangedType.VISIBLE:
            model = self.model()
            if model is not None:
                index = self.index(column=1)
                model.dataChanged.emit(index, index)

    def item(self):
        """Returns the :class:`Item3D` item or None"""
        return self._item()

    def data(self, column, role):
        if column == 0 and role == qt.Qt.CheckStateRole:
            item = self.item()
            if item is not None and item.isVisible():
                return qt.Qt.Checked
            else:
                return qt.Qt.Unchecked
        elif column == 0 and role == qt.Qt.DecorationRole:
            return icons.getQIcon('item-3dim')
        else:
            return super(Item3DRow, self).data(column, role)

    def setData(self, column, value, role):
        if column == 0 and role == qt.Qt.CheckStateRole:
            item = self.item()
            if item is not None:
                item.setVisible(value == qt.Qt.Checked)
                return True
            else:
                return False
        return super(Item3DRow, self).setData(column, value, role)


class AngleDegreeRow(ProxyRow):
    """ProxyRow patching display of column 1 to add degree symbol

    See :class:`ProxyRow`
    """

    def __init__(self, *args, **kwargs):
        super(AngleDegreeRow, self).__init__(*args, **kwargs)

    def data(self, column, role):
        if column == 1 and role == qt.Qt.DisplayRole:
            return u'%g°' % super(AngleDegreeRow, self).data(column, role)
        else:
            return super(AngleDegreeRow, self).data(column, role)


class DataItem3DTransformRow(StaticRow):
    """Represents :class:`DataItem3D` transform parameters

    :param DataItem3D item: The item for which to display/control transform
    """

    _ROTATION_CENTER_OPTIONS = 'Origin', 'Lower', 'Center', 'Upper'

    def __init__(self, item):
        super(DataItem3DTransformRow, self).__init__(('Transform', None))
        self._item = weakref.ref(item)

        translation = ProxyRow(name='Translation',
                               fget=item.getTranslation,
                               fset=self._setTranslation,
                               notify=item.sigItemChanged,
                               toModelData=lambda data: qt.QVector3D(*data))
        self.addRow(translation)

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

        rotateCenter = StaticRow(
            ('Center', None),
            children=(
                ProxyRow(name='X axis',
                         fget=item.getRotationCenter,
                         fset=self._xSetCenter,
                         notify=item.sigItemChanged,
                         toModelData=self._xCenterToModelData,
                         editorHint=self._ROTATION_CENTER_OPTIONS),
                ProxyRow(name='Y axis',
                         fget=item.getRotationCenter,
                         fset=self._ySetCenter,
                         notify=item.sigItemChanged,
                         toModelData=self._yCenterToModelData,
                         editorHint=self._ROTATION_CENTER_OPTIONS),
                ProxyRow(name='Z axis',
                         fget=item.getRotationCenter,
                         fset=self._zSetCenter,
                         notify=item.sigItemChanged,
                         toModelData=self._zCenterToModelData,
                         editorHint=self._ROTATION_CENTER_OPTIONS),
            ))

        rotate = StaticRow(
            ('Rotation', None),
            children=(
                AngleDegreeRow(name='Angle',
                               fget=item.getRotation,
                               fset=self._setAngle,
                               notify=item.sigItemChanged,
                               toModelData=lambda data: data[0]),
                ProxyRow(name='Axis',
                         fget=item.getRotation,
                         fset=self._setAxis,
                         notify=item.sigItemChanged,
                         toModelData=lambda data: qt.QVector3D(*data[1])),
                rotateCenter
            ))
        self.addRow(rotate)

        scale = ProxyRow(name='Scale',
                         fget=item.getScale,
                         fset=self._setScale,
                         notify=item.sigItemChanged,
                         toModelData=lambda data: qt.QVector3D(*data))
        self.addRow(scale)

    def item(self):
        """Returns the :class:`Item3D` item or None"""
        return self._item()

    @staticmethod
    def _centerToModelData(center, index):
        """Convert rotation center information from scene to model.

        :param center: The center info from the scene
        :param int index: dimension to convert
        """
        value = center[index]
        if isinstance(value, six.string_types):
            return value.title()
        elif value == 0.:
            return 'Origin'
        else:
            return six.text_type(value)

    def _setCenter(self, value, index):
        """Set one dimension of the rotation center.

        :param value: Value received through the model.
        :param int index: dimension to set
        """
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
        """Set rotation angle.

        :param float angle:
        """
        item = self.item()
        if item is not None:
            _, axis = item.getRotation()
            item.setRotation(angle, axis)

    def _setAxis(self, axis):
        """Set rotation axis.

        :param QVector3D axis:
        """
        item = self.item()
        if item is not None:
            angle, _ = item.getRotation()
            item.setRotation(angle, (axis.x(), axis.y(), axis.z()))

    def _setTranslation(self, translation):
        """Set translation transform.

        :param QVector3D translation:
        """
        item = self.item()
        if item is not None:
            item.setTranslation(translation.x(), translation.y(), translation.z())

    def _setScale(self, scale):
        """Set scale transform.

        :param QVector3D scale:
        """
        item = self.item()
        if item is not None:
            item.setScale(scale.x(), scale.y(), scale.z())


class GroupItemRow(Item3DRow):
    """Represents a :class:`GroupItem` with transforms and children

    :param GroupItem item: The scene group to represent.
    :param str name: The optional name of the group
    """

    def __init__(self, item, name=None):
        super(GroupItemRow, self).__init__(item, name)
        self.addRow(DataItem3DTransformRow(item))

        item.sigItemAdded.connect(self._itemAdded)
        item.sigItemRemoved.connect(self._itemRemoved)

        for child in item.getItems():
            self.addRow(nodeFromItem(child))

    def _itemAdded(self, item):
        """Handle item addition to the group and add it to the model.

        :param Item3D item: added item
        """
        group = self.item()
        if group is None:
            return

        row = group.getItems().index(item)
        self.addRow(nodeFromItem(item), row + 1)

    def _itemRemoved(self, item):
        """Handle item removal from the group and remove it from the model.

        :param Item3D item: removed item
        """
        group = self.item()
        if group is None:
            return

        # Find item
        for row in self.children():
            if row.item() is item:
                self.removeRow(row)
                break  # Got it
        else:
            raise RuntimeError("Model does not correspond to scene content")


class InterpolationRow(ProxyRow):
    """Represents :class:`InterpolationMixIn` property.

    :param Item3D item: Scene item with interpolation property
    """

    def __init__(self, item):
        modes = [mode.title() for mode in item.INTERPOLATION_MODES]
        super(InterpolationRow, self).__init__(
            name='Interpolation',
            fget=item.getInterpolation,
            fset=item.setInterpolation,
            notify=item.sigItemChanged,
            toModelData=lambda mode: mode.title(),
            fromModelData=lambda mode: mode.lower(),
            editorHint=modes)


class _RangeProxyRow(ProxyRow):
    """ProxyRow for colormap min and max

    It disable editing when colormap is autoscale.
    """

    def __init__(self, *args, **kwargs):
        super(_RangeProxyRow, self).__init__(*args, **kwargs)

    def _notified(self, *args, **kwargs):
        topLeft = self.index(column=0)
        bottomRight = self.index(column=1)
        model = self.model()
        model.dataChanged.emit(topLeft, bottomRight)

    def flags(self, column):
        flags = super(_RangeProxyRow, self).flags(column)

        parent = self.parent()
        if parent is not None:
            item = parent.item()
            if item is not None:
                colormap = item.getColormap()
                if colormap.isAutoscale():
                    # Remove item is enabled flag
                    flags = qt.Qt.ItemFlags(flags) & ~qt.Qt.ItemIsEnabled
        return flags


class ColormapRow(StaticRow):
    """Represents :class:`ColormapMixIn` property.

    :param Item3D item: Scene item with colormap property
    """

    _sigColormapChanged = qt.Signal()
    """Signal used internally to notify colormap (or data) update"""

    def __init__(self, item):
        super(ColormapRow, self).__init__(('Colormap', None))
        self._item = weakref.ref(item)
        item.sigItemChanged.connect(self._itemChanged)

        self._colormap = item.getColormap()
        self._colormap.sigChanged.connect(self._sigColormapChanged)

        self._colormapImage = None
        self._dataRange = None

        self._colormapsMapping = {}
        for cmap in preferredColormaps():
            self._colormapsMapping[cmap.title()] = cmap

        self.addRow(ProxyRow(
            name='Name',
            fget=self._getName,
            fset=self._setName,
            notify=self._sigColormapChanged,
            editorHint=list(self._colormapsMapping.keys())))

        norms = [norm.title() for norm in self._colormap.NORMALIZATIONS]
        self.addRow(ProxyRow(
            name='Normalization',
            fget=self._getNormalization,
            fset=self._setNormalization,
            notify=self._sigColormapChanged,
            editorHint=norms))

        self.addRow(ProxyRow(
            name='Autoscale',
            fget=self._isAutoscale,
            fset=self._setAutoscale,
            notify=self._sigColormapChanged))
        self.addRow(_RangeProxyRow(
            name='Min.',
            fget=self._getVMin,
            fset=self._setVMin,
            notify=self._sigColormapChanged))
        self.addRow(_RangeProxyRow(
            name='Max.',
            fget=self._getVMax,
            fset=self._setVMax,
            notify=self._sigColormapChanged))

        self._sigColormapChanged.connect(self._updateColormapImage)

    def _getName(self):
        """Proxy for :meth:`Colormap.getName`"""
        return self._colormap.getName().title()

    def _setName(self, name):
        """Proxy for :meth:`Colormap.setName`"""
        # Convert back from titled to name if possible
        name = self._colormapsMapping.get(name, name)
        self._colormap.setName(name)

    def _getNormalization(self):
        """Proxy for :meth:`Colormap.getNormalization`"""
        return self._colormap.getNormalization().title()

    def _setNormalization(self, normalization):
        """Proxy for :meth:`Colormap.setNormalization`"""
        return self._colormap.setNormalization(normalization.lower())

    def _isAutoscale(self):
        """Proxy for :meth:`Colormap.isAutoscale`"""
        return self._colormap.isAutoscale()

    def _updateColormapImage(self, *args, **kwargs):
        """Notify colormap update to update the image in the tree"""
        if self._colormapImage is not None:
            self._colormapImage = None
            index = self.index(column=1)
            self.model().dataChanged.emit(index, index)

    def data(self, column, role):
        if column == 1 and role == qt.Qt.DecorationRole:
            if self._colormapImage is None:
                image = numpy.zeros((16, 130, 3), dtype=numpy.uint8)
                image[1:-1, 1:-1] = self._colormap.getNColors(image.shape[1] - 2)[:, :3]
                self._colormapImage = convertArrayToQImage(image)
            return self._colormapImage

        return super(ColormapRow, self).data(column, role)

    def _getColormapRange(self):
        """Returns the range of the colormap for the current data.

        :return: Colormap range (min, max)
        """
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
        """Proxy to get colormap min value

        :rtype: float
        """
        min_ = self._colormap.getVMin()
        if min_ is None:
            min_ = self._getColormapRange()[0]
        return min_

    def _setVMin(self, min_):
        """Proxy to set colormap min.

        :param float min_:
        """
        max_ = self._colormap.getVMax()
        if max_ is not None and min_ > max_:
            min_, max_ = max_, min_
        self._colormap.setVRange(min_, max_)

    def _getVMax(self):
        """Proxy to get colormap max value

        :rtype: float
        """
        max_ = self._colormap.getVMax()
        if max_ is None:
            max_ = self._getColormapRange()[1]
        return max_

    def _setVMax(self, max_):
        """Proxy to set colormap max.

        :param float max_:
        """
        min_ = self._colormap.getVMin()
        if min_ is not None and min_ > max_:
            min_, max_ = max_, min_
        self._colormap.setVRange(min_, max_)

    def _setAutoscale(self, autoscale):
        """Proxy to set autscale

        :param bool autoscale:
        """
        item = self.item()
        if item is not None:
            colormap = item.getColormap()
            if autoscale:
                vmin, vmax = None, None
            else:
                vmin, vmax = self._getColormapRange()
            colormap.setVRange(vmin, vmax)

    def item(self):
        """Returns the :class:`ColormapMixIn` item or None"""
        return self._item()

    def _itemChanged(self, event):
        """Handle change of colormap or data in the item.

        :param ItemChangedType event:
        """
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


class SymbolRow(ProxyRow):
    """Represents :class:`SymbolMixIn` symbol property.

    :param Item3D item: Scene item with symbol property
    """

    def __init__(self, item):
        names = [item.getSymbolName(s) for s in item.getSupportedSymbols()]
        super(SymbolRow, self).__init__(
             name='Marker',
             fget=item.getSymbolName,
             fset=item.setSymbol,
             notify=item.sigItemChanged,
             editorHint=names)


class SymbolSizeRow(ProxyRow):
    """Represents :class:`SymbolMixIn` symbol size property.

    :param Item3D item: Scene item with symbol size property
    """

    def __init__(self, item):
        super(SymbolSizeRow, self).__init__(
            name='Marker size',
            fget=item.getSymbolSize,
            fset=item.setSymbolSize,
            notify=item.sigItemChanged,
            editorHint=(1, 50))  # TODO link with OpenGL max point size


class PlaneRow(ProxyRow):
    """Represents :class:`PlaneMixIn` property.

    :param Item3D item: Scene item with plane equation property
    """

    def __init__(self, item):
        super(PlaneRow, self).__init__(
            name='Equation',
            fget=item.getParameters,
            fset=item.setParameters,
            notify=item.sigItemChanged,
            toModelData=lambda data: qt.QVector4D(*data),
            fromModelData=lambda data: (data.x(), data.y(), data.z(), data.w()))
        self._item = weakref.ref(item)

    def data(self, column, role):
        if column == 1 and role == qt.Qt.DisplayRole:
            item = self._item()
            if item is not None:
                params = item.getParameters()
                return ('%gx %+gy %+gz %+g = 0' %
                        (params[0], params[1], params[2], params[3]))
        return super(PlaneRow, self).data(column, role)


class RemoveIsosurfaceRow(BaseRow):
    """Class for Isosurface Delete button

    :param Isosurface isosurface: The isosurface item to attach the button to.
    """

    def __init__(self, isosurface):
        super(RemoveIsosurfaceRow, self).__init__()
        self._isosurface = weakref.ref(isosurface)

    def createEditor(self):
        """Specific editor factory provided to the model"""
        editor = qt.QWidget()
        layout = qt.QHBoxLayout(editor)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        removeBtn = qt.QToolButton()
        removeBtn.setText('Delete')
        removeBtn.setToolButtonStyle(qt.Qt.ToolButtonTextOnly)
        layout.addWidget(removeBtn)
        removeBtn.clicked.connect(self._removeClicked)

        layout.addStretch(1)
        return editor

    def isosurface(self):
        """Returns the controlled isosurface

        :rtype: Isosurface
        """
        return self._isosurface()

    def data(self, column, role):
        if column == 0 and role == qt.Qt.UserRole:  # editor hint
            return self.createEditor

        return super(RemoveIsosurfaceRow, self).data(column, role)

    def flags(self, column):
        flags = super(RemoveIsosurfaceRow, self).flags(column)
        if column == 0:
            flags |= qt.Qt.ItemIsEditable
        return flags

    def _removeClicked(self):
        """Handle Delete button clicked"""
        isosurface = self.isosurface()
        if isosurface is not None:
            scalarField3D = isosurface.parent()
            if scalarField3D is not None:
                scalarField3D.removeIsosurface(isosurface)


class IsosurfaceRow(Item3DRow):
    """Represents an :class:`Isosurface` item.

    :param Isosurface item: Isosurface item
    """

    _LEVEL_SLIDER_RANGE = 0, 1000
    """Range given as editor hint"""

    def __init__(self, item):
        super(IsosurfaceRow, self).__init__(item, name=item.getLevel())

        self.setFlags(self.flags(1) | qt.Qt.ItemIsEditable, 1)

        item.sigItemChanged.connect(self._levelChanged)

        self.addRow(ProxyRow(
            name='Level',
            fget=self._getValueForLevelSlider,
            fset=self._setLevelFromSliderValue,
            notify=item.sigItemChanged,
            editorHint=self._LEVEL_SLIDER_RANGE))

        self.addRow(ColorProxyRow(
            name='Color',
            fget=self._rgbColor,
            fset=self._setRgbColor,
            notify=item.sigItemChanged))

        self.addRow(ProxyRow(
            name='Opacity',
            fget=self._opacity,
            fset=self._setOpacity,
            notify=item.sigItemChanged,
            editorHint=(0, 255)))

        self.addRow(RemoveIsosurfaceRow(item))

    def _getValueForLevelSlider(self):
        """Convert iso level to slider value.

        :rtype: int
        """
        item = self.item()
        if item is not None:
            scalarField3D = item.parent()
            if scalarField3D is not None:
                dataRange = scalarField3D.getDataRange()
                if dataRange is not None:
                    dataMin, dataMax = dataRange[0], dataRange[-1]
                    offset = (item.getLevel() - dataMin) / (dataMax - dataMin)

                    sliderMin, sliderMax = self._LEVEL_SLIDER_RANGE
                    value = sliderMin + (sliderMax - sliderMin) * offset
                    return value
        return 0

    def _setLevelFromSliderValue(self, value):
        """Convert slider value to isolevel.

        :param int value:
        """
        item = self.item()
        if item is not None:
            scalarField3D = item.parent()
            if scalarField3D is not None:
                dataRange = scalarField3D.getDataRange()
                if dataRange is not None:
                    sliderMin, sliderMax = self._LEVEL_SLIDER_RANGE
                    offset = (value - sliderMin) / (sliderMax - sliderMin)

                    dataMin, dataMax = dataRange[0], dataRange[-1]
                    level = dataMin + (dataMax - dataMin) * offset
                    item.setLevel(level)

    def _rgbColor(self):
        """Proxy to get the isosurface's RGB color without transparency

        :rtype: QColor
        """
        item = self.item()
        if item is None:
            return None
        else:
            color = item.getColor()
            color.setAlpha(255)
            return color

    def _setRgbColor(self, color):
        """Proxy to set the isosurface's RGB color without transparency

        :param QColor color:
        """
        item = self.item()
        if item is not None:
            color.setAlpha(item.getColor().alpha())
            item.setColor(color)

    def _opacity(self):
        """Proxy to get the isosurface's transparency

        :rtype: int
        """
        item = self.item()
        return 255 if item is None else item.getColor().alpha()

    def _setOpacity(self, opacity):
        """Proxy to set the isosurface's transparency.

        :param int opacity:
        """
        item = self.item()
        if item is not None:
            color = item.getColor()
            color.setAlpha(opacity)
            item.setColor(color)

    def _levelChanged(self, event):
        """Handle isosurface level changed and notify model

        :param ItemChangedType event:
        """
        if event == items.Item3DChangedType.ISO_LEVEL:
            model = self.model()
            if model is not None:
                index = self.index(column=1)
                model.dataChanged.emit(index, index)

    def data(self, column, role):
        if column == 0:  # Show color as decoration, not text
            if role == qt.Qt.DisplayRole:
                return None
            elif role == qt.Qt.DecorationRole:
                return self._rgbColor()

        elif column == 1 and role in (qt.Qt.DisplayRole, qt.Qt.EditRole):
                item = self.item()
                return None if item is None else item.getLevel()

        return super(IsosurfaceRow, self).data(column, role)

    def setData(self, column, value, role):
        if column == 1 and role == qt.Qt.EditRole:
            item = self.item()
            if item is not None:
                item.setLevel(value)
            return True

        return super(IsosurfaceRow, self).setData(column, value, role)


class AddIsosurfaceRow(BaseRow):
    """Class for Isosurface create button

    :param ScalarField3D scalarField3D:
        The ScalarField3D item to attach the button to.
    """

    def __init__(self, scalarField3D):
        super(AddIsosurfaceRow, self).__init__()
        self._scalarField3D = weakref.ref(scalarField3D)

    def createEditor(self):
        """Specific editor factory provided to the model"""
        editor = qt.QWidget()
        layout = qt.QHBoxLayout(editor)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        addBtn = qt.QToolButton()
        addBtn.setText('+')
        addBtn.setToolButtonStyle(qt.Qt.ToolButtonTextOnly)
        layout.addWidget(addBtn)
        addBtn.clicked.connect(self._addClicked)

        layout.addStretch(1)
        return editor

    def scalarField3D(self):
        """Returns the controlled ScalarField3D

        :rtype: ScalarField3D
        """
        return self._scalarField3D()

    def data(self, column, role):
        if column == 0 and role == qt.Qt.UserRole:  # editor hint
            return self.createEditor

        return super(AddIsosurfaceRow, self).data(column, role)

    def flags(self, column):
        flags = super(AddIsosurfaceRow, self).flags(column)
        if column == 0:
            flags |= qt.Qt.ItemIsEditable
        return flags

    def _addClicked(self):
        """Handle Delete button clicked"""
        scalarField3D = self.scalarField3D()
        if scalarField3D is not None:
            dataRange = scalarField3D.getDataRange()
            if dataRange is None:
                dataRange = 0., 1.

            scalarField3D.addIsosurface(
                numpy.mean((dataRange[0], dataRange[-1])),
                '#0000FF')


class ScalarField3DIsoSurfacesRow(StaticRow):
    """Represents  :class:`ScalarFieldView`'s isosurfaces

    :param ScalarFieldView scalarField3D: ScalarFieldView to control
    """

    def __init__(self, scalarField3D):
        super(ScalarField3DIsoSurfacesRow, self).__init__(
            ('Isosurfaces', None))
        self._scalarField3D = weakref.ref(scalarField3D)

        scalarField3D.sigIsosurfaceAdded.connect(self._isosurfaceAdded)
        scalarField3D.sigIsosurfaceRemoved.connect(self._isosurfaceRemoved)

        for item in scalarField3D.getIsosurfaces():
            self.addRow(nodeFromItem(item))

        self.addRow(AddIsosurfaceRow(scalarField3D))

    def scalarField3D(self):
        """Returns the controlled ScalarField3D

        :rtype: ScalarField3D
        """
        return self._scalarField3D()

    def _isosurfaceAdded(self, item):
        """Handle isosurface addition

        :param Isosurface item: added isosurface
        """
        scalarField3D = self.scalarField3D()
        if scalarField3D is None:
            return

        row = scalarField3D.getIsosurfaces().index(item)
        self.addRow(nodeFromItem(item), row)

    def _isosurfaceRemoved(self, item):
        """Handle isosurface removal

        :param Isosurface item: removed isosurface
        """
        scalarField3D = self.scalarField3D()
        if scalarField3D is None:
            return

        # Find item
        for row in self.children():
            if row.item() is item:
                self.removeRow(row)
                break  # Got it
        else:
            raise RuntimeError("Model does not correspond to scene content")


def initScatter2DNode(node, item):
    """Specific node init for Scatter2D to set order of parameters

    :param Item3DRow node: The model node to setup
    :param Scatter2D item: The Scatter2D the node is representing
    """
    node.addRow(ProxyRow(
        name='Mode',
        fget=item.getVisualization,
        fset=item.setVisualization,
        notify=item.sigItemChanged,
        editorHint=[m.title() for m in item.supportedVisualizations()],
        toModelData=lambda data: data.title(),
        fromModelData=lambda data: data.lower()))

    node.addRow(ProxyRow(
        name='Height map',
        fget=item.isHeightMap,
        fset=item.setHeightMap,
        notify=item.sigItemChanged))

    node.addRow(ColormapRow(item))

    node.addRow(SymbolRow(item))
    node.addRow(SymbolSizeRow(item))

    node.addRow(ProxyRow(
        name='Line width',
        fget=item.getLineWidth,
        fset=item.setLineWidth,
        notify=item.sigItemChanged,
        editorHint=(1, 10)))  # TODO link with OpenGL max line width


def initScalarField3DNode(node, item):
    """Specific node init for ScalarField3D

    :param Item3DRow node: The model node to setup
    :param ScalarField3D item: The ScalarField3D the node is representing
    """
    node.addRow(nodeFromItem(item.getCutPlanes()[0]))  # Add cut plane
    node.addRow(ScalarField3DIsoSurfacesRow(item))


NODE_SPECIFIC_INIT = [  # class, init(node, item)
    (items.Scatter2D, initScatter2DNode),
    (items.ScalarField3D, initScalarField3DNode),
]
"""List of specific node init for different item class"""


def nodeFromItem(item, name=None):
    """Create :class:`Item3DRow` subclass corresponding to item

    :param Item3D item: The item fow which to create the node
    :param str name: The name of the subtree for this item (optional)
    :rtype: Item3DRow
    """
    assert isinstance(item, items.Item3D)

    # Item with specific model row class
    if isinstance(item, items.GroupItem):
        return GroupItemRow(item)
    elif isinstance(item, Isosurface):
        return IsosurfaceRow(item)

    # Create Item3DRow and populate it
    node = Item3DRow(item, name)

    if isinstance(item, items.DataItem3D):
        node.addRow(DataItem3DTransformRow(item))

    # Specific extra init
    for cls, specificInit in NODE_SPECIFIC_INIT:
        if isinstance(item, cls):
            specificInit(node, item)
            break

    else:  # Generic case: handle mixins
        for cls in item.__class__.__mro__:
            if cls is items.ColormapMixIn:
                node.addRow(ColormapRow(item))

            elif cls is items.InterpolationMixIn:
                node.addRow(InterpolationRow(item))

            elif cls is items.SymbolMixIn:
                node.addRow(SymbolRow(item))
                node.addRow(SymbolSizeRow(item))

            elif cls is items.PlaneMixIn:
                node.addRow(PlaneRow(item))

    return node


class Root(BaseRow):
    """Root node of :class:`SceneWidget` parameters.

    It has two children:
    - Settings
    - Scene group
    """

    def __init__(self, model, sceneWidget):
        super(Root, self).__init__()
        self._sceneWidget = weakref.ref(sceneWidget)
        self.setParent(model)  # Needed for Root

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
        self._sceneWidget = weakref.ref(parent)

        super(SceneModel, self).__init__(parent)
        self._root = Root(self, parent)
        self._root.addRow(Settings(parent))
        self._root.addRow(nodeFromItem(parent.getSceneGroup(), name='Data'))

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
        """See :meth:`QAbstractItemModel.index`"""
        if column >= self.columnCount(parent) or row >= self.rowCount(parent):
            return qt.QModelIndex()

        item = self._itemFromIndex(parent)
        return self.createIndex(row, column, item.children()[row])

    def parent(self, index):
        """See :meth:`QAbstractItemModel.parent`"""
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
        """See :meth:`QAbstractItemModel.rowCount`"""
        item = self._itemFromIndex(parent)
        return item.rowCount()

    def columnCount(self, parent=qt.QModelIndex()):
        """See :meth:`QAbstractItemModel.columnCount`"""
        item = self._itemFromIndex(parent)
        return item.columnCount()

    def data(self, index, role=qt.Qt.DisplayRole):
        """See :meth:`QAbstractItemModel.data`"""
        item = self._itemFromIndex(index)
        column = index.column()
        return item.data(column, role)

    def setData(self, index, value, role=qt.Qt.EditRole):
        """See :meth:`QAbstractItemModel.setData`"""
        item = self._itemFromIndex(index)
        column = index.column()
        if item.setData(column, value, role):
            self.dataChanged.emit(index, index)
            return True
        return False

    def flags(self, index):
        """See :meth:`QAbstractItemModel.flags`"""
        item = self._itemFromIndex(index)
        column = index.column()
        return item.flags(column)

    def headerData(self, section, orientation, role=qt.Qt.DisplayRole):
        """See :meth:`QAbstractItemModel.headerData`"""
        if orientation == qt.Qt.Horizontal and role == qt.Qt.DisplayRole:
            return 'Item' if section == 0 else 'Value'
        else:
            return None
