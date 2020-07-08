# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017-2020 European Synchrotron Radiation Facility
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
This module provides base classes to implement models for 3D scene content
"""

from __future__ import absolute_import, division

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "24/04/2018"


from collections import OrderedDict
import functools
import logging
import weakref

import numpy
import six

from ...utils.image import convertArrayToQImage
from ...colors import preferredColormaps
from ... import qt, icons
from .. import items
from ..items.volume import Isosurface, CutPlane, ComplexIsosurface
from ..Plot3DWidget import Plot3DWidget


from .core import AngleDegreeRow, BaseRow, ColorProxyRow, ProxyRow, StaticRow


_logger = logging.getLogger(__name__)


class ItemProxyRow(ProxyRow):
    """Provides a node to proxy a data accessible through functions.

    It listens on sigItemChanged to trigger the update.

    Warning: Only weak reference are kept on fget and fset.

    :param Item3D item: The item to
    :param str name: The name of this node
    :param callable fget: A callable returning the data
    :param callable fset:
        An optional callable setting the data with data as a single argument.
    :param events:
        An optional event kind or list of event kinds to react upon.
    :param callable toModelData:
        An optional callable to convert from fget
        callable to data returned by the model.
    :param callable fromModelData:
        An optional callable converting data provided to the model to
        data for fset.
    :param editorHint: Data to provide as UserRole for editor selection/setup
    """

    def __init__(self,
                 item,
                 name='',
                 fget=None,
                 fset=None,
                 events=None,
                 toModelData=None,
                 fromModelData=None,
                 editorHint=None):
        super(ItemProxyRow, self).__init__(
            name=name,
            fget=fget,
            fset=fset,
            notify=None,
            toModelData=toModelData,
            fromModelData=fromModelData,
            editorHint=editorHint)

        if isinstance(events, (items.ItemChangedType,
                               items.Item3DChangedType)):
            events = (events,)
        self.__events = events
        item.sigItemChanged.connect(self.__itemChanged)

    def __itemChanged(self, event):
        """Handle item changed

        :param Union[ItemChangedType,Item3DChangedType] event:
        """
        if self.__events is None or event in self.__events:
            self._notified()


class ItemColorProxyRow(ColorProxyRow, ItemProxyRow):
    """Combines :class:`ColorProxyRow` and :class:`ItemProxyRow`"""

    def __init__(self, *args, **kwargs):
        ItemProxyRow.__init__(self, *args, **kwargs)


class ItemAngleDegreeRow(AngleDegreeRow, ItemProxyRow):
    """Combines :class:`AngleDegreeRow` and :class:`ItemProxyRow`"""

    def __init__(self, *args, **kwargs):
        ItemProxyRow.__init__(self, *args, **kwargs)


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
        self._azimuth = 0
        self._altitude = 0

    def getAzimuthAngle(self):
        """Returns the signed angle in the horizontal plane.

         Unit: degrees.
        The 0 angle corresponds to the axis perpendicular to the screen.

        :rtype: int
        """
        return self._azimuth

    def getAltitudeAngle(self):
        """Returns the signed vertical angle from the horizontal plane.

        Unit: degrees.
        Range: [-90, +90]

        :rtype: int
        """
        return self._altitude

    def setAzimuthAngle(self, angle):
        """Set the horizontal angle.

        :param int angle: Angle from -z axis in zx plane in degrees.
        """
        angle = int(round(angle))
        if angle != self._azimuth:
            self._azimuth = angle
            self._updateLight()
            self.sigAzimuthAngleChanged.emit()

    def setAltitudeAngle(self, angle):
        """Set the horizontal angle.

        :param int angle: Angle from -z axis in zy plane in degrees.
        """
        angle = int(round(angle))
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
        azimuth = int(round(numpy.degrees(numpy.arctan2(x, z))))
        altitude = int(round(numpy.degrees(numpy.pi/2. - numpy.arccos(y))))

        if azimuth != self.getAzimuthAngle():
            self.setAzimuthAngle(azimuth)

        if altitude != self.getAltitudeAngle():
            self.setAltitudeAngle(altitude)

    def _updateLight(self):
        """Update light direction in the scene"""
        azimuth = numpy.radians(self._azimuth)
        delta = numpy.pi/2. - numpy.radians(self._altitude)
        if delta == 0.:  # Avoids zenith position
            delta = 0.0001
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
            fset=sceneWidget.setBackgroundColor,
            notify=sceneWidget.sigStyleChanged)

        foreground = ColorProxyRow(
            name='Foreground',
            fget=sceneWidget.getForegroundColor,
            fset=sceneWidget.setForegroundColor,
            notify=sceneWidget.sigStyleChanged)

        text = ColorProxyRow(
            name='Text',
            fget=sceneWidget.getTextColor,
            fset=sceneWidget.setTextColor,
            notify=sceneWidget.sigStyleChanged)

        highlight = ColorProxyRow(
            name='Highlight',
            fget=sceneWidget.getHighlightColor,
            fset=sceneWidget.setHighlightColor,
            notify=sceneWidget.sigStyleChanged)

        axesIndicator = ProxyRow(
            name='Axes Indicator',
            fget=sceneWidget.isOrientationIndicatorVisible,
            fset=sceneWidget.setOrientationIndicatorVisible,
            notify=sceneWidget.sigStyleChanged)

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

        # Fog
        fog = ProxyRow(
            name='Fog',
            fget=sceneWidget.getFogMode,
            fset=sceneWidget.setFogMode,
            notify=sceneWidget.sigStyleChanged,
            toModelData=lambda mode: mode is Plot3DWidget.FogMode.LINEAR,
            fromModelData=lambda mode: Plot3DWidget.FogMode.LINEAR if mode else Plot3DWidget.FogMode.NONE)

        # Settings row
        children = (background, foreground, text, highlight,
                    axesIndicator, lightDirection, fog)
        super(Settings, self).__init__(('Settings', None), children=children)


class Item3DRow(BaseRow):
    """Represents an :class:`Item3D` with checkable visibility

    :param Item3D item: The scene item to represent.
    :param str name: The optional name of the item
    """

    _EVENTS = items.ItemChangedType.VISIBLE, items.Item3DChangedType.LABEL
    """Events for which to update the first column in the tree"""

    def __init__(self, item, name=None):
        self.__name = None if name is None else six.text_type(name)
        super(Item3DRow, self).__init__()

        self.setFlags(
            self.flags(0) | qt.Qt.ItemIsUserCheckable | qt.Qt.ItemIsSelectable,
            0)
        self.setFlags(self.flags(1) | qt.Qt.ItemIsSelectable, 1)

        self._item = weakref.ref(item)
        item.sigItemChanged.connect(self._itemChanged)

    def _itemChanged(self, event):
        """Handle model update upon change"""
        if event in self._EVENTS:
            model = self.model()
            if model is not None:
                index = self.index(column=0)
                model.dataChanged.emit(index, index)

    def item(self):
        """Returns the :class:`Item3D` item or None"""
        return self._item()

    def data(self, column, role):
        if column == 0:
            if role == qt.Qt.CheckStateRole:
                item = self.item()
                if item is not None and item.isVisible():
                    return qt.Qt.Checked
                else:
                    return qt.Qt.Unchecked

            elif role == qt.Qt.DecorationRole:
                return icons.getQIcon('item-3dim')

            elif role == qt.Qt.DisplayRole:
                if self.__name is None:
                    item = self.item()
                    return '' if item is None else item.getLabel()
                else:
                    return self.__name

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

    def columnCount(self):
        return 2


class DataItem3DBoundingBoxRow(ItemProxyRow):
    """Represents :class:`DataItem3D` bounding box visibility

    :param DataItem3D item: The item for which to display/control bounding box
    """

    def __init__(self, item):
        super(DataItem3DBoundingBoxRow, self).__init__(
            item=item,
            name='Bounding box',
            fget=item.isBoundingBoxVisible,
            fset=item.setBoundingBoxVisible,
            events=items.Item3DChangedType.BOUNDING_BOX_VISIBLE)


class MatrixProxyRow(ItemProxyRow):
    """Proxy for a row of a DataItem3D 3x3 matrix transform

    :param DataItem3D item:
    :param int index: Matrix row index
    """

    def __init__(self, item, index):
        self._item = weakref.ref(item)
        self._index = index

        super(MatrixProxyRow, self).__init__(
            item=item,
            name='',
            fget=self._getMatrixRow,
            fset=self._setMatrixRow,
            events=items.Item3DChangedType.TRANSFORM)

    def _getMatrixRow(self):
        """Returns the matrix row.

        :rtype: QVector3D
        """
        item = self._item()
        if item is not None:
            matrix = item.getMatrix()
            return qt.QVector3D(*matrix[self._index, :])
        else:
            return None

    def _setMatrixRow(self, row):
        """Set the row of the matrix

        :param QVector3D row: Row values to set
        """
        item = self._item()
        if item is not None:
            matrix = item.getMatrix()
            matrix[self._index, :] = row.x(), row.y(), row.z()
            item.setMatrix(matrix)

    def data(self, column, role):
        data = super(MatrixProxyRow, self).data(column, role)

        if column == 1 and role == qt.Qt.DisplayRole:
            # Convert QVector3D to text
            data = "%g; %g; %g" % (data.x(), data.y(), data.z())

        return data


class DataItem3DTransformRow(StaticRow):
    """Represents :class:`DataItem3D` transform parameters

    :param DataItem3D item: The item for which to display/control transform
    """

    _ROTATION_CENTER_OPTIONS = 'Origin', 'Lower', 'Center', 'Upper'

    def __init__(self, item):
        super(DataItem3DTransformRow, self).__init__(('Transform', None))
        self._item = weakref.ref(item)

        translation = ItemProxyRow(
            item=item,
            name='Translation',
            fget=item.getTranslation,
            fset=self._setTranslation,
            events=items.Item3DChangedType.TRANSFORM,
            toModelData=lambda data: qt.QVector3D(*data))
        self.addRow(translation)

        # Here to keep a reference
        self._xSetCenter = functools.partial(self._setCenter, index=0)
        self._ySetCenter = functools.partial(self._setCenter, index=1)
        self._zSetCenter = functools.partial(self._setCenter, index=2)

        rotateCenter = StaticRow(
            ('Center', None),
            children=(
                ItemProxyRow(item=item,
                             name='X axis',
                             fget=item.getRotationCenter,
                             fset=self._xSetCenter,
                             events=items.Item3DChangedType.TRANSFORM,
                             toModelData=functools.partial(
                                 self._centerToModelData, index=0),
                             editorHint=self._ROTATION_CENTER_OPTIONS),
                ItemProxyRow(item=item,
                             name='Y axis',
                             fget=item.getRotationCenter,
                             fset=self._ySetCenter,
                             events=items.Item3DChangedType.TRANSFORM,
                             toModelData=functools.partial(
                                 self._centerToModelData, index=1),
                             editorHint=self._ROTATION_CENTER_OPTIONS),
                ItemProxyRow(item=item,
                             name='Z axis',
                             fget=item.getRotationCenter,
                             fset=self._zSetCenter,
                             events=items.Item3DChangedType.TRANSFORM,
                             toModelData=functools.partial(
                                 self._centerToModelData, index=2),
                             editorHint=self._ROTATION_CENTER_OPTIONS),
            ))

        rotate = StaticRow(
            ('Rotation', None),
            children=(
                ItemAngleDegreeRow(
                    item=item,
                    name='Angle',
                    fget=item.getRotation,
                    fset=self._setAngle,
                    events=items.Item3DChangedType.TRANSFORM,
                    toModelData=lambda data: data[0]),
                ItemProxyRow(
                    item=item,
                    name='Axis',
                    fget=item.getRotation,
                    fset=self._setAxis,
                    events=items.Item3DChangedType.TRANSFORM,
                    toModelData=lambda data: qt.QVector3D(*data[1])),
                rotateCenter
            ))
        self.addRow(rotate)

        scale = ItemProxyRow(
            item=item,
            name='Scale',
            fget=item.getScale,
            fset=self._setScale,
            events=items.Item3DChangedType.TRANSFORM,
            toModelData=lambda data: qt.QVector3D(*data))
        self.addRow(scale)

        matrix = StaticRow(
            ('Matrix', None),
            children=(MatrixProxyRow(item, 0),
                      MatrixProxyRow(item, 1),
                      MatrixProxyRow(item, 2)))
        self.addRow(matrix)

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
            sx, sy, sz = scale.x(), scale.y(), scale.z()
            if sx == 0. or sy == 0. or sz == 0.:
                _logger.warning('Cannot set scale to 0: ignored')
            else:
                item.setScale(scale.x(), scale.y(), scale.z())


class GroupItemRow(Item3DRow):
    """Represents a :class:`GroupItem` with transforms and children

    :param GroupItem item: The scene group to represent.
    :param str name: The optional name of the group
    """

    _CHILDREN_ROW_OFFSET = 2
    """Number of rows for group parameters. Children are added after"""

    def __init__(self, item, name=None):
        super(GroupItemRow, self).__init__(item, name)
        self.addRow(DataItem3DBoundingBoxRow(item))
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
        self.addRow(nodeFromItem(item), row + self._CHILDREN_ROW_OFFSET)

    def _itemRemoved(self, item):
        """Handle item removal from the group and remove it from the model.

        :param Item3D item: removed item
        """
        group = self.item()
        if group is None:
            return

        # Find item
        for row in self.children():
            if isinstance(row, Item3DRow) and row.item() is item:
                self.removeRow(row)
                break  # Got it
        else:
            raise RuntimeError("Model does not correspond to scene content")


class InterpolationRow(ItemProxyRow):
    """Represents :class:`InterpolationMixIn` property.

    :param Item3D item: Scene item with interpolation property
    """

    def __init__(self, item):
        modes = [mode.title() for mode in item.INTERPOLATION_MODES]
        super(InterpolationRow, self).__init__(
            item=item,
            name='Interpolation',
            fget=item.getInterpolation,
            fset=item.setInterpolation,
            events=items.Item3DChangedType.INTERPOLATION,
            toModelData=lambda mode: mode.title(),
            fromModelData=lambda mode: mode.lower(),
            editorHint=modes)


class _ColormapBaseProxyRow(ProxyRow):
    """Base class for colormap model row

    This class handle synchronization and signals from the item and the colormap
    """

    _sigColormapChanged = qt.Signal()
    """Signal used internally to notify colormap (or data) update"""

    def __init__(self, item, *args, **kwargs):
        self._item = weakref.ref(item)
        self._colormap = item.getColormap()

        ProxyRow.__init__(self, *args, **kwargs)

        self._colormap.sigChanged.connect(self._colormapChanged)
        item.sigItemChanged.connect(self._itemChanged)
        self._sigColormapChanged.connect(self._modelUpdated)

    def item(self):
        """Returns the :class:`ColormapMixIn` item or None"""
        return self._item()

    def _getColormapRange(self):
        """Returns the range of the colormap for the current data.

        :return: Colormap range (min, max)
        """
        item = self.item()
        if item is not None and self._colormap is not None:
            return self._colormap.getColormapRange(item)
        else:
            return 1, 100  # Fallback

    def _modelUpdated(self, *args, **kwargs):
        """Emit dataChanged in the model"""
        topLeft = self.index(column=0)
        bottomRight = self.index(column=1)
        model = self.model()
        if model is not None:
            model.dataChanged.emit(topLeft, bottomRight)

    def _colormapChanged(self):
        self._sigColormapChanged.emit()

    def _itemChanged(self, event):
        """Handle change of colormap or data in the item.

        :param ItemChangedType event:
        """
        if event == items.ItemChangedType.COLORMAP:
            self._sigColormapChanged.emit()
            if self._colormap is not None:
                self._colormap.sigChanged.disconnect(self._colormapChanged)

            item = self.item()
            if item is not None:
                self._colormap = item.getColormap()
                self._colormap.sigChanged.connect(self._colormapChanged)
            else:
                self._colormap = None

        elif event == items.ItemChangedType.DATA:
            self._sigColormapChanged.emit()


class _ColormapBoundRow(_ColormapBaseProxyRow):
    """ProxyRow for colormap min or max

    :param ColormapMixIn item: The item to handle
    :param str name: Name of the raw
    :param int index: 0 for Min and 1 of Max
    """

    def __init__(self, item, name, index):
        self._index = index
        _ColormapBaseProxyRow.__init__(
            self,
            item,
            name=name,
            fget=self._getBound,
            fset=self._setBound)

        self.setToolTip('Colormap %s bound:\n'
                        'Check to set bound manually, '
                        'uncheck for autoscale' % name.lower())

    def _getRawBound(self):
        """Proxy to get raw colormap bound

        :rtype: float or None
        """
        if self._colormap is None:
            return None
        elif self._index == 0:
            return self._colormap.getVMin()
        else:  # self._index == 1
            return self._colormap.getVMax()

    def _getBound(self):
        """Proxy to get colormap effective bound value

        :rtype: float
        """
        if self._colormap is not None:
            bound = self._getRawBound()

            if bound is None:
                bound = self._getColormapRange()[self._index]
            return bound
        else:
            return 1.  # Fallback

    def _setBound(self, value):
        """Proxy to set colormap bound.

        :param float value:
        """
        if self._colormap is not None:
            if self._index == 0:
                min_ = value
                max_ = self._colormap.getVMax()
            else:  # self._index == 1
                min_ = self._colormap.getVMin()
                max_ = value

            if max_ is not None and min_ is not None and min_ > max_:
                min_, max_ = max_, min_
            self._colormap.setVRange(min_, max_)

    def flags(self, column):
        if column == 0:
            return qt.Qt.ItemIsEnabled | qt.Qt.ItemIsUserCheckable

        elif column == 1:
            if self._getRawBound() is not None:
                flags = qt.Qt.ItemIsEditable | qt.Qt.ItemIsEnabled
            else:
                flags = qt.Qt.NoItemFlags  # Disabled if autoscale
            return flags

        else:  # Never event
            return super(_ColormapBoundRow, self).flags(column)

    def data(self, column, role):
        if column == 0 and role == qt.Qt.CheckStateRole:
            if self._getRawBound() is None:
                return qt.Qt.Unchecked
            else:
                return qt.Qt.Checked

        else:
            return super(_ColormapBoundRow, self).data(column, role)

    def setData(self, column, value, role):
        if column == 0 and role == qt.Qt.CheckStateRole:
            if self._colormap is not None:
                bound = self._getBound() if value == qt.Qt.Checked else None
                self._setBound(bound)
                return True
            else:
                return False

        return super(_ColormapBoundRow, self).setData(column, value, role)


class _ColormapGammaRow(_ColormapBaseProxyRow):
    """ProxyRow for colormap gamma normalization parameter

    :param ColormapMixIn item: The item to handle
    :param str name: Name of the raw
    """

    def __init__(self, item):
        _ColormapBaseProxyRow.__init__(
            self,
            item,
            name="Gamma",
            fget=self._getGammaNormalizationParameter,
            fset=self._setGammaNormalizationParameter)

        self.setToolTip('Colormap gamma correction parameter:\n'
                        'Only meaningful for gamma normalization.')

    def _getGammaNormalizationParameter(self):
        """Proxy for :meth:`Colormap.getGammaNormalizationParameter`"""
        if self._colormap is not None:
            return self._colormap.getGammaNormalizationParameter()
        else:
            return 0.0

    def _setGammaNormalizationParameter(self, gamma):
        """Proxy for :meth:`Colormap.setGammaNormalizationParameter`"""
        if self._colormap is not None:
            return self._colormap.setGammaNormalizationParameter(gamma)

    def _getNormalization(self):
        """Proxy for :meth:`Colormap.getNormalization`"""
        if self._colormap is not None:
            return self._colormap.getNormalization()
        else:
            return ''

    def flags(self, column):
        if column in (0, 1):
            if self._getNormalization() == 'gamma':
                flags = qt.Qt.ItemIsEditable | qt.Qt.ItemIsEnabled
            else:
                flags = qt.Qt.NoItemFlags  # Disabled if not gamma correction
            return flags

        else:  # Never event
            return super(_ColormapGammaRow, self).flags(column)


class ColormapRow(_ColormapBaseProxyRow):
    """Represents :class:`ColormapMixIn` property.

    :param Item3D item: Scene item with colormap property
    """

    def __init__(self, item):
        super(ColormapRow, self).__init__(
            item,
            name='Colormap',
            fget=self._get)

        self._colormapImage = None

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

        self.addRow(_ColormapGammaRow(item))

        modes = [mode.title() for mode in self._colormap.AUTOSCALE_MODES]
        self.addRow(ProxyRow(
            name='Autoscale Mode',
            fget=self._getAutoscaleMode,
            fset=self._setAutoscaleMode,
            notify=self._sigColormapChanged,
            editorHint=modes))

        self.addRow(_ColormapBoundRow(item, name='Min.', index=0))
        self.addRow(_ColormapBoundRow(item, name='Max.', index=1))

        self._sigColormapChanged.connect(self._updateColormapImage)

    def getColormapImage(self):
        """Returns image representing the colormap or None

        :rtype: Union[QImage,None]
        """
        if self._colormapImage is None and self._colormap is not None:
            image = numpy.zeros((16, 130, 3), dtype=numpy.uint8)
            image[1:-1, 1:-1] = self._colormap.getNColors(image.shape[1] - 2)[:, :3]
            self._colormapImage = convertArrayToQImage(image)
        return self._colormapImage

    def _get(self):
        """Getter for ProxyRow subclass"""
        return None

    def _getName(self):
        """Proxy for :meth:`Colormap.getName`"""
        if self._colormap is not None and self._colormap.getName() is not None:
            return self._colormap.getName().title()
        else:
            return ''

    def _setName(self, name):
        """Proxy for :meth:`Colormap.setName`"""
        # Convert back from titled to name if possible
        if self._colormap is not None:
            name = self._colormapsMapping.get(name, name)
            self._colormap.setName(name)

    def _getNormalization(self):
        """Proxy for :meth:`Colormap.getNormalization`"""
        if self._colormap is not None:
            return self._colormap.getNormalization().title()
        else:
            return ''

    def _setNormalization(self, normalization):
        """Proxy for :meth:`Colormap.setNormalization`"""
        if self._colormap is not None:
            return self._colormap.setNormalization(normalization.lower())

    def _getAutoscaleMode(self):
        """Proxy for :meth:`Colormap.getAutoscaleMode`"""
        if self._colormap is not None:
            return self._colormap.getAutoscaleMode().title()
        else:
            return ''

    def _setAutoscaleMode(self, mode):
        """Proxy for :meth:`Colormap.setAutoscaleMode`"""
        if self._colormap is not None:
            return self._colormap.setAutoscaleMode(mode.lower())

    def _updateColormapImage(self, *args, **kwargs):
        """Notify colormap update to update the image in the tree"""
        if self._colormapImage is not None:
            self._colormapImage = None
            model = self.model()
            if model is not None:
                index = self.index(column=1)
                model.dataChanged.emit(index, index)

    def data(self, column, role):
        if column == 1 and role == qt.Qt.DecorationRole:
            return self.getColormapImage()
        else:
            return super(ColormapRow, self).data(column, role)


class SymbolRow(ItemProxyRow):
    """Represents :class:`SymbolMixIn` symbol property.

    :param Item3D item: Scene item with symbol property
    """

    def __init__(self, item):
        names = [item.getSymbolName(s) for s in item.getSupportedSymbols()]
        super(SymbolRow, self).__init__(
            item=item,
            name='Marker',
            fget=item.getSymbolName,
            fset=item.setSymbol,
            events=items.ItemChangedType.SYMBOL,
            editorHint=names)


class SymbolSizeRow(ItemProxyRow):
    """Represents :class:`SymbolMixIn` symbol size property.

    :param Item3D item: Scene item with symbol size property
    """

    def __init__(self, item):
        super(SymbolSizeRow, self).__init__(
            item=item,
            name='Marker size',
            fget=item.getSymbolSize,
            fset=item.setSymbolSize,
            events=items.ItemChangedType.SYMBOL_SIZE,
            editorHint=(1, 20))  # TODO link with OpenGL max point size


class PlaneEquationRow(ItemProxyRow):
    """Represents :class:`PlaneMixIn` as plane equation.

    :param Item3D item: Scene item with plane equation property
    """

    def __init__(self, item):
        super(PlaneEquationRow, self).__init__(
            item=item,
            name='Equation',
            fget=item.getParameters,
            fset=item.setParameters,
            events=items.ItemChangedType.POSITION,
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
        return super(PlaneEquationRow, self).data(column, role)


class PlaneRow(ItemProxyRow):
    """Represents :class:`PlaneMixIn` property.

    :param Item3D item: Scene item with plane equation property
    """

    _PLANES = OrderedDict((('Plane 0', (1., 0., 0.)),
                           ('Plane 1', (0., 1., 0.)),
                           ('Plane 2', (0., 0., 1.)),
                           ('-', None)))
    """Mapping of plane names to normals"""

    _PLANE_ICONS = {'Plane 0': '3d-plane-normal-x',
                    'Plane 1': '3d-plane-normal-y',
                    'Plane 2': '3d-plane-normal-z',
                    '-': '3d-plane'}
    """Mapping of plane names to normals"""

    def __init__(self, item):
        super(PlaneRow, self).__init__(
            item=item,
            name='Plane',
            fget=self.__getPlaneName,
            fset=self.__setPlaneName,
            events=items.ItemChangedType.POSITION,
            editorHint=tuple(self._PLANES.keys()))
        self._item = weakref.ref(item)
        self._lastName = None

        self.addRow(PlaneEquationRow(item))

    def _notified(self, *args, **kwargs):
        """Handle notification of modification

        Here only send if plane name actually changed
        """
        if self._lastName != self.__getPlaneName():
            super(PlaneRow, self)._notified()

    def __getPlaneName(self):
        """Returns name of plane // to axes or '-'

        :rtype: str
        """
        item = self._item()
        planeNormal = item.getNormal() if item is not None else None

        for name, normal in self._PLANES.items():
            if numpy.array_equal(planeNormal, normal):
                return name
        return '-'

    def __setPlaneName(self, data):
        """Set plane normal according to given plane name

        :param str data: Selected plane name
        """
        item = self._item()
        if item is not None:
            for name, normal in self._PLANES.items():
                if data == name and normal is not None:
                    item.setNormal(normal)

    def data(self, column, role):
        if column == 1 and role == qt.Qt.DecorationRole:
            return icons.getQIcon(self._PLANE_ICONS[self.__getPlaneName()])
        data = super(PlaneRow, self).data(column, role)
        if column == 1 and role == qt.Qt.DisplayRole:
            self._lastName = data
        return data


class ComplexModeRow(ItemProxyRow):
    """Represents :class:`items.ComplexMixIn` symbol property.

    :param Item3D item: Scene item with symbol property
    """

    def __init__(self, item, name='Mode'):
        names = [m.value.replace('_', ' ').title()
                 for m in item.supportedComplexModes()]
        super(ComplexModeRow, self).__init__(
            item=item,
            name=name,
            fget=item.getComplexMode,
            fset=item.setComplexMode,
            events=items.ItemChangedType.COMPLEX_MODE,
            toModelData=lambda data: data.value.replace('_', ' ').title(),
            fromModelData=lambda data: data.lower().replace(' ', '_'),
            editorHint=names)


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
            volume = isosurface.parent()
            if volume is not None:
                volume.removeIsosurface(isosurface)


class IsosurfaceRow(Item3DRow):
    """Represents an :class:`Isosurface` item.

    :param Isosurface item: Isosurface item
    """

    _LEVEL_SLIDER_RANGE = 0, 1000
    """Range given as editor hint"""

    _EVENTS = items.ItemChangedType.VISIBLE, items.ItemChangedType.COLOR
    """Events for which to update the first column in the tree"""

    def __init__(self, item):
        super(IsosurfaceRow, self).__init__(item, name=item.getLevel())

        self.setFlags(self.flags(1) | qt.Qt.ItemIsEditable, 1)

        item.sigItemChanged.connect(self._levelChanged)

        self.addRow(ItemProxyRow(
            item=item,
            name='Level',
            fget=self._getValueForLevelSlider,
            fset=self._setLevelFromSliderValue,
            events=items.Item3DChangedType.ISO_LEVEL,
            editorHint=self._LEVEL_SLIDER_RANGE))

        self.addRow(ItemColorProxyRow(
            item=item,
            name='Color',
            fget=self._rgbColor,
            fset=self._setRgbColor,
            events=items.ItemChangedType.COLOR))

        self.addRow(ItemProxyRow(
            item=item,
            name='Opacity',
            fget=self._opacity,
            fset=self._setOpacity,
            events=items.ItemChangedType.COLOR,
            editorHint=(0, 255)))

        self.addRow(RemoveIsosurfaceRow(item))

    def _getValueForLevelSlider(self):
        """Convert iso level to slider value.

        :rtype: int
        """
        item = self.item()
        if item is not None:
            volume = item.parent()
            if volume is not None:
                dataRange = volume.getDataRange()
                if dataRange is not None:
                    dataMin, dataMax = dataRange[0], dataRange[-1]
                    if dataMax != dataMin:
                        offset = (item.getLevel() - dataMin) / (dataMax - dataMin)
                    else:
                        offset = 0.

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
            volume = item.parent()
            if volume is not None:
                dataRange = volume.getDataRange()
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


class ComplexIsosurfaceRow(IsosurfaceRow):
    """Represents an :class:`ComplexIsosurface` item.

    :param ComplexIsosurface item:
    """

    _EVENTS = (items.ItemChangedType.VISIBLE,
               items.ItemChangedType.COLOR,
               items.ItemChangedType.COMPLEX_MODE)
    """Events for which to update the first column in the tree"""

    def __init__(self, item):
        super(ComplexIsosurfaceRow, self).__init__(item)

        self.addRow(ComplexModeRow(item, "Color Complex Mode"), index=1)
        for row in self.children():
            if isinstance(row, ColorProxyRow):
                self._colorRow = row
                break
        else:
            raise RuntimeError("Cannot retrieve Color tree row")
        self._colormapRow = ColormapRow(item)

        self.__updateRowsForItem(item)
        item.sigItemChanged.connect(self.__itemChanged)

    def __itemChanged(self, event):
        """Update enabled/disabled rows"""
        if event == items.ItemChangedType.COMPLEX_MODE:
            item = self.sender()
            self.__updateRowsForItem(item)

    def __updateRowsForItem(self, item):
        """Update rows for item

        :param item:
        """
        if not isinstance(item, ComplexIsosurface):
            return

        if item.getComplexMode() == items.ComplexMixIn.ComplexMode.NONE:
            removed = self._colormapRow
            added = self._colorRow
        else:
            removed = self._colorRow
            added = self._colormapRow

        # Remove unwanted rows
        if removed in self.children():
            self.removeRow(removed)

        # Add required rows
        if added not in self.children():
            self.addRow(added, index=2)

    def data(self, column, role):
        if column == 0 and role == qt.Qt.DecorationRole:
            item = self.item()
            if (item is not None and
                    item.getComplexMode() != items.ComplexMixIn.ComplexMode.NONE):
                return self._colormapRow.getColormapImage()

        return super(ComplexIsosurfaceRow, self).data(column, role)


class AddIsosurfaceRow(BaseRow):
    """Class for Isosurface create button

    :param Union[ScalarField3D,ComplexField3D] volume:
        The volume item to attach the button to.
    """

    def __init__(self, volume):
        super(AddIsosurfaceRow, self).__init__()
        self._volume = weakref.ref(volume)

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

    def volume(self):
        """Returns the controlled volume item

        :rtype: Union[ScalarField3D,ComplexField3D]
        """
        return self._volume()

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
        volume = self.volume()
        if volume is not None:
            dataRange = volume.getDataRange()
            if dataRange is None:
                dataRange = 0., 1.

            volume.addIsosurface(
                numpy.mean((dataRange[0], dataRange[-1])),
                '#0000FF')


class VolumeIsoSurfacesRow(StaticRow):
    """Represents  :class:`ScalarFieldView`'s isosurfaces

    :param Union[ScalarField3D,ComplexField3D] volume:
        Volume item to control
    """

    def __init__(self, volume):
        super(VolumeIsoSurfacesRow, self).__init__(
            ('Isosurfaces', None))
        self._volume = weakref.ref(volume)

        volume.sigIsosurfaceAdded.connect(self._isosurfaceAdded)
        volume.sigIsosurfaceRemoved.connect(self._isosurfaceRemoved)

        if isinstance(volume, items.ComplexMixIn):
            self.addRow(ComplexModeRow(volume, "Complex Mode"))

        for item in volume.getIsosurfaces():
            self.addRow(nodeFromItem(item))

        self.addRow(AddIsosurfaceRow(volume))

    def volume(self):
        """Returns the controlled volume item

        :rtype: Union[ScalarField3D,ComplexField3D]
        """
        return self._volume()

    def _isosurfaceAdded(self, item):
        """Handle isosurface addition

        :param Isosurface item: added isosurface
        """
        volume = self.volume()
        if volume is None:
            return

        row = volume.getIsosurfaces().index(item)
        if isinstance(volume, items.ComplexMixIn):
            row += 1  # Offset for the ComplexModeRow
        self.addRow(nodeFromItem(item), row)

    def _isosurfaceRemoved(self, item):
        """Handle isosurface removal

        :param Isosurface item: removed isosurface
        """
        volume = self.volume()
        if volume is None:
            return

        # Find item
        for row in self.children():
            if isinstance(row, IsosurfaceRow) and row.item() is item:
                self.removeRow(row)
                break  # Got it
        else:
            raise RuntimeError("Model does not correspond to scene content")


class Scatter2DPropertyMixInRow(object):
    """Mix-in class that enable/disable row according to Scatter2D mode.

    :param Scatter2D item:
    :param str propertyName: Name of the Scatter2D property of this row
    """

    def __init__(self, item, propertyName):
        assert propertyName in ('lineWidth', 'symbol', 'symbolSize')
        self.__propertyName = propertyName

        self.__isEnabled = item.isPropertyEnabled(propertyName)
        self.__updateFlags()

        item.sigItemChanged.connect(self.__itemChanged)

    def data(self, column, role):
        if column == 1 and not self.__isEnabled:
            # Discard data and editorHint if disabled
            return None
        else:
            return super(Scatter2DPropertyMixInRow, self).data(column, role)

    def __updateFlags(self):
        """Update model flags"""
        if self.__isEnabled:
            self.setFlags(qt.Qt.ItemIsEnabled, 0)
            self.setFlags(qt.Qt.ItemIsEnabled | qt.Qt.ItemIsEditable, 1)
        else:
            self.setFlags(qt.Qt.NoItemFlags)

    def __itemChanged(self, event):
        """Set flags to enable/disable the row"""
        if event == items.ItemChangedType.VISUALIZATION_MODE:
            item = self.sender()
            if item is not None:  # This occurs with PySide/python2.7
                self.__isEnabled = item.isPropertyEnabled(self.__propertyName)
                self.__updateFlags()

            # Notify model
            model = self.model()
            if model is not None:
                begin = self.index(column=0)
                end = self.index(column=1)
                model.dataChanged.emit(begin, end)


class Scatter2DSymbolRow(Scatter2DPropertyMixInRow, SymbolRow):
    """Specific class for Scatter2D symbol.

    It is enabled/disabled according to visualization mode.

    :param Scatter2D item:
    """

    def __init__(self, item):
        SymbolRow.__init__(self, item)
        Scatter2DPropertyMixInRow.__init__(self, item, 'symbol')


class Scatter2DSymbolSizeRow(Scatter2DPropertyMixInRow, SymbolSizeRow):
    """Specific class for Scatter2D symbol size.

    It is enabled/disabled according to visualization mode.

    :param Scatter2D item:
    """

    def __init__(self, item):
        SymbolSizeRow.__init__(self, item)
        Scatter2DPropertyMixInRow.__init__(self, item, 'symbolSize')


class Scatter2DLineWidth(Scatter2DPropertyMixInRow, ItemProxyRow):
    """Specific class for Scatter2D symbol size.

    It is enabled/disabled according to visualization mode.

    :param Scatter2D item:
    """

    def __init__(self, item):
        # TODO link editorHint with OpenGL max line width
        ItemProxyRow.__init__(self,
                              item=item,
                              name='Line width',
                              fget=item.getLineWidth,
                              fset=item.setLineWidth,
                              events=items.ItemChangedType.LINE_WIDTH,
                              editorHint=(1, 10))
        Scatter2DPropertyMixInRow.__init__(self, item, 'lineWidth')


def initScatter2DNode(node, item):
    """Specific node init for Scatter2D to set order of parameters

    :param Item3DRow node: The model node to setup
    :param Scatter2D item: The Scatter2D the node is representing
    """
    node.addRow(ItemProxyRow(
        item=item,
        name='Mode',
        fget=item.getVisualization,
        fset=item.setVisualization,
        events=items.ItemChangedType.VISUALIZATION_MODE,
        editorHint=[m.value.title() for m in item.supportedVisualizations()],
        toModelData=lambda data: data.value.title(),
        fromModelData=lambda data: data.lower()))

    node.addRow(ItemProxyRow(
        item=item,
        name='Height map',
        fget=item.isHeightMap,
        fset=item.setHeightMap,
        events=items.Item3DChangedType.HEIGHT_MAP))

    node.addRow(ColormapRow(item))

    node.addRow(Scatter2DSymbolRow(item))
    node.addRow(Scatter2DSymbolSizeRow(item))

    node.addRow(Scatter2DLineWidth(item))


def initVolumeNode(node, item):
    """Specific node init for volume items

    :param Item3DRow node: The model node to setup
    :param Union[ScalarField3D,ComplexField3D] item:
        The volume item represented by the node
    """
    node.addRow(nodeFromItem(item.getCutPlanes()[0]))  # Add cut plane
    node.addRow(VolumeIsoSurfacesRow(item))


def initVolumeCutPlaneNode(node, item):
    """Specific node init for volume CutPlane

    :param Item3DRow node: The model node to setup
    :param CutPlane item: The CutPlane the node is representing
    """
    if isinstance(item, items.ComplexMixIn):
        node.addRow(ComplexModeRow(item))

    node.addRow(PlaneRow(item))

    node.addRow(ColormapRow(item))

    node.addRow(ItemProxyRow(
        item=item,
        name='Show <=Min',
        fget=item.getDisplayValuesBelowMin,
        fset=item.setDisplayValuesBelowMin,
        events=items.ItemChangedType.ALPHA))

    node.addRow(InterpolationRow(item))


NODE_SPECIFIC_INIT = [  # class, init(node, item)
    (items.Scatter2D, initScatter2DNode),
    (items.ScalarField3D, initVolumeNode),
    (CutPlane, initVolumeCutPlaneNode),
]
"""List of specific node init for different item class"""


def nodeFromItem(item):
    """Create :class:`Item3DRow` subclass corresponding to item

    :param Item3D item: The item fow which to create the node
    :rtype: Item3DRow
    """
    assert isinstance(item, items.Item3D)

    # Item with specific model row class
    if isinstance(item, (items.GroupItem, items.GroupWithAxesItem)):
        return GroupItemRow(item)
    elif isinstance(item, ComplexIsosurface):
        return ComplexIsosurfaceRow(item)
    elif isinstance(item, Isosurface):
        return IsosurfaceRow(item)

    # Create Item3DRow and populate it
    node = Item3DRow(item)

    if isinstance(item, items.DataItem3D):
        node.addRow(DataItem3DBoundingBoxRow(item))
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
