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
"""
This module provides base classes to implement models for 3D scene content
"""

from __future__ import absolute_import, division

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "24/04/2018"


import functools
import logging
import weakref

import numpy

from silx.third_party import six

from ...utils.image import convertArrayToQImage
from ...colors import preferredColormaps
from ... import qt, icons
from .. import items
from ..items.volume import Isosurface, CutPlane


from .core import AngleDegreeRow, BaseRow, ColorProxyRow, ProxyRow, StaticRow


_logger = logging.getLogger(__name__)


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

        # Settings row
        children = (background, foreground, text, highlight,
                    axesIndicator, lightDirection)
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
            self.flags(0) | qt.Qt.ItemIsUserCheckable | qt.Qt.ItemIsSelectable,
            0)
        self.setFlags(self.flags(1) | qt.Qt.ItemIsSelectable, 1)

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


class DataItem3DBoundingBoxRow(ProxyRow):
    """Represents :class:`DataItem3D` bounding box visibility

    :param DataItem3D item: The item for which to display/control bounding box
    """

    def __init__(self, item):
        super(DataItem3DBoundingBoxRow, self).__init__(
            name='Bounding box',
            fget=item.isBoundingBoxVisible,
            fset=item.setBoundingBoxVisible,
            notify=item.sigItemChanged)


class MatrixProxyRow(ProxyRow):
    """Proxy for a row of a DataItem3D 3x3 matrix transform

    :param DataItem3D item:
    :param int index: Matrix row index
    """

    def __init__(self, item, index):
        self._item = weakref.ref(item)
        self._index = index

        super(MatrixProxyRow, self).__init__(
            name='',
            fget=self._getMatrixRow,
            fset=self._setMatrixRow,
            notify=item.sigItemChanged)

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

        translation = ProxyRow(name='Translation',
                               fget=item.getTranslation,
                               fset=self._setTranslation,
                               notify=item.sigItemChanged,
                               toModelData=lambda data: qt.QVector3D(*data))
        self.addRow(translation)

        # Here to keep a reference
        self._xSetCenter = functools.partial(self._setCenter, index=0)
        self._ySetCenter = functools.partial(self._setCenter, index=1)
        self._zSetCenter = functools.partial(self._setCenter, index=2)

        rotateCenter = StaticRow(
            ('Center', None),
            children=(
                ProxyRow(name='X axis',
                         fget=item.getRotationCenter,
                         fset=self._xSetCenter,
                         notify=item.sigItemChanged,
                         toModelData=functools.partial(
                             self._centerToModelData, index=0),
                         editorHint=self._ROTATION_CENTER_OPTIONS),
                ProxyRow(name='Y axis',
                         fget=item.getRotationCenter,
                         fset=self._ySetCenter,
                         notify=item.sigItemChanged,
                         toModelData=functools.partial(
                             self._centerToModelData, index=1),
                         editorHint=self._ROTATION_CENTER_OPTIONS),
                ProxyRow(name='Z axis',
                         fget=item.getRotationCenter,
                         fset=self._zSetCenter,
                         notify=item.sigItemChanged,
                         toModelData=functools.partial(
                             self._centerToModelData, index=2),
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


class _ColormapBaseProxyRow(ProxyRow):
    """Base class for colormap model row

    This class handle synchronization and signals from the item and the colormap
    """

    _sigColormapChanged = qt.Signal()
    """Signal used internally to notify colormap (or data) update"""

    def __init__(self, item, *args, **kwargs):
        self._dataRange = None
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
        if self._dataRange is None:
            item = self.item()
            if item is not None and self._colormap is not None:
                if hasattr(item, 'getDataRange'):
                    data = item.getDataRange()
                else:
                    data = item.getData(copy=False)

                self._dataRange = self._colormap.getColormapRange(data)

            else:  # Fallback
                self._dataRange = 1, 100
        return self._dataRange

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
            self._dataRange = None
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

        self.addRow(_ColormapBoundRow(item, name='Min.', index=0))
        self.addRow(_ColormapBoundRow(item, name='Max.', index=1))

        self._sigColormapChanged.connect(self._updateColormapImage)

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
            if self._colormapImage is None:
                image = numpy.zeros((16, 130, 3), dtype=numpy.uint8)
                image[1:-1, 1:-1] = self._colormap.getNColors(image.shape[1] - 2)[:, :3]
                self._colormapImage = convertArrayToQImage(image)
            return self._colormapImage

        return super(ColormapRow, self).data(column, role)


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
            editorHint=(1, 20))  # TODO link with OpenGL max point size


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


class Scatter2DLineWidth(Scatter2DPropertyMixInRow, ProxyRow):
    """Specific class for Scatter2D symbol size.

    It is enabled/disabled according to visualization mode.

    :param Scatter2D item:
    """

    def __init__(self, item):
        # TODO link editorHint with OpenGL max line width
        ProxyRow.__init__(self,
                          name='Line width',
                          fget=item.getLineWidth,
                          fset=item.setLineWidth,
                          notify=item.sigItemChanged,
                          editorHint=(1, 10))
        Scatter2DPropertyMixInRow.__init__(self, item, 'lineWidth')


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

    node.addRow(Scatter2DSymbolRow(item))
    node.addRow(Scatter2DSymbolSizeRow(item))

    node.addRow(Scatter2DLineWidth(item))


def initScalarField3DNode(node, item):
    """Specific node init for ScalarField3D

    :param Item3DRow node: The model node to setup
    :param ScalarField3D item: The ScalarField3D the node is representing
    """
    node.addRow(nodeFromItem(item.getCutPlanes()[0]))  # Add cut plane
    node.addRow(ScalarField3DIsoSurfacesRow(item))


def initScalarField3DCutPlaneNode(node, item):
    """Specific node init for ScalarField3D CutPlane

    :param Item3DRow node: The model node to setup
    :param CutPlane item: The CutPlane the node is representing
    """
    node.addRow(PlaneRow(item))

    node.addRow(ColormapRow(item))

    node.addRow(ProxyRow(
        name='Values<=Min',
        fget=item.getDisplayValuesBelowMin,
        fset=item.setDisplayValuesBelowMin,
        notify=item.sigItemChanged))

    node.addRow(InterpolationRow(item))


NODE_SPECIFIC_INIT = [  # class, init(node, item)
    (items.Scatter2D, initScatter2DNode),
    (items.ScalarField3D, initScalarField3DNode),
    (CutPlane, initScalarField3DCutPlaneNode),
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
