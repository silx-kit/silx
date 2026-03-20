"""pygfx-based SceneWidget, replacement for SceneWidget."""

import logging

import numpy

from .. import qt
from ..colors import rgba

from .Plot3DWidgetPygfx import Plot3DWidgetPygfx
from . import items
from .items._pygfx_sync import sync_item

_logger = logging.getLogger(__name__)


class _StubSelection(qt.QObject):
    """Stub selection for pygfx backend."""

    sigCurrentChanged = qt.Signal(object, object)

    def getCurrentItem(self):
        return None

    def setCurrentItem(self, item):
        pass

    def _setSyncSelectionModel(self, model):
        pass


class SceneWidgetPygfx(Plot3DWidgetPygfx):
    """Widget displaying data sets in 3D using pygfx backend.

    Provides the same public API as SceneWidget for item management.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pygfxObjects = {}  # item -> pygfx WorldObject mapping

        self._textColor = (1.0, 1.0, 1.0, 1.0)
        self._foregroundColor = (1.0, 1.0, 1.0, 1.0)
        self._highlightColor = (0.7, 0.7, 0.0, 1.0)

        self._selection = _StubSelection(self)

        # Item management via GroupItem (reuses SceneModel infrastructure)
        self._sceneGroup = items.GroupItem()
        self._sceneGroup.setLabel("Data")
        self._sceneGroup.sigItemAdded.connect(self._onGroupItemAdded)
        self._sceneGroup.sigItemRemoved.connect(self._onGroupItemRemoved)
        self._model = None

        # Axes and bounding box
        gfx = self._gfx
        self._axesGroup = gfx.Group()
        self._scene.add(self._axesGroup)
        self._rulers = None  # (ruler_x, ruler_y, ruler_z)
        self._bboxLine = None

    # --- Item management ---

    def addItem(self, item, index=None):
        """Add an Item3D to the scene.

        :param Item3D item: The item to add
        :param int index: Index at which to place the item (default: end)
        """
        self._sceneGroup.addItem(item, index)
        # Auto-fit camera to scene after adding items
        self._camera.show_object(self._scene)

    def removeItem(self, item):
        """Remove an Item3D from the scene.

        :param Item3D item: The item to remove
        """
        self._sceneGroup.removeItem(item)

    def getItems(self):
        """Return the list of items in the scene.

        :rtype: tuple
        """
        return self._sceneGroup.getItems()

    def clearItems(self):
        """Remove all items from the scene."""
        self._sceneGroup.clearItems()

    # --- visit (for GroupPropertiesWidget compatibility) ---

    def visit(self, included=True):
        """Generator visiting scene items recursively.

        :param bool included: Whether to include self
        """
        if included:
            yield self
        for item in self._sceneGroup.getItems():
            if hasattr(item, "visit"):
                yield from item.visit(included=True)
            else:
                yield item

    # --- Convenience add methods ---

    def addVolume(self, data, copy=True, index=None):
        """Add 3D data volume to the scene.

        :param data: 3D array (zyx order)
        :param bool copy: Whether to copy the data
        :param int index: Position index
        :returns: ScalarField3D or ComplexField3D
        """
        if data is not None:
            data = numpy.asarray(data)

        if numpy.iscomplexobj(data):
            volume = items.ComplexField3D()
        else:
            volume = items.ScalarField3D()
        volume.setData(data, copy=copy)
        self.addItem(volume, index)
        return volume

    def add3DScatter(self, x, y, z, value, copy=True, index=None):
        """Add 3D scatter data to the scene.

        :returns: Scatter3D item
        """
        scatter3d = items.Scatter3D()
        scatter3d.setData(x=x, y=y, z=z, value=value, copy=copy)
        self.addItem(scatter3d, index)
        return scatter3d

    def add2DScatter(self, x, y, value, copy=True, index=None):
        """Add 2D scatter data to the scene.

        :returns: Scatter2D item
        """
        scatter2d = items.Scatter2D()
        scatter2d.setData(x=x, y=y, value=value, copy=copy)
        self.addItem(scatter2d, index)
        return scatter2d

    def addImage(self, data, copy=True, index=None):
        """Add 2D image or RGBA image to the scene.

        :returns: ImageData or ImageRgba
        """
        data = numpy.asarray(data)
        if data.ndim == 2:
            image = items.ImageData()
        elif data.ndim == 3:
            image = items.ImageRgba()
        else:
            raise ValueError("Unsupported array dimensions: %d" % data.ndim)
        image.setData(data, copy=copy)
        self.addItem(image, index)
        return image

    # --- Colors ---

    def getTextColor(self):
        return qt.QColor.fromRgbF(*self._textColor)

    def setTextColor(self, color):
        color = rgba(color)
        if color != self._textColor:
            self._textColor = color
            self.sigStyleChanged.emit("textColor")

    def getForegroundColor(self):
        return qt.QColor.fromRgbF(*self._foregroundColor)

    def setForegroundColor(self, color):
        color = rgba(color)
        if color != self._foregroundColor:
            self._foregroundColor = color
            self.sigStyleChanged.emit("foregroundColor")

    def getHighlightColor(self):
        return qt.QColor.fromRgbF(*self._highlightColor)

    def setHighlightColor(self, color):
        color = rgba(color)
        if color != self._highlightColor:
            self._highlightColor = color
            self.sigStyleChanged.emit("highlightColor")

    # --- Scene group ---

    def getSceneGroup(self):
        """Return the GroupItem managing scene items.

        :rtype: GroupItem
        """
        return self._sceneGroup

    # --- Picking (stub) ---

    def pickItems(self, x, y, condition=None):
        """Stub for picking - not yet implemented for pygfx backend."""
        return iter([])

    # --- Selection (stub) ---

    def selection(self):
        """Return a stub selection object."""
        return self._selection

    def model(self):
        """Return the SceneModel for the parameter tree.

        :rtype: SceneModel
        """
        if self._model is None:
            from ._model.model import SceneModel

            self._model = SceneModel(parent=self)
        return self._model

    # --- Internal sync ---

    def _onGroupItemAdded(self, item):
        """Handle item added to GroupItem."""
        item.sigItemChanged.connect(self._onItemChanged)
        self._syncItem(item)
        self._updateAxesAndBBox()

    def _onGroupItemRemoved(self, item):
        """Handle item removed from GroupItem."""
        item.sigItemChanged.disconnect(self._onItemChanged)
        self._unsyncItem(item)
        self._updateAxesAndBBox()

    def _syncItem(self, item):
        """Synchronize an Item3D to pygfx scene objects."""
        self._unsyncItem(item)

        obj = sync_item(item)
        if obj is not None:
            self._pygfxObjects[id(item)] = obj
            self._dataGroup.add(obj)

    def _unsyncItem(self, item):
        """Remove pygfx objects for an item."""
        key = id(item)
        if key in self._pygfxObjects:
            obj = self._pygfxObjects.pop(key)
            try:
                self._dataGroup.remove(obj)
            except ValueError:
                pass

    def _onItemChanged(self, event):
        """Handle item property changes by re-syncing."""
        item = self.sender()
        if item in self._sceneGroup.getItems():
            self._syncItem(item)
            self._updateAxesAndBBox()

    def _resyncAll(self):
        """Re-synchronize all items."""
        for obj in list(self._pygfxObjects.values()):
            try:
                self._dataGroup.remove(obj)
            except ValueError:
                pass
        self._pygfxObjects.clear()

        for item in self._sceneGroup.getItems():
            obj = sync_item(item)
            if obj is not None:
                self._pygfxObjects[id(item)] = obj
                self._dataGroup.add(obj)

    # --- 3D Axes and Bounding Box ---

    def _getDataBounds(self):
        """Get bounding box of all data in the scene.

        :returns: (min, max) arrays or None if no data
        """
        try:
            bbox = self._dataGroup.get_world_bounding_box()
            if bbox is not None:
                mn = numpy.array(bbox[0], dtype=numpy.float64)
                mx = numpy.array(bbox[1], dtype=numpy.float64)
                if numpy.all(numpy.isfinite(mn)) and numpy.all(numpy.isfinite(mx)):
                    return mn, mx
        except Exception:
            pass
        return None

    def _updateAxesAndBBox(self):
        """Update 3D axes rulers and bounding box wireframe."""
        gfx = self._gfx
        bounds = self._getDataBounds()

        if bounds is None:
            # Remove existing axes/bbox
            if self._rulers is not None:
                for ruler in self._rulers:
                    self._axesGroup.remove(ruler)
                self._rulers = None
            if self._bboxLine is not None:
                self._axesGroup.remove(self._bboxLine)
                self._bboxLine = None
            return

        mn, mx = bounds

        # Rulers: X (red), Y (green), Z (blue) at bbox edges, labels facing outward
        if self._rulers is None:
            self._rulers = (
                gfx.Ruler(
                    start_pos=tuple(mn),
                    end_pos=(mx[0], mn[1], mn[2]),
                    start_value=mn[0],
                    tick_side="right",
                    color=(1, 0, 0, 1),
                    line_width=2,
                ),
                gfx.Ruler(
                    start_pos=tuple(mn),
                    end_pos=(mn[0], mx[1], mn[2]),
                    start_value=mn[1],
                    tick_side="left",
                    color=(0, 1, 0, 1),
                    line_width=2,
                ),
                gfx.Ruler(
                    start_pos=tuple(mn),
                    end_pos=(mn[0], mn[1], mx[2]),
                    start_value=mn[2],
                    tick_side="right",
                    color=(0, 0, 1, 1),
                    line_width=2,
                ),
            )
            for ruler in self._rulers:
                self._axesGroup.add(ruler)
        else:
            self._rulers[0].start_pos = tuple(mn)
            self._rulers[0].end_pos = (mx[0], mn[1], mn[2])
            self._rulers[0].start_value = mn[0]
            self._rulers[1].start_pos = tuple(mn)
            self._rulers[1].end_pos = (mn[0], mx[1], mn[2])
            self._rulers[1].start_value = mn[1]
            self._rulers[2].start_pos = tuple(mn)
            self._rulers[2].end_pos = (mn[0], mn[1], mx[2])
            self._rulers[2].start_value = mn[2]

        # Bounding box wireframe (12 edges)
        corners = numpy.array(
            [
                [mn[0], mn[1], mn[2]],
                [mx[0], mn[1], mn[2]],
                [mx[0], mx[1], mn[2]],
                [mn[0], mx[1], mn[2]],
                [mn[0], mn[1], mx[2]],
                [mx[0], mn[1], mx[2]],
                [mx[0], mx[1], mx[2]],
                [mn[0], mx[1], mx[2]],
            ],
            dtype=numpy.float32,
        )
        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),  # bottom face
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),  # top face
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),  # vertical edges
        ]
        positions = numpy.array(
            [corners[i] for edge in edges for i in edge],
            dtype=numpy.float32,
        )

        if self._bboxLine is not None:
            self._axesGroup.remove(self._bboxLine)

        self._bboxLine = gfx.Line(
            gfx.Geometry(positions=positions),
            gfx.LineSegmentMaterial(color=(0.6, 0.6, 0.6, 0.5), thickness=1),
        )
        self._axesGroup.add(self._bboxLine)

    def _animate(self):
        """Render callback with ruler updates."""
        if self._rulers:
            size = self._renderWidget.get_logical_size()
            for ruler in self._rulers:
                ruler.update(self._camera, size)
        super()._animate()
