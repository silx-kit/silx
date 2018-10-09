# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2018 European Synchrotron Radiation Facility
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
"""This module provides a widget that displays data values of a SceneWidget.
"""

from __future__ import absolute_import

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "01/10/2018"


import logging
import weakref

from ... import qt
from .. import items
from ..items import volume
from ..SceneWidget import SceneWidget


_logger = logging.getLogger(__name__)


class PositionInfoWidget(qt.QWidget):
    """Widget displaying information about picked position

    :param QWidget parent: See :class:`QWidget`
    """

    def __init__(self, parent=None):
        super(PositionInfoWidget, self).__init__(parent)
        self._sceneWidgetRef = None

        self.setToolTip("Double-click on a data point to show its value")
        layout = qt.QBoxLayout(qt.QBoxLayout.LeftToRight, self)

        self._xLabel = self._addInfoField('X')
        self._yLabel = self._addInfoField('Y')
        self._zLabel = self._addInfoField('Z')
        self._dataLabel = self._addInfoField('Data')
        self._itemLabel = self._addInfoField('Item')

        layout.addStretch(1)

    def _addInfoField(self, label):
        """Add a description: info widget to this widget

        :param str label: Description label
        :return: The QLabel used to display the info
        :rtype: QLabel
        """
        subLayout = qt.QHBoxLayout()
        subLayout.setContentsMargins(0, 0, 0, 0)

        subLayout.addWidget(qt.QLabel(label + ':'))

        widget = qt.QLabel('-')
        widget.setAlignment(qt.Qt.AlignLeft | qt.Qt.AlignVCenter)
        widget.setTextInteractionFlags(qt.Qt.TextSelectableByMouse)
        widget.setMinimumWidth(widget.fontMetrics().width('#######'))
        subLayout.addWidget(widget)

        subLayout.addStretch(1)

        layout = self.layout()
        layout.addLayout(subLayout)
        return widget

    def getSceneWidget(self):
        """Returns the associated :class:`SceneWidget` or None.

        :rtype: Union[None,~silx.gui.plot3d.SceneWidget.SceneWidget]
        """
        if self._sceneWidgetRef is None:
            return None
        else:
            return self._sceneWidgetRef()

    def setSceneWidget(self, widget):
        """Set the associated :class:`SceneWidget`

        :param ~silx.gui.plot3d.SceneWidget.SceneWidget widget:
            3D scene for which to display information
        """
        if widget is not None and not isinstance(widget, SceneWidget):
            raise ValueError("widget must be a SceneWidget or None")

        previous = self.getSceneWidget()
        if previous is not None:
            previous.removeEventFilter(self)

        if widget is None:
            self._sceneWidgetRef = None
        else:
            widget.installEventFilter(self)
            self._sceneWidgetRef = weakref.ref(widget)

    def eventFilter(self, watched, event):
        # Filter events of SceneWidget to react on mouse events.
        if (event.type() == qt.QEvent.MouseButtonDblClick and
                event.button() == qt.Qt.LeftButton):
            self.pick(event.x(), event.y())

        return super(PositionInfoWidget, self).eventFilter(watched, event)

    def clear(self):
        """Clean-up displayed values"""
        for widget in (self._xLabel, self._yLabel, self._zLabel,
                       self._dataLabel, self._itemLabel):
            widget.setText('-')

    _SUPPORTED_ITEMS = (items.Scatter3D,
                        items.Scatter2D,
                        items.ImageData,
                        items.ImageRgba,
                        items.Mesh,
                        items.Box,
                        items.Cylinder,
                        items.Hexagon,
                        volume.CutPlane,
                        volume.Isosurface)
    """Type of items that are picked"""

    def _isSupportedItem(self, item):
        """Returns True if item is of supported type

        :param Item3D item: The Item3D to check
        :rtype: bool
        """
        return isinstance(item, self._SUPPORTED_ITEMS)

    def pick(self, x, y):
        """Pick items in the associated SceneWidget and display result

        Only the closest point is displayed.

        :param int x: X coordinate in pixel in the SceneWidget
        :param int y: Y coordinate in pixel in the SceneWidget
        """
        self.clear()

        sceneWidget = self.getSceneWidget()
        if sceneWidget is None:  # No associated widget
            _logger.info('Picking without associated SceneWidget')
            return

        # Find closest (and latest in the tree) supported item
        closestNdcZ = float('inf')
        picking = None
        for result in sceneWidget.pickItems(x, y,
                                            condition=self._isSupportedItem):
            ndcZ = result.getPositions('ndc', copy=False)[0, 2]
            if ndcZ <= closestNdcZ:
                closestNdcZ = ndcZ
                picking = result

        if picking is None:
            return  # No picked item

        item = picking.getItem()
        self._itemLabel.setText(item.getLabel())
        positions = picking.getPositions('scene', copy=False)
        x, y, z = positions[0]
        self._xLabel.setText("%g" % x)
        self._yLabel.setText("%g" % y)
        self._zLabel.setText("%g" % z)

        data = picking.getData(copy=False)
        if data is not None:
            data = data[0]
            if hasattr(data, '__len__'):
                text = ' '.join(["%.3g"] * len(data)) % tuple(data)
            else:
                text = "%g" % data
            self._dataLabel.setText(text)

    def updateInfo(self):
        """Update information according to cursor position"""
        widget = self.getSceneWidget()
        if widget is None:
            _logger.info('Update without associated SceneWidget')
            self.clear()
            return

        position = widget.mapFromGlobal(qt.QCursor.pos())
        self.pick(position.x(), position.y())
