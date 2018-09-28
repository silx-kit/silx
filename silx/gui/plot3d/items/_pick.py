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
"""This module provides classes supporting item picking.
"""

from __future__ import absolute_import

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "24/09/2018"

import numpy

from ..scene import Viewport, Base


class PickContext(object):
    """Store information related to current picking

    :param int x: Widget coordinate
    :param int y: Widget coordinate
    :param ~silx.gui.plot3d.scene.Viewport viewport:
        Viewport where picking occurs
    :param Union[None,callable] condition:
        Test whether each item needs to be picked or not.
    """

    def __init__(self, x, y, viewport, condition):
        self._widgetPosition = x, y
        assert isinstance(viewport, Viewport)
        self._viewport = viewport
        self._ndcZRange = -1., 1.
        self._enabled = True
        self._condition = condition

    def copy(self):
        """Returns a copy

        :rtype: PickContent
        """
        x, y = self.getWidgetPosition()
        context = PickContext(x, y, self.getViewport(), self._condition)
        context.setNDCZRange(*self._ndcZRange)
        context.setEnabled(self.isEnabled())
        return context

    def isItemPickable(self, item):
        """Check condition for the given item.

        :param Item3D item:
        :return: Whether to process the item (True) or to skip it (False)
        :rtype: bool
        """
        return self._condition is None or self._condition(item)

    def getViewport(self):
        """Returns viewport where picking occurs

        :rtype: ~silx.gui.plot3d.scene.Viewport
        """
        return self._viewport

    def getWidgetPosition(self):
        """Returns (x, y) position in pixel in the widget

        Origin is at the top-left corner of the widget,
        X from left to right, Y goes downward.

        :rtype: List[int]
        """
        return self._widgetPosition

    def setEnabled(self, enabled):
        """Set whether picking is enabled or not

        :param bool enabled: True to enable picking, False otherwise
        """
        self._enabled = bool(enabled)

    def isEnabled(self):
        """Returns True if picking is currently enabled, False otherwise.

        :rtype: bool
        """
        return self._enabled

    def setNDCZRange(self, near=-1., far=1.):
        """Set near and far Z value in normalized device coordinates

        This allows to clip the ray to a subset of the NDC range

        :param float near: Near segment end point Z coordinate
        :param float far: Far segment end point Z coordinate
        """
        self._ndcZRange = near, far

    def getNDCPosition(self):
        """Return Normalized device coordinates of picked point.

        :return: (x, y) in NDC coordinates or None if outside viewport.
        :rtype: Union[None,List[float]]
        """
        if not self.isEnabled():
            return None

        # Convert x, y from window to NDC
        x, y = self.getWidgetPosition()
        return self.getViewport().windowToNdc(x, y, checkInside=True)

    def getPickingSegment(self, frame):
        """Returns picking segment in requested coordinate frame.

        :param Union[str,Base] frame:
            The frame in which to get the picking segment,
            either a keyword: 'ndc', 'camera', 'scene' or a scene
            :class:`~silx.gui.plot3d.scene.Base` object.
        :return: Near and far points of the segment as (x, y, z, w)
            or None if picked point is outside viewport
        :rtype: Union[None,numpy.ndarray]
        """
        assert frame in ('ndc', 'camera', 'scene') or isinstance(frame, Base)

        positionNdc = self.getNDCPosition()
        if positionNdc is None:
            return None

        near, far = self._ndcZRange
        rayNdc = numpy.array((positionNdc + (near, 1.),
                              positionNdc + (far, 1.)),
                             dtype=numpy.float64)
        if frame == 'ndc':
            return rayNdc

        viewport = self.getViewport()

        rayCamera = viewport.camera.intrinsic.transformPoints(
            rayNdc,
            direct=False,
            perspectiveDivide=True)
        if frame == 'camera':
            return rayCamera

        rayScene = viewport.camera.extrinsic.transformPoints(
            rayCamera, direct=False)
        if frame == 'scene':
            return rayScene

        # frame is a scene Base object
        rayObject = frame.objectToSceneTransform.transformPoints(
            rayScene, direct=False)
        return rayObject


class PickingResult(object):
    """Class to access picking information in a 3D scene

    :param ~silx.gui.plot3d.items.Item3D item: The picked item
    :param numpy.ndarray indices: Array-like of indices of picked data.
        Either 1D or 2D with dim0: data dimension and dim1: indices.
        No copy is made.
    """

    def __init__(self, item, indices):
        self._item = item
        self._indices = numpy.array(indices)

    def getItem(self):
        """Returns the item this results corresponds to.

        :rtype: ~silx.gui.plot3d.items.Item3D
        """
        return self._item

    def getIndices(self, copy=True):
        """Returns indices of picked data.

        If data is 1D, it returns a numpy.ndarray, otherwise
        it returns a tuple with as many numpy.ndarray as there are
        dimensions in the data.

        :param bool copy: True (default) to get a copy,
            False to return internal arrays
        :rtype: Union[numpy.ndarray,List[numpy.ndarray]]
        """
        indices = numpy.array(self._indices, copy=copy)
        return indices if indices.ndim == 1 else tuple(indices)
