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
    """

    def __init__(self, x, y, viewport):
        self._widgetPosition = x, y
        assert isinstance(viewport, Viewport)
        self._viewport = viewport

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

    def getNDCPosition(self):
        """Return Normalized device coordinates of picked point.

        :return: (x, y) in NDC coordinates or None if outside viewport.
        :rtype: Union[None,List[float]]
        """
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

        rayNdc = numpy.array((positionNdc + (-1., 1.),
                              positionNdc + (1., 1.)),
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
