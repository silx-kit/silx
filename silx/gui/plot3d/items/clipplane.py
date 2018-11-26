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
"""This module provides a scene clip plane class.
"""

from __future__ import absolute_import

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "15/11/2017"


import numpy

from ..scene import primitives, utils

from ._pick import PickingResult
from .core import Item3D
from .mixins import PlaneMixIn


class ClipPlane(Item3D, PlaneMixIn):
    """Represents a clipping plane that clips following items within the group.

    For now only on clip plane is allowed at once in a scene.
    """

    def __init__(self, parent=None):
        plane = primitives.ClipPlane()
        Item3D.__init__(self, parent=parent, primitive=plane)
        PlaneMixIn.__init__(self, plane=plane)

    def __pickPreProcessing(self, context):
        """Common processing for :meth:`_pickPostProcess` and :meth:`_pickFull`

        :param PickContext context: Current picking context
        :return None or (bounds, intersection points, rayObject)
        """
        plane = self._getPlane()
        planeParent = plane.parent
        if planeParent is None:
            return None

        rayObject = context.getPickingSegment(frame=plane)
        if rayObject is None:
            return None

        bounds = planeParent.bounds(dataBounds=True)
        rayClip = utils.clipSegmentToBounds(rayObject[:, :3], bounds)
        if rayClip is None:
            return None  # Ray is outside parent's bounding box

        points = utils.segmentPlaneIntersect(
            rayObject[0, :3],
            rayObject[1, :3],
            planeNorm=self.getNormal(),
            planePt=self.getPoint())

        # A single intersection inside bounding box
        picked = (len(points) == 1 and
                  numpy.all(bounds[0] <= points[0]) and
                  numpy.all(points[0] <= bounds[1]))

        return picked, points, rayObject

    def _pick(self, context):
        # Perform picking before modifying context
        result = super(ClipPlane, self)._pick(context)

        # Modify context if needed
        if self.isVisible() and context.isEnabled():
            info = self.__pickPreProcessing(context)
            if info is not None:
                picked, points, rayObject = info
                plane = self._getPlane()

                if picked:  # A single intersection inside bounding box
                    # Clip NDC z range for following brother items
                    ndcIntersect = plane.objectToNDCTransform.transformPoint(
                        points[0], perspectiveDivide=True)
                    ndcNormal = plane.objectToNDCTransform.transformNormal(
                        self.getNormal())
                    if ndcNormal[2] < 0:
                        context.setNDCZRange(-1., ndcIntersect[2])
                    else:
                        context.setNDCZRange(ndcIntersect[2], 1.)

                else:
                    # TODO check this might not be correct
                    rayObject[:, 3] = 1.  # Make sure 4h coordinate is one
                    if numpy.sum(rayObject[0] * self.getParameters()) < 0.:
                        # Disable picking for remaining brothers
                        context.setEnabled(False)

        return result

    def _pickFastCheck(self, context):
        return True

    def _pickFull(self, context):
        """Perform picking in this item at given widget position.

        :param PickContext context: Current picking context
        :return: Object holding the results or None
        :rtype: Union[None,PickingResult]
        """
        info = self.__pickPreProcessing(context)
        if info is not None:
            picked, points, _ = info

            if picked:
                return PickingResult(self, positions=[points[0]])

        return None
