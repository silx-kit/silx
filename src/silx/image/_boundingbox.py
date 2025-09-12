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
"""offer some generic 2D bounding box features"""

__authors__ = ["H. Payno"]
__license__ = "MIT"
__date__ = "07/09/2019"

from typing import Union, Sequence

from silx.math.combo import min_max
import numpy
from numpy.typing import ArrayLike


class _BoundingBox:
    """
    Simple 2D bounding box

    :param tuple bottom_left: (y, x) bottom left point
    :param tuple top_right: (y, x) top right point
    """

    def __init__(self, bottom_left, top_right):
        self.bottom_left = bottom_left
        self.top_right = top_right
        self.min_x = bottom_left[1]
        self.min_y = bottom_left[0]
        self.max_x = top_right[1]
        self.max_y = top_right[0]

    def contains(
        self,
        items: Union[
            tuple[float, float], "_BoundingBox", ArrayLike, Sequence["_BoundingBox"]
        ],
    ) -> Union[bool, numpy.ndarray]:
        """Check if a point or bounding box is inside this bounding box.

        :param items: array-like of shape (N, 2) or (2,), each row is (x, y)
                      or `_BoundingBox` or list of `_BoundingBox`
        :return: boolean or boolean array of shape (N,)
        """
        is_single = False
        is_box = False
        err_msg = "items must be points of shape (N,2) or (2,) or a single _BoundingBox or list of _BoundingBox"
        if isinstance(items, _BoundingBox):
            is_single = True
            is_box = True
            items = [items]
        elif isinstance(items, (list, tuple, numpy.ndarray)):
            if isinstance(items[0], _BoundingBox):
                is_box = True
            else:
                items = numpy.asarray(items)
                ndim_org = items.ndim
                is_single = ndim_org == 1
                if ndim_org == 1:
                    if items.shape[0] != 2:
                        raise ValueError(err_msg)
                    items = items.reshape(1, 2)
                elif ndim_org != 2 or items.shape[1] != 2:
                    raise ValueError(err_msg)
        else:
            raise TypeError(err_msg)

        if is_box:
            bottom_lefts = numpy.array([item.bottom_left for item in items])
            top_rights = numpy.array([item.top_right for item in items])

            inside_x = (self.min_x <= bottom_lefts[:, 1]) & (
                top_rights[:, 1] <= self.max_x
            )
            inside_y = (self.min_y <= bottom_lefts[:, 0]) & (
                top_rights[:, 0] <= self.max_y
            )
        else:
            inside_x = (self.min_x <= items[:, 1]) & (items[:, 1] <= self.max_x)
            inside_y = (self.min_y <= items[:, 0]) & (items[:, 0] <= self.max_y)

        is_inside = inside_x & inside_y
        return is_inside[0] if is_single else is_inside

    def collide(self, bb):
        """
        Check if the two bounding box collide

        :param bb: bounding box to compare with
        :type: :class:BoundingBox
        :return: True if the two boxes collides
        :rtype: bool
        """
        assert isinstance(bb, _BoundingBox)
        return (self.min_x < bb.max_x and self.max_x > bb.min_x) and (
            self.min_y < bb.max_y and self.max_y > bb.min_y
        )

    @staticmethod
    def from_points(points):
        """

        :param numpy.array tuple points: list of points. Should be 2D:
                                         [(y1, x1), (y2, x2), (y3, x3), ...]
        :return: bounding box from two points
        :rtype: _BoundingBox
        """
        if not isinstance(points, numpy.ndarray):
            points_ = numpy.ndarray(points)
        else:
            points_ = points
        x = points_[:, 1]
        y = points_[:, 0]
        x_min, x_max = min_max(x)
        y_min, y_max = min_max(y)
        return _BoundingBox(bottom_left=(y_min, x_min), top_right=(y_max, x_max))
