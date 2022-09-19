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
"""This module contains utils class for axes management.
"""

__authors__ = ["H. Payno", ]
__license__ = "MIT"
__date__ = "18/05/2020"


import numpy


def lines_intersection(line1_pt1, line1_pt2, line2_pt1, line2_pt2):
    """
    line segment intersection using vectors (Computer Graphics by F.S. Hill)

    :param tuple line1_pt1:
    :param tuple line1_pt2:
    :param tuple line2_pt1:
    :param tuple line2_pt2:
    :return: Union[None,numpy.array]
    """
    dir_line1 = line1_pt2[0] - line1_pt1[0], line1_pt2[1] - line1_pt1[1]
    dir_line2 = line2_pt2[0] - line2_pt1[0], line2_pt2[1] - line2_pt1[1]
    dp = line1_pt1 - line2_pt1

    def perp(a):
        b = numpy.empty_like(a)
        b[0] = -a[1]
        b[1] = a[0]
        return b

    dap = perp(dir_line1)
    denom = numpy.dot(dap, dir_line2)
    num = numpy.dot(dap, dp)
    if denom == 0:
        return None
    return (
        (num / denom.astype(float)) * dir_line2[0] + line2_pt1[0],
        (num / denom.astype(float)) * dir_line2[1] + line2_pt1[1])


def segments_intersection(seg1_start_pt, seg1_end_pt, seg2_start_pt,
                          seg2_end_pt):
    """
    Compute intersection between two segments

    :param seg1_start_pt:
    :param seg1_end_pt:
    :param seg2_start_pt:
    :param seg2_end_pt:
    :return: numpy.array if an intersection exists, else None
    :rtype: Union[None,numpy.array]
    """
    intersection = lines_intersection(line1_pt1=seg1_start_pt,
                                      line1_pt2=seg1_end_pt,
                                      line2_pt1=seg2_start_pt,
                                      line2_pt2=seg2_end_pt)
    if intersection is not None:
        max_x_seg1 = max(seg1_start_pt[0], seg1_end_pt[0])
        max_x_seg2 = max(seg2_start_pt[0], seg2_end_pt[0])
        max_y_seg1 = max(seg1_start_pt[1], seg1_end_pt[1])
        max_y_seg2 = max(seg2_start_pt[1], seg2_end_pt[1])

        min_x_seg1 = min(seg1_start_pt[0], seg1_end_pt[0])
        min_x_seg2 = min(seg2_start_pt[0], seg2_end_pt[0])
        min_y_seg1 = min(seg1_start_pt[1], seg1_end_pt[1])
        min_y_seg2 = min(seg2_start_pt[1], seg2_end_pt[1])

        min_tmp_x = max(min_x_seg1, min_x_seg2)
        max_tmp_x = min(max_x_seg1, max_x_seg2)
        min_tmp_y = max(min_y_seg1, min_y_seg2)
        max_tmp_y = min(max_y_seg1, max_y_seg2)
        if (min_tmp_x <= intersection[0] <= max_tmp_x and
                min_tmp_y <= intersection[1] <= max_tmp_y):
            return intersection
        else:
            return None
