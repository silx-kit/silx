# /*##########################################################################
#
# Copyright (c) 2016-2026 European Synchrotron Radiation Facility
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
__authors__ = ["woutdenolf"]
__license__ = "MIT"
__date__ = "21/11/2026"

import numpy

from silx.gui.data._utils import getAxesCalib


def test_image_axis_of_two_values_is_coordinates():
    """An NXdata axis provides the coordinates of the pixels along the axis.

    A 2-value axis is a pair of coordinates, not a linear calibration
    (slope, intercept).
    """
    # 5 columns at x = 0..4, 2 rows at y = [10, 20]
    image_shape = (2, 5)
    x_axis = numpy.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y_axis = numpy.array([10.0, 20.0])

    xcalib, ycalib = getAxesCalib(image_shape, x_axis, y_axis)
    x_positions = [xcalib(i) for i in range(x_axis.size)]
    y_positions = [ycalib(i) for i in range(y_axis.size)]

    # A 2-value axis is a pair of coordinates, not a linear calibration, so
    # the positions are the axis values.
    numpy.testing.assert_allclose(x_positions, x_axis)
    numpy.testing.assert_allclose(y_positions, y_axis)


def test_image_axis_not_matching_dimension_is_ignored():
    """An axis that is not a 1D array matching the dimension is not used.

    A single-value axis or a wrong-length axis is ignored: the pixels keep
    their index coordinates instead of being mapped to a constant or a linear
    calibration.
    """
    # 3 rows, but a 1-value y axis: it is not a coordinate array of the rows.
    image_shape = (3, 3)
    x_axis = numpy.array([0.0, 1.0, 2.0])
    y_axis = numpy.array([7.0])

    xcalib, ycalib = getAxesCalib(image_shape, x_axis, y_axis)
    y_positions = [ycalib(i) for i in range(3)]

    # An axis that is not a 1D array matching the rows is ignored, so the
    # rows keep their index coordinates.
    numpy.testing.assert_allclose(y_positions, [0.0, 1.0, 2.0])

    # 3 rows, but a 2-value y axis: again not matching, so it is ignored.
    y_axis = numpy.array([10.0, 20.0])
    xcalib, ycalib = getAxesCalib(image_shape, x_axis, y_axis)
    y_positions = [ycalib(i) for i in range(3)]

    # A wrong-length axis is ignored too, so the rows keep the index
    # coordinates.
    numpy.testing.assert_allclose(y_positions, [0.0, 1.0, 2.0])
