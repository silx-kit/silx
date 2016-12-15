# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2004-2016 European Synchrotron Radiation Facility
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
"""utility functions to create profile on images and volumes"""

__authors__ = ["V.A. Sole", "T. Vincent", "P. Knobel", "H. Payno"]
__license__ = "MIT"
__date__ = "15/12/2016"

import numpy 
from silx.image.bilinear import BilinearImage


def _alignedFullProfile(data, origin, scale, position, roiWidth, axis):
    """Get a profile along one axis on a stack of images

    :param numpy.ndarray data: 3D volume (stack of 2D images)
        The first dimension is the image index.
    :param origin: Origin of image in plot (ox, oy)
    :param scale: Scale of image in plot (sx, sy)
    :param float position: Position of profile line in plot coords
                           on the axis orthogonal to the profile direction.
    :param int roiWidth: Width of the profile in image pixels.
    :param int axis: 0 for horizontal profile, 1 for vertical.
    :return: profile image + effective ROI area corners in plot coords
    """
    assert axis in (0, 1)
    assert len(data.shape) == 3

    # Convert from plot to image coords
    imgPos = int((position - origin[1 - axis]) / scale[1 - axis])

    if axis == 1:  # Vertical profile
        # Transpose image to always do a horizontal profile
        data = numpy.transpose(data, (0, 2, 1))

    nimages, height, width = data.shape

    roiWidth = min(height, roiWidth)  # Clip roi width to image size

    # Get [start, end[ coords of the roi in the data
    start = int(int(imgPos) + 0.5 - roiWidth / 2.)
    start = min(max(0, start), height - roiWidth)
    end = start + roiWidth

    if start < height and end > 0:
        profile = data[:, max(0, start):min(end, height), :].mean(
            axis=1, dtype=numpy.float32)
    else:
        profile = numpy.zeros((nimages, width), dtype=numpy.float32)

    # Compute effective ROI in plot coords
    profileBounds = numpy.array(
        (0, width, width, 0),
        dtype=numpy.float32) * scale[axis] + origin[axis]
    roiBounds = numpy.array(
        (start, start, end, end),
        dtype=numpy.float32) * scale[1 - axis] + origin[1 - axis]

    if axis == 0:  # Horizontal profile
        area = profileBounds, roiBounds
    else:  # vertical profile
        area = roiBounds, profileBounds

    return profile, area


def _alignedPartialProfile(data, rowRange, colRange, axis):
    """Mean of a rectangular region (ROI) of a stack of images
    along a given axis.

    Returned values and all parameters are in image coordinates.

    :param numpy.ndarray data: 3D volume (stack of 2D images)
        The first dimension is the image index.
    :param rowRange: [min, max[ of ROI rows (upper bound excluded).
    :type rowRange: 2-tuple of int (min, max) with min < max
    :param colRange: [min, max[ of ROI columns (upper bound excluded).
    :type colRange: 2-tuple of int (min, max) with min < max
    :param int axis: The axis along which to take the profile of the ROI.
                     0: Sum rows along columns.
                     1: Sum columns along rows.
    :return: Profile image along the ROI as the mean of the intersection
             of the ROI and the image.
    """
    assert axis in (0, 1)
    assert len(data.shape) == 3
    assert rowRange[0] < rowRange[1]
    assert colRange[0] < colRange[1]

    nimages, height, width = data.shape

    # Range aligned with the integration direction
    profileRange = colRange if axis == 0 else rowRange

    profileLength = abs(profileRange[1] - profileRange[0])

    # Subset of the image to use as intersection of ROI and image
    rowStart = min(max(0, rowRange[0]), height)
    rowEnd = min(max(0, rowRange[1]), height)
    colStart = min(max(0, colRange[0]), width)
    colEnd = min(max(0, colRange[1]), width)

    imgProfile = numpy.mean(data[:, rowStart:rowEnd, colStart:colEnd],
                            axis=axis+1, dtype=numpy.float32)

    # Profile including out of bound area
    profile = numpy.zeros((nimages, profileLength), dtype=numpy.float32)

    # Place imgProfile in full profile
    offset = - min(0, profileRange[0])
    profile[:, offset:offset + profileLength] = imgProfile

    return profile


def createProfile(roiInfo, currentData, params, lineWidth):
    """Create the profile line for the the given image.

    :param roiInfo: information about the ROI (Store start and end points and type)
    :param numpy.ndarray currentData: the image or the stack of images
        on which we compute the profile
    :param params: parameters of the plot, such as origin, scale
        and colormap
    :param int lineWidth: width of the profile line
    """
    if currentData is None or params is None or roiInfo is None or lineWidth is None:
        return

    dataIs3D = len(currentData.shape) > 2
    # the rest of the method works on 3D data (stack of images)
    if len(currentData.shape) == 2:
        currentData3D = currentData.reshape((1,) + currentData.shape)
    elif len(currentData.shape) == 3:
        currentData3D = currentData

    origin, scale = params['origin'], params['scale']
    zActiveImage = params['z']

    roiWidth = max(1, lineWidth)
    roiStart, roiEnd, lineProjectionMode = roiInfo

    if lineProjectionMode == 'X':  # Horizontal profile on the whole image
        profile, area = _alignedFullProfile(currentData3D,
                                                 origin, scale,
                                                 roiStart[1], roiWidth,
                                                 axis=0)

        yMin, yMax = min(area[1]), max(area[1]) - 1
        if roiWidth <= 1:
            profileName = 'Y = %g' % yMin
        else:
            profileName = 'Y = [%g, %g]' % (yMin, yMax)
        xLabel = 'Columns'

    elif lineProjectionMode == 'Y':  # Vertical profile on the whole image
        profile, area = _alignedFullProfile(currentData3D,
                                                 origin, scale,
                                                 roiStart[0], roiWidth,
                                                 axis=1)

        xMin, xMax = min(area[0]), max(area[0]) - 1
        if roiWidth <= 1:
            profileName = 'X = %g' % xMin
        else:
            profileName = 'X = [%g, %g]' % (xMin, xMax)
        xLabel = 'Rows'

    else:  # Free line profile

        # Convert start and end points in image coords as (row, col)
        startPt = ((roiStart[1] - origin[1]) / scale[1],
                   (roiStart[0] - origin[0]) / scale[0])
        endPt = ((roiEnd[1] - origin[1]) / scale[1],
                 (roiEnd[0] - origin[0]) / scale[0])

        if (int(startPt[0]) == int(endPt[0]) or
                int(startPt[1]) == int(endPt[1])):
            # Profile is aligned with one of the axes

            # Convert to int
            startPt = int(startPt[0]), int(startPt[1])
            endPt = int(endPt[0]), int(endPt[1])

            # Ensure startPt <= endPt
            if startPt[0] > endPt[0] or startPt[1] > endPt[1]:
                startPt, endPt = endPt, startPt

            if startPt[0] == endPt[0]:  # Row aligned
                rowRange = (int(startPt[0] + 0.5 - 0.5 * roiWidth),
                            int(startPt[0] + 0.5 + 0.5 * roiWidth))
                colRange = startPt[1], endPt[1] + 1
                profile = _alignedPartialProfile(currentData3D,
                                                      rowRange, colRange,
                                                      axis=0)

            else:  # Column aligned
                rowRange = startPt[0], endPt[0] + 1
                colRange = (int(startPt[1] + 0.5 - 0.5 * roiWidth),
                            int(startPt[1] + 0.5 + 0.5 * roiWidth))
                profile = _alignedPartialProfile(currentData3D,
                                                      rowRange, colRange,
                                                      axis=1)

            # Convert ranges to plot coords to draw ROI area
            area = (
                numpy.array(
                    (colRange[0], colRange[1], colRange[1], colRange[0]),
                    dtype=numpy.float32) * scale[0] + origin[0],
                numpy.array(
                    (rowRange[0], rowRange[0], rowRange[1], rowRange[1]),
                    dtype=numpy.float32) * scale[1] + origin[1])

        else:  # General case: use bilinear interpolation

            # Ensure startPt <= endPt
            if (startPt[1] > endPt[1] or (
                    startPt[1] == endPt[1] and startPt[0] > endPt[0])):
                startPt, endPt = endPt, startPt

            profile = []
            for slice_idx in range(currentData3D.shape[0]):
                bilinear = BilinearImage(currentData3D[slice_idx, :, :])

                profile.append(bilinear.profile_line(
                                        (startPt[0] - 0.5, startPt[1] - 0.5),
                                        (endPt[0] - 0.5, endPt[1] - 0.5),
                                        roiWidth))
            profile = numpy.array(profile)


            # Extend ROI with half a pixel on each end, and
            # Convert back to plot coords (x, y)
            length = numpy.sqrt((endPt[0] - startPt[0]) ** 2 +
                                (endPt[1] - startPt[1]) ** 2)
            dRow = (endPt[0] - startPt[0]) / length
            dCol = (endPt[1] - startPt[1]) / length

            # Extend ROI with half a pixel on each end
            startPt = startPt[0] - 0.5 * dRow, startPt[1] - 0.5 * dCol
            endPt = endPt[0] + 0.5 * dRow, endPt[1] + 0.5 * dCol

            # Rotate deltas by 90 degrees to apply line width
            dRow, dCol = dCol, -dRow

            area = (
                numpy.array((startPt[1] - 0.5 * roiWidth * dCol,
                             startPt[1] + 0.5 * roiWidth * dCol,
                             endPt[1] + 0.5 * roiWidth * dCol,
                             endPt[1] - 0.5 * roiWidth * dCol),
                            dtype=numpy.float32) * scale[0] + origin[0],
                numpy.array((startPt[0] - 0.5 * roiWidth * dRow,
                             startPt[0] + 0.5 * roiWidth * dRow,
                             endPt[0] + 0.5 * roiWidth * dRow,
                             endPt[0] - 0.5 * roiWidth * dRow),
                            dtype=numpy.float32) * scale[1] + origin[1])

        y0, x0 = startPt
        y1, x1 = endPt
        if x1 == x0 or y1 == y0:
            profileName = 'From (%g, %g) to (%g, %g)' % (x0, y0, x1, y1)
        else:
            m = (y1 - y0) / (x1 - x0)
            b = y0 - m * x0
            profileName = 'y = %g * x %+g ; width=%d' % (m, b, roiWidth)
        xLabel = 'Distance'

    return profile, area, profileName, xLabel