# coding: utf-8
# /*##########################################################################
# Copyright (C) 2018 European Synchrotron Radiation Facility
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
# ############################################################################*/

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "05/04/2018"


import numpy
import skimage.measure


class MarchingSquaresSciKitImage(object):
    """Reference implementation of a marching squares using sci-kit image.

    It uses `skimage.measure.find_contours` to find iso contours taking care of
    an optional mask. As result the computation is not accurate but can be used
    as reference for benchmark or for testing the API without compiling the
    cython part of silx.

    :param numpy.ndarray image: 2d-image containing the values
    :param numpy.ndarray mask: Optional 2d-image containing mask to cancel
        mask on part of the image. A `0` means the pixel at this location is
        valid, else the pixel from the image will not be used.
    """

    def __init__(self, image, mask=None):
        self._image = image
        self._mask = mask

    _deltas = [(0.0, 0.0), (0.99, 0.0), (0.0, 0.99), (0.99, 0.99)]

    def _flag_coord_over_mask(self, coord):
        """Flag coord over the mask as NaN"""
        for dx, dy in self._deltas:
            if self._mask[int(coord[0] + dx), int(coord[1] + dy)] != 0:
                return float("nan"), float("nan")
        return coord

    def find_pixels(self, level):
        """
        Compute the pixels from the image over the requested iso contours
        at this `level`.

        This implementation have to use `skimage.measure.find_contours` then
        it is not accurate nor efficient.

        :param float level: Level of the requested iso contours.
        :returns: An array of y-x coordinates.
        :rtype: numpy.ndarray
        """
        polylines = skimage.measure.find_contours(self._image, level=level)
        size = 0
        for polyline in polylines:
            size += len(polyline)
        result = numpy.empty((size, 2), dtype=numpy.int32)
        size = 0
        delta = numpy.array([0.5, 0.5])
        for polyline in polylines:
            if len(polyline) == 0:
                continue
            integer_polyline = numpy.floor(polyline + delta)
            result[size:size + len(polyline)] = integer_polyline
            size += len(polyline)

        if len(result) == 0:
            return result

        if self._mask is not None:
            # filter out pixels over the mask
            x_dim = self._image.shape[1]
            indexes = result[:, 0] * x_dim + result[:, 1]
            indexes = indexes.ravel()
            mask = self._mask.ravel()
            indexes = numpy.unique(indexes)
            indexes = indexes[mask[indexes] == 0]
            pixels = numpy.concatenate((indexes // x_dim, indexes % x_dim))
            pixels.shape = 2, -1
            pixels = pixels.T
            result = pixels
        else:
            # Note: Cound be done using a single line numpy.unique(result, axis=0)
            # But here it supports Debian 8
            x_dim = self._image.shape[1]
            indexes = result[:, 0] * x_dim + result[:, 1]
            indexes = indexes.ravel()
            indexes = numpy.unique(indexes)
            pixels = numpy.concatenate((indexes // x_dim, indexes % x_dim))
            pixels.shape = 2, -1
            pixels = pixels.T
            result = pixels
        return result

    def find_contours(self, level):
        """
        Compute the list of polygons of the iso contours at this `level`.

        If no mask is involved, the result is the same as
        `skimage.measure.find_contours`.

        If the result have to be filtered with a mask, the result is not
        accurate nor efficient. Polygons are not splited, but only points are
        filtered out using NaN coordinates. This could create artefacts.

        :param float level: Level of the requested iso contours.
        :returns: A list of array containg y-x coordinates of points
        :rtype: List[numpy.ndarray]
        """
        polylines = skimage.measure.find_contours(self._image, level=level)
        if self._mask is None:
            return polylines
        result = []
        for polyline in polylines:
            polyline = map(self._flag_coord_over_mask, polyline)
            polyline = list(polyline)
            polyline = numpy.array(polyline)
            result.append(polyline)
        return result
