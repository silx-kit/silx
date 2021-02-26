# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017-2021 European Synchrotron Radiation Facility
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
"""This module provides 2D data and RGB(A) image item class.
"""

from __future__ import absolute_import

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "15/11/2017"

import numpy

from ....image.bilinear import BilinearImage
from ..scene import primitives, utils
from .core import DataItem3D, ItemChangedType
from .mixins import ColormapMixIn, InterpolationMixIn
from ._pick import PickingResult


class _Image(DataItem3D, InterpolationMixIn):
    """Base class for images

    :param parent: The View widget this item belongs to.
    """

    def __init__(self, parent=None):
        DataItem3D.__init__(self, parent=parent)
        InterpolationMixIn.__init__(self)

    def _setPrimitive(self, primitive):
        InterpolationMixIn._setPrimitive(self, primitive)

    def getData(self, copy=True):
        raise NotImplementedError()

    def _pickFull(self, context):
        """Perform picking in this item at given widget position.

        :param PickContext context: Current picking context
        :return: Object holding the results or None
        :rtype: Union[None,PickingResult]
        """
        rayObject = context.getPickingSegment(frame=self._getScenePrimitive())
        if rayObject is None:
            return None

        points = utils.segmentPlaneIntersect(
            rayObject[0, :3],
            rayObject[1, :3],
            planeNorm=numpy.array((0., 0., 1.), dtype=numpy.float64),
            planePt=numpy.array((0., 0., 0.), dtype=numpy.float64))

        if len(points) == 1:  # Single intersection
            if points[0][0] < 0. or points[0][1] < 0.:
                return None  # Outside image
            row, column = int(points[0][1]), int(points[0][0])
            data = self.getData(copy=False)
            height, width = data.shape[:2]
            if row < height and column < width:
                return PickingResult(
                    self,
                    positions=[(points[0][0], points[0][1], 0.)],
                    indices=([row], [column]))
            else:
                return None  # Outside image
        else:  # Either no intersection or segment and image are coplanar
            return None


class ImageData(_Image, ColormapMixIn):
    """Description of a 2D image data.

    :param parent: The View widget this item belongs to.
    """

    def __init__(self, parent=None):
        _Image.__init__(self, parent=parent)
        ColormapMixIn.__init__(self)

        self._data = numpy.zeros((0, 0), dtype=numpy.float32)

        self._image = primitives.ImageData(self._data)
        self._getScenePrimitive().children.append(self._image)

        # Connect scene primitive to mix-in class
        ColormapMixIn._setSceneColormap(self, self._image.colormap)
        _Image._setPrimitive(self, self._image)

    def setData(self, data, copy=True):
        """Set the image data to display.

        The data will be casted to float32.

        :param numpy.ndarray data: The image data
        :param bool copy: True (default) to copy the data,
                          False to use as is (do not modify!).
        """
        self._image.setData(data, copy=copy)
        self._setColormappedData(self.getData(copy=False), copy=False)
        self._updated(ItemChangedType.DATA)

    def getData(self, copy=True):
        """Get the image data.

        :param bool copy:
            True (default) to get a copy,
            False to get internal representation (do not modify!).
        :rtype: numpy.ndarray
        :return: The image data
        """
        return self._image.getData(copy=copy)


class ImageRgba(_Image, InterpolationMixIn):
    """Description of a 2D data RGB(A) image.

    :param parent: The View widget this item belongs to.
    """

    def __init__(self, parent=None):
        _Image.__init__(self, parent=parent)
        InterpolationMixIn.__init__(self)

        self._data = numpy.zeros((0, 0, 3), dtype=numpy.float32)

        self._image = primitives.ImageRgba(self._data)
        self._getScenePrimitive().children.append(self._image)

        # Connect scene primitive to mix-in class
        _Image._setPrimitive(self, self._image)

    def setData(self, data, copy=True):
        """Set the RGB(A) image data to display.

        Supported array format: float32 in [0, 1], uint8.

        :param numpy.ndarray data:
            The RGBA image data as an array of shape (H, W, Channels)
        :param bool copy: True (default) to copy the data,
                          False to use as is (do not modify!).
        """
        self._image.setData(data, copy=copy)
        self._updated(ItemChangedType.DATA)

    def getData(self, copy=True):
        """Get the image data.

        :param bool copy:
            True (default) to get a copy,
            False to get internal representation (do not modify!).
        :rtype: numpy.ndarray
        :return: The image data
        """
        return self._image.getData(copy=copy)


class _HeightMap(DataItem3D):
    """Base class for 2D data array displayed as a height field.

    :param parent: The View widget this item belongs to.
    """

    def __init__(self, parent=None):
        DataItem3D.__init__(self, parent=parent)
        self.__data = numpy.zeros((0, 0), dtype=numpy.float32)

    def setData(self, data, copy: bool=True):
        """"Set the height field data.

        :param data:
        :param copy: True (default) to copy the data,
            False to use as is (do not modify!).
        """
        data = numpy.array(data, copy=copy)
        assert data.ndim == 2

        self.__data = data
        self._updated(ItemChangedType.DATA)

    def getData(self, copy: bool=True) -> numpy.ndarray:
        """Get the height field 2D data.

        :param bool copy:
            True (default) to get a copy,
            False to get internal representation (do not modify!).
        """
        return numpy.array(self.__data, copy=copy)


class HeightMapData(_HeightMap, ColormapMixIn):
    """Description of a 2D height field associated to a colormapped dataset.

    :param parent: The View widget this item belongs to.
    """

    def __init__(self, parent=None):
        _HeightMap.__init__(self, parent=parent)
        ColormapMixIn.__init__(self)

        self.__data = numpy.zeros((0, 0), dtype=numpy.float32)

    def _updated(self, event=None):
        if event == ItemChangedType.DATA:
            self.__updateScene()
        super()._updated(event=event)

    def __updateScene(self):
        """Update display primitive to use"""
        self._getScenePrimitive().children = []  # Remove previous primitives
        ColormapMixIn._setSceneColormap(self, None)

        if not self.isVisible():
            return  # Update when visible

        data = self.getColormappedData(copy=False)
        heightData = self.getData(copy=False)

        if data.size == 0 or heightData.size == 0:
            return  # Nothing to display

        # Display as a set of points
        height, width = heightData.shape
        # Generates coordinates
        y, x = numpy.mgrid[0:height, 0:width]
        x = numpy.ravel(x)
        y = numpy.ravel(y)

        if data.shape != heightData.shape:  # data and height size miss-match
            # Colormapped data is interpolated to match the height field
            data = BilinearImage(data).map_coordinates(
                (y * data.shape[0] / height, x * data.shape[1] / width))

        primitive = primitives.Points(
            x=x,
            y=y,
            z=numpy.ravel(heightData),
            value=numpy.ravel(data),
            size=1)
        primitive.marker = 's'
        ColormapMixIn._setSceneColormap(self, primitive.colormap)
        self._getScenePrimitive().children = [primitive]

    def setColormappedData(self, data, copy: bool=True):
        """Set the 2D data used to compute colors.

        :param data: 2D array of data
        :param copy: True (default) to copy the data,
            False to use as is (do not modify!).
        """
        data = numpy.array(data, copy=copy)
        assert data.ndim == 2

        self.__data = data
        self._updated(ItemChangedType.DATA)

    def getColormappedData(self, copy: bool=True) -> numpy.ndarray:
        """Returns the 2D data used to compute colors.

        :param copy:
            True (default) to get a copy,
            False to get internal representation (do not modify!).
        """
        return numpy.array(self.__data, copy=copy)


class HeightMapRGBA(_HeightMap):
    """Description of a 2D height field associated to a RGB(A) image.

    :param parent: The View widget this item belongs to.
    """

    def __init__(self, parent=None):
        _HeightMap.__init__(self, parent=parent)

        self.__rgba = numpy.zeros((0, 0, 3), dtype=numpy.float32)

    def _updated(self, event=None):
        if event == ItemChangedType.DATA:
            self.__updateScene()
        super()._updated(event=event)

    def __updateScene(self):
        """Update display primitive to use"""
        self._getScenePrimitive().children = []  # Remove previous primitives

        if not self.isVisible():
            return  # Update when visible

        rgba = self.getColorData(copy=False)
        heightData = self.getData(copy=False)
        if rgba.size == 0 or heightData.size == 0:
            return  # Nothing to display

        # Display as a set of points
        height, width = rgba.shape[:2]
        # Generates coordinates
        y, x = numpy.mgrid[0:height, 0:width]
        x = numpy.ravel(x)
        y = numpy.ravel(y)

        if rgba.shape[:2] != heightData.shape:  # image and height size miss-match
            # We interpolate the height map to match the RGBA size
            # Best to do it the other way around, but simpler to implement this way
            heightData = BilinearImage(heightData).map_coordinates(
                (y * heightData.shape[0] / height, x * heightData.shape[1] / width))

        primitive = primitives.ColorPoints(
            x=x,
            y=y,
            z=numpy.ravel(heightData),
            color=rgba.reshape(-1, rgba.shape[-1]),
            size=1)
        primitive.marker = 's'
        self._getScenePrimitive().children = [primitive]

    def setColorData(self, data, copy: bool=True):
        """Set the RGB(A) image to use.

        Supported array format: float32 in [0, 1], uint8.

        :param data:
            The RGBA image data as an array of shape (H, W, Channels)
        :param copy: True (default) to copy the data,
            False to use as is (do not modify!).
        """
        data = numpy.array(data, copy=copy)
        assert data.ndim == 3
        assert data.shape[-1] in (3, 4)
        # TODO check type

        self.__rgba = data
        self._updated(ItemChangedType.DATA)

    def getColorData(self, copy: bool=True) -> numpy.ndarray:
        """Get the RGB(A) image data.

        :param copy: True (default) to get a copy,
            False to get internal representation (do not modify!).
        """
        return numpy.array(self.__rgba, copy=copy)
