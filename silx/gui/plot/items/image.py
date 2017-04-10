# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017 European Synchrotron Radiation Facility
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
"""This module provides the :class:`ImageData` and :class:`ImageRgba` items
of the :class:`Plot`.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "06/03/2017"


from collections import Sequence
import logging

import numpy

from .core import Item, LabelsMixIn, DraggableMixIn, ColormapMixIn, AlphaMixIn
from ..Colors import applyColormapToData
from ....utils.decorators import deprecated


_logger = logging.getLogger(__name__)


def _convertImageToRgba32(image, copy=True):
    """Convert an RGB or RGBA image to RGBA32.

    It converts from floats in [0, 1], bool, integer and uint in [0, 255]

    If the input image is already an RGBA32 image,
    the returned image shares the same data.

    :param image: Image to convert to
    :type image: numpy.ndarray with 3 dimensions: height, width, color channels
    :param bool copy: True (Default) to get a copy, False, avoid copy if possible
    :return: The image converted to RGBA32 with dimension: (height, width, 4)
    :rtype: numpy.ndarray of uint8
    """
    assert image.ndim == 3
    assert image.shape[-1] in (3, 4)

    # Convert type to uint8
    if image.dtype.name != 'uin8':
        if image.dtype.kind == 'f':  # Float in [0, 1]
            image = (numpy.clip(image, 0., 1.) * 255).astype(numpy.uint8)
        elif image.dtype.kind == 'b':  # boolean
            image = image.astype(numpy.uint8) * 255
        elif image.dtype.kind in ('i', 'u'):  # int, uint
            image = numpy.clip(image, 0, 255).astype(numpy.uint8)
        else:
            raise ValueError('Unsupported image dtype: %s', image.dtype.name)
        copy = False  # A copy as already been done, avoid next one

    # Convert RGB to RGBA
    if image.shape[-1] == 3:
        new_image = numpy.empty((image.shape[0], image.shape[1], 4),
                                dtype=numpy.uint8)
        new_image[:, :, :3] = image
        new_image[:, :, 3] = 255
        return new_image  # This is a copy anyway
    else:
        return numpy.array(image, copy=copy)


class ImageBase(Item, LabelsMixIn, DraggableMixIn, AlphaMixIn):
    """Description of an image"""

    def __init__(self):
        Item.__init__(self)
        LabelsMixIn.__init__(self)
        DraggableMixIn.__init__(self)
        AlphaMixIn.__init__(self)
        self._data = numpy.zeros((0, 0, 4), dtype=numpy.uint8)

        self._origin = (0., 0.)
        self._scale = (1., 1.)

    @deprecated
    def __getitem__(self, item):
        """Compatibility with PyMca and silx <= 0.4.0"""
        if isinstance(item, slice):
            return [self[index] for index in range(*item.indices(5))]
        elif item == 0:
            return self.getData(copy=False)
        elif item == 1:
            return self.getLegend()
        elif item == 2:
            info = self.getInfo(copy=False)
            return {} if info is None else info
        elif item == 3:
            return None
        elif item == 4:
            params = {
                'info': self.getInfo(),
                'origin': self.getOrigin(),
                'scale': self.getScale(),
                'z': self.getZValue(),
                'selectable': self.isSelectable(),
                'draggable': self.isDraggable(),
                'colormap': None,
                'xlabel': self.getXLabel(),
                'ylabel': self.getYLabel(),
            }
            return params
        else:
            raise IndexError("Index out of range: %s" % str(item))

    def setVisible(self, visible):
        """Set visibility of item.

        :param bool visible: True to display it, False otherwise
        """
        visibleChanged = self.isVisible() != bool(visible)
        super(ImageBase, self).setVisible(visible)

        # TODO hackish data range implementation
        if visibleChanged:
            plot = self.getPlot()
            if plot is not None:
                plot._invalidateDataRange()

    def _getBounds(self):
        if self.getData(copy=False).size == 0:  # Empty data
            return None

        height, width = self.getData(copy=False).shape[:2]
        origin = self.getOrigin()
        scale = self.getScale()
        # Taking care of scale might be < 0
        xmin, xmax = origin[0], origin[0] + width * scale[0]
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        # Taking care of scale might be < 0
        ymin, ymax = origin[1], origin[1] + height * scale[1]
        if ymin > ymax:
            ymin, ymax = ymax, ymin

        plot = self.getPlot()
        if (plot is not None and
                plot.isXAxisLogarithmic() or plot.isYAxisLogarithmic()):
            return None
        else:
            return xmin, xmax, ymin, ymax

    def getData(self, copy=True):
        """Returns the image data

        :param bool copy: True (Default) to get a copy,
                          False to use internal representation (do not modify!)
        :rtype: numpy.ndarray
        """
        return numpy.array(self._data, copy=copy)

    def getRgbaImageData(self, copy=True):
        """Get the displayed RGB(A) image

        :returns: numpy.ndarray of uint8 of shape (height, width, 4)
        """
        raise NotImplementedError('This MUST be implemented in sub-class')

    def getOrigin(self):
        """Returns the offset from origin at which to display the image.

        :rtype: 2-tuple of float
        """
        return self._origin

    def setOrigin(self, origin):
        """Set the offset from origin at which to display the image.

        :param origin: (ox, oy) Offset from origin
        :type origin: float or 2-tuple of float
        """
        if isinstance(origin, Sequence):
            origin = float(origin[0]), float(origin[1])
        else:  # single value origin
            origin = float(origin), float(origin)
        if origin != self._origin:
            self._origin = origin
            self._updated()

            # TODO hackish data range implementation
            if self.isVisible():
                plot = self.getPlot()
                if plot is not None:
                    plot._invalidateDataRange()

    def getScale(self):
        """Returns the scale of the image in data coordinates.

        :rtype: 2-tuple of float
        """
        return self._scale

    def setScale(self, scale):
        """Set the scale of the image

        :param scale: (sx, sy) Scale of the image
        :type scale: float or 2-tuple of float
        """
        if isinstance(scale, Sequence):
            scale = float(scale[0]), float(scale[1])
        else:  # single value scale
            scale = float(scale), float(scale)
        if scale != self._scale:
            self._scale = scale
            self._updated()


class ImageData(ImageBase, ColormapMixIn):
    """Description of a data image with a colormap"""

    def __init__(self):
        ImageBase.__init__(self)
        ColormapMixIn.__init__(self)
        self._data = numpy.zeros((0, 0), dtype=numpy.float32)
        self._alternativeImage = None

    def _addBackendRenderer(self, backend):
        """Update backend renderer"""
        plot = self.getPlot()
        assert plot is not None
        if plot.isXAxisLogarithmic() or plot.isYAxisLogarithmic():
            return None  # Do not render with log scales

        if self.getAlternativeImageData(copy=False) is not None:
            dataToUse = self.getAlternativeImageData(copy=False)
        else:
            dataToUse = self.getData(copy=False)

        if dataToUse.size == 0:
            return None  # No data to display

        return backend.addImage(dataToUse,
                                legend=self.getLegend(),
                                origin=self.getOrigin(),
                                scale=self.getScale(),
                                z=self.getZValue(),
                                selectable=self.isSelectable(),
                                draggable=self.isDraggable(),
                                colormap=self.getColormap(),
                                alpha=self.getAlpha())

    @deprecated
    def __getitem__(self, item):
        """Compatibility with PyMca and silx <= 0.4.0"""
        if item == 3:
            return self.getAlternativeImageData(copy=False)

        params = ImageBase.__getitem__(self, item)
        if item == 4:
            params['colormap'] = self.getColormap()

        return params

    def getRgbaImageData(self, copy=True):
        """Get the displayed RGB(A) image

        :returns: numpy.ndarray of uint8 of shape (height, width, 4)
        """
        if self._alternativeImage is not None:
            return _convertImageToRgba32(
                self.getAlternativeImageData(copy=False), copy=copy)
        else:
            # Apply colormap, in this case an new array is always returned
            colormap = self.getColormap()
            image = applyColormapToData(self.getData(copy=False),
                                        **colormap)
            return image

    def getAlternativeImageData(self, copy=True):
        """Get the optional RGBA image that is displayed instead of the data

        :param copy: True (Default) to get a copy,
                     False to use internal representation (do not modify!)
        :returns: None or numpy.ndarray
        :rtype: numpy.ndarray or None
        """
        if self._alternativeImage is None:
            return None
        else:
            return numpy.array(self._alternativeImage, copy=copy)

    def setData(self, data, alternative=None, copy=True):
        """"Set the image data and optionally an alternative RGB(A) representation

        :param numpy.ndarray data: Data array with 2 dimensions (h, w)
        :param alternative: RGB(A) image to display instead of data,
                            shape: (h, w, 3 or 4)
        :type alternative: None or numpy.ndarray
        :param bool copy: True (Default) to get a copy,
                          False to use internal representation (do not modify!)
        """
        data = numpy.array(data, copy=copy)
        assert data.ndim == 2
        self._data = data

        if alternative is not None:
            alternative = numpy.array(alternative, copy=copy)
            assert alternative.ndim == 3
            assert alternative.shape[2] in (3, 4)
            assert alternative.shape[:2] == data.shape[:2]
        self._alternativeImage = alternative
        self._updated()

        # TODO hackish data range implementation
        if self.isVisible():
            plot = self.getPlot()
            if plot is not None:
                plot._invalidateDataRange()


class ImageRgba(ImageBase):
    """Description of an RGB(A) image"""

    def __init__(self):
        ImageBase.__init__(self)

    def _addBackendRenderer(self, backend):
        """Update backend renderer"""
        plot = self.getPlot()
        assert plot is not None
        if plot.isXAxisLogarithmic() or plot.isYAxisLogarithmic():
            return None  # Do not render with log scales

        data = self.getData(copy=False)

        if data.size == 0:
            return None  # No data to display

        return backend.addImage(data,
                                legend=self.getLegend(),
                                origin=self.getOrigin(),
                                scale=self.getScale(),
                                z=self.getZValue(),
                                selectable=self.isSelectable(),
                                draggable=self.isDraggable(),
                                colormap=None,
                                alpha=self.getAlpha())

    def getRgbaImageData(self, copy=True):
        """Get the displayed RGB(A) image

        :returns: numpy.ndarray of uint8 of shape (height, width, 4)
        """
        return _convertImageToRgba32(self.getData(copy=False), copy=copy)

    def setData(self, data, copy=True):
        """Set the image data

        :param data: RGB(A) image data to set
        :param bool copy: True (Default) to get a copy,
                          False to use internal representation (do not modify!)
        """
        data = numpy.array(data, copy=copy)
        assert data.ndim == 3
        assert data.shape[-1] in (3, 4)
        self._data = data

        self._updated()

        # TODO hackish data range implementation
        if self.isVisible():
            plot = self.getPlot()
            if plot is not None:
                plot._invalidateDataRange()
