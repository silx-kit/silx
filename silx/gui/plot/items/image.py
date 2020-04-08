# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017-2020 European Synchrotron Radiation Facility
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
__date__ = "20/10/2017"


try:
    from collections import abc
except ImportError:  # Python2 support
    import collections as abc
import logging

import numpy

from ....utils.proxy import docstring
from .core import (Item, LabelsMixIn, DraggableMixIn, ColormapMixIn,
                   AlphaMixIn, ItemChangedType)


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
    if image.dtype.name != 'uint8':
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

    def __getitem__(self, item):
        """Compatibility with PyMca and silx <= 0.4.0"""
        if isinstance(item, slice):
            return [self[index] for index in range(*item.indices(5))]
        elif item == 0:
            return self.getData(copy=False)
        elif item == 1:
            return self.getName()
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
        visible = bool(visible)
        # TODO hackish data range implementation
        if self.isVisible() != visible:
            plot = self.getPlot()
            if plot is not None:
                plot._invalidateDataRange()
        super(ImageBase, self).setVisible(visible)

    def _isPlotLinear(self, plot):
        """Return True if plot only uses linear scale for both of x and y
        axes."""
        linear = plot.getXAxis().LINEAR
        if plot.getXAxis().getScale() != linear:
            return False
        if plot.getYAxis().getScale() != linear:
            return False
        return True

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
        if plot is not None and not self._isPlotLinear(plot):
            return None
        else:
            return xmin, xmax, ymin, ymax

    @docstring(DraggableMixIn)
    def drag(self, from_, to):
        origin = self.getOrigin()
        self.setOrigin((origin[0] + to[0] - from_[0],
                        origin[1] + to[1] - from_[1]))

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
        if isinstance(origin, abc.Sequence):
            origin = float(origin[0]), float(origin[1])
        else:  # single value origin
            origin = float(origin), float(origin)
        if origin != self._origin:
            self._origin = origin

            # TODO hackish data range implementation
            if self.isVisible():
                plot = self.getPlot()
                if plot is not None:
                    plot._invalidateDataRange()

            self._updated(ItemChangedType.POSITION)

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
        if isinstance(scale, abc.Sequence):
            scale = float(scale[0]), float(scale[1])
        else:  # single value scale
            scale = float(scale), float(scale)

        if scale != self._scale:
            self._scale = scale

            # TODO hackish data range implementation
            if self.isVisible():
                plot = self.getPlot()
                if plot is not None:
                    plot._invalidateDataRange()

            self._updated(ItemChangedType.SCALE)


class ImageData(ImageBase, ColormapMixIn):
    """Description of a data image with a colormap"""

    def __init__(self):
        ImageBase.__init__(self)
        ColormapMixIn.__init__(self)
        self._data = numpy.zeros((0, 0), dtype=numpy.float32)
        self._alternativeImage = None
        self.__alpha = None

    def _addBackendRenderer(self, backend):
        """Update backend renderer"""
        plot = self.getPlot()
        assert plot is not None
        if not self._isPlotLinear(plot):
            # Do not render with non linear scales
            return None

        if (self.getAlternativeImageData(copy=False) is not None or
                self.getAlphaData(copy=False) is not None):
            dataToUse = self.getRgbaImageData(copy=False)
        else:
            dataToUse = self.getData(copy=False)

        if dataToUse.size == 0:
            return None  # No data to display

        colormap = self.getColormap()
        if colormap.isAutoscale():
            # Avoid backend to compute autoscale: use item cache
            colormap = colormap.copy()
            colormap.setVRange(*colormap.getColormapRange(self))

        return backend.addImage(dataToUse,
                                origin=self.getOrigin(),
                                scale=self.getScale(),
                                colormap=colormap,
                                alpha=self.getAlpha())

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

        :returns: Array of uint8 of shape (height, width, 4)
        :rtype: numpy.ndarray
        """
        alternative = self.getAlternativeImageData(copy=False)
        if alternative is not None:
            return _convertImageToRgba32(alternative, copy=copy)
        else:
            # Apply colormap, in this case an new array is always returned
            colormap = self.getColormap()
            image = colormap.applyToData(self)
            alphaImage = self.getAlphaData(copy=False)
            if alphaImage is not None:
                # Apply transparency
                image[:, :, 3] = image[:, :, 3] * alphaImage
            return image

    def getAlternativeImageData(self, copy=True):
        """Get the optional RGBA image that is displayed instead of the data

        :param bool copy: True (Default) to get a copy,
            False to use internal representation (do not modify!)
        :rtype: Union[None,numpy.ndarray]
        """
        if self._alternativeImage is None:
            return None
        else:
            return numpy.array(self._alternativeImage, copy=copy)

    def getAlphaData(self, copy=True):
        """Get the optional transparency image applied on the data

        :param bool copy: True (Default) to get a copy,
            False to use internal representation (do not modify!)
        :rtype: Union[None,numpy.ndarray]
        """
        if self.__alpha is None:
            return None
        else:
            return numpy.array(self.__alpha, copy=copy)

    def setData(self, data, alternative=None, alpha=None, copy=True):
        """"Set the image data and optionally an alternative RGB(A) representation

        :param numpy.ndarray data: Data array with 2 dimensions (h, w)
        :param alternative: RGB(A) image to display instead of data,
                            shape: (h, w, 3 or 4)
        :type alternative: Union[None,numpy.ndarray]
        :param alpha: An array of transparency value in [0, 1] to use for
                      display with shape: (h, w)
        :type alpha: Union[None,numpy.ndarray]
        :param bool copy: True (Default) to get a copy,
                          False to use internal representation (do not modify!)
        """
        data = numpy.array(data, copy=copy)
        assert data.ndim == 2
        if data.dtype.kind == 'b':
            _logger.warning(
                'Converting boolean image to int8 to plot it.')
            data = numpy.array(data, copy=False, dtype=numpy.int8)
        elif numpy.iscomplexobj(data):
            _logger.warning(
                'Converting complex image to absolute value to plot it.')
            data = numpy.absolute(data)
        self._data = data
        self._setColormappedData(data, copy=False)

        if alternative is not None:
            alternative = numpy.array(alternative, copy=copy)
            assert alternative.ndim == 3
            assert alternative.shape[2] in (3, 4)
            assert alternative.shape[:2] == data.shape[:2]
        self._alternativeImage = alternative

        if alpha is not None:
            alpha = numpy.array(alpha, copy=copy)
            assert alpha.shape == data.shape
            if alpha.dtype.kind != 'f':
                alpha = alpha.astype(numpy.float32)
            if numpy.any(numpy.logical_or(alpha < 0., alpha > 1.)):
                alpha = numpy.clip(alpha, 0., 1.)
        self.__alpha = alpha

        # TODO hackish data range implementation
        if self.isVisible():
            plot = self.getPlot()
            if plot is not None:
                plot._invalidateDataRange()

        self._updated(ItemChangedType.DATA)


class ImageRgba(ImageBase):
    """Description of an RGB(A) image"""

    def __init__(self):
        ImageBase.__init__(self)

    def _addBackendRenderer(self, backend):
        """Update backend renderer"""
        plot = self.getPlot()
        assert plot is not None
        if not self._isPlotLinear(plot):
            # Do not render with non linear scales
            return None

        data = self.getData(copy=False)

        if data.size == 0:
            return None  # No data to display

        return backend.addImage(data,
                                origin=self.getOrigin(),
                                scale=self.getScale(),
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

        # TODO hackish data range implementation
        if self.isVisible():
            plot = self.getPlot()
            if plot is not None:
                plot._invalidateDataRange()

        self._updated(ItemChangedType.DATA)


class MaskImageData(ImageData):
    """Description of an image used as a mask.

    This class is used to flag mask items. This information is used to improve
    internal silx widgets.
    """
    pass


class ImageStack(ImageData):
    """Item to store a stack of images and to show it in the plot as one
    of the images of the stack.

    The stack is a 3D array ordered this way: `frame id, y, x`.
    So the first image of the stack can be reached this way: `stack[0, :, :]`
    """

    def __init__(self):
        ImageData.__init__(self)
        self.__stack = None
        """A 3D numpy array (or a mimic one, see ListOfImages)"""
        self.__stackPosition = None
        """Displayed position in the cube"""

    def setStackData(self, stack, position=None, copy=True):
        """Set the stack data

        :param stack: A 3D numpy array like
        :param int position: The position of the displayed image in the stack
        :param bool copy: True (Default) to get a copy,
                          False to use internal representation (do not modify!)
        """
        if self.__stack is stack:
            return
        if copy:
            stack = numpy.array(stack)
        assert stack.ndim == 3
        self.__stack = stack
        if position is not None:
            self.__stackPosition = position
        if self.__stackPosition is None:
            self.__stackPosition = 0
        self.__updateDisplayedData()

    def getStackData(self, copy=True):
        """Get the stored stack array.

        :param bool copy: True (Default) to get a copy,
                          False to use internal representation (do not modify!)
        :rtype: A 3D numpy array, or numpy array like
        """
        if copy:
            return numpy.array(self.__stack)
        else:
            return self.__stack

    def setStackPosition(self, pos):
        """Set the displayed position on the stack.

        This function will clamp the stack position according to
        the real size of the first axis of the stack.

        :param int pos: A position on the first axis of the stack.
        """
        if self.__stackPosition == pos:
            return
        self.__stackPosition = pos
        self.__updateDisplayedData()

    def getStackPosition(self):
        """Get the displayed position of the stack.

        :rtype: int
        """
        return self.__stackPosition

    def __updateDisplayedData(self):
        """Update the displayed frame whenever the stack or the stack
        position are updated."""
        if self.__stack is None or self.__stackPosition is None:
            empty = numpy.array([]).reshape(0, 0)
            self.setData(empty, copy=False)
            return
        size = len(self.__stack)
        self.__stackPosition = numpy.clip(self.__stackPosition, 0, size)
        self.setData(self.__stack[self.__stackPosition], copy=False)
