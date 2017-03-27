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
"""This module provides the :class:`Image` item of the :class:`Plot`.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "06/03/2017"


from collections import Sequence
import logging

import numpy

from .core import Item, LabelsMixIn, DraggableMixIn, ColormapMixIn
from ....utils.decorators import deprecated


_logger = logging.getLogger(__name__)


class Image(Item, LabelsMixIn, DraggableMixIn, ColormapMixIn):
    """Description of an image"""

    # TODO method to get image of data converted to RGBA with current colormap

    def __init__(self):
        Item.__init__(self)
        LabelsMixIn.__init__(self)
        DraggableMixIn.__init__(self)
        ColormapMixIn.__init__(self)
        self._data = ()
        self._pixmap = None

        # TODO use calibration instead of origin and scale?
        self._origin = (0., 0.)
        self._scale = (1., 1.)

    def _addBackendRenderer(self, backend):
        """Update backend renderer"""
        plot = self.getPlot()
        assert plot is not None
        if plot.isXAxisLogarithmic() or plot.isYAxisLogarithmic():
            return None  # Do not render with log scales

        if self.getPixmap(copy=False) is not None:
            dataToSend = self.getPixmap(copy=False)
        else:
            dataToSend = self.getData(copy=False)

        if dataToSend.size == 0:
            return None  # No data to display

        return backend.addImage(dataToSend,
                                legend=self.getLegend(),
                                origin=self.getOrigin(),
                                scale=self.getScale(),
                                z=self.getZValue(),
                                selectable=self.isSelectable(),
                                draggable=self.isDraggable(),
                                colormap=self.getColormap())

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
            return self.getPixmap(copy=False)
        elif item == 4:
            params = {
                'info': self.getInfo(),
                'origin': self.getOrigin(),
                'scale': self.getScale(),
                'z': self.getZValue(),
                'selectable': self.isSelectable(),
                'draggable': self.isDraggable(),
                'colormap': self.getColormap(),
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
        super(Image, self).setVisible(visible)

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

    def getPixmap(self, copy=True):
        """Get the optional pixmap that is displayed instead of the data

        :param copy: True (Default) to get a copy,
                     False to use internal representation (do not modify!)
        :return: The pixmap representing the data if any
        :rtype: numpy.ndarray or None
        """
        if self._pixmap is None:
            return None
        else:
            return numpy.array(self._pixmap, copy=copy)

    def setData(self, data, pixmap=None, copy=True):
        """Set the image data

        :param data: Image data to set
        :param pixmap: Optional RGB(A) image representing the data
        :param bool copy: True (Default) to get a copy,
                          False to use internal representation (do not modify!)
        """
        data = numpy.array(data, copy=copy)
        assert data.ndim in (2, 3)
        if data.ndim == 3:
            assert data.shape[1] in (3, 4)
        self._data = data

        if pixmap is not None:
            pixmap = numpy.array(pixmap, copy=copy)
            assert pixmap.ndim == 3
            assert pixmap.shape[2] in (3, 4)
            assert pixmap.shape[:2] == data.shape[:2]
        self._pixmap = pixmap
        self._updated()

        # TODO hackish data range implementation
        if self.isVisible():
            plot = self.getPlot()
            if plot is not None:
                plot._invalidateDataRange()

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

            # TODO hackish data range implementation
            if self.isVisible():
                plot = self.getPlot()
                if plot is not None:
                    plot._invalidateDataRange()
