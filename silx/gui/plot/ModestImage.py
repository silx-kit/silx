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
# ############################################################################*/
"""Matplotlib computationally modest image class."""

__authors__ = ["V.A. Sole", "T. Vincent"]
__license__ = "MIT"
__date__ = "16/02/2016"


import numpy

from matplotlib import cbook
from matplotlib.image import AxesImage


class ModestImage(AxesImage):
    """Computationally modest image class.

Customization of https://github.com/ChrisBeaumont/ModestImage to allow
extent support.

ModestImage is an extension of the Matplotlib AxesImage class
better suited for the interactive display of larger images. Before
drawing, ModestImage resamples the data array based on the screen
resolution and view window. This has very little affect on the
appearance of the image, but can substantially cut down on
computation since calculations of unresolved or clipped pixels
are skipped.

The interface of ModestImage is the same as AxesImage. However, it
does not currently support setting the 'extent' property. There
may also be weird coordinate warping operations for images that
I'm not aware of. Don't expect those to work either.
"""
    def __init__(self, *args, **kwargs):
        self._full_res = None
        self._sx, self._sy = None, None
        self._bounds = (None, None, None, None)
        self._origExtent = None
        super(ModestImage, self).__init__(*args, **kwargs)
        if 'extent' in kwargs and kwargs['extent'] is not None:
            self.set_extent(kwargs['extent'])

    def set_extent(self, extent):
        super(ModestImage, self).set_extent(extent)
        if self._origExtent is None:
            self._origExtent = self.get_extent()

    def get_image_extent(self):
        """Returns the extent of the whole image.

        get_extent returns the extent of the drawn area and not of the full
        image.

        :return: Bounds of the image (x0, x1, y0, y1).
        :rtype: Tuple of 4 floats.
        """
        if self._origExtent is not None:
            return self._origExtent
        else:
            return self.get_extent()

    def set_data(self, A):
        """
        Set the image array

        ACCEPTS: numpy/PIL Image A
        """

        self._full_res = A
        self._A = A

        if (self._A.dtype != numpy.uint8 and
                not numpy.can_cast(self._A.dtype, numpy.float)):
            raise TypeError("Image data can not convert to float")

        if (self._A.ndim not in (2, 3) or
                (self._A.ndim == 3 and self._A.shape[-1] not in (3, 4))):
            raise TypeError("Invalid dimensions for image data")

        self._imcache = None
        self._rgbacache = None
        self._oldxslice = None
        self._oldyslice = None
        self._sx, self._sy = None, None

    def get_array(self):
        """Override to return the full-resolution array"""
        return self._full_res

    def _scale_to_res(self):
        """ Change self._A and _extent to render an image whose
resolution is matched to the eventual rendering."""
        # extent has to be set BEFORE set_data
        if self._origExtent is None:
            if self.origin == "upper":
                self._origExtent = (0, self._full_res.shape[1],
                                    self._full_res.shape[0], 0)
            else:
                self._origExtent = (0, self._full_res.shape[1],
                                    0, self._full_res.shape[0])

        if self.origin == "upper":
            origXMin, origXMax, origYMax, origYMin = self._origExtent[0:4]
        else:
            origXMin, origXMax, origYMin, origYMax = self._origExtent[0:4]
        ax = self.axes
        ext = ax.transAxes.transform([1, 1]) - ax.transAxes.transform([0, 0])
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        xlim = max(xlim[0], origXMin), min(xlim[1], origXMax)
        if ylim[0] > ylim[1]:
            ylim = max(ylim[1], origYMin), min(ylim[0], origYMax)
        else:
            ylim = max(ylim[0], origYMin), min(ylim[1], origYMax)
        # print("THOSE LIMITS ARE TO BE COMPARED WITH THE EXTENT")
        # print("IN ORDER TO KNOW WHAT IT IS LIMITING THE DISPLAY")
        # print("IF THE AXES OR THE EXTENT")
        dx, dy = xlim[1] - xlim[0], ylim[1] - ylim[0]

        y0 = max(0, ylim[0] - 5)
        y1 = min(self._full_res.shape[0], ylim[1] + 5)
        x0 = max(0, xlim[0] - 5)
        x1 = min(self._full_res.shape[1], xlim[1] + 5)
        y0, y1, x0, x1 = [int(a) for a in [y0, y1, x0, x1]]

        sy = int(max(1, min((y1 - y0) / 5., numpy.ceil(dy / ext[1]))))
        sx = int(max(1, min((x1 - x0) / 5., numpy.ceil(dx / ext[0]))))

        # have we already calculated what we need?
        if (self._sx is not None) and (self._sy is not None):
            if (sx >= self._sx and sy >= self._sy and
                    x0 >= self._bounds[0] and x1 <= self._bounds[1] and
                    y0 >= self._bounds[2] and y1 <= self._bounds[3]):
                return

        self._A = self._full_res[y0:y1:sy, x0:x1:sx]
        self._A = cbook.safe_masked_invalid(self._A)
        x1 = x0 + self._A.shape[1] * sx
        y1 = y0 + self._A.shape[0] * sy

        if self.origin == "upper":
            self.set_extent([x0, x1, y1, y0])
        else:
            self.set_extent([x0, x1, y0, y1])
        self._sx = sx
        self._sy = sy
        self._bounds = (x0, x1, y0, y1)
        self.changed()

    def draw(self, renderer, *args, **kwargs):
        self._scale_to_res()
        super(ModestImage, self).draw(renderer, *args, **kwargs)
