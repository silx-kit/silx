# /*#########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2004-2014 European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
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
__author__ = "T. Vincent - ESRF Data Analysis"
__contact__ = "thomas.vincent@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__doc__ = """
This module provides classes wrapping OpenGL texture.
"""


# import ######################################################################

from .gl import *  # noqa
from ctypes import c_void_p
import numpy as np


# texture #####################################################################

class Texture2D(object):
    """Wraps OpenGL texture2D"""
    def __init__(self, internalFormat, width, height,
                 format_=None, type_=GL_FLOAT, data=None, texUnit=0,
                 minFilter=None, magFilter=None, wrapS=None, wrapT=None,
                 unpackAlign=1,
                 unpackRowLength=0, unpackSkipPixels=0, unpackSkipRows=0):
        self._internalFormat = internalFormat
        self._width, self._height = width, height
        self.texUnit = texUnit

        self._tid = glGenTextures(1)
        self.bind(self.texUnit)

        if minFilter is not None:
            glTexParameter(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minFilter)
        if magFilter is not None:
            glTexParameter(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, magFilter)
        if wrapS is not None:
            glTexParameter(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrapS)
        if wrapT is not None:
            glTexParameter(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrapT)

        glPixelStorei(GL_UNPACK_ALIGNMENT, unpackAlign)
        if unpackRowLength:
            glPixelStorei(GL_UNPACK_ROW_LENGTH, unpackRowLength)
        if unpackSkipPixels:
            glPixelStorei(GL_UNPACK_SKIP_PIXELS, unpackSkipPixels)
        if unpackSkipRows:
            glPixelStorei(GL_UNPACK_SKIP_ROWS, unpackSkipRows)

        glTexImage2D(GL_TEXTURE_2D, 0, self.internalFormat,
                     self.width, self.height, 0,
                     format_ if format_ is not None else self.internalFormat,
                     type_,
                     data if data is not None else c_void_p(0))

        if unpackRowLength:
            glPixelStorei(GL_UNPACK_ROW_LENGTH, 0)
        if unpackSkipPixels:
            glPixelStorei(GL_UNPACK_SKIP_PIXELS, 0)
        if unpackSkipRows:
            glPixelStorei(GL_UNPACK_SKIP_ROWS, 0)

        glBindTexture(GL_TEXTURE_2D, 0)

    @property
    def internalFormat(self):
        return self._internalFormat

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def tid(self):
        """OpenGL texture ID"""
        try:
            return self._tid
        except AttributeError:
            raise RuntimeError("No OpenGL texture resource, \
                               discard has already been called")

    def discard(self):
        if hasattr(self, '_tid'):
            if bool(glDeleteTextures):  # Test for __del__
                glDeleteTextures(np.array((self._tid,)))
            del self._tid

    def __del__(self):
        self.discard()

    def bind(self, texUnit=None):
        texUnit = texUnit if texUnit is not None else self.texUnit
        glActiveTexture(GL_TEXTURE0 + texUnit)
        glBindTexture(GL_TEXTURE_2D, self.tid)

    def update(self, format_, type_, data,
               xOffset=0, yOffset=0, width=None, height=None,
               texUnit=None, unpackAlign=1,
               unpackRowLength=0, unpackSkipPixels=0, unpackSkipRows=0):

        if unpackRowLength:
            glPixelStorei(GL_UNPACK_ROW_LENGTH, unpackRowLength)
        if unpackSkipPixels:
            glPixelStorei(GL_UNPACK_SKIP_PIXELS, unpackSkipPixels)
        if unpackSkipRows:
            glPixelStorei(GL_UNPACK_SKIP_ROWS, unpackSkipRows)

        glPixelStorei(GL_UNPACK_ALIGNMENT, unpackAlign)

        self.bind(texUnit)
        glTexSubImage2D(GL_TEXTURE_2D, 0,
                        xOffset, yOffset,
                        width or self.width,
                        height or self.height,
                        format_, type_, data)
        glBindTexture(GL_TEXTURE_2D, 0)

        if unpackRowLength:
            glPixelStorei(GL_UNPACK_ROW_LENGTH, 0)
        if unpackSkipPixels:
            glPixelStorei(GL_UNPACK_SKIP_PIXELS, 0)
        if unpackSkipRows:
            glPixelStorei(GL_UNPACK_SKIP_ROWS, 0)


def _checkTexture2D(internalFormat, width, height,
                    format_=None, type_=GL_FLOAT, border=0):
    """Check if texture size with provided parameters is supported

    :rtype: bool
    """
    glTexImage2D(GL_PROXY_TEXTURE_2D, 0, internalFormat,
                 width, height, border,
                 format_ or internalFormat,
                 type_, c_void_p(0))
    width = glGetTexLevelParameteriv(GL_PROXY_TEXTURE_2D, 0, GL_TEXTURE_WIDTH)
    return bool(width)


MIN_TEXTURE_SIZE = 64


def _getMaxSquareTexture2DSize(internalFormat=GL_RGBA,
                               format_=None, type_=GL_FLOAT, border=0):
    """Returns a supported size for a corresponding square texture

    :returns: GL_MAX_TEXTURE_SIZE or a smaller supported size (not optimal)
    :rtype: tuple
    """
    # Is this useful?
    maxTexSize = glGetIntegerv(GL_MAX_TEXTURE_SIZE)
    while maxTexSize > MIN_TEXTURE_SIZE and \
        not _checkTexture2D(internalFormat, maxTexSize, maxTexSize,
                            format_, type_, border):
        maxTexSize = maxTexSize // 2
    return max(MIN_TEXTURE_SIZE, maxTexSize)


class Image(object):
    """Image of any size eventually using multiple textures or larger texture
    """

    _WRAP_S = GL_CLAMP_TO_EDGE
    _WRAP_T = GL_CLAMP_TO_EDGE
    _MIN_FILTER = GL_NEAREST
    _MAG_FILTER = GL_NEAREST

    def __init__(self, internalFormat, width, height,
                 format_=None, type_=GL_FLOAT, data=None,
                 texUnit=0, unpackAlign=1):
        self.internalFormat = internalFormat
        self.width, self.height = width, height

        if _checkTexture2D(internalFormat, width, height, format_, type_):
            texture = Texture2D(internalFormat, width, height,
                                format_, type_, data, texUnit=texUnit,
                                minFilter=self._MIN_FILTER,
                                magFilter=self._MAG_FILTER,
                                wrapS=self._WRAP_S, wrapT=self._WRAP_T,
                                unpackAlign=unpackAlign)
            vertices = np.array((
                (0., 0.,        0., 0.),
                (width, 0.,     1., 0.),
                (0., height,    0., 1.),
                (width, height, 1., 1.)), dtype=np.float32)
            self.tiles = ((texture, vertices,
                           {'xOrigData': 0, 'yOrigData': 0,
                            'wData': width, 'hData': height}),)

        else:
            # Handle dimension too large: make tiles
            maxTexSize = _getMaxSquareTexture2DSize(internalFormat,
                                                    format_, type_)

            nCols = (width+maxTexSize-1) // maxTexSize
            colWidths = [width // nCols] * nCols
            colWidths[-1] += width % nCols

            nRows = (height+maxTexSize-1) // maxTexSize
            rowHeights = [height//nRows] * nRows
            rowHeights[-1] += height % nRows

            tiles = []
            yOrig = 0
            for hData in rowHeights:
                xOrig = 0
                for wData in colWidths:
                    if (hData < MIN_TEXTURE_SIZE or wData < MIN_TEXTURE_SIZE) \
                        and not _checkTexture2D(internalFormat, wData, hData,
                                                format_, type_):
                        # Ensure texture size is at least MIN_TEXTURE_SIZE
                        tH = max(hData, MIN_TEXTURE_SIZE)
                        tW = max(wData, MIN_TEXTURE_SIZE)

                        uMax, vMax = float(wData)/tW, float(hData)/tH

                        texture = Texture2D(internalFormat, tW, tH,
                                            format_, type_,
                                            None, texUnit=texUnit,
                                            minFilter=self._MIN_FILTER,
                                            magFilter=self._MAG_FILTER,
                                            wrapS=self._WRAP_S,
                                            wrapT=self._WRAP_T,
                                            unpackAlign=unpackAlign)
                        texture.update(format_, type_, data,
                                       width=wData, height=hData,
                                       unpackRowLength=width,
                                       unpackSkipPixels=xOrig,
                                       unpackSkipRows=yOrig)
                    else:
                        uMax, vMax = 1, 1
                        texture = Texture2D(internalFormat, wData, hData,
                                            format_, type_,
                                            data, texUnit=texUnit,
                                            minFilter=self._MIN_FILTER,
                                            magFilter=self._MAG_FILTER,
                                            wrapS=self._WRAP_S,
                                            wrapT=self._WRAP_T,
                                            unpackAlign=unpackAlign,
                                            unpackRowLength=width,
                                            unpackSkipPixels=xOrig,
                                            unpackSkipRows=yOrig)
                    vertices = np.array((
                        (xOrig, yOrig,         0., 0.),
                        (xOrig + wData, yOrig,     uMax, 0.),
                        (xOrig, yOrig + hData,     0., vMax),
                        (xOrig + wData, yOrig + hData, uMax, vMax)),
                        dtype=np.float32)
                    tiles.append((texture, vertices,
                                  {'xOrigData': xOrig, 'yOrigData': yOrig,
                                   'wData': wData, 'hData': hData}))
                    xOrig += wData
                yOrig += hData
            self.tiles = tuple(tiles)

    def discard(self):
        for texture, vertices, _ in self.tiles:
            texture.discard()
        del self.tiles

    def updateAll(self, format_, type_, data, texUnit=0, unpackAlign=1):
        if not hasattr(self, 'tiles'):
            raise RuntimeError("No texture, discard has already been called")

        assert data.shape[:2] == (self.height, self.width)
        if len(self.tiles) == 1:
            self.tiles[0][0].update(format_, type_, data,
                                    width=self.width, height=self.height,
                                    texUnit=texUnit, unpackAlign=unpackAlign)
        else:
            for texture, _, info in self.tiles:
                texture.update(format_, type_, data,
                               width=info['wData'], height=info['hData'],
                               texUnit=texUnit, unpackAlign=unpackAlign,
                               unpackRowLength=self.width,
                               unpackSkipPixels=info['xOrigData'],
                               unpackSkipRows=info['yOrigData'])

    def render(self, posAttrib, texAttrib, texUnit=0):
        try:
            tiles = self.tiles
        except AttributeError:
            raise RuntimeError("No texture, discard has already been called")

        for texture, vertices, _ in tiles:
            texture.bind(texUnit)

            stride = vertices.shape[-1] * vertices.itemsize
            glEnableVertexAttribArray(posAttrib)
            glVertexAttribPointer(posAttrib,
                                  2,
                                  GL_FLOAT,
                                  GL_FALSE,
                                  stride, vertices)

            texCoordsPtr = c_void_p(vertices.ctypes.data +
                                    2 * vertices.itemsize)
            glEnableVertexAttribArray(texAttrib)
            glVertexAttribPointer(texAttrib,
                                  2,
                                  GL_FLOAT,
                                  GL_FALSE,
                                  stride, texCoordsPtr)
            glDrawArrays(GL_TRIANGLE_STRIP, 0, len(vertices))
