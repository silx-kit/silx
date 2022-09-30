# /*##########################################################################
#
# Copyright (c) 2014-2020 European Synchrotron Radiation Facility
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
"""This module provides classes wrapping OpenGL texture."""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "03/04/2017"


from ctypes import c_void_p
import logging

import numpy

from ...._glutils import gl, Texture, numpyToGLType


_logger = logging.getLogger(__name__)


def _checkTexture2D(internalFormat, shape,
                    format_=None, type_=gl.GL_FLOAT, border=0):
    """Check if texture size with provided parameters is supported

    :rtype: bool
    """
    height, width = shape
    gl.glTexImage2D(gl.GL_PROXY_TEXTURE_2D, 0, internalFormat,
                    width, height, border,
                    format_ or internalFormat,
                    type_, c_void_p(0))
    width = gl.glGetTexLevelParameteriv(
        gl.GL_PROXY_TEXTURE_2D, 0, gl.GL_TEXTURE_WIDTH)
    return bool(width)


MIN_TEXTURE_SIZE = 64


def _getMaxSquareTexture2DSize(internalFormat=gl.GL_RGBA,
                               format_=None,
                               type_=gl.GL_FLOAT,
                               border=0):
    """Returns a supported size for a corresponding square texture

    :returns: GL_MAX_TEXTURE_SIZE or a smaller supported size (not optimal)
    :rtype: int
    """
    # Is this useful?
    maxTexSize = gl.glGetIntegerv(gl.GL_MAX_TEXTURE_SIZE)
    while maxTexSize > MIN_TEXTURE_SIZE and \
        not _checkTexture2D(internalFormat, (maxTexSize, maxTexSize),
                            format_, type_, border):
        maxTexSize //= 2
    return max(MIN_TEXTURE_SIZE, maxTexSize)


class Image(object):
    """Image of any size eventually using multiple textures or larger texture
    """

    _WRAP = (gl.GL_CLAMP_TO_EDGE, gl.GL_CLAMP_TO_EDGE)
    _MIN_FILTER = gl.GL_NEAREST
    _MAG_FILTER = gl.GL_NEAREST

    def __init__(self, internalFormat, data, format_=None, texUnit=0):
        self.internalFormat = internalFormat
        self.height, self.width = data.shape[0:2]
        type_ = numpyToGLType(data.dtype)

        if _checkTexture2D(internalFormat, data.shape[0:2], format_, type_):
            texture = Texture(internalFormat,
                              data,
                              format_,
                              texUnit=texUnit,
                              minFilter=self._MIN_FILTER,
                              magFilter=self._MAG_FILTER,
                              wrap=self._WRAP)
            texture.prepare()
            vertices = numpy.array((
                (0., 0., 0., 0.),
                (self.width, 0., 1., 0.),
                (0., self.height, 0., 1.),
                (self.width, self.height, 1., 1.)), dtype=numpy.float32)
            self.tiles = ((texture, vertices,
                           {'xOrigData': 0, 'yOrigData': 0,
                            'wData': self.width, 'hData': self.height}),)

        else:
            # Handle dimension too large: make tiles
            maxTexSize = _getMaxSquareTexture2DSize(internalFormat,
                                                    format_, type_)

            nCols = (self.width+maxTexSize-1) // maxTexSize
            colWidths = [self.width // nCols] * nCols
            colWidths[-1] += self.width % nCols

            nRows = (self.height+maxTexSize-1) // maxTexSize
            rowHeights = [self.height//nRows] * nRows
            rowHeights[-1] += self.height % nRows

            tiles = []
            yOrig = 0
            for hData in rowHeights:
                xOrig = 0
                for wData in colWidths:
                    if (hData < MIN_TEXTURE_SIZE or wData < MIN_TEXTURE_SIZE) \
                        and not _checkTexture2D(internalFormat,
                                                (hData, wData),
                                                format_,
                                                type_):
                        # Ensure texture size is at least MIN_TEXTURE_SIZE
                        tH = max(hData, MIN_TEXTURE_SIZE)
                        tW = max(wData, MIN_TEXTURE_SIZE)

                        uMax, vMax = float(wData)/tW, float(hData)/tH

                        # TODO issue with type_ and alignment
                        texture = Texture(internalFormat,
                                          data=None,
                                          format_=format_,
                                          shape=(tH, tW),
                                          texUnit=texUnit,
                                          minFilter=self._MIN_FILTER,
                                          magFilter=self._MAG_FILTER,
                                          wrap=self._WRAP)
                        # TODO handle unpack
                        texture.update(format_,
                                       data[yOrig:yOrig+hData,
                                            xOrig:xOrig+wData])
                        # texture.update(format_, type_, data,
                        #                width=wData, height=hData,
                        #                unpackRowLength=width,
                        #                unpackSkipPixels=xOrig,
                        #                unpackSkipRows=yOrig)
                    else:
                        uMax, vMax = 1, 1
                        # TODO issue with type_ and unpacking tiles
                        # TODO idea to handle unpack: use array strides
                        # As it is now, it will make a copy
                        texture = Texture(internalFormat,
                                          data[yOrig:yOrig+hData,
                                               xOrig:xOrig+wData],
                                          format_,
                                          texUnit=texUnit,
                                          minFilter=self._MIN_FILTER,
                                          magFilter=self._MAG_FILTER,
                                          wrap=self._WRAP)
                        # TODO
                        # unpackRowLength=width,
                        # unpackSkipPixels=xOrig,
                        # unpackSkipRows=yOrig)
                    vertices = numpy.array((
                        (xOrig, yOrig, 0., 0.),
                        (xOrig + wData, yOrig, uMax, 0.),
                        (xOrig, yOrig + hData, 0., vMax),
                        (xOrig + wData, yOrig + hData, uMax, vMax)),
                        dtype=numpy.float32)
                    texture.prepare()
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

    def updateAll(self, format_, data, texUnit=0):
        if not hasattr(self, 'tiles'):
            raise RuntimeError("No texture, discard has already been called")

        assert data.shape[:2] == (self.height, self.width)
        if len(self.tiles) == 1:
            self.tiles[0][0].update(format_, data, texUnit=texUnit)
        else:
            for texture, _, info in self.tiles:
                yOrig, xOrig = info['yOrigData'], info['xOrigData']
                height, width = info['hData'], info['wData']
                texture.update(format_,
                               data[yOrig:yOrig+height, xOrig:xOrig+width],
                               texUnit=texUnit)
                texture.prepare()
                # TODO check
                # width=info['wData'], height=info['hData'],
                # texUnit=texUnit, unpackAlign=unpackAlign,
                # unpackRowLength=self.width,
                # unpackSkipPixels=info['xOrigData'],
                # unpackSkipRows=info['yOrigData'])

    def render(self, posAttrib, texAttrib, texUnit=0):
        try:
            tiles = self.tiles
        except AttributeError:
            raise RuntimeError("No texture, discard has already been called")

        for texture, vertices, _ in tiles:
            texture.bind(texUnit)

            stride = vertices.shape[-1] * vertices.itemsize
            gl.glEnableVertexAttribArray(posAttrib)
            gl.glVertexAttribPointer(posAttrib,
                                     2,
                                     gl.GL_FLOAT,
                                     gl.GL_FALSE,
                                     stride, vertices)

            texCoordsPtr = c_void_p(vertices.ctypes.data +
                                    2 * vertices.itemsize)
            gl.glEnableVertexAttribArray(texAttrib)
            gl.glVertexAttribPointer(texAttrib,
                                     2,
                                     gl.GL_FLOAT,
                                     gl.GL_FALSE,
                                     stride, texCoordsPtr)
            gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, len(vertices))
