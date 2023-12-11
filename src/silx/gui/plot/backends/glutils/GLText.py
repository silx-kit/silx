# /*##########################################################################
#
# Copyright (c) 2014-2023 European Synchrotron Radiation Facility
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
"""
This module provides minimalistic text support for OpenGL.
It provides Latin-1 (ISO8859-1) characters for one monospace font at one size.
"""

from __future__ import annotations

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "03/04/2017"


from collections import OrderedDict
import weakref

import numpy

from .... import qt
from ...._glutils import font, gl, Context, Program, Texture
from .GLSupport import mat4Translate


class _Cache:
    """LRU (Least Recent Used) cache.

    :param int maxsize: Maximum number of (key, value) pairs in the cache
    :param callable callback:
       Called when a (key, value) pair is removed from the cache.
       It must take 2 arguments: key and value.
    """

    def __init__(self, maxsize=128, callback=None):
        self._maxsize = int(maxsize)
        self._callback = callback
        self._cache = OrderedDict()  # Needed for popitem(last=False)

    def __contains__(self, item):
        return item in self._cache

    def __getitem__(self, key):
        if key in self._cache:
            # Remove/add key from ordered dict to store last access info
            value = self._cache.pop(key)
            self._cache[key] = value
            return value
        else:
            raise KeyError

    def __setitem__(self, key, value):
        """Add a key, value pair to the cache.

        :param key: The key to set
        :param value: The corresponding value
        """
        if key not in self._cache and len(self._cache) >= self._maxsize:
            removedKey, removedValue = self._cache.popitem(last=False)
            if self._callback is not None:
                self._callback(removedKey, removedValue)
        self._cache[key] = value


# Text2D ######################################################################

LEFT, CENTER, RIGHT = "left", "center", "right"
TOP, BASELINE, BOTTOM = "top", "baseline", "bottom"
ROTATE_90, ROTATE_180, ROTATE_270 = 90, 180, 270


class Text2D:
    _SHADERS = {
        "vertex": """
    #version 120

    attribute vec2 position;
    attribute vec2 texCoords;
    uniform mat4 matrix;

    varying vec2 vCoords;

    void main(void) {
        gl_Position = matrix * vec4(position, 0.0, 1.0);
        vCoords = texCoords;
    }
    """,
        "fragment": """
    #version 120

    uniform sampler2D texText;
    uniform vec4 color;
    uniform vec4 bgColor;

    varying vec2 vCoords;

    void main(void) {
        gl_FragColor = mix(bgColor, color, texture2D(texText, vCoords).r);
    }
    """,
    }

    _TEX_COORDS = numpy.array(
        ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)), dtype=numpy.float32
    ).ravel()

    _program = Program(_SHADERS["vertex"], _SHADERS["fragment"], attrib0="position")

    # Discard texture objects when removed from the cache
    _textures = weakref.WeakKeyDictionary()
    """Cache already created textures"""

    _sizes = _Cache()
    """Cache already computed sizes"""

    def __init__(
        self,
        text: str,
        font: qt.QFont,
        x: float = 0.0,
        y: float = 0.0,
        color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        bgColor: tuple[float, float, float, float] | None = None,
        align: str = LEFT,
        valign: str = BASELINE,
        rotate: float = 0.0,
        devicePixelRatio: float = 1.0,
    ):
        self.devicePixelRatio = devicePixelRatio
        self.font = font
        self._vertices = None
        self._text = text
        self.x = x
        self.y = y
        self.color = color
        self.bgColor = bgColor

        if align not in (LEFT, CENTER, RIGHT):
            raise ValueError("Horizontal alignment not supported: {0}".format(align))
        self._align = align

        if valign not in (TOP, CENTER, BASELINE, BOTTOM):
            raise ValueError("Vertical alignment not supported: {0}".format(valign))
        self._valign = valign

        self._rotate = numpy.radians(rotate)

    def _textureKey(self) -> tuple[str, str, float]:
        """Returns the current texture key"""
        return self.text, self.font.key(), self.devicePixelRatio

    def _getTexture(self) -> tuple[Texture, int]:
        # Retrieve/initialize texture cache for current context
        textureKey = self._textureKey()

        context = Context.getCurrent()
        if context not in self._textures:
            self._textures[context] = _Cache(
                callback=lambda key, value: value[0].discard()
            )
        textures = self._textures[context]

        if textureKey not in textures:
            image, offset = font.rasterText(
                self.text, self.font, devicePixelRatio=self.devicePixelRatio
            )
            if textureKey not in self._sizes:
                self._sizes[textureKey] = image.shape[1], image.shape[0]

            texture = Texture(
                gl.GL_RED,
                data=image,
                minFilter=gl.GL_NEAREST,
                magFilter=gl.GL_NEAREST,
                wrap=(gl.GL_CLAMP_TO_EDGE, gl.GL_CLAMP_TO_EDGE),
            )
            texture.prepare()
            textures[textureKey] = texture, offset

        return textures[textureKey]

    @property
    def text(self) -> str:
        return self._text

    @property
    def size(self) -> tuple[int, int]:
        textureKey = self._textureKey()
        if textureKey not in self._sizes:
            image, offset = font.rasterText(
                self.text, self.font, devicePixelRatio=self.devicePixelRatio
            )
            self._sizes[textureKey] = image.shape[1], image.shape[0]
        return self._sizes[textureKey]

    def getVertices(self, offset: int, shape: tuple[int, int]) -> numpy.ndarray:
        height, width = shape

        if self._align == LEFT:
            xOrig = 0
        elif self._align == RIGHT:
            xOrig = -width
        else:  # CENTER
            xOrig = -width // 2

        if self._valign == BASELINE:
            yOrig = -offset
        elif self._valign == TOP:
            yOrig = 0
        elif self._valign == BOTTOM:
            yOrig = -height
        else:  # CENTER
            yOrig = -height // 2

        vertices = numpy.array(
            (
                (xOrig, yOrig),
                (xOrig + width, yOrig),
                (xOrig, yOrig + height),
                (xOrig + width, yOrig + height),
            ),
            dtype=numpy.float32,
        )

        cos, sin = numpy.cos(self._rotate), numpy.sin(self._rotate)
        vertices = numpy.ascontiguousarray(
            numpy.transpose(
                numpy.array(
                    (
                        cos * vertices[:, 0] - sin * vertices[:, 1],
                        sin * vertices[:, 0] + cos * vertices[:, 1],
                    ),
                    dtype=numpy.float32,
                )
            )
        )

        return vertices

    def render(self, matrix: numpy.ndarray):
        if not self.text.strip():
            return

        prog = self._program
        prog.use()

        texUnit = 0
        texture, offset = self._getTexture()

        gl.glUniform1i(prog.uniforms["texText"], texUnit)

        mat = numpy.dot(matrix, mat4Translate(int(self.x), int(self.y)))
        gl.glUniformMatrix4fv(
            prog.uniforms["matrix"], 1, gl.GL_TRUE, mat.astype(numpy.float32)
        )

        gl.glUniform4f(prog.uniforms["color"], *self.color)
        if self.bgColor is not None:
            bgColor = self.bgColor
        else:
            bgColor = self.color[0], self.color[1], self.color[2], 0.0
        gl.glUniform4f(prog.uniforms["bgColor"], *bgColor)

        vertices = self.getVertices(offset, texture.shape)

        posAttrib = prog.attributes["position"]
        gl.glEnableVertexAttribArray(posAttrib)
        gl.glVertexAttribPointer(posAttrib, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, vertices)

        texAttrib = prog.attributes["texCoords"]
        gl.glEnableVertexAttribArray(texAttrib)
        gl.glVertexAttribPointer(
            texAttrib, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, self._TEX_COORDS
        )

        with texture:
            gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)
