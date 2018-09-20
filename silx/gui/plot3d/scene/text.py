# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2017 European Synchrotron Radiation Facility
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
"""Primitive displaying a text field in the scene."""

from __future__ import absolute_import, division, unicode_literals

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "24/04/2018"


import logging
import numpy

from silx.gui.colors import rgba

from ... import _glutils
from ..._glutils import gl

from ..._glutils import font as _font
from ...plot._utils import ticklayout

from . import event, primitives, core, transform


_logger = logging.getLogger(__name__)


class Font(event.Notifier):
    """Description of a font.

    :param str name: Family of the font
    :param int size: Size of the font in points
    :param int weight: Font weight
    :param bool italic: True for italic font, False (default) otherwise
    """

    def __init__(self, name=None, size=-1, weight=-1, italic=False):
        self._name = name if name is not None else _font.getDefaultFontFamily()
        self._size = size
        self._weight = weight
        self._italic = italic
        super(Font, self).__init__()

    name = event.notifyProperty(
        '_name',
        doc="""Name of the font (str)""",
        converter=str)

    size = event.notifyProperty(
        '_size',
        doc="""Font size in points (int)""",
        converter=int)

    weight = event.notifyProperty(
        '_weight',
        doc="""Font size in points (int)""",
        converter=int)

    italic = event.notifyProperty(
        '_italic',
        doc="""True for italic (bool)""",
        converter=bool)


class Text2D(primitives.Geometry):
    """Text field as a 2D texture displayed with bill-boarding

    :param str text: Text to display
    :param Font font: The font to use
    """

    # Text anchor values
    CENTER = 'center'

    LEFT = 'left'
    RIGHT = 'right'

    TOP = 'top'
    BASELINE = 'baseline'
    BOTTOM = 'bottom'

    _ALIGN = LEFT, CENTER, RIGHT
    _VALIGN = TOP, BASELINE, CENTER, BOTTOM

    _rasterTextCache = {}
    """Internal cache storing already rasterized text"""
    # TODO limit cache size and discard least recent used

    def __init__(self, text='', font=None):
        self._dirtyTexture = True
        self._dirtyAlign = True
        self._baselineOffset = 0
        self._text = text
        self._font = font if font is not None else Font()
        self._foreground = 1., 1., 1., 1.
        self._background = 0., 0., 0., 0.
        self._overlay = False
        self._align = 'left'
        self._valign = 'baseline'
        self._devicePixelRatio = 1.0  # Store it to check for changes

        self._texture = None
        self._textureDirty = True

        super(Text2D, self).__init__(
            'triangle_strip',
            copy=False,
            # Keep an array for position as it is bound to attr 0 and MUST
            # be active and an array at least on Mac OS X
            position=numpy.zeros((4, 3), dtype=numpy.float32),
            vertexID=numpy.arange(4., dtype=numpy.float32).reshape(4, 1),
            offsetInViewportCoords=(0., 0.))

    @property
    def text(self):
        """Text displayed by this primitive (str)"""
        return self._text

    @text.setter
    def text(self, text):
        text = str(text)
        if text != self._text:
            self._dirtyTexture = True
            self._text = text
            self.notify()

    @property
    def font(self):
        """Font to use to raster text (Font)"""
        return self._font

    @font.setter
    def font(self, font):
        self._font.removeListener(self._fontChanged)
        self._font = font
        self._font.addListener(self._fontChanged)
        self._fontChanged(self)  # Which calls notify and primitive as dirty

    def _fontChanged(self, source):
        """Listen for font change"""
        self._dirtyTexture = True
        self.notify()

    foreground = event.notifyProperty(
        '_foreground', doc="""RGBA color of the text: 4 float in [0, 1]""",
        converter=rgba)

    background = event.notifyProperty(
        '_background',
        doc="RGBA background color of the text field: 4 float in [0, 1]",
        converter=rgba)

    overlay = event.notifyProperty(
        '_overlay',
        doc="True to always display text on top of the scene (default: False)",
        converter=bool)

    def _setAlign(self, align):
        assert align in self._ALIGN
        self._align = align
        self._dirtyAlign = True
        self.notify()

    align = property(
        lambda self: self._align,
        _setAlign,
        doc="""Horizontal anchor position of the text field (str).

        Either 'left' (default), 'center' or 'right'.""")

    def _setVAlign(self, valign):
        assert valign in self._VALIGN
        self._valign = valign
        self._dirtyAlign = True
        self.notify()

    valign = property(
        lambda self: self._valign,
        _setVAlign,
        doc="""Vertical anchor position of the text field (str).

        Either 'top', 'baseline' (default), 'center' or 'bottom'""")

    def _raster(self, devicePixelRatio):
        """Raster current primitive to a bitmap

        :param float devicePixelRatio:
            The ratio between device and device-independent pixels
        :return: Corresponding image in grayscale and baseline offset from top
        :rtype: (HxW numpy.ndarray of uint8, int)
        """
        params = (self.text,
                  self.font.name,
                  self.font.size,
                  self.font.weight,
                  self.font.italic,
                  devicePixelRatio)

        if params not in self._rasterTextCache:  # Add to cache
            self._rasterTextCache[params] = _font.rasterText(*params)

        array, offset = self._rasterTextCache[params]
        return array.copy(), offset

    def _bounds(self, dataBounds=False):
        return None

    def prepareGL2(self, context):
        # Check if devicePixelRatio has changed since last rendering
        devicePixelRatio = context.glCtx.devicePixelRatio
        if self._devicePixelRatio != devicePixelRatio:
            self._devicePixelRatio = devicePixelRatio
            self._dirtyTexture = True

        if self._dirtyTexture:
            self._dirtyTexture = False

            if self._texture is not None:
                self._texture.discard()
            self._texture = None
            self._baselineOffset = 0

            if self.text:
                image, self._baselineOffset = self._raster(
                    self._devicePixelRatio)
                self._texture = _glutils.Texture(
                    gl.GL_R8, image, gl.GL_RED,
                    minFilter=gl.GL_NEAREST,
                    magFilter=gl.GL_NEAREST,
                    wrap=gl.GL_CLAMP_TO_EDGE)
                self._dirtyAlign = True  # To force update of offset

        if self._dirtyAlign:
            self._dirtyAlign = False

            if self._texture is not None:
                height, width = self._texture.shape

                if self._align == 'left':
                    ox = 0.
                elif self._align == 'center':
                    ox = - width // 2
                elif self._align == 'right':
                    ox = - width
                else:
                    _logger.error("Unsupported align: %s", self._align)
                    ox = 0.

                if self._valign == 'top':
                    oy = 0.
                elif self._valign == 'baseline':
                    oy = self._baselineOffset
                elif self._valign == 'center':
                    oy = height // 2
                elif self._valign == 'bottom':
                    oy = height
                else:
                    _logger.error("Unsupported valign: %s", self._valign)
                    oy = 0.

                offsets = (ox, oy) + numpy.array(
                    ((0., 0.), (width, 0.), (0., -height), (width, -height)),
                    dtype=numpy.float32)
                self.setAttribute('offsetInViewportCoords', offsets)

        super(Text2D, self).prepareGL2(context)

    def renderGL2(self, context):
        if not self.text:
            return  # Nothing to render

        program = context.glCtx.prog(*self._shaders)
        program.use()

        program.setUniformMatrix('matrix', context.objectToNDC.matrix)
        gl.glUniform2f(
            program.uniforms['viewportSize'], *context.viewport.size)
        gl.glUniform4f(program.uniforms['foreground'], *self.foreground)
        gl.glUniform4f(program.uniforms['background'], *self.background)
        gl.glUniform1i(program.uniforms['texture'], self._texture.texUnit)
        gl.glUniform1i(program.uniforms['isOverlay'],
                       1 if self._overlay else 0)

        self._texture.bind()

        if not self._overlay or not gl.glGetBoolean(gl.GL_DEPTH_TEST):
            self._draw(program)
        else:  # overlay and depth test currently enabled
            gl.glDisable(gl.GL_DEPTH_TEST)
            self._draw(program)
            gl.glEnable(gl.GL_DEPTH_TEST)

    # TODO texture atlas + viewportSize as attribute to chain text rendering

    _shaders = (
        """
    attribute vec3 position;
    attribute vec2 offsetInViewportCoords;  /* Offset in pixels (y upward) */
    attribute float vertexID;  /* Index of rectangle corner */

    uniform mat4 matrix;
    uniform vec2 viewportSize;  /* Width, height of the viewport */
    uniform int isOverlay;

    varying vec2 texCoords;

    void main(void)
    {
        vec4 clipPos = matrix * vec4(position, 1.0);
        vec4 ndcPos = clipPos / clipPos.w;  /* Perspective divide */

        /* Align ndcPos with pixels in viewport-like coords (origin useless) */
        vec2 viewportPos = floor((ndcPos.xy + vec2(1.0, 1.0)) * 0.5 * viewportSize);

        /* Apply offset in viewport coords */
        viewportPos += offsetInViewportCoords;

        /* Convert back to NDC */
        vec2 pointPos = 2.0 * viewportPos / viewportSize - vec2(1.0, 1.0);
        float z = (isOverlay != 0) ? -1.0 : ndcPos.z;
        gl_Position = vec4(pointPos, z, 1.0);

        /* Index : texCoords:
         * 0: (0., 0.)
         * 1: (1., 0.)
         * 2: (0., 1.)
         * 3: (1., 1.)
         */
        texCoords = vec2(vertexID == 0.0 || vertexID == 2.0 ? 0.0 : 1.0,
                         vertexID < 1.5 ? 0.0 : 1.0);
    }
    """,  # noqa

        """
    varying vec2 texCoords;

    uniform vec4 foreground;
    uniform vec4 background;
    uniform sampler2D texture;

    void main(void)
    {
        float value = texture2D(texture, texCoords).r;

        if (background.a != 0.0) {
            gl_FragColor = mix(background, foreground, value);
        } else {
            gl_FragColor = foreground;
            gl_FragColor.a *= value;
            if (gl_FragColor.a <= 0.01) {
                discard;
            }
        }
    }
    """)


class LabelledAxes(primitives.GroupBBox):
    """A group displaying a bounding box with axes labels around its children.
    """

    def __init__(self):
        super(LabelledAxes, self).__init__()
        self._ticksForBounds = None

        self._font = Font()

        # TODO offset labels from anchor in pixels

        self._xlabel = Text2D(font=self._font)
        self._xlabel.align = 'center'
        self._xlabel.transforms = [self._boxTransforms,
                                   transform.Translate(tx=0.5)]
        self._children.append(self._xlabel)

        self._ylabel = Text2D(font=self._font)
        self._ylabel.align = 'center'
        self._ylabel.transforms = [self._boxTransforms,
                                   transform.Translate(ty=0.5)]
        self._children.append(self._ylabel)

        self._zlabel = Text2D(font=self._font)
        self._zlabel.align = 'center'
        self._zlabel.transforms = [self._boxTransforms,
                                   transform.Translate(tz=0.5)]
        self._children.append(self._zlabel)

        self._tickLines = primitives.Lines(  # Init tick lines with dummy pos
            positions=((0., 0., 0.), (0., 0., 0.)),
            mode='lines')
        self._tickLines.visible = False
        self._children.append(self._tickLines)

        self._tickLabels = core.Group()
        self._children.append(self._tickLabels)

    @property
    def font(self):
        """Font of axes text labels (Font)"""
        return self._font

    @font.setter
    def font(self, font):
        self._font = font
        self._xlabel.font = font
        self._ylabel.font = font
        self._zlabel.font = font
        for label in self._tickLabels.children:
            label.font = font

    @property
    def xlabel(self):
        """Text label of the X axis (str)"""
        return self._xlabel.text

    @xlabel.setter
    def xlabel(self, text):
        self._xlabel.text = text

    @property
    def ylabel(self):
        """Text label of the Y axis (str)"""
        return self._ylabel.text

    @ylabel.setter
    def ylabel(self, text):
        self._ylabel.text = text

    @property
    def zlabel(self):
        """Text label of the Z axis (str)"""
        return self._zlabel.text

    @zlabel.setter
    def zlabel(self, text):
        self._zlabel.text = text

    def _updateTicks(self):
        """Check if ticks need update and update them if needed."""
        bounds = self._group.bounds(transformed=False, dataBounds=True)
        if bounds is None:  # No content
            if self._ticksForBounds is not None:
                self._ticksForBounds = None
                self._tickLines.visible = False
                self._tickLabels.children = []  # Reset previous labels

        elif (self._ticksForBounds is None or
                not numpy.all(numpy.equal(bounds, self._ticksForBounds))):
            self._ticksForBounds = bounds

            # Update ticks
            # TODO make ticks having a constant length on the screen
            ticklength = numpy.abs(bounds[1] - bounds[0]) / 20.

            xticks, xlabels = ticklayout.ticks(*bounds[:, 0])
            yticks, ylabels = ticklayout.ticks(*bounds[:, 1])
            zticks, zlabels = ticklayout.ticks(*bounds[:, 2])

            # Update tick lines
            coords = numpy.empty(
                ((len(xticks) + len(yticks) + len(zticks)), 4, 3),
                dtype=numpy.float32)
            coords[:, :, :] = bounds[0, :]  # account for offset from origin

            xcoords = coords[:len(xticks)]
            xcoords[:, :, 0] = numpy.asarray(xticks)[:, numpy.newaxis]
            xcoords[:, 1, 1] += ticklength[1]  # X ticks on XY plane
            xcoords[:, 3, 2] += ticklength[2]  # X ticks on XZ plane

            ycoords = coords[len(xticks):len(xticks) + len(yticks)]
            ycoords[:, :, 1] = numpy.asarray(yticks)[:, numpy.newaxis]
            ycoords[:, 1, 0] += ticklength[0]  # Y ticks on XY plane
            ycoords[:, 3, 2] += ticklength[2]  # Y ticks on YZ plane

            zcoords = coords[len(xticks) + len(yticks):]
            zcoords[:, :, 2] = numpy.asarray(zticks)[:, numpy.newaxis]
            zcoords[:, 1, 0] += ticklength[0]  # Z ticks on XZ plane
            zcoords[:, 3, 1] += ticklength[1]  # Z ticks on YZ plane

            self._tickLines.setAttribute('position', coords.reshape(-1, 3))
            self._tickLines.visible = True

            # Update labels
            offsets = bounds[0] - ticklength
            labels = []
            for tick, label in zip(xticks, xlabels):
                text = Text2D(text=label, font=self.font)
                text.align = 'center'
                text.transforms = [transform.Translate(
                    tx=tick, ty=offsets[1], tz=offsets[2])]
                labels.append(text)

            for tick, label in zip(yticks, ylabels):
                text = Text2D(text=label, font=self.font)
                text.align = 'center'
                text.transforms = [transform.Translate(
                    tx=offsets[0], ty=tick, tz=offsets[2])]
                labels.append(text)

            for tick, label in zip(zticks, zlabels):
                text = Text2D(text=label, font=self.font)
                text.align = 'center'
                text.transforms = [transform.Translate(
                    tx=offsets[0], ty=offsets[1], tz=tick)]
                labels.append(text)

            self._tickLabels.children = labels  # Reset previous labels

    def prepareGL2(self, context):
        self._updateTicks()
        super(LabelledAxes, self).prepareGL2(context)
