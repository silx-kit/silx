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
This module provides a class to render 2D array as a colormap or RGB(A) image
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "03/04/2017"


import math
import numpy


from ...._glutils import gl, Program, Texture
from ...items.types import AxisScaleType
from ..._utils import FLOAT32_MINPOS
from .GLSupport import mat4Translate, mat4Scale
from .GLTexture import Image
from .GLPlotItem import GLPlotItem


_SHADER_AXIS_SCALES: dict[AxisScaleType, int] = {"linear": 0, "log": 1, "asinh": 2}


class _GLPlotData2D(GLPlotItem):
    def __init__(self, data, origin, scale):
        super().__init__()
        self.data = data
        assert len(origin) == 2
        self.origin = tuple(origin)
        assert len(scale) == 2
        self.scale = tuple(scale)

    def pick(self, x, y):
        if self.xMin <= x <= self.xMax and self.yMin <= y <= self.yMax:
            ox, oy = self.origin
            sx, sy = self.scale
            col = int((x - ox) / sx)
            row = int((y - oy) / sy)
            return (row,), (col,)
        else:
            return None

    @property
    def xMin(self):
        ox, sx = self.origin[0], self.scale[0]
        return ox if sx >= 0.0 else ox + sx * self.data.shape[1]

    @property
    def yMin(self):
        oy, sy = self.origin[1], self.scale[1]
        return oy if sy >= 0.0 else oy + sy * self.data.shape[0]

    @property
    def xMax(self):
        ox, sx = self.origin[0], self.scale[0]
        return ox + sx * self.data.shape[1] if sx >= 0.0 else ox

    @property
    def yMax(self):
        oy, sy = self.origin[1], self.scale[1]
        return oy + sy * self.data.shape[0] if sy >= 0.0 else oy


class GLPlotColormap(_GLPlotData2D):
    _SHADERS = {
        "linear": {
            "vertex": """
    #version 120

    uniform mat4 matrix;
    attribute vec2 texCoords;
    attribute vec2 position;

    varying vec2 coords;

    void main(void) {
        coords = texCoords;
        gl_Position = matrix * vec4(position, 0.0, 1.0);
    }
    """,
            "fragTransform": """
    vec2 textureCoords(void) {
        return coords;
    }
    """,
        },
        "non_linear": {
            "vertex": """
    #version 120

    attribute vec2 position;
    uniform mat4 matrix;
    uniform mat4 matOffset;
    uniform ivec2 axisScale;

    varying vec2 coords;

    const float oneOverLog10 = 0.43429448190325176;

    /* asinh = log(x + sqrt(x*x + 1)) for compatibility with GLSL 1.20 */
    float asinh(float value) {
        return log(value + sqrt(value*value + 1.0));
    }

    void main(void) {
        vec4 dataPos = matOffset * vec4(position, 0.0, 1.0);
        if (axisScale.x == 1) {  /* log scale */
            dataPos.x = oneOverLog10 * log(dataPos.x);
        } else if (axisScale.x == 2) {  /* asinh scale */
            dataPos.x = asinh(dataPos.x);
        }
        if (axisScale.y == 1) {  /* log scale */
            dataPos.y = oneOverLog10 * log(dataPos.y);
        } else if (axisScale.y == 2) {  /* asinh scale */
            dataPos.y = asinh(dataPos.y);
        }
        coords = dataPos.xy;
        gl_Position = matrix * dataPos;
    }
    """,
            "fragTransform": """
    uniform ivec2 axisScale;
    uniform vec2 bounds_oneOverRange;
    uniform vec2 bounds_originOverRange;

    /* sinh = (e^x - e^-x)/2 for compatibility with GLSL 1.20 */
    float sinh(float value) {
        return (exp(value) - exp(-value)) / 2.0;
    }

    vec2 textureCoords(void) {
        vec2 pos = coords;
        if (axisScale.x == 1) {  /* log scale */
            pos.x = pow(10., coords.x);
        } else if (axisScale.x == 2) {  /* asinh scale */
            pos.x = sinh(coords.x);
        }
        if (axisScale.y == 1) {  /* log scale */
            pos.y = pow(10., coords.y);
        } else if (axisScale.y == 2) {  /* asinh scale */
            pos.y = sinh(coords.y);
        }
        return pos * bounds_oneOverRange - bounds_originOverRange;
        // TODO texture coords in range different from [0, 1]
    }
    """,
        },
        "fragment": """
    #version 120

    /* isnan declaration for compatibility with GLSL 1.20 */
    bool isnan(float value) {
        return (value != value);
    }

    uniform sampler2D data;
    uniform float data_scale;
    uniform sampler2D cmap_texture;
    uniform int cmap_normalization;
    uniform float cmap_parameter;
    uniform float cmap_min;
    uniform float cmap_oneOverRange;
    uniform float alpha;
    uniform vec4 nancolor;

    varying vec2 coords;

    %s

    const float oneOverLog10 = 0.43429448190325176;

    void main(void) {
        float raw_data = texture2D(data, textureCoords()).r * data_scale;
        float value = 0.;
        if (cmap_normalization == 1) { /*Logarithm mapping*/
            if (raw_data > 0.) {
                value = clamp(cmap_oneOverRange *
                              (oneOverLog10 * log(raw_data) - cmap_min),
                              0., 1.);
            } else {
                value = 0.;
            }
        } else if (cmap_normalization == 2) { /*Square root mapping*/
            if (raw_data >= 0.) {
                value = clamp(cmap_oneOverRange * (sqrt(raw_data) - cmap_min),
                              0., 1.);
            } else {
                value = 0.;
            }
        } else if (cmap_normalization == 3) { /*Gamma correction mapping*/
            value = pow(
                clamp(cmap_oneOverRange * (raw_data - cmap_min), 0., 1.),
                cmap_parameter);
        } else if (cmap_normalization == 4) { /* arcsinh mapping */
            /* asinh = log(x + sqrt(x*x + 1) for compatibility with GLSL 1.20 */
            value = clamp(cmap_oneOverRange * (log(raw_data + sqrt(raw_data*raw_data + 1.0)) - cmap_min), 0., 1.);
        } else { /*Linear mapping and fallback*/
            value = clamp(cmap_oneOverRange * (raw_data - cmap_min), 0., 1.);
        }

        if (isnan(raw_data)) {
            gl_FragColor = nancolor;
        } else {
            gl_FragColor = texture2D(cmap_texture, vec2(value, 0.5));
        }
        gl_FragColor.a *= alpha;
    }
    """,
    }

    _DATA_TEX_UNIT = 0
    _CMAP_TEX_UNIT = 1

    _INTERNAL_FORMATS = {
        numpy.dtype(numpy.float32): gl.GL_R32F,
        numpy.dtype(numpy.float16): gl.GL_R16F,
        # Use normalized integer for unsigned int formats
        numpy.dtype(numpy.uint16): gl.GL_R16,
        numpy.dtype(numpy.uint8): gl.GL_R8,
    }

    _linearProgram = Program(
        _SHADERS["linear"]["vertex"],
        _SHADERS["fragment"] % _SHADERS["linear"]["fragTransform"],
        attrib0="position",
    )

    _nonLinearProgram = Program(
        _SHADERS["non_linear"]["vertex"],
        _SHADERS["fragment"] % _SHADERS["non_linear"]["fragTransform"],
        attrib0="position",
    )

    SUPPORTED_NORMALIZATIONS = "linear", "log", "sqrt", "gamma", "arcsinh"

    def __init__(
        self,
        data,
        origin,
        scale,
        colormap,
        normalization="linear",
        gamma=0.0,
        cmapRange=None,
        alpha=1.0,
        nancolor=(1.0, 1.0, 1.0, 0.0),
    ):
        """Create a 2D colormap

        :param data: The 2D scalar data array to display
        :type data: numpy.ndarray with 2 dimensions (dtype=numpy.float32)
        :param origin: (x, y) coordinates of the origin of the data array
        :type origin: 2-tuple of floats.
        :param scale: (sx, sy) scale factors of the data array.
                      This is the size of a data pixel in plot data space.
        :type scale: 2-tuple of floats.
        :param str colormap: Name of the colormap to use
            TODO: Accept a 1D scalar array as the colormap
        :param str normalization: The colormap normalization.
            One of: 'linear', 'log', 'sqrt', 'gamma'
        ;param float gamma: The gamma parameter (for 'gamma' normalization)
        :param cmapRange: The range of colormap or None for autoscale colormap
            For logarithmic colormap, the range is in the untransformed data
            TODO: check consistency with matplotlib
        :type cmapRange: (float, float) or None
        :param float alpha: Opacity from 0 (transparent) to 1 (opaque)
        :param nancolor: RGBA color for Not-A-Number values
        :type nancolor: 4-tuple of float in [0., 1.]
        """
        assert data.dtype in self._INTERNAL_FORMATS
        assert normalization in self.SUPPORTED_NORMALIZATIONS

        super().__init__(data, origin, scale)
        self.colormap = numpy.asarray(colormap)
        self.normalization = normalization
        self.gamma = gamma
        self._cmapRange = (1.0, 10.0)  # Colormap range
        self.cmapRange = cmapRange  # Update _cmapRange
        self._alpha = numpy.clip(alpha, 0.0, 1.0)
        self._nancolor = numpy.clip(nancolor, 0.0, 1.0)

        self._cmap_texture = None
        self._texture = None
        self._textureIsDirty = False

    def discard(self):
        if self._cmap_texture is not None:
            self._cmap_texture.discard()
            self._cmap_texture = None

        if self._texture is not None:
            self._texture.discard()
            self._texture = None
        self._textureIsDirty = False

    def isInitialized(self):
        return self._cmap_texture is not None or self._texture is not None

    @property
    def cmapRange(self):
        if self.normalization == "log":
            assert self._cmapRange[0] > 0.0 and self._cmapRange[1] > 0.0
        elif self.normalization == "sqrt":
            assert self._cmapRange[0] >= 0.0 and self._cmapRange[1] >= 0.0
        return self._cmapRange

    @cmapRange.setter
    def cmapRange(self, cmapRange):
        assert len(cmapRange) == 2
        assert cmapRange[0] <= cmapRange[1]
        self._cmapRange = float(cmapRange[0]), float(cmapRange[1])

    @property
    def alpha(self):
        return self._alpha

    def updateData(self, data):
        assert data.dtype in self._INTERNAL_FORMATS
        oldData = self.data
        self.data = data

        if self._texture is not None:
            if self.data.shape != oldData.shape or self.data.dtype != oldData.dtype:
                self.discard()
            else:
                self._textureIsDirty = True

    def prepare(self):
        if self._cmap_texture is None:
            # TODO share cmap texture accross Images
            # put all cmaps in one texture
            colormap = numpy.empty(
                (16, 256, self.colormap.shape[1]), dtype=self.colormap.dtype
            )
            colormap[:] = self.colormap
            format_ = gl.GL_RGBA if colormap.shape[-1] == 4 else gl.GL_RGB
            self._cmap_texture = Texture(
                internalFormat=format_,
                data=colormap,
                format_=format_,
                texUnit=self._CMAP_TEX_UNIT,
                minFilter=gl.GL_NEAREST,
                magFilter=gl.GL_NEAREST,
                wrap=gl.GL_MIRRORED_REPEAT,
            )
            self._cmap_texture.prepare()

        if self._texture is None:
            internalFormat = self._INTERNAL_FORMATS[self.data.dtype]

            self._texture = Image(
                internalFormat,
                self.data,
                format_=gl.GL_RED,
                texUnit=self._DATA_TEX_UNIT,
            )
        elif self._textureIsDirty:
            self._textureIsDirty = True
            self._texture.updateAll(format_=gl.GL_RED, data=self.data)

    def _setCMap(self, prog):
        dataMin, dataMax = self.cmapRange  # If log, it is stricly positive
        param = 0.0

        if self.data.dtype in (numpy.uint16, numpy.uint8):
            # Using unsigned int as normalized integer in OpenGL
            # So revert normalization in the shader
            dataScale = float(numpy.iinfo(self.data.dtype).max)
        else:
            dataScale = 1.0

        if self.normalization == "log":
            dataMin = math.log10(dataMin)
            dataMax = math.log10(dataMax)
            normID = 1
        elif self.normalization == "sqrt":
            dataMin = math.sqrt(dataMin)
            dataMax = math.sqrt(dataMax)
            normID = 2
        elif self.normalization == "gamma":
            # Keep dataMin, dataMax as is
            param = self.gamma
            normID = 3
        elif self.normalization == "arcsinh":
            dataMin = numpy.arcsinh(dataMin)
            dataMax = numpy.arcsinh(dataMax)
            normID = 4
        else:  # Linear and fallback
            normID = 0

        gl.glUniform1f(prog.uniforms["data_scale"], dataScale)
        gl.glUniform1i(prog.uniforms["cmap_texture"], self._cmap_texture.texUnit)
        gl.glUniform1i(prog.uniforms["cmap_normalization"], normID)
        gl.glUniform1f(prog.uniforms["cmap_parameter"], param)
        gl.glUniform1f(prog.uniforms["cmap_min"], dataMin)
        if dataMax > dataMin:
            oneOverRange = 1.0 / (dataMax - dataMin)
        else:
            oneOverRange = 0.0  # Fall-back
        gl.glUniform1f(prog.uniforms["cmap_oneOverRange"], oneOverRange)

        gl.glUniform4f(prog.uniforms["nancolor"], *self._nancolor)

        self._cmap_texture.bind()

    def _renderLinear(self, context):
        """Perform rendering when both axes have linear scales

        :param RenderContext context: Rendering information
        """
        self.prepare()

        prog = self._linearProgram
        prog.use()

        gl.glUniform1i(prog.uniforms["data"], self._DATA_TEX_UNIT)

        mat = numpy.dot(
            numpy.dot(context.matrix, mat4Translate(*self.origin)),
            mat4Scale(*self.scale),
        )
        gl.glUniformMatrix4fv(
            prog.uniforms["matrix"], 1, gl.GL_TRUE, mat.astype(numpy.float32)
        )

        gl.glUniform1f(prog.uniforms["alpha"], self.alpha)

        self._setCMap(prog)

        self._texture.render(
            prog.attributes["position"],
            prog.attributes["texCoords"],
            self._DATA_TEX_UNIT,
        )

    def _renderNonLinear(self, context):
        """Perform rendering when one axis is non linear

        :param RenderContext context: Rendering information
        """
        xMin, yMin = self.xMin, self.yMin
        if (context.xAxisScale == "log" and xMin < FLOAT32_MINPOS) or (
            context.yAxisScale == "log" and yMin < FLOAT32_MINPOS
        ):
            # Do not render images that are partly or totally <= 0
            return

        self.prepare()

        prog = self._nonLinearProgram
        prog.use()

        ox, oy = self.origin

        gl.glUniform1i(prog.uniforms["data"], self._DATA_TEX_UNIT)

        gl.glUniformMatrix4fv(
            prog.uniforms["matrix"], 1, gl.GL_TRUE, context.matrix.astype(numpy.float32)
        )
        mat = numpy.dot(mat4Translate(ox, oy), mat4Scale(*self.scale))
        gl.glUniformMatrix4fv(
            prog.uniforms["matOffset"], 1, gl.GL_TRUE, mat.astype(numpy.float32)
        )

        gl.glUniform2i(
            prog.uniforms["axisScale"],
            _SHADER_AXIS_SCALES[context.xAxisScale],
            _SHADER_AXIS_SCALES[context.yAxisScale],
        )

        ex = ox + self.scale[0] * self.data.shape[1]
        ey = oy + self.scale[1] * self.data.shape[0]

        xOneOverRange = 1.0 / (ex - ox)
        yOneOverRange = 1.0 / (ey - oy)
        gl.glUniform2f(
            prog.uniforms["bounds_originOverRange"],
            ox * xOneOverRange,
            oy * yOneOverRange,
        )
        gl.glUniform2f(
            prog.uniforms["bounds_oneOverRange"], xOneOverRange, yOneOverRange
        )

        gl.glUniform1f(prog.uniforms["alpha"], self.alpha)

        self._setCMap(prog)

        try:
            tiles = self._texture.tiles
        except AttributeError:
            raise RuntimeError("No texture, discard has already been called")
        if len(tiles) > 1:
            raise NotImplementedError(
                "Image over multiple textures not supported with log scale"
            )

        texture, vertices, info = tiles[0]

        texture.bind(self._DATA_TEX_UNIT)

        posAttrib = prog.attributes["position"]
        stride = vertices.shape[-1] * vertices.itemsize
        gl.glEnableVertexAttribArray(posAttrib)
        gl.glVertexAttribPointer(
            posAttrib, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, vertices
        )

        gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, len(vertices))

    def render(self, context):
        """Perform rendering

        :param RenderContext context: Rendering information
        """
        if any((context.xAxisScale != "linear", context.yAxisScale != "linear")):
            self._renderNonLinear(context)
        else:
            self._renderLinear(context)

        # Unbind colormap texture
        gl.glActiveTexture(gl.GL_TEXTURE0 + self._cmap_texture.texUnit)
        gl.glBindTexture(self._cmap_texture.target, 0)


# image #######################################################################


class GLPlotRGBAImage(_GLPlotData2D):
    _SHADERS = {
        "linear": {
            "vertex": """
    #version 120

    attribute vec2 position;
    attribute vec2 texCoords;
    uniform mat4 matrix;

    varying vec2 coords;

    void main(void) {
        gl_Position = matrix * vec4(position, 0.0, 1.0);
        coords = texCoords;
    }
    """,
            "fragment": """
    #version 120

    uniform sampler2D tex;
    uniform float alpha;

    varying vec2 coords;

    void main(void) {
        gl_FragColor = texture2D(tex, coords);
        gl_FragColor.a *= alpha;
    }
    """,
        },
        "non_linear": {
            "vertex": """
    #version 120

    attribute vec2 position;
    uniform mat4 matrix;
    uniform mat4 matOffset;
    uniform ivec2 axisScale;

    varying vec2 coords;

    const float oneOverLog10 = 0.43429448190325176;

    /* asinh = log(x + sqrt(x*x + 1)) for compatibility with GLSL 1.20 */
    float asinh(float value) {
        return log(value + sqrt(value*value + 1.0));
    }

    void main(void) {
        vec4 dataPos = matOffset * vec4(position, 0.0, 1.0);
        if (axisScale.x == 1) {  /* log scale */
            dataPos.x = oneOverLog10 * log(dataPos.x);
        } else if (axisScale.x == 2) {  /* asinh scale */
            dataPos.x = asinh(dataPos.x);
        }
        if (axisScale.y == 1) {  /* log scale */
            dataPos.y = oneOverLog10 * log(dataPos.y);
        } else if (axisScale.y == 2) {  /* asinh scale */
            dataPos.y = asinh(dataPos.y);
        }
        coords = dataPos.xy;
        gl_Position = matrix * dataPos;
    }
    """,
            "fragment": """
    #version 120

    uniform sampler2D tex;
    uniform ivec2 axisScale;
    uniform vec2 bounds_oneOverRange;
    uniform vec2 bounds_originOverRange;
    uniform float alpha;

    varying vec2 coords;

    /* sinh = (e^x - e^-x)/2 for compatibility with GLSL 1.20 */
    float sinh(float value) {
        return (exp(value) - exp(-value)) / 2.0;
    }

    vec2 textureCoords(void) {
        vec2 pos = coords;
        if (axisScale.x == 1) {  /* log scale */
            pos.x = pow(10., coords.x);
        } else if (axisScale.x == 2) {  /* asinh scale */
            pos.x = sinh(coords.x);
        }
        if (axisScale.y == 1) {  /* log scale */
            pos.y = pow(10., coords.y);
        } else if (axisScale.y == 2) {  /* asinh scale */
            pos.y = sinh(coords.y);
        }
        return pos * bounds_oneOverRange - bounds_originOverRange;
        // TODO texture coords in range different from [0, 1]
    }

    void main(void) {
        gl_FragColor = texture2D(tex, textureCoords());
        gl_FragColor.a *= alpha;
    }
    """,
        },
    }

    _DATA_TEX_UNIT = 0

    _SUPPORTED_DTYPES = (
        numpy.dtype(numpy.float32),
        numpy.dtype(numpy.uint8),
        numpy.dtype(numpy.uint16),
    )

    _linearProgram = Program(
        _SHADERS["linear"]["vertex"], _SHADERS["linear"]["fragment"], attrib0="position"
    )

    _nonLinearProgram = Program(
        _SHADERS["non_linear"]["vertex"],
        _SHADERS["non_linear"]["fragment"],
        attrib0="position",
    )

    def __init__(self, data, origin, scale, alpha):
        """Create a 2D RGB(A) image from data

        :param data: The 2D image data array to display
        :type data: numpy.ndarray with 3 dimensions
                    (dtype=numpy.uint8 or numpy.float32)
        :param origin: (x, y) coordinates of the origin of the data array
        :type origin: 2-tuple of floats.
        :param scale: (sx, sy) scale factors of the data array.
                      This is the size of a data pixel in plot data space.
        :type scale: 2-tuple of floats.
        :param float alpha: Opacity from 0 (transparent) to 1 (opaque)
        """
        assert data.dtype in self._SUPPORTED_DTYPES
        super().__init__(data, origin, scale)
        self._texture = None
        self._textureIsDirty = False
        self._alpha = numpy.clip(alpha, 0.0, 1.0)

    @property
    def alpha(self):
        return self._alpha

    def discard(self):
        if self.isInitialized():
            self._texture.discard()
            self._texture = None
        self._textureIsDirty = False

    def isInitialized(self):
        return self._texture is not None

    def updateData(self, data):
        assert data.dtype in self._SUPPORTED_DTYPES
        oldData = self.data
        self.data = data

        if self._texture is not None:
            if self.data.shape != oldData.shape:
                self.discard()
            else:
                self._textureIsDirty = True

    def prepare(self):
        if self._texture is None:
            formatName = "GL_RGBA" if self.data.shape[2] == 4 else "GL_RGB"
            format_ = getattr(gl, formatName)

            if self.data.dtype == numpy.uint16:
                formatName += "16"  # Use sized internal format for uint16
            internalFormat = getattr(gl, formatName)

            self._texture = Image(
                internalFormat, self.data, format_=format_, texUnit=self._DATA_TEX_UNIT
            )
        elif self._textureIsDirty:
            self._textureIsDirty = False

            # We should check that internal format is the same
            format_ = gl.GL_RGBA if self.data.shape[2] == 4 else gl.GL_RGB
            self._texture.updateAll(format_=format_, data=self.data)

    def _renderLinear(self, context):
        """Perform rendering with both axes having linear scales

        :param RenderContext context: Rendering information
        """
        self.prepare()

        prog = self._linearProgram
        prog.use()

        gl.glUniform1i(prog.uniforms["tex"], self._DATA_TEX_UNIT)

        mat = numpy.dot(
            numpy.dot(context.matrix, mat4Translate(*self.origin)),
            mat4Scale(*self.scale),
        )
        gl.glUniformMatrix4fv(
            prog.uniforms["matrix"], 1, gl.GL_TRUE, mat.astype(numpy.float32)
        )

        gl.glUniform1f(prog.uniforms["alpha"], self.alpha)

        self._texture.render(
            prog.attributes["position"],
            prog.attributes["texCoords"],
            self._DATA_TEX_UNIT,
        )

    def _renderNonLinear(self, context):
        """Perform rendering with non linear axes

        :param RenderContext context: Rendering information
        """
        self.prepare()

        prog = self._nonLinearProgram
        prog.use()

        ox, oy = self.origin

        gl.glUniform1i(prog.uniforms["tex"], self._DATA_TEX_UNIT)

        gl.glUniformMatrix4fv(
            prog.uniforms["matrix"], 1, gl.GL_TRUE, context.matrix.astype(numpy.float32)
        )
        mat = numpy.dot(mat4Translate(ox, oy), mat4Scale(*self.scale))
        gl.glUniformMatrix4fv(
            prog.uniforms["matOffset"], 1, gl.GL_TRUE, mat.astype(numpy.float32)
        )

        gl.glUniform2i(
            prog.uniforms["axisScale"],
            _SHADER_AXIS_SCALES[context.xAxisScale],
            _SHADER_AXIS_SCALES[context.yAxisScale],
        )

        gl.glUniform1f(prog.uniforms["alpha"], self.alpha)

        ex = ox + self.scale[0] * self.data.shape[1]
        ey = oy + self.scale[1] * self.data.shape[0]

        xOneOverRange = 1.0 / (ex - ox)
        yOneOverRange = 1.0 / (ey - oy)
        gl.glUniform2f(
            prog.uniforms["bounds_originOverRange"],
            ox * xOneOverRange,
            oy * yOneOverRange,
        )
        gl.glUniform2f(
            prog.uniforms["bounds_oneOverRange"], xOneOverRange, yOneOverRange
        )

        try:
            tiles = self._texture.tiles
        except AttributeError:
            raise RuntimeError("No texture, discard has already been called")
        if len(tiles) > 1:
            raise NotImplementedError(
                "Image over multiple textures not supported with log scale"
            )

        texture, vertices, info = tiles[0]

        texture.bind(self._DATA_TEX_UNIT)

        posAttrib = prog.attributes["position"]
        stride = vertices.shape[-1] * vertices.itemsize
        gl.glEnableVertexAttribArray(posAttrib)
        gl.glVertexAttribPointer(
            posAttrib, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, vertices
        )

        gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, len(vertices))

    def render(self, context):
        """Perform rendering

        :param RenderContext context: Rendering information
        """
        if any((context.xAxisScale != "linear", context.yAxisScale != "linear")):
            self._renderNonLinear(context)
        else:
            self._renderLinear(context)
