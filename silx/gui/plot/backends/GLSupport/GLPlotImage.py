# /*#########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2004-2015 European Synchrotron Radiation Facility
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
This module provides a class to render 2D array as a colormap or RGB(A) image
"""


# import ######################################################################

from .gl import *  # noqa

import math
from .GLSupport import mat4Translate, mat4Scale, FLOAT32_MINPOS
from .GLProgram import GLProgram
from .GLTexture import Image

try:
    from ....ctools import minMax
except ImportError:
    from PyMca5.PyMcaGraph.ctools import minMax


# colormap ####################################################################

class _GLPlotData2D(object):
    def __init__(self, data, origin, scale):
        self.data = data
        assert len(origin) == 2
        self.origin = tuple(origin)
        assert len(scale) == 2
        self.scale = tuple(scale)

    def pick(self, x, y):
        if (self.xMin <= x and x <= self.xMax and
                self.yMin <= y and y <= self.yMax):
            ox, oy = self.origin
            sx, sy = self.scale
            col = int((x - ox) / sx)
            row = int((y - oy) / sy)
            return col, row
        else:
            return None

    @property
    def xMin(self):
        ox, sx = self.origin[0], self.scale[0]
        return ox if sx >= 0. else ox + sx * self.data.shape[1]

    @property
    def yMin(self):
        oy, sy = self.origin[1], self.scale[1]
        return oy if sy >= 0. else oy + sy * self.data.shape[0]

    @property
    def xMax(self):
        ox, sx = self.origin[0], self.scale[0]
        return ox + sx * self.data.shape[1] if sx >= 0. else ox

    @property
    def yMax(self):
        oy, sy = self.origin[1], self.scale[1]
        return oy + sy * self.data.shape[0] if sy >= 0. else oy

    def discard(self):
        pass

    def prepare(self):
        pass

    def render(self, matrix):
        pass


class GLPlotColormap(_GLPlotData2D):

    _SHADERS = {
        'linear': {
            'vertex': """
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
            'fragTransform': """
    vec2 textureCoords(void) {
        return coords;
    }
    """},

        'log': {
            'vertex': """
    #version 120

    attribute vec2 position;
    uniform mat4 matrix;
    uniform mat4 matOffset;
    uniform bvec2 isLog;

    varying vec2 coords;

    const float oneOverLog10 = 0.43429448190325176;

    void main(void) {
        vec4 dataPos = matOffset * vec4(position, 0.0, 1.0);
        if (isLog.x) {
            dataPos.x = oneOverLog10 * log(dataPos.x);
        }
        if (isLog.y) {
            dataPos.y = oneOverLog10 * log(dataPos.y);
        }
        coords = dataPos.xy;
        gl_Position = matrix * dataPos;
    }
    """,
            'fragTransform': """
    uniform bvec2 isLog;
    uniform struct {
        vec2 oneOverRange;
        vec2 originOverRange;
    } bounds;

    vec2 textureCoords(void) {
        vec2 pos = coords;
        if (isLog.x) {
            pos.x = pow(10., coords.x);
        }
        if (isLog.y) {
            pos.y = pow(10., coords.y);
        }
        return pos * bounds.oneOverRange - bounds.originOverRange;
        // TODO texture coords in range different from [0, 1]
    }
    """},

        'fragment': """
    #version 120

    #define CMAP_GRAY   0
    #define CMAP_R_GRAY 1
    #define CMAP_RED    2
    #define CMAP_GREEN  3
    #define CMAP_BLUE   4
    #define CMAP_TEMP   5

    uniform sampler2D data;
    uniform struct {
        int id;
        bool isLog;
        float min;
        float oneOverRange;
    } cmap;

    varying vec2 coords;

    %s

    vec4 cmapGray(float normValue) {
        return vec4(normValue, normValue, normValue, 1.);
    }

    vec4 cmapReversedGray(float normValue) {
        float invValue = 1. - normValue;
        return vec4(invValue, invValue, invValue, 1.);
    }

    vec4 cmapRed(float normValue) {
        return vec4(normValue, 0., 0., 1.);
    }

    vec4 cmapGreen(float normValue) {
        return vec4(0., normValue, 0., 1.);
    }

    vec4 cmapBlue(float normValue) {
        return vec4(0., 0., normValue, 1.);
    }

    //red: 0.5->0.75: 0->1
    //green: 0.->0.25: 0->1; 0.75->1.: 1->0
    //blue: 0.25->0.5: 1->0
    vec4 cmapTemperature(float normValue) {
        float red = clamp(4. * normValue - 2., 0., 1.);
        float green = 1. - clamp(4. * abs(normValue - 0.5) - 1., 0., 1.);
        float blue = 1. - clamp(4. * normValue - 1., 0., 1.);
        return vec4(red, green, blue, 1.);
    }

    const float oneOverLog10 = 0.43429448190325176;

    void main(void) {
        float value = texture2D(data, textureCoords()).r;
        if (cmap.isLog) {
            if (value > 0.) {
                value = clamp(cmap.oneOverRange *
                              (oneOverLog10 * log(value) - cmap.min),
                              0., 1.);
            } else {
                value = 0.;
            }
        } else { /*Linear mapping*/
            value = clamp(cmap.oneOverRange * (value - cmap.min), 0., 1.);
        }

        if (cmap.id == CMAP_GRAY) {
            gl_FragColor = cmapGray(value);
        } else if (cmap.id == CMAP_R_GRAY) {
            gl_FragColor = cmapReversedGray(value);
        } else if (cmap.id == CMAP_RED) {
            gl_FragColor = cmapRed(value);
        } else if (cmap.id == CMAP_GREEN) {
            gl_FragColor = cmapGreen(value);
        } else if (cmap.id == CMAP_BLUE) {
            gl_FragColor = cmapBlue(value);
        } else if (cmap.id == CMAP_TEMP) {
            gl_FragColor = cmapTemperature(value);
        }
    }
    """
    }

    _SHADER_CMAP_IDS = {
        'gray': 0,
        'reversed gray': 1,
        'red': 2,
        'green': 3,
        'blue': 4,
        'temperature': 5
    }

    COLORMAPS = tuple(_SHADER_CMAP_IDS.keys())

    _DATA_TEX_UNIT = 0

    _INTERNAL_FORMATS = {
        np.dtype(np.float32): GL_R32F,
        # Use normalized integer for unsigned int formats
        np.dtype(np.uint16): GL_R16,
        np.dtype(np.uint8): GL_R8,
    }

    _linearProgram = GLProgram(_SHADERS['linear']['vertex'],
                               _SHADERS['fragment'] %
                               _SHADERS['linear']['fragTransform'])

    _logProgram = GLProgram(_SHADERS['log']['vertex'],
                            _SHADERS['fragment'] %
                            _SHADERS['log']['fragTransform'])

    def __init__(self, data, origin, scale,
                 colormap, cmapIsLog=False, cmapRange=None):
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
        :param bool cmapIsLog: If True, uses log10 of the data value
        :param cmapRange: The range of colormap or None for autoscale colormap
            For logarithmic colormap, the range is in the untransformed data
            TODO: check consistency with matplotlib
        :type cmapRange: (float, float) or None
        """
        assert data.dtype in self._INTERNAL_FORMATS

        super(GLPlotColormap, self).__init__(data, origin, scale)
        self.colormap = colormap
        self.cmapIsLog = cmapIsLog
        self._cmapRange = None  # User-provided range info
        self._cmapRangeCache = None  # Store extra data for range
        self.cmapRange = cmapRange  # Update _cmapRange

        self._textureIsDirty = False

    def __del__(self):
        self.discard()

    def discard(self):
        if hasattr(self, '_texture'):
            self._texture.discard()
            del self._texture
        self._textureIsDirty = False

    @property
    def cmapRange(self):
        if self._cmapRange is None:  # Auto-scale mode
            if self._cmapRangeCache is None:
                # Build data , positive ranges
                min_, minPos, max_ = minMax(self.data, minPositive=True)
                maxPos = max_ if max_ > 0. else 1.
                if minPos is None:
                    minPos = maxPos
                self._cmapRangeCache = {'range': (min_, max_),
                                        'pos': (minPos, maxPos)}

            return self._cmapRangeCache['pos' if self.cmapIsLog else 'range']

        else:
            if not self.cmapIsLog:
                return self._cmapRange  # Return range as is
            else:
                if self._cmapRangeCache is None:
                    # Build a strictly positive range from cmapRange
                    min_, max_ = self._cmapRange
                    if min_ > 0. and max_ > 0.:
                        minPos, maxPos = min_, max_
                    else:
                        dataMin, minPos, dataMax = minMax(self.data,
                                                          minPositive=True)
                        if max_ > 0.:
                            maxPos = max_
                        elif dataMax > 0.:
                            maxPos = dataMax
                        else:
                            maxPos = 1.  # Arbitrary fallback
                        if minPos is None:
                            minPos = maxPos
                    self._cmapRangeCache = minPos, maxPos
                return self._cmapRangeCache  # Strictly positive range

    @cmapRange.setter
    def cmapRange(self, cmapRange):
        self._cmapRangeCache = None
        if cmapRange is None:
            self._cmapRange = None
        else:
            assert len(cmapRange) == 2
            assert cmapRange[0] <= cmapRange[1]
            self._cmapRange = tuple(cmapRange)

    def updateData(self, data):
        assert data.dtype in self._INTERNAL_FORMATS
        oldData = self.data
        self.data = data

        self._cmapRangeCache = None

        if hasattr(self, '_texture'):
            if (self.data.shape != oldData.shape or
                    self.data.dtype != oldData.dtype):
                self.discard()
            else:
                self._textureIsDirty = True

    def prepare(self):
        if not hasattr(self, '_texture'):
            internalFormat = self._INTERNAL_FORMATS[self.data.dtype]
            height, width = self.data.shape

            self._texture = Image(internalFormat, width, height,
                                  format_=GL_RED,
                                  type_=numpyToGLType(self.data.dtype),
                                  data=self.data,
                                  texUnit=self._DATA_TEX_UNIT)
        elif self._textureIsDirty:
            self._textureIsDirty = True
            self._texture.updateAll(format_=GL_RED,
                                    type_=numpyToGLType(self.data.dtype),
                                    data=self.data)

    def _setCMap(self, prog):
        dataMin, dataMax = self.cmapRange  # If log, it is stricly positive

        if self.data.dtype in (np.uint16, np.uint8):
            # Using unsigned int as normalized integer in OpenGL
            # So normalize range
            maxInt = float(np.iinfo(self.data.dtype).max)
            dataMin, dataMax = dataMin / maxInt, dataMax / maxInt

        if self.cmapIsLog:
            dataMin = math.log10(dataMin)
            dataMax = math.log10(dataMax)

        glUniform1i(prog.uniforms['cmap.id'],
                    self._SHADER_CMAP_IDS[self.colormap])
        glUniform1i(prog.uniforms['cmap.isLog'], self.cmapIsLog)
        glUniform1f(prog.uniforms['cmap.min'], dataMin)
        if dataMax > dataMin:
            oneOverRange = 1. / (dataMax - dataMin)
        else:
            oneOverRange = 0.  # Fall-back
        glUniform1f(prog.uniforms['cmap.oneOverRange'], oneOverRange)

    def _renderLinear(self, matrix):
        self.prepare()

        prog = self._linearProgram
        prog.use()

        glUniform1i(prog.uniforms['data'], self._DATA_TEX_UNIT)

        mat = matrix * mat4Translate(*self.origin) * mat4Scale(*self.scale)
        glUniformMatrix4fv(prog.uniforms['matrix'], 1, GL_TRUE, mat)

        self._setCMap(prog)

        self._texture.render(prog.attributes['position'],
                             prog.attributes['texCoords'],
                             self._DATA_TEX_UNIT)

    def _renderLog10(self, matrix, isXLog, isYLog):
        xMin, yMin = self.xMin, self.yMin
        if ((isXLog and xMin < FLOAT32_MINPOS) or
                (isYLog and yMin < FLOAT32_MINPOS)):
            # Do not render images that are partly or totally <= 0
            return

        self.prepare()

        prog = self._logProgram
        prog.use()

        ox, oy = self.origin

        glUniform1i(prog.uniforms['data'], self._DATA_TEX_UNIT)

        glUniformMatrix4fv(prog.uniforms['matrix'], 1, GL_TRUE, matrix)
        mat = mat4Translate(ox, oy) * mat4Scale(*self.scale)
        glUniformMatrix4fv(prog.uniforms['matOffset'], 1, GL_TRUE, mat)

        glUniform2i(prog.uniforms['isLog'], isXLog, isYLog)

        ex = ox + self.scale[0] * self.data.shape[1]
        ey = oy + self.scale[1] * self.data.shape[0]

        xOneOverRange = 1. / (ex - ox)
        yOneOverRange = 1. / (ey - oy)
        glUniform2f(prog.uniforms['bounds.originOverRange'],
                    ox * xOneOverRange, oy * yOneOverRange)
        glUniform2f(prog.uniforms['bounds.oneOverRange'],
                    xOneOverRange, yOneOverRange)

        self._setCMap(prog)

        try:
            tiles = self._texture.tiles
        except AttributeError:
            raise RuntimeError("No texture, discard has already been called")
        if len(tiles) > 1:
            raise NotImplementedError(
                "Image over multiple textures not supported with log scale")

        texture, vertices, info = tiles[0]

        texture.bind(self._DATA_TEX_UNIT)

        posAttrib = prog.attributes['position']
        stride = vertices.shape[-1] * vertices.itemsize
        glEnableVertexAttribArray(posAttrib)
        glVertexAttribPointer(posAttrib,
                              2,
                              GL_FLOAT,
                              GL_FALSE,
                              stride, vertices)

        glDrawArrays(GL_TRIANGLE_STRIP, 0, len(vertices))

    def render(self, matrix, isXLog, isYLog):
        if any((isXLog, isYLog)):
            self._renderLog10(matrix, isXLog, isYLog)
        else:
            self._renderLinear(matrix)


# image #######################################################################

class GLPlotRGBAImage(_GLPlotData2D):

    _SHADERS = {
        'linear': {
            'vertex': """
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
            'fragment': """
    #version 120

    uniform sampler2D tex;

    varying vec2 coords;

    void main(void) {
        gl_FragColor = texture2D(tex, coords);
    }
    """},

        'log': {
            'vertex': """
    #version 120

    attribute vec2 position;
    uniform mat4 matrix;
    uniform mat4 matOffset;
    uniform bvec2 isLog;

    varying vec2 coords;

    const float oneOverLog10 = 0.43429448190325176;

    void main(void) {
        vec4 dataPos = matOffset * vec4(position, 0.0, 1.0);
        if (isLog.x) {
            dataPos.x = oneOverLog10 * log(dataPos.x);
        }
        if (isLog.y) {
            dataPos.y = oneOverLog10 * log(dataPos.y);
        }
        coords = dataPos.xy;
        gl_Position = matrix * dataPos;
    }
    """,
            'fragment': """
    #version 120

    uniform sampler2D tex;
    uniform bvec2 isLog;
    uniform struct {
        vec2 oneOverRange;
        vec2 originOverRange;
    } bounds;

    varying vec2 coords;

    vec2 textureCoords(void) {
        vec2 pos = coords;
        if (isLog.x) {
            pos.x = pow(10., coords.x);
        }
        if (isLog.y) {
            pos.y = pow(10., coords.y);
        }
        return pos * bounds.oneOverRange - bounds.originOverRange;
        // TODO texture coords in range different from [0, 1]
    }

    void main(void) {
        gl_FragColor = texture2D(tex, textureCoords());
    }
    """}
    }

    _DATA_TEX_UNIT = 0

    _SUPPORTED_DTYPES = (np.dtype(np.float32), np.dtype(np.uint8))

    _linearProgram = GLProgram(_SHADERS['linear']['vertex'],
                               _SHADERS['linear']['fragment'])

    _logProgram = GLProgram(_SHADERS['log']['vertex'],
                            _SHADERS['log']['fragment'])

    def __init__(self, data, origin, scale):
        """Create a 2D RGB(A) image from data

        :param data: The 2D image data array to display
        :type data: numpy.ndarray with 3 dimensions
                    (dtype=numpy.uint8 or numpy.float32)
        :param origin: (x, y) coordinates of the origin of the data array
        :type origin: 2-tuple of floats.
        :param scale: (sx, sy) scale factors of the data array.
                      This is the size of a data pixel in plot data space.
        :type scale: 2-tuple of floats.
        """
        assert data.dtype in self._SUPPORTED_DTYPES
        super(GLPlotRGBAImage, self).__init__(data, origin, scale)
        self._textureIsDirty = False

    def __del__(self):
        self.discard()

    def discard(self):
        if hasattr(self, '_texture'):
            self._texture.discard()
            del self._texture
        self._textureIsDirty = False

    def updateData(self, data):
        assert data.dtype in self._SUPPORTED_DTYPES
        oldData = self.data
        self.data = data

        if hasattr(self, '_texture'):
            if self.data.shape != oldData.shape:
                self.discard()
            else:
                self._textureIsDirty = True

    def prepare(self):
        if not hasattr(self, '_texture'):
            height, width, depth = self.data.shape
            format_ = GL_RGBA if depth == 4 else GL_RGB
            type_ = numpyToGLType(self.data.dtype)

            self._texture = Image(format_, width, height,
                                  format_=format_, type_=type_,
                                  data=self.data,
                                  texUnit=self._DATA_TEX_UNIT)
        elif self._textureIsDirty:
            self._textureIsDirty = False

            # We should check that internal format is the same
            format_ = GL_RGBA if self.data.shape[2] == 4 else GL_RGB
            type_ = numpyToGLType(self.data.dtype)
            self._texture.updateAll(format_=format_, type_=type_,
                                    data=self.data)

    def _renderLinear(self, matrix):
        self.prepare()

        prog = self._linearProgram
        prog.use()

        glUniform1i(prog.uniforms['tex'], self._DATA_TEX_UNIT)

        mat = matrix * mat4Translate(*self.origin) * mat4Scale(*self.scale)
        glUniformMatrix4fv(prog.uniforms['matrix'], 1, GL_TRUE, mat)

        self._texture.render(prog.attributes['position'],
                             prog.attributes['texCoords'],
                             self._DATA_TEX_UNIT)

    def _renderLog(self, matrix, isXLog, isYLog):
        self.prepare()

        prog = self._logProgram
        prog.use()

        ox, oy = self.origin

        glUniform1i(prog.uniforms['tex'], self._DATA_TEX_UNIT)

        glUniformMatrix4fv(prog.uniforms['matrix'], 1, GL_TRUE, matrix)
        mat = mat4Translate(ox, oy) * mat4Scale(*self.scale)
        glUniformMatrix4fv(prog.uniforms['matOffset'], 1, GL_TRUE, mat)

        glUniform2i(prog.uniforms['isLog'], isXLog, isYLog)

        ex = ox + self.scale[0] * self.data.shape[1]
        ey = oy + self.scale[1] * self.data.shape[0]

        xOneOverRange = 1. / (ex - ox)
        yOneOverRange = 1. / (ey - oy)
        glUniform2f(prog.uniforms['bounds.originOverRange'],
                    ox * xOneOverRange, oy * yOneOverRange)
        glUniform2f(prog.uniforms['bounds.oneOverRange'],
                    xOneOverRange, yOneOverRange)

        try:
            tiles = self._texture.tiles
        except AttributeError:
            raise RuntimeError("No texture, discard has already been called")
        if len(tiles) > 1:
            raise NotImplementedError(
                "Image over multiple textures not supported with log scale")

        texture, vertices, info = tiles[0]

        texture.bind(self._DATA_TEX_UNIT)

        posAttrib = prog.attributes['position']
        stride = vertices.shape[-1] * vertices.itemsize
        glEnableVertexAttribArray(posAttrib)
        glVertexAttribPointer(posAttrib,
                              2,
                              GL_FLOAT,
                              GL_FALSE,
                              stride, vertices)

        glDrawArrays(GL_TRIANGLE_STRIP, 0, len(vertices))

    def render(self, matrix, isXLog, isYLog):
        if any((isXLog, isYLog)):
            self._renderLog(matrix, isXLog, isYLog)
        else:
            self._renderLinear(matrix)
