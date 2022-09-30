# /*##########################################################################
#
# Copyright (c) 2020-2022 European Synchrotron Radiation Facility
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
This module provides a base class for PlotWidget OpenGL backend primitives
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "02/07/2020"


class RenderContext:
    """Context with which to perform OpenGL rendering.

    :param numpy.ndarray matrix: 4x4 transform matrix to use for rendering
    :param bool isXLog: Whether X axis is log scale or not
    :param bool isYLog: Whether Y axis is log scale or not
    :param float dpi: Number of device pixels per inch
    """

    def __init__(self, matrix=None, isXLog=False, isYLog=False, dpi=96., plotFrame=None):
        self.matrix = matrix
        """Current transformation matrix"""

        self.__isXLog = isXLog
        self.__isYLog = isYLog
        self.__dpi = dpi
        self.__plotFrame = plotFrame

    @property
    def isXLog(self):
        """True if X axis is using log scale"""
        return self.__isXLog

    @property
    def isYLog(self):
        """True if Y axis is using log scale"""
        return self.__isYLog

    @property
    def dpi(self):
        """Number of device pixels per inch"""
        return self.__dpi

    @property
    def plotFrame(self):
        """Current PlotFrame"""
        return self.__plotFrame


class GLPlotItem:
    """Base class for primitives used in the PlotWidget OpenGL backend"""

    def __init__(self):
        self.yaxis = 'left'
        "YAxis this item is attached to (either 'left' or 'right')"

    def pick(self, x, y):
        """Perform picking at given position.

        :param float x: X coordinate in plot data frame of reference
        :param float y: Y coordinate in plot data frame of reference
        :returns:
           Result of picking as a list of indices or None if nothing picked
        :rtype: Union[List[int],None]
        """
        return None

    def render(self, context):
        """Performs OpenGL rendering of the item.

        :param RenderContext context: Rendering context information
        """
        pass

    def discard(self):
        """Discards OpenGL resources this item has created."""
        pass

    def isInitialized(self) -> bool:
        """Returns True if resources where initialized and requires `discard`.
        """
        return True
