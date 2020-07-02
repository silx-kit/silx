# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2020 European Synchrotron Radiation Facility
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


class GLPlotItem:
    """Base class for primitives used in the PlotWidget OpenGL backend"""
    def __init__(self):
        pass

    def pick(self, x, y):
        """Perform picking at given position.

        :param float x: X coordinate in plot data frame of reference
        :param float y: Y coordinate in plot data frame of reference
        :returns:
           Result of picking as a list of indices or None if nothing picked
        :rtype: Union[List[int],None]
        """
        return None

    def render(self, matrix, isXLog, isYLog):
        """Performs OpenGL rendering of the item.

        :param numpy.ndarray matrix: 4x4 transform matrix to use for rendering
        :param bool isXLog: Whether X axis is log scale or not
        :param bool isYLog: Whether Y axis is log scale or not
        """
        pass

    def discard(self):
        """Discards OpenGL resources this item has created."""
        pass
