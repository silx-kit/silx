# /*##########################################################################
#
# Copyright (c) 2018-2019 European Synchrotron Radiation Facility
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
"""This module provides main objects shared by the compare image plot.
"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "09/06/2023"


from silx.gui.plot.items.image import ImageBase
from silx.gui.plot.items.core import ItemChangedType


import enum
from typing import NamedTuple


from silx.opencl import ocl
if ocl is not None:
    try:
        from silx.opencl import sift
    except ImportError:
        # sift module is not available (e.g., in official Debian packages)
        sift = None
else:  # No OpenCL device or no pyopencl
    sift = None


@enum.unique
class VisualizationMode(enum.Enum):
    """Enum for each visualization mode available."""
    ONLY_A = 'a'
    ONLY_B = 'b'
    VERTICAL_LINE = 'vline'
    HORIZONTAL_LINE = 'hline'
    COMPOSITE_RED_BLUE_GRAY = "rbgchannel"
    COMPOSITE_RED_BLUE_GRAY_NEG = "rbgnegchannel"
    COMPOSITE_A_MINUS_B = "aminusb"


@enum.unique
class AlignmentMode(enum.Enum):
    """Enum for each alignment mode available."""
    ORIGIN = 'origin'
    CENTER = 'center'
    STRETCH = 'stretch'
    AUTO = 'auto'


class AffineTransformation(NamedTuple):
    """Description of a 2D affine transformation: translation, scale and
    rotation.
    """
    tx: float
    ty: float
    sx: float
    sy: float
    rot: float


class _CompareImageItem(ImageBase):
    """Description of a virtual item of images to compare, in order to share
    to other places.
    """

    def __init__(self):
        super(_CompareImageItem, self).__init__()
        self.__image1 = None
        self.__image2 = None

    def getImageData1(self):
        return self.__image1

    def getImageData2(self):
        return self.__image2

    def setImageData1(self, image1):
        if self.__image1 is image1:
            return
        self.__image1 = image1
        self._updated(ItemChangedType.DATA)

    def setImageData2(self, image2):
        if self.__image2 is image2:
            return
        self.__image2 = image2
        self._updated(ItemChangedType.DATA)
