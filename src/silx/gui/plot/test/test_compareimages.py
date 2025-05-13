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
"""Tests for CompareImages widget"""

__authors__ = ["H. Payno"]
__license__ = "MIT"
__date__ = "23/07/2018"

import pytest
import numpy
import weakref

from silx.gui import qt
from silx.gui.plot.CompareImages import CompareImages


@pytest.fixture
def compareImages(qapp, qapp_utils):
    widget = CompareImages()
    widget.setAttribute(qt.Qt.WA_DeleteOnClose)
    yield widget
    widget.close()
    ref = weakref.ref(widget)
    widget = None
    qapp_utils.qWaitForDestroy(ref)


def testIntensityImage(compareImages):
    image1 = numpy.random.rand(10, 10)
    image2 = numpy.random.rand(10, 10)
    compareImages.setData(image1, image2)


def testRgbImage(compareImages):
    image1 = numpy.random.randint(0, 255, size=(10, 10, 3))
    image2 = numpy.random.randint(0, 255, size=(10, 10, 3))
    compareImages.setData(image1, image2)


def testRgbaImage(compareImages):
    image1 = numpy.random.randint(0, 255, size=(10, 10, 4))
    image2 = numpy.random.randint(0, 255, size=(10, 10, 4))
    compareImages.setData(image1, image2)


def testAlignemnt(compareImages):
    image1 = numpy.random.rand(10, 10)
    image2 = numpy.random.rand(5, 5)
    compareImages.setData(image1, image2)
    for mode in CompareImages.AlignmentMode:
        compareImages.setAlignmentMode(mode)


def testGetPixel(compareImages):
    image1 = numpy.random.rand(11, 11)
    image2 = numpy.random.rand(5, 5)
    image1[5, 5] = 111.111
    image2[2, 2] = 222.222
    compareImages.setData(image1, image2)
    expectedValue = {}
    expectedValue[CompareImages.AlignmentMode.CENTER] = 222.222
    expectedValue[CompareImages.AlignmentMode.STRETCH] = 222.222
    expectedValue[CompareImages.AlignmentMode.ORIGIN] = None
    for mode in expectedValue.keys():
        compareImages.setAlignmentMode(mode)
        data = compareImages.getRawPixelData(11 / 2.0, 11 / 2.0)
        data1, data2 = data
        assert data1 == 111.111
        assert data2 == expectedValue[mode]


def testImageEmpty(compareImages):
    compareImages.setData(image1=None, image2=None)


def testSetImageSeparately(compareImages):
    compareImages.setImage1(numpy.random.rand(10, 10))
    compareImages.setImage2(numpy.random.rand(10, 10))


@pytest.mark.parametrize(
    "data",
    [
        (CompareImages.VisualizationMode.COMPOSITE_A_MINUS_B,),
        (CompareImages.VisualizationMode.COMPOSITE_RED_BLUE_GRAY,),
        (CompareImages.VisualizationMode.HORIZONTAL_LINE,),
        (CompareImages.VisualizationMode.VERTICAL_LINE,),
        (CompareImages.VisualizationMode.ONLY_A,),
        (CompareImages.VisualizationMode.ONLY_B,),
    ],
)
def testVisualizationMode(compareImages, data):
    (visualizationMode,) = data
    compareImages.setImage1(numpy.random.rand(10, 10))
    compareImages.setImage2(numpy.random.rand(10, 10))
    compareImages.setVisualizationMode(visualizationMode)


@pytest.mark.parametrize(
    "data",
    [
        (CompareImages.VisualizationMode.COMPOSITE_A_MINUS_B,),
        (CompareImages.VisualizationMode.COMPOSITE_RED_BLUE_GRAY,),
        (CompareImages.VisualizationMode.HORIZONTAL_LINE,),
        (CompareImages.VisualizationMode.VERTICAL_LINE,),
        (CompareImages.VisualizationMode.ONLY_A,),
        (CompareImages.VisualizationMode.ONLY_B,),
    ],
)
def testVisualizationModeWithoutImage(compareImages, data):
    (visualizationMode,) = data
    compareImages.setImage1(None)
    compareImages.setImage2(None)
    compareImages.setVisualizationMode(visualizationMode)


@pytest.mark.parametrize(
    "data",
    [
        (CompareImages.VisualizationMode.COMPOSITE_A_MINUS_B,),
        (CompareImages.VisualizationMode.COMPOSITE_RED_BLUE_GRAY,),
        (CompareImages.VisualizationMode.HORIZONTAL_LINE,),
        (CompareImages.VisualizationMode.VERTICAL_LINE,),
        (CompareImages.VisualizationMode.ONLY_A,),
        (CompareImages.VisualizationMode.ONLY_B,),
    ],
)
def testVisualizationModeWithOnlyImage1(compareImages, data):
    (visualizationMode,) = data
    compareImages.setImage1(numpy.random.rand(10, 10))
    compareImages.setImage2(None)
    compareImages.setVisualizationMode(visualizationMode)


@pytest.mark.parametrize(
    "data",
    [
        (CompareImages.VisualizationMode.COMPOSITE_A_MINUS_B,),
        (CompareImages.VisualizationMode.COMPOSITE_RED_BLUE_GRAY,),
        (CompareImages.VisualizationMode.HORIZONTAL_LINE,),
        (CompareImages.VisualizationMode.VERTICAL_LINE,),
        (CompareImages.VisualizationMode.ONLY_A,),
        (CompareImages.VisualizationMode.ONLY_B,),
    ],
)
def testVisualizationModeWithOnlyImage2(compareImages, data):
    (visualizationMode,) = data
    compareImages.setImage1(None)
    compareImages.setImage2(numpy.random.rand(10, 10))
    compareImages.setVisualizationMode(visualizationMode)


@pytest.mark.parametrize(
    "data",
    [
        (CompareImages.VisualizationMode.COMPOSITE_A_MINUS_B,),
        (CompareImages.VisualizationMode.COMPOSITE_RED_BLUE_GRAY,),
        (CompareImages.VisualizationMode.HORIZONTAL_LINE,),
        (CompareImages.VisualizationMode.VERTICAL_LINE,),
        (CompareImages.VisualizationMode.ONLY_A,),
        (CompareImages.VisualizationMode.ONLY_B,),
    ],
)
def testVisualizationModeWithRGBImage(compareImages, data):
    (visualizationMode,) = data
    image1 = numpy.random.rand(10, 10)
    image2 = numpy.random.randint(0, 255, size=(10, 10, 3))
    compareImages.setData(image1, image2)
    compareImages.setVisualizationMode(visualizationMode)


@pytest.mark.parametrize(
    "data",
    [
        (CompareImages.AlignmentMode.STRETCH,),
        (CompareImages.AlignmentMode.AUTO,),
        (CompareImages.AlignmentMode.CENTER,),
        (CompareImages.AlignmentMode.ORIGIN,),
    ],
)
def testAlignemntModeWithoutImages(compareImages, data):
    (alignmentMode,) = data
    compareImages.setAlignmentMode(alignmentMode)


@pytest.mark.parametrize(
    "data",
    [
        (CompareImages.AlignmentMode.STRETCH,),
        (CompareImages.AlignmentMode.AUTO,),
        (CompareImages.AlignmentMode.CENTER,),
        (CompareImages.AlignmentMode.ORIGIN,),
    ],
)
def testAlignemntModeWithSingleImage(compareImages, data):
    (alignmentMode,) = data
    compareImages.setImage1(numpy.arange(9).reshape(3, 3))
    compareImages.setAlignmentMode(alignmentMode)


def testTooltip(compareImages):
    compareImages.setImage1(numpy.arange(9).reshape(3, 3))
    compareImages.setImage2(numpy.arange(9).reshape(3, 3))
    compareImages.getRawPixelData(1.5, 1.5)


def testTooltipWithoutImage(compareImages):
    compareImages.setImage1(numpy.arange(9).reshape(3, 3))
    compareImages.setImage2(numpy.arange(9).reshape(3, 3))
    compareImages.getRawPixelData(1.5, 1.5)


def testTooltipWithSingleImage(compareImages):
    compareImages.setImage1(numpy.arange(9).reshape(3, 3))
    compareImages.getRawPixelData(1.5, 1.5)
