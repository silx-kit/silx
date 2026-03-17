# /*##########################################################################
#
# Copyright (c) 2017-2024 European Synchrotron Radiation Facility
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

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "24/04/2018"


import numpy
import pytest

from ...colors import Colormap
from .. import items
from ..ImageView import ImageView


@pytest.fixture
def imageView(qWidgetFactory):
    yield qWidgetFactory(ImageView)


def testSetImage(imageView, qapp_utils):
    image = numpy.arange(100).reshape(10, 10)

    imageView.setImage(image, resetzoom=True)
    qapp_utils.qWait(100)
    assert imageView.getXAxis().getLimits() == (0, 10)
    assert imageView.getYAxis().getLimits() == (0, 10)

    # With resetzoom=False
    imageView.setImage(image[::2, ::2], resetzoom=False)
    qapp_utils.qWait(100)
    assert imageView.getXAxis().getLimits() == (0, 10)
    assert imageView.getYAxis().getLimits() == (0, 10)

    imageView.setImage(image, origin=(10, 20), scale=(2, 4), resetzoom=False)
    qapp_utils.qWait(100)
    assert imageView.getXAxis().getLimits() == (0, 10)
    assert imageView.getYAxis().getLimits() == (0, 10)

    # With resetzoom=True
    imageView.setImage(image, origin=(1, 2), scale=(1, 0.5), resetzoom=True)
    qapp_utils.qWait(100)
    assert imageView.getXAxis().getLimits() == (1, 11)
    assert imageView.getYAxis().getLimits() == (2, 7)

    imageView.setImage(image[::2, ::2], resetzoom=True)
    qapp_utils.qWait(100)
    assert imageView.getXAxis().getLimits() == (0, 5)
    assert imageView.getYAxis().getLimits() == (0, 5)


def testColormap(imageView):
    image = numpy.arange(100).reshape(10, 10)
    imageView.setImage(image)

    # Colormap as dict
    imageView.setColormap(
        {
            "name": "viridis",
            "normalization": "log",
            "autoscale": False,
            "vmin": 0,
            "vmax": 1,
        }
    )
    colormap = imageView.getColormap()
    assert colormap.getName() == "viridis"
    assert colormap.getNormalization() == "log"
    assert colormap.getVMin() == 0
    assert colormap.getVMax() == 1

    # Colormap as keyword arguments
    imageView.setColormap(
        colormap="magma", normalization="linear", autoscale=True, vmin=1, vmax=2
    )
    assert colormap.getName() == "magma"
    assert colormap.getNormalization() == "linear"
    assert colormap.getVMin() is None
    assert colormap.getVMax() is None

    # Update colormap with keyword argument
    imageView.setColormap(normalization="log")
    assert colormap.getNormalization() == "log"

    # Colormap as Colormap object
    cmap = Colormap()
    imageView.setColormap(cmap)
    assert imageView.getColormap() is cmap


def testSetProfileWindowBehavior(imageView):
    """Test change of profile window display behavior"""
    assert imageView.getProfileWindowBehavior() is ImageView.ProfileWindowBehavior.POPUP

    imageView.setProfileWindowBehavior("embedded")
    assert (
        imageView.getProfileWindowBehavior() is ImageView.ProfileWindowBehavior.EMBEDDED
    )

    image = numpy.arange(100).reshape(10, 10)
    imageView.setImage(image)

    imageView.setProfileWindowBehavior(ImageView.ProfileWindowBehavior.POPUP)
    assert imageView.getProfileWindowBehavior() is ImageView.ProfileWindowBehavior.POPUP


def testRGBImage(imageView, qapp_utils):
    image = numpy.arange(100 * 3, dtype=numpy.uint8).reshape(10, 10, 3)

    imageView.setImage(image, resetzoom=True)
    qapp_utils.qWait(100)
    assert imageView.getXAxis().getLimits() == (0, 10)
    assert imageView.getYAxis().getLimits() == (0, 10)


def testRGBAImage(imageView, qapp_utils):
    image = numpy.arange(100 * 4, dtype=numpy.uint8).reshape(10, 10, 4)

    imageView.setImage(image, resetzoom=True)
    qapp_utils.qWait(100)
    assert imageView.getXAxis().getLimits() == (0, 10)
    assert imageView.getYAxis().getLimits() == (0, 10)


def testImageAggregationMode(imageView, qapp_utils):
    image = numpy.arange(100).reshape(10, 10)
    imageView.setImage(image, resetzoom=True)
    qapp_utils.qWait(100)
    imageView.getAggregationModeAction().setAggregationMode(
        items.ImageDataAggregated.Aggregation.MAX
    )
    qapp_utils.qWait(100)


def testImageAggregationModeBackToNormalMode(imageView, qapp_utils):
    image = numpy.arange(100).reshape(10, 10)
    imageView.setImage(image, resetzoom=True)
    qapp_utils.qWait(100)
    imageView.getAggregationModeAction().setAggregationMode(
        items.ImageDataAggregated.Aggregation.MAX
    )
    qapp_utils.qWait(100)
    imageView.getAggregationModeAction().setAggregationMode(
        items.ImageDataAggregated.Aggregation.NONE
    )
    qapp_utils.qWait(100)


def testRGBAInAggregationMode(imageView, qapp_utils):
    """Test setImage"""
    image = numpy.arange(100 * 3, dtype=numpy.uint8).reshape(10, 10, 3)

    imageView.setImage(image, resetzoom=True)
    qapp_utils.qWait(100)
    imageView.getAggregationModeAction().setAggregationMode(
        items.ImageDataAggregated.Aggregation.MAX
    )
    qapp_utils.qWait(100)
