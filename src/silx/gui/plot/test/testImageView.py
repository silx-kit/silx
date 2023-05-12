# /*##########################################################################
#
# Copyright (c) 2017-2021 European Synchrotron Radiation Facility
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
"""Basic tests for PlotWindow"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "24/04/2018"


import numpy

from silx.gui import qt
from silx.gui.utils.testutils import TestCaseQt
from silx.gui.plot import items

from silx.gui.plot.ImageView import ImageView
from silx.gui.colors import Colormap


class TestImageView(TestCaseQt):
    """Tests of ImageView widget."""

    def setUp(self):
        super(TestImageView, self).setUp()
        self.plot = ImageView()
        self.plot.show()
        self.qWaitForWindowExposed(self.plot)

    def tearDown(self):
        self.qapp.processEvents()
        self.plot.setAttribute(qt.Qt.WA_DeleteOnClose)
        self.plot.close()
        del self.plot
        self.qapp.processEvents()
        super(TestImageView, self).tearDown()

    def testSetImage(self):
        """Test setImage"""
        image = numpy.arange(100).reshape(10, 10)

        self.plot.setImage(image, reset=True)
        self.qWait(100)
        self.assertEqual(self.plot.getXAxis().getLimits(), (0, 10))
        self.assertEqual(self.plot.getYAxis().getLimits(), (0, 10))

        # With reset=False
        self.plot.setImage(image[::2, ::2], reset=False)
        self.qWait(100)
        self.assertEqual(self.plot.getXAxis().getLimits(), (0, 10))
        self.assertEqual(self.plot.getYAxis().getLimits(), (0, 10))

        self.plot.setImage(image, origin=(10, 20), scale=(2, 4), reset=False)
        self.qWait(100)
        self.assertEqual(self.plot.getXAxis().getLimits(), (0, 10))
        self.assertEqual(self.plot.getYAxis().getLimits(), (0, 10))

        # With reset=True
        self.plot.setImage(image, origin=(1, 2), scale=(1, 0.5), reset=True)
        self.qWait(100)
        self.assertEqual(self.plot.getXAxis().getLimits(), (1, 11))
        self.assertEqual(self.plot.getYAxis().getLimits(), (2, 7))

        self.plot.setImage(image[::2, ::2], reset=True)
        self.qWait(100)
        self.assertEqual(self.plot.getXAxis().getLimits(), (0, 5))
        self.assertEqual(self.plot.getYAxis().getLimits(), (0, 5))

    def testColormap(self):
        """Test get|setColormap"""
        image = numpy.arange(100).reshape(10, 10)
        self.plot.setImage(image)

        # Colormap as dict
        self.plot.setColormap({'name': 'viridis',
                               'normalization': 'log',
                               'autoscale': False,
                               'vmin': 0,
                               'vmax': 1})
        colormap = self.plot.getColormap()
        self.assertEqual(colormap.getName(), 'viridis')
        self.assertEqual(colormap.getNormalization(), 'log')
        self.assertEqual(colormap.getVMin(), 0)
        self.assertEqual(colormap.getVMax(), 1)

        # Colormap as keyword arguments
        self.plot.setColormap(colormap='magma',
                              normalization='linear',
                              autoscale=True,
                              vmin=1,
                              vmax=2)
        self.assertEqual(colormap.getName(), 'magma')
        self.assertEqual(colormap.getNormalization(), 'linear')
        self.assertEqual(colormap.getVMin(), None)
        self.assertEqual(colormap.getVMax(), None)

        # Update colormap with keyword argument
        self.plot.setColormap(normalization='log')
        self.assertEqual(colormap.getNormalization(), 'log')

        # Colormap as Colormap object
        cmap = Colormap()
        self.plot.setColormap(cmap)
        self.assertIs(self.plot.getColormap(), cmap)

    def testSetProfileWindowBehavior(self):
        """Test change of profile window display behavior"""
        self.assertIs(
            self.plot.getProfileWindowBehavior(),
            ImageView.ProfileWindowBehavior.POPUP,
        )

        self.plot.setProfileWindowBehavior('embedded')
        self.assertIs(
            self.plot.getProfileWindowBehavior(),
            ImageView.ProfileWindowBehavior.EMBEDDED,
        )

        image = numpy.arange(100).reshape(10, 10)
        self.plot.setImage(image)

        self.plot.setProfileWindowBehavior(
            ImageView.ProfileWindowBehavior.POPUP
        )
        self.assertIs(
            self.plot.getProfileWindowBehavior(),
            ImageView.ProfileWindowBehavior.POPUP,
        )

    def testRGBImage(self):
        """Test setImage"""
        image = numpy.arange(100 * 3, dtype=numpy.uint8).reshape(10, 10, 3)

        self.plot.setImage(image, reset=True)
        self.qWait(100)
        self.assertEqual(self.plot.getXAxis().getLimits(), (0, 10))
        self.assertEqual(self.plot.getYAxis().getLimits(), (0, 10))

    def testRGBAImage(self):
        """Test setImage"""
        image = numpy.arange(100 * 4, dtype=numpy.uint8).reshape(10, 10, 4)

        self.plot.setImage(image, reset=True)
        self.qWait(100)
        self.assertEqual(self.plot.getXAxis().getLimits(), (0, 10))
        self.assertEqual(self.plot.getYAxis().getLimits(), (0, 10))

    def testImageAggregationMode(self):
        """Test setImage"""
        image = numpy.arange(100).reshape(10, 10)
        self.plot.setImage(image, reset=True)
        self.qWait(100)
        self.plot.getAggregationModeAction().setAggregationMode(items.ImageDataAggregated.Aggregation.MAX)
        self.qWait(100)

    def testImageAggregationModeBackToNormalMode(self):
        """Test setImage"""
        image = numpy.arange(100).reshape(10, 10)
        self.plot.setImage(image, reset=True)
        self.qWait(100)
        self.plot.getAggregationModeAction().setAggregationMode(items.ImageDataAggregated.Aggregation.MAX)
        self.qWait(100)
        self.plot.getAggregationModeAction().setAggregationMode(items.ImageDataAggregated.Aggregation.NONE)
        self.qWait(100)

    def testRGBAInAggregationMode(self):
        """Test setImage"""
        image = numpy.arange(100 * 3, dtype=numpy.uint8).reshape(10, 10, 3)

        self.plot.setImage(image, reset=True)
        self.qWait(100)
        self.plot.getAggregationModeAction().setAggregationMode(items.ImageDataAggregated.Aggregation.MAX)
        self.qWait(100)
