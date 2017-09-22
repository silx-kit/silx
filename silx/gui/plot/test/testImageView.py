# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017 European Synchrotron Radiation Facility
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
__date__ = "22/09/2017"


import unittest
import numpy

from .utils import PlotWidgetTestCase

from silx.gui.plot import ImageView
from silx.gui.plot.Colormap import Colormap


class TestImageView(PlotWidgetTestCase):
    """Tests of ImageView widget."""

    def _createPlot(self):
        return ImageView()

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


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestImageView))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
