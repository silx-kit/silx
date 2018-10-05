# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2018 European Synchrotron Radiation Facility
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
"""Tests for RangeSlider"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "01/08/2018"

import unittest

from silx.gui import qt, colors
from silx.gui.widgets.RangeSlider import RangeSlider
from silx.gui.utils.testutils import TestCaseQt
from silx.utils.testutils import ParametricTestCase


class TestRangeSlider(TestCaseQt, ParametricTestCase):
    """Tests for TestRangeSlider"""

    def setUp(self):
        self.slider = RangeSlider()
        self.slider.show()
        self.qWaitForWindowExposed(self.slider)

    def tearDown(self):
        self.slider.setAttribute(qt.Qt.WA_DeleteOnClose)
        self.slider.close()
        del self.slider
        self.qapp.processEvents()

    def testRangeValue(self):
        """Test slider range and values"""

        # Play with range
        self.slider.setRange(1, 2)
        self.assertEqual(self.slider.getRange(), (1., 2.))
        self.assertEqual(self.slider.getValues(), (1., 1.))

        self.slider.setMinimum(-1)
        self.assertEqual(self.slider.getRange(), (-1., 2.))
        self.assertEqual(self.slider.getValues(), (1., 1.))

        self.slider.setMaximum(0)
        self.assertEqual(self.slider.getRange(), (-1., 0.))
        self.assertEqual(self.slider.getValues(), (0., 0.))

        # Play with values
        self.slider.setFirstValue(-2.)
        self.assertEqual(self.slider.getValues(), (-1., 0.))

        self.slider.setFirstValue(-0.5)
        self.assertEqual(self.slider.getValues(), (-0.5, 0.))

        self.slider.setSecondValue(2.)
        self.assertEqual(self.slider.getValues(), (-0.5, 0.))

        self.slider.setSecondValue(-0.1)
        self.assertEqual(self.slider.getValues(), (-0.5, -0.1))

    def testStepCount(self):
        """Test related to step count"""
        self.slider.setPositionCount(11)
        self.assertEqual(self.slider.getPositionCount(), 11)
        self.slider.setFirstValue(0.32)
        self.assertEqual(self.slider.getFirstValue(), 0.3)
        self.assertEqual(self.slider.getFirstPosition(), 3)

        self.slider.setPositionCount(3)  # Value is adjusted
        self.assertEqual(self.slider.getValues(), (0.5, 1.))
        self.assertEqual(self.slider.getPositions(), (1, 2))

    def testGroove(self):
        """Test Groove pixmap"""
        profile = list(range(100))

        for cmap in ('jet', colors.Colormap('viridis')):
            with self.subTest(str(cmap)):
                self.slider.setGroovePixmapFromProfile(profile, cmap)
                pixmap = self.slider.getGroovePixmap()
                self.assertIsInstance(pixmap, qt.QPixmap)
                self.assertEqual(pixmap.width(), len(profile))


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    test_suite = unittest.TestSuite()
    test_suite.addTest(loader(TestRangeSlider))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
