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
# ###########################################################################*/
"""Tests for ElidedLabel"""

__license__ = "MIT"
__date__ = "08/06/2020"

import unittest

from silx.gui import qt
from silx.gui.widgets.ElidedLabel import ElidedLabel
from silx.gui.utils import testutils


class TestElidedLabel(testutils.TestCaseQt):

    def setUp(self):
        self.label = ElidedLabel()
        self.label.show()
        self.qWaitForWindowExposed(self.label)

    def tearDown(self):
        self.label.setAttribute(qt.Qt.WA_DeleteOnClose)
        self.label.close()
        del self.label
        self.qapp.processEvents()

    def testElidedValue(self):
        """Test elided text"""
        raw = "mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm"
        self.label.setText(raw)
        self.label.setFixedWidth(30)
        displayedText = qt.QLabel.text(self.label)
        self.assertNotEqual(raw, displayedText)
        self.assertIn("…", displayedText)
        self.assertIn("m", displayedText)

    def testNotElidedValue(self):
        """Test elided text"""
        raw = "mmmmmmm"
        self.label.setText(raw)
        self.label.setFixedWidth(200)
        displayedText = qt.QLabel.text(self.label)
        self.assertNotIn("…", displayedText)
        self.assertEqual(raw, displayedText)

    def testUpdateFromElidedToNotElided(self):
        """Test tooltip when not elided"""
        raw1 = "mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm"
        raw2 = "nn"
        self.label.setText(raw1)
        self.label.setFixedWidth(30)
        self.label.setText(raw2)
        displayedTooltip = qt.QLabel.toolTip(self.label)
        self.assertNotIn(raw1, displayedTooltip)
        self.assertNotIn(raw2, displayedTooltip)

    def testUpdateFromNotElidedToElided(self):
        """Test tooltip when elided"""
        raw1 = "nn"
        raw2 = "mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm"
        self.label.setText(raw1)
        self.label.setFixedWidth(30)
        self.label.setText(raw2)
        displayedTooltip = qt.QLabel.toolTip(self.label)
        self.assertNotIn(raw1, displayedTooltip)
        self.assertIn(raw2, displayedTooltip)

    def testUpdateFromElidedToElided(self):
        """Test tooltip when elided"""
        raw1 = "nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn"
        raw2 = "mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm"
        self.label.setText(raw1)
        self.label.setFixedWidth(30)
        self.label.setText(raw2)
        displayedTooltip = qt.QLabel.toolTip(self.label)
        self.assertNotIn(raw1, displayedTooltip)
        self.assertIn(raw2, displayedTooltip)


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    test_suite = unittest.TestSuite()
    test_suite.addTest(loader(TestElidedLabel))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
