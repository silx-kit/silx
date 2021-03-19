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
__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "23/03/2018"


import unittest

from silx.gui.utils.testutils import TestCaseQt
from silx.gui.widgets.FrameBrowser import FrameBrowser


class TestFrameBrowser(TestCaseQt):
    """Test for FrameBrowser"""

    def test(self):
        """Test FrameBrowser"""
        widget = FrameBrowser()
        widget.show()
        self.qWaitForWindowExposed(widget)

        nFrames = 20
        widget.setNFrames(nFrames)
        self.assertEqual(widget.getRange(), (0, nFrames - 1))
        self.assertEqual(widget.getValue(), 0)

        range_ = -100, 100
        widget.setRange(*range_)
        self.assertEqual(widget.getRange(), range_)
        self.assertEqual(widget.getValue(), range_[0])

        widget.setValue(0)
        self.assertEqual(widget.getValue(), 0)

        widget.setValue(range_[1] + 100)
        self.assertEqual(widget.getValue(), range_[1])

        widget.setValue(range_[0] - 100)
        self.assertEqual(widget.getValue(), range_[0])


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    test_suite = unittest.TestSuite()
    test_suite.addTest(loader(TestFrameBrowser))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
