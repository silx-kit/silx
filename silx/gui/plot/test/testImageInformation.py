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
"""Basic tests for ImageInformation"""


__authors__ = ["H. Payno"]
__license__ = "MIT"
__date__ = "05/17/2019"


from silx.gui import qt
from silx.gui.utils.testutils import TestCaseQt
from silx.gui.plot import PlotWindow
from silx.gui.plot.ImageInformation import ImageInformationWidget

import unittest
import numpy


class TestImageInformation(TestCaseQt):
    """Tests of ImageInformation widget."""
    def setUp(self):
        self.plot = PlotWindow(position=False, control=False)
        self.informationWidget = ImageInformationWidget()
        self.informationWidget.layout().setContentsMargins(0, 0, 0, 0)
        self.plot.setImageInfoWidget(self.informationWidget)

    def tearDown(self):
        self.plot.setAttribute(qt.Qt.WA_DeleteOnClose)
        self.plot.close()

    def testBehavior(self):
        self.assertTrue(self.getDimsDisplayed() == '')
        self.plot.addImage(numpy.random.random((246, 5000)))
        self.assertTrue(self.getDimsDisplayed() == 'dims: 5000 x 246')
        self.plot.addImage(numpy.random.random((32, 32)))
        self.assertTrue(self.getDimsDisplayed() == 'dims: 32 x 32')
        assert self.plot.getImageInfoWidget() == self.informationWidget

    def getDimsDisplayed(self):
        """Return dims currently displayed by the image information widget"""
        return self.plot.getImageInfoWidget()._qLabel.text()


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestImageInformation))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')