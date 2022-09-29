# /*##########################################################################
#
# Copyright (c) 2016 European Synchrotron Radiation Facility
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
import unittest

from silx.gui.utils.testutils import TestCaseQt

from .. import BackgroundWidget

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "05/12/2016"


class TestBackgroundWidget(TestCaseQt):
    def setUp(self):
        super(TestBackgroundWidget, self).setUp()
        self.bgdialog = BackgroundWidget.BackgroundDialog()
        self.bgdialog.setData(list([0, 1, 2, 3]),
                              list([0, 1, 4, 8]))
        self.qWaitForWindowExposed(self.bgdialog)

    def tearDown(self):
        del self.bgdialog
        super(TestBackgroundWidget, self).tearDown()

    def testShow(self):
        self.bgdialog.show()
        self.bgdialog.hide()

    def testAccept(self):
        self.bgdialog.accept()
        self.assertTrue(self.bgdialog.result())

    def testReject(self):
        self.bgdialog.reject()
        self.assertFalse(self.bgdialog.result())

    def testDefaultOutput(self):
        self.bgdialog.accept()
        output = self.bgdialog.output

        for key in ["algorithm", "StripThreshold",  "SnipWidth",
                    "StripIterations", "StripWidth", "SmoothingFlag",
                    "SmoothingWidth", "AnchorsFlag", "AnchorsList"]:
            self.assertIn(key, output)

        self.assertFalse(output["AnchorsFlag"])
        self.assertEqual(output["StripWidth"], 1)
        self.assertEqual(output["SmoothingFlag"], False)
        self.assertEqual(output["SmoothingWidth"], 3)
