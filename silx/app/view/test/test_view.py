# coding: utf-8
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
"""Module testing silx.app.view"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "06/06/2018"


import unittest
from silx.test.utils import test_options
from silx.app.view.Viewer import Viewer


if not test_options.WITH_QT_TEST:
    TestCaseQt = unittest.TestCase
else:
    from silx.gui.test.utils import TestCaseQt


class TestViewer(TestCaseQt):
    """Test for Viewer class"""

    @unittest.skipUnless(test_options.WITH_QT_TEST, test_options.WITH_QT_TEST_REASON)
    def testConstruct(self):
        widget = Viewer()
        self.qWaitForWindowExposed(widget)


def suite():
    test_suite = unittest.TestSuite()
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    test_suite.addTest(loader(TestViewer))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
