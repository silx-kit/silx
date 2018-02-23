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
__date__ = "09/11/2017"


import unittest
import sys
from silx.test.utils import test_options


if not test_options.WITH_QT_TEST:
    view = None
    TestCaseQt = unittest.TestCase
else:
    from silx.gui.test.utils import TestCaseQt
    from .. import view


class QApplicationMock(object):

    def __init__(self, args):
        pass

    def exec_(self):
        return 0

    def deleteLater(self):
        pass


class ViewerMock(object):

    def __init__(self):
        super(ViewerMock, self).__init__()
        self.__class__._instance = self
        self.appendFileCalls = []

    def appendFile(self, filename):
        self.appendFileCalls.append(filename)

    def setAttribute(self, attr, value):
        pass

    def resize(self, size):
        pass

    def show(self):
        pass


@unittest.skipUnless(test_options.WITH_QT_TEST, test_options.WITH_QT_TEST_REASON)
class TestLauncher(unittest.TestCase):
    """Test command line parsing"""

    @classmethod
    def setUpClass(cls):
        super(TestLauncher, cls).setUpClass()
        cls._Viewer = view.Viewer
        view.Viewer = ViewerMock
        cls._QApplication = view.qt.QApplication
        view.qt.QApplication = QApplicationMock

    @classmethod
    def tearDownClass(cls):
        view.Viewer = cls._Viewer
        view.qt.QApplication = cls._QApplication
        cls._Viewer = None
        super(TestLauncher, cls).tearDownClass()

    def testHelp(self):
        # option -h must cause a raise SystemExit or a return 0
        try:
            result = view.main(["view", "--help"])
        except SystemExit as e:
            result = e.args[0]
        self.assertEqual(result, 0)

    def testWrongOption(self):
        try:
            result = view.main(["view", "--foo"])
        except SystemExit as e:
            result = e.args[0]
        self.assertNotEqual(result, 0)

    def testWrongFile(self):
        try:
            result = view.main(["view", "__file.not.found__"])
        except SystemExit as e:
            result = e.args[0]
        self.assertEqual(result, 0)

    def testFile(self):
        # sys.executable is an existing readable file
        result = view.main(["view", sys.executable])
        self.assertEqual(result, 0)
        viewer = ViewerMock._instance
        self.assertEqual(viewer.appendFileCalls, [sys.executable])
        ViewerMock._instance = None


class TestViewer(TestCaseQt):
    """Test for Viewer class"""

    @unittest.skipUnless(test_options.WITH_QT_TEST, test_options.WITH_QT_TEST_REASON)
    def testConstruct(self):
        if view is not None:
            widget = view.Viewer()
        self.qWaitForWindowExposed(widget)


def suite():
    test_suite = unittest.TestSuite()
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    test_suite.addTest(loader(TestViewer))
    test_suite.addTest(loader(TestLauncher))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
