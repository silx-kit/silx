# coding: utf-8
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
"""Basic tests for IPython console widget"""

from __future__ import print_function

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "05/12/2016"


import unittest

from silx.gui.utils.testutils import TestCaseQt

from silx.gui import qt
try:
    from silx.gui.console import IPythonDockWidget
except ImportError:
    console_missing = True
else:
    console_missing = False


# dummy objects to test pushing variables to the interactive namespace
_a = 1


def _f():
    print("Hello World!")


@unittest.skipIf(console_missing, "Could not import Ipython and/or qtconsole")
class TestConsole(TestCaseQt):
    """Basic test for ``module.IPythonDockWidget``"""

    def setUp(self):
        super(TestConsole, self).setUp()
        self.console = IPythonDockWidget(
            available_vars={"a": _a, "f": _f},
            custom_banner="Welcome!\n")
        self.console.show()
        self.qWaitForWindowExposed(self.console)

    def tearDown(self):
        self.console.setAttribute(qt.Qt.WA_DeleteOnClose)
        self.console.close()
        del self.console
        super(TestConsole, self).tearDown()

    def testShow(self):
        pass

    def testInteract(self):
        self.mouseClick(self.console, qt.Qt.LeftButton)
        self.keyClicks(self.console, 'import silx')
        self.keyClick(self.console, qt.Qt.Key_Enter)
        self.qapp.processEvents()


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestConsole))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
