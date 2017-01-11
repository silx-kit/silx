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
"""Basic tests for Colorbar"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "23/08/2016"


import doctest
import unittest

from silx.gui import qt
from silx.gui.plot import Colorbar


# Makes sure a QApplication exists
_qapp = qt.QApplication.instance() or qt.QApplication([])


def _tearDownQt(docTest):
    """Tear down to use for test from docstring.

    Checks that dialog widget is displayed
    """
    # Needed twice to display both windows
    _qapp.processEvents()
    _qapp.processEvents()
    for widgetName in ('plot', 'colorbar'):
        widget = docTest.globs[widgetName]
        widget.setAttribute(qt.Qt.WA_DeleteOnClose)
        widget.close()
        del widget


colorbarDocTestSuite = doctest.DocTestSuite(Colorbar, tearDown=_tearDownQt)
"""Test suite of tests from the module's docstrings."""


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(colorbarDocTestSuite)
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
