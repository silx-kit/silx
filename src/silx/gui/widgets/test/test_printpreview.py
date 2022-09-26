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
"""Test PrintPreview"""

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "19/07/2017"


import unittest
from silx.gui.utils.testutils import TestCaseQt
from silx.gui.widgets.PrintPreview import PrintPreviewDialog
from silx.gui import qt

from silx.resources import resource_filename


class TestPrintPreview(TestCaseQt):
    def testShow(self):
        p = qt.QPrinter()
        d = PrintPreviewDialog(printer=p)
        d.show()
        self.qapp.processEvents()

    def testAddImage(self):
        p = qt.QPrinter()
        d = PrintPreviewDialog(printer=p)
        d.addImage(qt.QImage(resource_filename("gui/icons/clipboard.png")))
        self.qapp.processEvents()

    def testAddSvg(self):
        p = qt.QPrinter()
        d = PrintPreviewDialog(printer=p)
        d.addSvgItem(qt.QSvgRenderer(resource_filename("gui/icons/clipboard.svg"), d.page))
        self.qapp.processEvents()

    def testAddPixmap(self):
        p = qt.QPrinter()
        d = PrintPreviewDialog(printer=p)
        d.addPixmap(qt.QPixmap.fromImage(qt.QImage(resource_filename("gui/icons/clipboard.png"))))
        self.qapp.processEvents()
