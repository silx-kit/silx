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
import weakref
import numpy
import tempfile
import shutil
import os.path
try:
    import h5py
except ImportError:
    h5py = None

from silx.app.view.Viewer import Viewer
from silx.app.view.About import About
from silx.app.view.DataPanel import DataPanel
from silx.gui.test.utils import TestCaseQt

_tmpDirectory = None


def setUpModule():
    global _tmpDirectory
    _tmpDirectory = tempfile.mkdtemp(prefix=__name__)

    if h5py is not None:
        # create h5 data
        filename = _tmpDirectory + "/data.h5"
        f = h5py.File(filename, "w")
        g = f.create_group("arrays")
        g.create_dataset("scalar", data=10)
        f.close()

        # create h5 data
        filename = _tmpDirectory + "/data2.h5"
        f = h5py.File(filename, "w")
        g = f.create_group("arrays")
        g.create_dataset("scalar", data=20)
        f.close()


def tearDownModule():
    global _tmpDirectory
    shutil.rmtree(_tmpDirectory)
    _tmpDirectory = None


class TestViewer(TestCaseQt):
    """Test for Viewer class"""

    def testConstruct(self):
        widget = Viewer()
        self.qWaitForWindowExposed(widget)

    def testDestroy(self):
        widget = Viewer()
        ref = weakref.ref(widget)
        widget = None
        self.qWaitForDestroy(ref)


class TestAbout(TestCaseQt):
    """Test for About box class"""

    def testConstruct(self):
        widget = About()
        self.qWaitForWindowExposed(widget)

    def testLicense(self):
        widget = About()
        widget.getHtmlLicense()
        self.qWaitForWindowExposed(widget)

    def testDestroy(self):
        widget = About()
        ref = weakref.ref(widget)
        widget = None
        self.qWaitForDestroy(ref)


class TestDataPanel(TestCaseQt):

    def testConstruct(self):
        widget = DataPanel()
        self.qWaitForWindowExposed(widget)

    def testDestroy(self):
        widget = DataPanel()
        ref = weakref.ref(widget)
        widget = None
        self.qWaitForDestroy(ref)

    def testHeaderLabelPaintEvent(self):
        widget = DataPanel()
        data = numpy.array([1, 2, 3, 4, 5])
        widget.setData(data)
        # Expected to execute HeaderLabel.paintEvent
        widget.setVisible(True)
        self.qWaitForWindowExposed(widget)

    def testData(self):
        widget = DataPanel()
        data = numpy.array([1, 2, 3, 4, 5])
        widget.setData(data)
        self.assertIs(widget.getData(), data)
        self.assertIs(widget.getCustomNxdataItem(), None)

    def testDataNone(self):
        widget = DataPanel()
        widget.setData(None)
        self.assertIs(widget.getData(), None)
        self.assertIs(widget.getCustomNxdataItem(), None)

    def testCustomDataItem(self):
        class CustomDataItemMock(object):
            def getVirtualGroup(self):
                return None

            def text(self):
                return ""

        data = CustomDataItemMock()
        widget = DataPanel()
        widget.setCustomDataItem(data)
        self.assertIs(widget.getData(), None)
        self.assertIs(widget.getCustomNxdataItem(), data)

    def testCustomDataItemNone(self):
        data = None
        widget = DataPanel()
        widget.setCustomDataItem(data)
        self.assertIs(widget.getData(), None)
        self.assertIs(widget.getCustomNxdataItem(), data)

    @unittest.skipIf(h5py is None, "Could not import h5py")
    def testRemoveDatasetsFrom(self):
        f = h5py.File(os.path.join(_tmpDirectory, "data.h5"))
        try:
            widget = DataPanel()
            widget.setData(f["arrays/scalar"])
            widget.removeDatasetsFrom(f)
            self.assertIs(widget.getData(), None)
        finally:
            widget.setData(None)
            f.close()

    @unittest.skipIf(h5py is None, "Could not import h5py")
    def testReplaceDatasetsFrom(self):
        f = h5py.File(os.path.join(_tmpDirectory, "data.h5"))
        f2 = h5py.File(os.path.join(_tmpDirectory, "data2.h5"))
        try:
            widget = DataPanel()
            widget.setData(f["arrays/scalar"])
            self.assertEqual(widget.getData()[()], 10)
            widget.replaceDatasetsFrom(f, f2)
            self.assertEqual(widget.getData()[()], 20)
        finally:
            widget.setData(None)
            f.close()
            f2.close()


def suite():
    test_suite = unittest.TestSuite()
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    test_suite.addTest(loader(TestViewer))
    test_suite.addTest(loader(TestAbout))
    test_suite.addTest(loader(TestDataPanel))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
