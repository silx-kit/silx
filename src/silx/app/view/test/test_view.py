# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2020 European Synchrotron Radiation Facility
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
__date__ = "07/06/2018"


import unittest
import weakref
import numpy
import tempfile
import shutil
import os.path
import h5py

from silx.gui import qt
from silx.app.view.Viewer import Viewer
from silx.app.view.About import About
from silx.app.view.DataPanel import DataPanel
from silx.app.view.CustomNxdataWidget import CustomNxdataWidget
from silx.gui.hdf5._utils import Hdf5DatasetMimeData
from silx.gui.utils.testutils import TestCaseQt
from silx.io import commonh5

_tmpDirectory = None


def setUpModule():
    global _tmpDirectory
    _tmpDirectory = tempfile.mkdtemp(prefix=__name__)

    # create h5 data
    filename = _tmpDirectory + "/data.h5"
    f = h5py.File(filename, "w")
    g = f.create_group("arrays")
    g.create_dataset("scalar", data=10)
    g.create_dataset("integers", data=numpy.array([10, 20, 30]))
    f.close()

    # create h5 data
    filename = _tmpDirectory + "/data2.h5"
    f = h5py.File(filename, "w")
    g = f.create_group("arrays")
    g.create_dataset("scalar", data=20)
    g.create_dataset("integers", data=numpy.array([10, 20, 30]))
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

    def testRemoveDatasetsFrom(self):
        f = h5py.File(os.path.join(_tmpDirectory, "data.h5"), mode='r')
        try:
            widget = DataPanel()
            widget.setData(f["arrays/scalar"])
            widget.removeDatasetsFrom(f)
            self.assertIs(widget.getData(), None)
        finally:
            widget.setData(None)
            f.close()

    def testReplaceDatasetsFrom(self):
        f = h5py.File(os.path.join(_tmpDirectory, "data.h5"), mode='r')
        f2 = h5py.File(os.path.join(_tmpDirectory, "data2.h5"), mode='r')
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


class TestCustomNxdataWidget(TestCaseQt):

    def testConstruct(self):
        widget = CustomNxdataWidget()
        self.qWaitForWindowExposed(widget)

    def testDestroy(self):
        widget = CustomNxdataWidget()
        ref = weakref.ref(widget)
        widget = None
        self.qWaitForDestroy(ref)

    def testCreateNxdata(self):
        widget = CustomNxdataWidget()
        model = widget.model()
        model.createNewNxdata()
        model.createNewNxdata("Foo")
        widget.setVisible(True)
        self.qWaitForWindowExposed(widget)

    def testCreateNxdataFromDataset(self):
        widget = CustomNxdataWidget()
        model = widget.model()
        signal = commonh5.Dataset("foo", data=numpy.array([[[5]]]))
        model.createFromSignal(signal)
        widget.setVisible(True)
        self.qWaitForWindowExposed(widget)

    def testCreateNxdataFromNxdata(self):
        widget = CustomNxdataWidget()
        model = widget.model()
        data = numpy.array([[[5]]])
        nxdata = commonh5.Group("foo")
        nxdata.attrs["NX_class"] = "NXdata"
        nxdata.attrs["signal"] = "signal"
        nxdata.create_dataset("signal", data=data)
        model.createFromNxdata(nxdata)
        widget.setVisible(True)
        self.qWaitForWindowExposed(widget)

    def testCreateBadNxdata(self):
        widget = CustomNxdataWidget()
        model = widget.model()
        signal = commonh5.Dataset("foo", data=numpy.array([[[5]]]))
        model.createFromSignal(signal)
        axis = commonh5.Dataset("foo", data=numpy.array([[[5]]]))
        nxdataIndex = model.index(0, 0)
        item = model.itemFromIndex(nxdataIndex)
        item.setAxesDatasets([axis])
        nxdata = item.getVirtualGroup()
        self.assertIsNotNone(nxdata)
        self.assertFalse(item.isValid())

    def testRemoveDatasetsFrom(self):
        f = h5py.File(os.path.join(_tmpDirectory, "data.h5"), mode='r')
        try:
            widget = CustomNxdataWidget()
            model = widget.model()
            dataset = f["arrays/integers"]
            model.createFromSignal(dataset)
            widget.removeDatasetsFrom(f)
        finally:
            model.clear()
            f.close()

    def testReplaceDatasetsFrom(self):
        f = h5py.File(os.path.join(_tmpDirectory, "data.h5"), mode='r')
        f2 = h5py.File(os.path.join(_tmpDirectory, "data2.h5"), mode='r')
        try:
            widget = CustomNxdataWidget()
            model = widget.model()
            dataset = f["arrays/integers"]
            model.createFromSignal(dataset)
            widget.replaceDatasetsFrom(f, f2)
        finally:
            model.clear()
            f.close()
            f2.close()


class TestCustomNxdataWidgetInteraction(TestCaseQt):
    """Test CustomNxdataWidget with user interaction"""

    def setUp(self):
        TestCaseQt.setUp(self)

        self.widget = CustomNxdataWidget()
        self.model = self.widget.model()
        data = numpy.array([[[5]]])
        dataset = commonh5.Dataset("foo", data=data)
        self.model.createFromSignal(dataset)
        self.selectionModel = self.widget.selectionModel()

    def tearDown(self):
        self.selectionModel = None
        self.model.clear()
        self.model = None
        self.widget = None
        TestCaseQt.tearDown(self)

    def testSelectedNxdata(self):
        index = self.model.index(0, 0)
        self.selectionModel.setCurrentIndex(index, qt.QItemSelectionModel.ClearAndSelect)
        nxdata = self.widget.selectedNxdata()
        self.assertEqual(len(nxdata), 1)
        self.assertIsNot(nxdata[0], None)

    def testSelectedItems(self):
        index = self.model.index(0, 0)
        self.selectionModel.setCurrentIndex(index, qt.QItemSelectionModel.ClearAndSelect)
        items = self.widget.selectedItems()
        self.assertEqual(len(items), 1)
        self.assertIsNot(items[0], None)
        self.assertIsInstance(items[0], qt.QStandardItem)

    def testRowsAboutToBeRemoved(self):
        self.model.removeRow(0)
        self.qWaitForWindowExposed(self.widget)

    def testPaintItems(self):
        self.widget.expandAll()
        self.widget.setVisible(True)
        self.qWaitForWindowExposed(self.widget)

    def testCreateDefaultContextMenu(self):
        nxDataIndex = self.model.index(0, 0)
        menu = self.widget.createDefaultContextMenu(nxDataIndex)
        self.assertIsNot(menu, None)
        self.assertIsInstance(menu, qt.QMenu)

        signalIndex = self.model.index(0, 0, nxDataIndex)
        menu = self.widget.createDefaultContextMenu(signalIndex)
        self.assertIsNot(menu, None)
        self.assertIsInstance(menu, qt.QMenu)

        axesIndex = self.model.index(1, 0, nxDataIndex)
        menu = self.widget.createDefaultContextMenu(axesIndex)
        self.assertIsNot(menu, None)
        self.assertIsInstance(menu, qt.QMenu)

    def testDropNewDataset(self):
        dataset = commonh5.Dataset("foo", numpy.array([1, 2, 3, 4]))
        mimedata = Hdf5DatasetMimeData(dataset=dataset)
        self.model.dropMimeData(mimedata, qt.Qt.CopyAction, -1, -1, qt.QModelIndex())
        self.assertEqual(self.model.rowCount(qt.QModelIndex()), 2)

    def testDropNewNxdata(self):
        data = numpy.array([[[5]]])
        nxdata = commonh5.Group("foo")
        nxdata.attrs["NX_class"] = "NXdata"
        nxdata.attrs["signal"] = "signal"
        nxdata.create_dataset("signal", data=data)
        mimedata = Hdf5DatasetMimeData(dataset=nxdata)
        self.model.dropMimeData(mimedata, qt.Qt.CopyAction, -1, -1, qt.QModelIndex())
        self.assertEqual(self.model.rowCount(qt.QModelIndex()), 2)

    def testDropAxisDataset(self):
        dataset = commonh5.Dataset("foo", numpy.array([1, 2, 3, 4]))
        mimedata = Hdf5DatasetMimeData(dataset=dataset)
        nxDataIndex = self.model.index(0, 0)
        axesIndex = self.model.index(1, 0, nxDataIndex)
        self.model.dropMimeData(mimedata, qt.Qt.CopyAction, -1, -1, axesIndex)
        self.assertEqual(self.model.rowCount(qt.QModelIndex()), 1)
        item = self.model.itemFromIndex(axesIndex)
        self.assertIsNot(item.getDataset(), None)

    def testMimeData(self):
        nxDataIndex = self.model.index(0, 0)
        signalIndex = self.model.index(0, 0, nxDataIndex)
        mimeData = self.model.mimeData([signalIndex])
        self.assertIsNot(mimeData, None)
        self.assertIsInstance(mimeData, qt.QMimeData)

    def testRemoveNxdataItem(self):
        nxdataIndex = self.model.index(0, 0)
        item = self.model.itemFromIndex(nxdataIndex)
        self.model.removeNxdataItem(item)

    def testAppendAxisToNxdataItem(self):
        nxdataIndex = self.model.index(0, 0)
        item = self.model.itemFromIndex(nxdataIndex)
        self.model.appendAxisToNxdataItem(item)

    def testRemoveAxisItem(self):
        nxdataIndex = self.model.index(0, 0)
        axesIndex = self.model.index(1, 0, nxdataIndex)
        item = self.model.itemFromIndex(axesIndex)
        self.model.removeAxisItem(item)


def suite():
    test_suite = unittest.TestSuite()
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    test_suite.addTest(loader(TestViewer))
    test_suite.addTest(loader(TestAbout))
    test_suite.addTest(loader(TestDataPanel))
    test_suite.addTest(loader(TestCustomNxdataWidget))
    test_suite.addTest(loader(TestCustomNxdataWidgetInteraction))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
