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
"""Test for silx.gui.hdf5 module"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "06/10/2016"


import time
import unittest
import tempfile
import numpy
from contextlib import contextmanager
from silx.gui import qt
from silx.gui import testutils
from silx.gui import hdf5
from . import _mock

try:
    import h5py
except ImportError:
    h5py = None


_called = 0


class _Holder(object):
    def callback(self, *args, **kvargs):
        _called += 1


class TestHdf5TreeModel(testutils.TestCaseQt):

    def setUp(self):
        super(TestHdf5TreeModel, self).setUp()
        if h5py is None:
            self.skipTest("h5py is not available")

    @contextmanager
    def h5TempFile(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".h5", delete=True)
        tmp.file.close()

        self.h5 = h5py.File(tmp.name, "w")
        g = self.h5.create_group("arrays")
        g.create_dataset("scalar", data=10)
        self.h5.close()
        yield tmp.name

    def testCreate(self):
        model = hdf5.Hdf5TreeModel()
        self.assertIsNotNone(model)

    def testAppendFilename(self):
        with self.h5TempFile() as filename:
            model = hdf5.Hdf5TreeModel()
            self.assertEquals(model.rowCount(qt.QModelIndex()), 0)
            model.appendFile(filename)
            self.assertEquals(model.rowCount(qt.QModelIndex()), 1)

    def testAppendBadFilename(self):
        model = hdf5.Hdf5TreeModel()
        self.assertRaises(IOError, model.appendFile, "#%$")

    def testInsertFilename(self):
        with self.h5TempFile() as filename:
            model = hdf5.Hdf5TreeModel()
            self.assertEquals(model.rowCount(qt.QModelIndex()), 0)
            model.insertFile(filename)
            self.assertEquals(model.rowCount(qt.QModelIndex()), 1)

    def testInsertFilenameAsync(self):
        with self.h5TempFile() as filename:
            model = hdf5.Hdf5TreeModel()
            self.assertEquals(model.rowCount(qt.QModelIndex()), 0)
            model.insertFileAsync(filename)
            index = model.index(0, 0, qt.QModelIndex())
            self.assertIsInstance(model.nodeFromIndex(index), hdf5.Hdf5LoadingItem.Hdf5LoadingItem)
            time.sleep(0.1)
            self.qapp.processEvents()
            time.sleep(0.1)
            index = model.index(0, 0, qt.QModelIndex())
            self.assertIsInstance(model.nodeFromIndex(index), hdf5.Hdf5Item.Hdf5Item)
            self.assertEquals(model.rowCount(qt.QModelIndex()), 1)

    def testInsertObject(self):
        h5 = _mock.File("/foo/bar/1.mock")
        model = hdf5.Hdf5TreeModel()
        self.assertEquals(model.rowCount(qt.QModelIndex()), 0)
        model.insertH5pyObject(h5)
        self.assertEquals(model.rowCount(qt.QModelIndex()), 1)

    def testRemoveObject(self):
        h5 = _mock.File("/foo/bar/1.mock")
        model = hdf5.Hdf5TreeModel()
        self.assertEquals(model.rowCount(qt.QModelIndex()), 0)
        model.insertH5pyObject(h5)
        self.assertEquals(model.rowCount(qt.QModelIndex()), 1)
        model.removeH5pyObject(h5)
        self.assertEquals(model.rowCount(qt.QModelIndex()), 0)

    def testSynchronizeObject(self):
        with self.h5TempFile() as filename:
            h5 = h5py.File(filename)
            model = hdf5.Hdf5TreeModel()
            model.insertH5pyObject(h5)
            self.assertEquals(model.rowCount(qt.QModelIndex()), 1)
            index = model.index(0, 0, qt.QModelIndex())
            node1 = model.nodeFromIndex(index)
            model.synchronizeH5pyObject(h5)
            index = model.index(0, 0, qt.QModelIndex())
            node2 = model.nodeFromIndex(index)
            self.assertIsNot(node1, node2)

    def testFileMoveState(self):
        model = hdf5.Hdf5TreeModel()
        self.assertEquals(model.isFileMoveEnabled(), True)
        model.setFileMoveEnabled(False)
        self.assertEquals(model.isFileMoveEnabled(), False)

    def testFileDropState(self):
        model = hdf5.Hdf5TreeModel()
        self.assertEquals(model.isFileDropEnabled(), True)
        model.setFileDropEnabled(False)
        self.assertEquals(model.isFileDropEnabled(), False)

    def testSupportedDrop(self):
        model = hdf5.Hdf5TreeModel()
        self.assertNotEquals(model.supportedDropActions(), 0)

        model.setFileMoveEnabled(False)
        model.setFileDropEnabled(False)
        self.assertEquals(model.supportedDropActions(), 0)

        model.setFileMoveEnabled(False)
        model.setFileDropEnabled(True)
        self.assertNotEquals(model.supportedDropActions(), 0)

        model.setFileMoveEnabled(True)
        model.setFileDropEnabled(False)
        self.assertNotEquals(model.supportedDropActions(), 0)

    def testDropExternalFile(self):
        with self.h5TempFile() as filename:
            model = hdf5.Hdf5TreeModel()
            mimeData = qt.QMimeData()
            mimeData.setUrls([qt.QUrl.fromLocalFile(filename)])
            model.dropMimeData(mimeData, qt.Qt.CopyAction, 0, 0, qt.QModelIndex())
            self.assertEquals(model.rowCount(qt.QModelIndex()), 1)

    def getRowDataAsDict(self, model, row):
        displayed = {}
        roles = [qt.Qt.DisplayRole, qt.Qt.DecorationRole, qt.Qt.ToolTipRole, qt.Qt.TextAlignmentRole]
        for column in range(0, model.columnCount(qt.QModelIndex())):
            index = model.index(0, column, qt.QModelIndex())
            for role in roles:
                datum = model.data(index, role)
                displayed[column, role] = datum
        return displayed

    def getItemName(self, model, row):
        index = model.index(row, hdf5.Hdf5TreeModel.NAME_COLUMN, qt.QModelIndex())
        return model.data(index, qt.Qt.DisplayRole)

    def testFileData(self):
        h5 = _mock.File("/foo/bar/1.mock")
        model = hdf5.Hdf5TreeModel()
        model.insertH5pyObject(h5)
        displayed = self.getRowDataAsDict(model, row=0)
        self.assertEquals(displayed[hdf5.Hdf5TreeModel.NAME_COLUMN, qt.Qt.DisplayRole], "1.mock")
        self.assertIsInstance(displayed[hdf5.Hdf5TreeModel.NAME_COLUMN, qt.Qt.DecorationRole], qt.QIcon)
        self.assertEquals(displayed[hdf5.Hdf5TreeModel.TYPE_COLUMN, qt.Qt.DisplayRole], "")
        self.assertEquals(displayed[hdf5.Hdf5TreeModel.SHAPE_COLUMN, qt.Qt.DisplayRole], "")
        self.assertEquals(displayed[hdf5.Hdf5TreeModel.VALUE_COLUMN, qt.Qt.DisplayRole], "")
        self.assertEquals(displayed[hdf5.Hdf5TreeModel.DESCRIPTION_COLUMN, qt.Qt.DisplayRole], "")
        self.assertEquals(displayed[hdf5.Hdf5TreeModel.NODE_COLUMN, qt.Qt.DisplayRole], "File")

    def testGroupData(self):
        h5 = _mock.File("/foo/bar/1.mock")
        d = h5.create_group("foo")
        d.attrs["desc"] = "fooo"

        model = hdf5.Hdf5TreeModel()
        model.insertH5pyObject(d)
        displayed = self.getRowDataAsDict(model, row=0)
        self.assertEquals(displayed[hdf5.Hdf5TreeModel.NAME_COLUMN, qt.Qt.DisplayRole], "1.mock::foo")
        self.assertIsInstance(displayed[hdf5.Hdf5TreeModel.NAME_COLUMN, qt.Qt.DecorationRole], qt.QIcon)
        self.assertEquals(displayed[hdf5.Hdf5TreeModel.TYPE_COLUMN, qt.Qt.DisplayRole], "")
        self.assertEquals(displayed[hdf5.Hdf5TreeModel.SHAPE_COLUMN, qt.Qt.DisplayRole], "")
        self.assertEquals(displayed[hdf5.Hdf5TreeModel.VALUE_COLUMN, qt.Qt.DisplayRole], "")
        self.assertEquals(displayed[hdf5.Hdf5TreeModel.DESCRIPTION_COLUMN, qt.Qt.DisplayRole], "fooo")
        self.assertEquals(displayed[hdf5.Hdf5TreeModel.NODE_COLUMN, qt.Qt.DisplayRole], "Group")

    def testDatasetData(self):
        h5 = _mock.File("/foo/bar/1.mock")
        value = numpy.array([1, 2, 3])
        d = h5.create_dataset("foo", value)

        model = hdf5.Hdf5TreeModel()
        model.insertH5pyObject(d)
        displayed = self.getRowDataAsDict(model, row=0)
        self.assertEquals(displayed[hdf5.Hdf5TreeModel.NAME_COLUMN, qt.Qt.DisplayRole], "1.mock::foo")
        self.assertIsInstance(displayed[hdf5.Hdf5TreeModel.NAME_COLUMN, qt.Qt.DecorationRole], qt.QIcon)
        self.assertEquals(displayed[hdf5.Hdf5TreeModel.TYPE_COLUMN, qt.Qt.DisplayRole], value.dtype.name)
        self.assertEquals(displayed[hdf5.Hdf5TreeModel.SHAPE_COLUMN, qt.Qt.DisplayRole], "3")
        self.assertEquals(displayed[hdf5.Hdf5TreeModel.VALUE_COLUMN, qt.Qt.DisplayRole], "[1 2 3]")
        self.assertEquals(displayed[hdf5.Hdf5TreeModel.DESCRIPTION_COLUMN, qt.Qt.DisplayRole], "")
        self.assertEquals(displayed[hdf5.Hdf5TreeModel.NODE_COLUMN, qt.Qt.DisplayRole], "Dataset")

    def testDropLastAsFirst(self):
        model = hdf5.Hdf5TreeModel()
        h5_1 = _mock.File("/foo/bar/1.mock")
        h5_2 = _mock.File("/foo/bar/2.mock")
        model.insertH5pyObject(h5_1)
        model.insertH5pyObject(h5_2)
        self.assertEquals(self.getItemName(model, 0), "1.mock")
        self.assertEquals(self.getItemName(model, 1), "2.mock")
        index = model.index(1, 0, qt.QModelIndex())
        mimeData = model.mimeData([index])
        model.dropMimeData(mimeData, qt.Qt.MoveAction, 0, 0, qt.QModelIndex())
        self.assertEquals(self.getItemName(model, 0), "2.mock")
        self.assertEquals(self.getItemName(model, 1), "1.mock")

    def testDropFirstAsLast(self):
        model = hdf5.Hdf5TreeModel()
        h5_1 = _mock.File("/foo/bar/1.mock")
        h5_2 = _mock.File("/foo/bar/2.mock")
        model.insertH5pyObject(h5_1)
        model.insertH5pyObject(h5_2)
        self.assertEquals(self.getItemName(model, 0), "1.mock")
        self.assertEquals(self.getItemName(model, 1), "2.mock")
        index = model.index(0, 0, qt.QModelIndex())
        mimeData = model.mimeData([index])
        model.dropMimeData(mimeData, qt.Qt.MoveAction, 2, 0, qt.QModelIndex())
        self.assertEquals(self.getItemName(model, 0), "2.mock")
        self.assertEquals(self.getItemName(model, 1), "1.mock")

    def testRootParent(self):
        model = hdf5.Hdf5TreeModel()
        h5_1 = _mock.File("/foo/bar/1.mock")
        model.insertH5pyObject(h5_1)
        index = model.index(0, 0, qt.QModelIndex())
        index = model.parent(index)
        self.assertEquals(index, qt.QModelIndex())


class TestNexusSortFilterProxyModel(testutils.TestCaseQt):

    def getChildNames(self, model, index):
        count = model.rowCount(index)
        result = []
        for row in range(0, count):
            itemIndex = model.index(row, hdf5.Hdf5TreeModel.NAME_COLUMN, index)
            name = model.data(itemIndex, qt.Qt.DisplayRole)
            result.append(name)
        return result

    def testStartTime(self):
        model = hdf5.Hdf5TreeModel()
        h5 = _mock.File("/foo/bar/1.mock")
        h5.create_group("a").create_dataset("start_time", numpy.string_("2015"))
        h5.create_group("b").create_dataset("start_time", numpy.string_("2013"))
        h5.create_group("c").create_dataset("start_time", numpy.string_("2014"))
        model.insertH5pyObject(h5)

        proxy = hdf5.NexusSortFilterProxyModel()
        proxy.setSourceModel(model)
        proxy.sort(0, qt.Qt.DescendingOrder)
        names = self.getChildNames(proxy, proxy.index(0, 0, qt.QModelIndex()))
        self.assertListEqual(names, ["a", "c", "b"])

    def testStartTimeInArray(self):
        model = hdf5.Hdf5TreeModel()
        h5 = _mock.File("/foo/bar/1.mock")
        h5.create_group("a").create_dataset("start_time", numpy.array([numpy.string_("2015")]))
        h5.create_group("b").create_dataset("start_time", numpy.array([numpy.string_("2013")]))
        h5.create_group("c").create_dataset("start_time", numpy.array([numpy.string_("2014")]))
        model.insertH5pyObject(h5)

        proxy = hdf5.NexusSortFilterProxyModel()
        proxy.setSourceModel(model)
        proxy.sort(0, qt.Qt.DescendingOrder)
        names = self.getChildNames(proxy, proxy.index(0, 0, qt.QModelIndex()))
        self.assertListEqual(names, ["a", "c", "b"])

    def testEndTimeInArray(self):
        model = hdf5.Hdf5TreeModel()
        h5 = _mock.File("/foo/bar/1.mock")
        h5.create_group("a").create_dataset("end_time", numpy.array([numpy.string_("2015")]))
        h5.create_group("b").create_dataset("end_time", numpy.array([numpy.string_("2013")]))
        h5.create_group("c").create_dataset("end_time", numpy.array([numpy.string_("2014")]))
        model.insertH5pyObject(h5)

        proxy = hdf5.NexusSortFilterProxyModel()
        proxy.setSourceModel(model)
        proxy.sort(0, qt.Qt.DescendingOrder)
        names = self.getChildNames(proxy, proxy.index(0, 0, qt.QModelIndex()))
        self.assertListEqual(names, ["a", "c", "b"])

    def testName(self):
        model = hdf5.Hdf5TreeModel()
        h5 = _mock.File("/foo/bar/1.mock")
        h5.create_group("a")
        h5.create_group("c")
        h5.create_group("b")
        model.insertH5pyObject(h5)

        proxy = hdf5.NexusSortFilterProxyModel()
        proxy.setSourceModel(model)
        proxy.sort(0, qt.Qt.AscendingOrder)
        names = self.getChildNames(proxy, proxy.index(0, 0, qt.QModelIndex()))
        self.assertListEqual(names, ["a", "b", "c"])

    def testNumber(self):
        model = hdf5.Hdf5TreeModel()
        h5 = _mock.File("/foo/bar/1.mock")
        h5.create_group("a1")
        h5.create_group("a20")
        h5.create_group("a3")
        model.insertH5pyObject(h5)

        proxy = hdf5.NexusSortFilterProxyModel()
        proxy.setSourceModel(model)
        proxy.sort(0, qt.Qt.AscendingOrder)
        names = self.getChildNames(proxy, proxy.index(0, 0, qt.QModelIndex()))
        self.assertListEqual(names, ["a1", "a3", "a20"])

    def testMultiNumber(self):
        model = hdf5.Hdf5TreeModel()
        h5 = _mock.File("/foo/bar/1.mock")
        h5.create_group("a1-1")
        h5.create_group("a20-1")
        h5.create_group("a3-1")
        h5.create_group("a3-20")
        h5.create_group("a3-3")
        model.insertH5pyObject(h5)

        proxy = hdf5.NexusSortFilterProxyModel()
        proxy.setSourceModel(model)
        proxy.sort(0, qt.Qt.AscendingOrder)
        names = self.getChildNames(proxy, proxy.index(0, 0, qt.QModelIndex()))
        self.assertListEqual(names, ["a1-1", "a3-1", "a3-3", "a3-20", "a20-1"])

    def testUnconsistantTypes(self):
        model = hdf5.Hdf5TreeModel()
        h5 = _mock.File("/foo/bar/1.mock")
        h5.create_group("aaa100")
        h5.create_group("100aaa")
        model.insertH5pyObject(h5)

        proxy = hdf5.NexusSortFilterProxyModel()
        proxy.setSourceModel(model)
        proxy.sort(0, qt.Qt.AscendingOrder)
        names = self.getChildNames(proxy, proxy.index(0, 0, qt.QModelIndex()))
        self.assertListEqual(names, ["100aaa", "aaa100"])


class TestHdf5(testutils.TestCaseQt):
    """Test to check that icons module."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.NamedTemporaryFile(suffix=".h5", delete=True)
        cls.tmp.file.close()

        cls.h5 = h5py.File(cls.tmp.name, "w")
        g = cls.h5.create_group("arrays")
        g.create_dataset("scalar", data=10)

    def setUp(self):
        super(TestHdf5, self).setUp()
        if h5py is None:
            self.skipTest("h5py is not available")

    def testCreate(self):
        view = hdf5.Hdf5TreeView()
        self.assertIsNotNone(view)

    def testContextMenu(self):
        view = hdf5.Hdf5TreeView()
        view._createContextMenu(qt.QPoint(0, 0))


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestHdf5TreeModel))
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestNexusSortFilterProxyModel))
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestHdf5))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
