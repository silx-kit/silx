# /*##########################################################################
#
# Copyright (c) 2016-2023 European Synchrotron Radiation Facility
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
__date__ = "12/03/2019"


import time
import os
import tempfile
import numpy
from packaging.version import Version
from contextlib import contextmanager
from silx.gui import qt
from silx.gui.utils.testutils import TestCaseQt
from silx.gui import hdf5
from silx.gui.utils.testutils import SignalListener
from silx.io import commonh5
from silx.io.url import DataUrl
import weakref

import h5py
import pytest


h5py2_9 = Version(h5py.version.version) >= Version("2.9.0")


@pytest.fixture(scope="class")
def useH5File(request, tmpdir_factory):
    tmp = tmpdir_factory.mktemp("test_hdf5")
    request.cls.filename = os.path.join(tmp, "data.h5")
    # create h5 data
    with h5py.File(request.cls.filename, "w") as f:
        g = f.create_group("arrays")
        g.create_dataset("scalar", data=10)
    yield


def create_NXentry(group, name):
    attrs = {"NX_class": "NXentry"}
    node = commonh5.Group(name, parent=group, attrs=attrs)
    group.add_node(node)
    return node


@pytest.mark.usefixtures("useH5File")
class TestHdf5TreeModel(TestCaseQt):
    def setUp(self):
        super().setUp()

    def waitForPendingOperations(self, model):
        for _ in range(20):
            if not model.hasPendingOperations():
                break
            self.qWait(200)
        else:
            raise RuntimeError("Still waiting for a pending operation")

    @contextmanager
    def h5TempFile(self):
        # create tmp file
        fd, tmp_name = tempfile.mkstemp(suffix=".h5")
        os.close(fd)
        # create h5 data
        h5file = h5py.File(tmp_name, "w")
        g = h5file.create_group("arrays")
        g.create_dataset("scalar", data=10)
        h5file.close()
        yield tmp_name
        # clean up
        os.unlink(tmp_name)

    def testCreate(self):
        model = hdf5.Hdf5TreeModel()
        self.assertIsNotNone(model)

    def testAppendFilename(self):
        model = hdf5.Hdf5TreeModel()
        self.assertEqual(model.rowCount(qt.QModelIndex()), 0)
        model.appendFile(self.filename)
        self.assertEqual(model.rowCount(qt.QModelIndex()), 1)
        # clean up
        ref = weakref.ref(model)
        model = None
        self.qWaitForDestroy(ref)

    def testAppendBadFilename(self):
        model = hdf5.Hdf5TreeModel()
        self.assertRaises(IOError, model.appendFile, "#%$")

    def testInsertFilename(self):
        try:
            model = hdf5.Hdf5TreeModel()
            self.assertEqual(model.rowCount(qt.QModelIndex()), 0)
            model.insertFile(self.filename)
            self.assertEqual(model.rowCount(qt.QModelIndex()), 1)
            # clean up
            index = model.index(0, 0, qt.QModelIndex())
            h5File = model.data(index, hdf5.Hdf5TreeModel.H5PY_OBJECT_ROLE)
            self.assertIsNotNone(h5File)
        finally:
            ref = weakref.ref(model)
            model = None
            self.qWaitForDestroy(ref)

    def testInsertFilenameAsync(self):
        try:
            model = hdf5.Hdf5TreeModel()
            self.assertEqual(model.rowCount(qt.QModelIndex()), 0)
            model.insertFileAsync(self.filename)
            index = model.index(0, 0, qt.QModelIndex())
            self.assertIsInstance(
                model.nodeFromIndex(index), hdf5.Hdf5LoadingItem.Hdf5LoadingItem
            )
            self.waitForPendingOperations(model)
            index = model.index(0, 0, qt.QModelIndex())
            self.assertIsInstance(model.nodeFromIndex(index), hdf5.Hdf5Item.Hdf5Item)
        finally:
            ref = weakref.ref(model)
            model = None
            self.qWaitForDestroy(ref)

    def testInsertObject(self):
        h5 = commonh5.File("/foo/bar/1.mock", "w")
        model = hdf5.Hdf5TreeModel()
        self.assertEqual(model.rowCount(qt.QModelIndex()), 0)
        model.insertH5pyObject(h5)
        self.assertEqual(model.rowCount(qt.QModelIndex()), 1)

    def testRemoveObject(self):
        h5 = commonh5.File("/foo/bar/1.mock", "w")
        model = hdf5.Hdf5TreeModel()
        self.assertEqual(model.rowCount(qt.QModelIndex()), 0)
        model.insertH5pyObject(h5)
        self.assertEqual(model.rowCount(qt.QModelIndex()), 1)
        model.removeH5pyObject(h5)
        self.assertEqual(model.rowCount(qt.QModelIndex()), 0)

    def testSynchronizeObject(self):
        h5 = h5py.File(self.filename, mode="r")
        model = hdf5.Hdf5TreeModel()
        model.insertH5pyObject(h5)
        self.assertEqual(model.rowCount(qt.QModelIndex()), 1)
        index = model.index(0, 0, qt.QModelIndex())
        node1 = model.nodeFromIndex(index)
        model.synchronizeH5pyObject(h5)
        self.waitForPendingOperations(model)
        # Now h5 was loaded from it's filename
        # Another ref is owned by the model
        h5.close()

        index = model.index(0, 0, qt.QModelIndex())
        node2 = model.nodeFromIndex(index)
        self.assertIsNot(node1, node2)
        # after sync
        time.sleep(0.1)
        self.qapp.processEvents()
        time.sleep(0.1)
        index = model.index(0, 0, qt.QModelIndex())
        self.assertIsInstance(model.nodeFromIndex(index), hdf5.Hdf5Item.Hdf5Item)
        # clean up
        index = model.index(0, 0, qt.QModelIndex())
        h5File = model.data(index, hdf5.Hdf5TreeModel.H5PY_OBJECT_ROLE)
        self.assertIsNotNone(h5File)
        h5File = None
        # delete the model
        ref = weakref.ref(model)
        model = None
        self.qWaitForDestroy(ref)

    def testFileMoveState(self):
        model = hdf5.Hdf5TreeModel()
        self.assertEqual(model.isFileMoveEnabled(), True)
        model.setFileMoveEnabled(False)
        self.assertEqual(model.isFileMoveEnabled(), False)

    def testFileDropState(self):
        model = hdf5.Hdf5TreeModel()
        self.assertEqual(model.isFileDropEnabled(), True)
        model.setFileDropEnabled(False)
        self.assertEqual(model.isFileDropEnabled(), False)

    def testSupportedDrop(self):
        model = hdf5.Hdf5TreeModel()
        self.assertNotEqual(model.supportedDropActions(), 0)

        model.setFileMoveEnabled(False)
        model.setFileDropEnabled(False)
        self.assertEqual(model.supportedDropActions(), 0)

        model.setFileMoveEnabled(False)
        model.setFileDropEnabled(True)
        self.assertNotEqual(model.supportedDropActions(), 0)

        model.setFileMoveEnabled(True)
        model.setFileDropEnabled(False)
        self.assertNotEqual(model.supportedDropActions(), 0)

    def testCloseFile(self):
        """A file inserted as a filename is open and closed internally."""
        model = hdf5.Hdf5TreeModel()
        self.assertEqual(model.rowCount(qt.QModelIndex()), 0)
        model.insertFile(self.filename)
        self.assertEqual(model.rowCount(qt.QModelIndex()), 1)
        index = model.index(0, 0)
        h5File = model.data(index, role=hdf5.Hdf5TreeModel.H5PY_OBJECT_ROLE)
        model.removeIndex(index)
        self.assertEqual(model.rowCount(qt.QModelIndex()), 0)
        self.assertFalse(bool(h5File.id.valid), "The HDF5 file was not closed")

    def testNotCloseFile(self):
        """A file inserted as an h5py object is not open (then not closed)
        internally."""
        try:
            h5File = h5py.File(self.filename, mode="r")
            model = hdf5.Hdf5TreeModel()
            self.assertEqual(model.rowCount(qt.QModelIndex()), 0)
            model.insertH5pyObject(h5File)
            self.assertEqual(model.rowCount(qt.QModelIndex()), 1)
            index = model.index(0, 0)
            h5File = model.data(index, role=hdf5.Hdf5TreeModel.H5PY_OBJECT_ROLE)
            model.removeIndex(index)
            self.assertEqual(model.rowCount(qt.QModelIndex()), 0)
            self.assertTrue(
                bool(h5File.id.valid), "The HDF5 file was unexpetedly closed"
            )
        finally:
            h5File.close()

    def testDropExternalFile(self):
        model = hdf5.Hdf5TreeModel()
        mimeData = qt.QMimeData()
        mimeData.setUrls([qt.QUrl.fromLocalFile(self.filename)])
        model.dropMimeData(mimeData, qt.Qt.CopyAction, 0, 0, qt.QModelIndex())
        self.assertEqual(model.rowCount(qt.QModelIndex()), 1)
        # after sync
        self.waitForPendingOperations(model)
        index = model.index(0, 0, qt.QModelIndex())
        self.assertIsInstance(model.nodeFromIndex(index), hdf5.Hdf5Item.Hdf5Item)
        # clean up
        index = model.index(0, 0, qt.QModelIndex())
        h5File = model.data(index, role=hdf5.Hdf5TreeModel.H5PY_OBJECT_ROLE)
        self.assertIsNotNone(h5File)
        h5File = None
        ref = weakref.ref(model)
        model = None
        self.qWaitForDestroy(ref)

    def getRowDataAsDict(self, model, row):
        displayed = {}
        roles = [
            qt.Qt.DisplayRole,
            qt.Qt.DecorationRole,
            qt.Qt.ToolTipRole,
            qt.Qt.TextAlignmentRole,
        ]
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
        h5 = commonh5.File("/foo/bar/1.mock", "w")
        model = hdf5.Hdf5TreeModel()
        model.insertH5pyObject(h5)
        displayed = self.getRowDataAsDict(model, row=0)
        self.assertEqual(
            displayed[hdf5.Hdf5TreeModel.NAME_COLUMN, qt.Qt.DisplayRole], "1.mock"
        )
        self.assertIsInstance(
            displayed[hdf5.Hdf5TreeModel.NAME_COLUMN, qt.Qt.DecorationRole], qt.QIcon
        )
        self.assertEqual(
            displayed[hdf5.Hdf5TreeModel.TYPE_COLUMN, qt.Qt.DisplayRole], ""
        )
        self.assertEqual(
            displayed[hdf5.Hdf5TreeModel.SHAPE_COLUMN, qt.Qt.DisplayRole], ""
        )
        self.assertEqual(
            displayed[hdf5.Hdf5TreeModel.VALUE_COLUMN, qt.Qt.DisplayRole], ""
        )
        self.assertEqual(
            displayed[hdf5.Hdf5TreeModel.DESCRIPTION_COLUMN, qt.Qt.DisplayRole], None
        )
        self.assertEqual(
            displayed[hdf5.Hdf5TreeModel.NODE_COLUMN, qt.Qt.DisplayRole], "File"
        )

    def testGroupData(self):
        h5 = commonh5.File("/foo/bar/1.mock", "w")
        d = h5.create_group("foo")
        d.attrs["desc"] = "fooo"

        model = hdf5.Hdf5TreeModel()
        model.insertH5pyObject(d)
        displayed = self.getRowDataAsDict(model, row=0)
        self.assertEqual(
            displayed[hdf5.Hdf5TreeModel.NAME_COLUMN, qt.Qt.DisplayRole], "1.mock::foo"
        )
        self.assertIsInstance(
            displayed[hdf5.Hdf5TreeModel.NAME_COLUMN, qt.Qt.DecorationRole], qt.QIcon
        )
        self.assertEqual(
            displayed[hdf5.Hdf5TreeModel.TYPE_COLUMN, qt.Qt.DisplayRole], ""
        )
        self.assertEqual(
            displayed[hdf5.Hdf5TreeModel.SHAPE_COLUMN, qt.Qt.DisplayRole], ""
        )
        self.assertEqual(
            displayed[hdf5.Hdf5TreeModel.VALUE_COLUMN, qt.Qt.DisplayRole], ""
        )
        self.assertEqual(
            displayed[hdf5.Hdf5TreeModel.DESCRIPTION_COLUMN, qt.Qt.DisplayRole], "fooo"
        )
        self.assertEqual(
            displayed[hdf5.Hdf5TreeModel.NODE_COLUMN, qt.Qt.DisplayRole], "Group"
        )

    def testDatasetData(self):
        h5 = commonh5.File("/foo/bar/1.mock", "w")
        value = numpy.array([1, 2, 3])
        d = h5.create_dataset("foo", data=value)

        model = hdf5.Hdf5TreeModel()
        model.insertH5pyObject(d)
        displayed = self.getRowDataAsDict(model, row=0)
        self.assertEqual(
            displayed[hdf5.Hdf5TreeModel.NAME_COLUMN, qt.Qt.DisplayRole], "1.mock::foo"
        )
        self.assertIsInstance(
            displayed[hdf5.Hdf5TreeModel.NAME_COLUMN, qt.Qt.DecorationRole], qt.QIcon
        )
        self.assertEqual(
            displayed[hdf5.Hdf5TreeModel.TYPE_COLUMN, qt.Qt.DisplayRole],
            value.dtype.name,
        )
        self.assertEqual(
            displayed[hdf5.Hdf5TreeModel.SHAPE_COLUMN, qt.Qt.DisplayRole], "3"
        )
        self.assertEqual(
            displayed[hdf5.Hdf5TreeModel.VALUE_COLUMN, qt.Qt.DisplayRole], "[1 2 3]"
        )
        self.assertEqual(
            displayed[hdf5.Hdf5TreeModel.DESCRIPTION_COLUMN, qt.Qt.DisplayRole],
            "[1 2 3]",
        )
        self.assertEqual(
            displayed[hdf5.Hdf5TreeModel.NODE_COLUMN, qt.Qt.DisplayRole], "Dataset"
        )

    def testDropLastAsFirst(self):
        model = hdf5.Hdf5TreeModel()
        h5_1 = commonh5.File("/foo/bar/1.mock", "w")
        h5_2 = commonh5.File("/foo/bar/2.mock", "w")
        model.insertH5pyObject(h5_1)
        self.qapp.processEvents()
        model.insertH5pyObject(h5_2)
        self.qapp.processEvents()
        self.assertEqual(self.getItemName(model, 0), "1.mock")
        self.assertEqual(self.getItemName(model, 1), "2.mock")
        index = model.index(1, 0, qt.QModelIndex())
        mimeData = model.mimeData([index])
        model.dropMimeData(mimeData, qt.Qt.MoveAction, 0, 0, qt.QModelIndex())
        self.assertEqual(self.getItemName(model, 0), "2.mock")
        self.assertEqual(self.getItemName(model, 1), "1.mock")

    def testDropFirstAsLast(self):
        model = hdf5.Hdf5TreeModel()
        h5_1 = commonh5.File("/foo/bar/1.mock", "w")
        h5_2 = commonh5.File("/foo/bar/2.mock", "w")
        model.insertH5pyObject(h5_1)
        self.qapp.processEvents()
        model.insertH5pyObject(h5_2)
        self.qapp.processEvents()
        self.assertEqual(self.getItemName(model, 0), "1.mock")
        self.assertEqual(self.getItemName(model, 1), "2.mock")
        index = model.index(0, 0, qt.QModelIndex())
        mimeData = model.mimeData([index])
        model.dropMimeData(mimeData, qt.Qt.MoveAction, 2, 0, qt.QModelIndex())
        self.assertEqual(self.getItemName(model, 0), "2.mock")
        self.assertEqual(self.getItemName(model, 1), "1.mock")

    def testRootParent(self):
        model = hdf5.Hdf5TreeModel()
        h5_1 = commonh5.File("/foo/bar/1.mock", "w")
        model.insertH5pyObject(h5_1)
        index = model.index(0, 0, qt.QModelIndex())
        index = model.parent(index)
        self.assertEqual(index, qt.QModelIndex())


@pytest.mark.usefixtures("useH5File")
class TestHdf5TreeModelSignals(TestCaseQt):
    def setUp(self):
        TestCaseQt.setUp(self)
        self.model = hdf5.Hdf5TreeModel()
        self.h5 = h5py.File(self.filename, mode="r")
        self.model.insertH5pyObject(self.h5)

        self.listener = SignalListener()
        self.model.sigH5pyObjectLoaded.connect(self.listener.partial(signal="loaded"))
        self.model.sigH5pyObjectRemoved.connect(self.listener.partial(signal="removed"))
        self.model.sigH5pyObjectSynchronized.connect(
            self.listener.partial(signal="synchronized")
        )

    def tearDown(self):
        self.signals = None
        ref = weakref.ref(self.model)
        self.model = None
        self.qWaitForDestroy(ref)
        self.h5.close()
        self.h5 = None
        TestCaseQt.tearDown(self)

    def waitForPendingOperations(self, model):
        for _ in range(20):
            if not model.hasPendingOperations():
                break
            self.qWait(200)
        else:
            raise RuntimeError("Still waiting for a pending operation")

    def testInsert(self):
        h5 = h5py.File(self.filename, mode="r")
        self.model.insertH5pyObject(h5)
        self.assertEqual(self.listener.callCount(), 0)

    def testLoaded(self):
        for data_path in [None, "/arrays/scalar"]:
            with self.subTest(data_path=data_path):
                url = DataUrl(file_path=self.filename, data_path=data_path)
                insertedFilename = url.path()
                self.model.insertFile(insertedFilename)
                self.assertEqual(self.listener.callCount(), 1)
                self.assertEqual(
                    self.listener.karguments(argumentName="signal")[0], "loaded"
                )
                self.assertIsNot(self.listener.arguments(callIndex=0)[0], self.h5)
                self.assertEqual(
                    self.listener.arguments(callIndex=0)[0].file.filename, self.filename
                )
                self.assertEqual(
                    self.listener.arguments(callIndex=0)[1], insertedFilename
                )
                self.listener.clear()

    def testRemoved(self):
        self.model.removeH5pyObject(self.h5)
        self.assertEqual(self.listener.callCount(), 1)
        self.assertEqual(self.listener.karguments(argumentName="signal")[0], "removed")
        self.assertIs(self.listener.arguments(callIndex=0)[0], self.h5)

    def testSynchonized(self):
        self.model.synchronizeH5pyObject(self.h5)
        self.waitForPendingOperations(self.model)
        self.assertEqual(self.listener.callCount(), 1)
        self.assertEqual(
            self.listener.karguments(argumentName="signal")[0], "synchronized"
        )
        self.assertIs(self.listener.arguments(callIndex=0)[0], self.h5)
        self.assertIsNot(self.listener.arguments(callIndex=0)[1], self.h5)


class TestNexusSortFilterProxyModel(TestCaseQt):
    def getChildNames(self, model, index):
        count = model.rowCount(index)
        result = []
        for row in range(0, count):
            itemIndex = model.index(row, hdf5.Hdf5TreeModel.NAME_COLUMN, index)
            name = model.data(itemIndex, qt.Qt.DisplayRole)
            result.append(name)
        return result

    def testNXentryStartTime(self):
        """Test NXentry with start_time"""
        model = hdf5.Hdf5TreeModel()
        h5 = commonh5.File("/foo/bar/1.mock", "w")
        create_NXentry(h5, "a").create_dataset("start_time", data=numpy.bytes_("2015"))
        create_NXentry(h5, "b").create_dataset("start_time", data=numpy.bytes_("2013"))
        create_NXentry(h5, "c").create_dataset("start_time", data=numpy.bytes_("2014"))
        model.insertH5pyObject(h5)

        proxy = hdf5.NexusSortFilterProxyModel()
        proxy.setSourceModel(model)
        proxy.sort(0, qt.Qt.DescendingOrder)
        names = self.getChildNames(proxy, proxy.index(0, 0, qt.QModelIndex()))
        self.assertListEqual(names, ["a", "c", "b"])

    def testNXentryStartTimeInArray(self):
        """Test NXentry with start_time"""
        model = hdf5.Hdf5TreeModel()
        h5 = commonh5.File("/foo/bar/1.mock", "w")
        create_NXentry(h5, "a").create_dataset(
            "start_time", data=numpy.array([numpy.bytes_("2015")])
        )
        create_NXentry(h5, "b").create_dataset(
            "start_time", data=numpy.array([numpy.bytes_("2013")])
        )
        create_NXentry(h5, "c").create_dataset(
            "start_time", data=numpy.array([numpy.bytes_("2014")])
        )
        model.insertH5pyObject(h5)

        proxy = hdf5.NexusSortFilterProxyModel()
        proxy.setSourceModel(model)
        proxy.sort(0, qt.Qt.DescendingOrder)
        names = self.getChildNames(proxy, proxy.index(0, 0, qt.QModelIndex()))
        self.assertListEqual(names, ["a", "c", "b"])

    def testNXentryEndTimeInArray(self):
        """Test NXentry with end_time"""
        model = hdf5.Hdf5TreeModel()
        h5 = commonh5.File("/foo/bar/1.mock", "w")
        create_NXentry(h5, "a").create_dataset(
            "end_time", data=numpy.array([numpy.bytes_("2015")])
        )
        create_NXentry(h5, "b").create_dataset(
            "end_time", data=numpy.array([numpy.bytes_("2013")])
        )
        create_NXentry(h5, "c").create_dataset(
            "end_time", data=numpy.array([numpy.bytes_("2014")])
        )
        model.insertH5pyObject(h5)

        proxy = hdf5.NexusSortFilterProxyModel()
        proxy.setSourceModel(model)
        proxy.sort(0, qt.Qt.DescendingOrder)
        names = self.getChildNames(proxy, proxy.index(0, 0, qt.QModelIndex()))
        self.assertListEqual(names, ["a", "c", "b"])

    def testNXentryName(self):
        """Test NXentry without start_time  or end_time"""
        model = hdf5.Hdf5TreeModel()
        h5 = commonh5.File("/foo/bar/1.mock", "w")
        create_NXentry(h5, "a")
        create_NXentry(h5, "c")
        create_NXentry(h5, "b")
        model.insertH5pyObject(h5)

        proxy = hdf5.NexusSortFilterProxyModel()
        proxy.setSourceModel(model)
        proxy.sort(0, qt.Qt.AscendingOrder)
        names = self.getChildNames(proxy, proxy.index(0, 0, qt.QModelIndex()))
        self.assertListEqual(names, ["a", "b", "c"])

    def testStartTime(self):
        """If it is not NXentry, start_time is not used"""
        model = hdf5.Hdf5TreeModel()
        h5 = commonh5.File("/foo/bar/1.mock", "w")
        h5.create_group("a").create_dataset("start_time", data=numpy.bytes_("2015"))
        h5.create_group("b").create_dataset("start_time", data=numpy.bytes_("2013"))
        h5.create_group("c").create_dataset("start_time", data=numpy.bytes_("2014"))
        model.insertH5pyObject(h5)

        proxy = hdf5.NexusSortFilterProxyModel()
        proxy.setSourceModel(model)
        proxy.sort(0, qt.Qt.AscendingOrder)
        names = self.getChildNames(proxy, proxy.index(0, 0, qt.QModelIndex()))
        self.assertListEqual(names, ["a", "b", "c"])

    def testName(self):
        model = hdf5.Hdf5TreeModel()
        h5 = commonh5.File("/foo/bar/1.mock", "w")
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
        h5 = commonh5.File("/foo/bar/1.mock", "w")
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
        h5 = commonh5.File("/foo/bar/1.mock", "w")
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
        h5 = commonh5.File("/foo/bar/1.mock", "w")
        h5.create_group("aaa100")
        h5.create_group("100aaa")
        model.insertH5pyObject(h5)

        proxy = hdf5.NexusSortFilterProxyModel()
        proxy.setSourceModel(model)
        proxy.sort(0, qt.Qt.AscendingOrder)
        names = self.getChildNames(proxy, proxy.index(0, 0, qt.QModelIndex()))
        self.assertListEqual(names, ["100aaa", "aaa100"])


@pytest.fixture(scope="class")
def useH5Model(request, tmpdir_factory):
    # Create HDF5 files
    tmp = tmpdir_factory.mktemp("test_hdf5")
    filename = os.path.join(tmp, "base.h5")
    extH5FileName = os.path.join(tmp, "base__external.h5")
    extDatFileName = os.path.join(tmp, "base__external.dat")

    externalh5 = h5py.File(extH5FileName, mode="w")
    externalh5["target/dataset"] = 50
    externalh5["target/link"] = h5py.SoftLink("/target/dataset")
    externalh5["/ext/vds0"] = [0, 1]
    externalh5["/ext/vds1"] = [2, 3]
    externalh5.close()

    numpy.array([0, 1, 10, 10, 2, 3]).tofile(extDatFileName)

    h5 = h5py.File(filename, mode="w")
    h5["group/dataset"] = 50
    h5["link/soft_link"] = h5py.SoftLink("/group/dataset")
    h5["link/soft_link_to_group"] = h5py.SoftLink("/group")
    h5["link/soft_link_to_soft_link"] = h5py.SoftLink("/link/soft_link")
    h5["link/soft_link_to_external_link"] = h5py.SoftLink("/link/external_link")
    h5["link/soft_link_to_file"] = h5py.SoftLink("/")
    h5["group/soft_link_relative"] = h5py.SoftLink("dataset")
    h5["link/external_link"] = h5py.ExternalLink(extH5FileName, "/target/dataset")
    h5["link/external_link_to_soft_link"] = h5py.ExternalLink(
        extH5FileName, "/target/link"
    )
    h5["broken_link/external_broken_file"] = h5py.ExternalLink(
        extH5FileName + "_not_exists", "/target/link"
    )
    h5["broken_link/external_broken_link"] = h5py.ExternalLink(
        extH5FileName, "/target/not_exists"
    )
    h5["broken_link/soft_broken_link"] = h5py.SoftLink("/group/not_exists")
    h5["broken_link/soft_link_to_broken_link"] = h5py.SoftLink("/group/not_exists")
    if h5py2_9:
        layout = h5py.VirtualLayout((2, 2), dtype=int)
        layout[0] = h5py.VirtualSource(
            "base__external.h5", name="/ext/vds0", shape=(2,), dtype=int
        )
        layout[1] = h5py.VirtualSource(
            "base__external.h5", name="/ext/vds1", shape=(2,), dtype=int
        )
        h5.create_group("/ext")
        h5["/ext"].create_virtual_dataset("virtual", layout)
        external = [
            ("base__external.dat", 0, 2 * 8),
            ("base__external.dat", 4 * 8, 2 * 8),
        ]
        h5["/ext"].create_dataset("raw", shape=(2, 2), dtype=int, external=external)
        h5.close()

    with h5py.File(filename, mode="r") as h5File:
        # Create model
        request.cls.model = hdf5.Hdf5TreeModel()
        request.cls.model.insertH5pyObject(h5File)
        yield
        ref = weakref.ref(request.cls.model)
        request.cls.model = None
        TestCaseQt.qWaitForDestroy(ref)


@pytest.mark.usefixtures("useH5Model")
class _TestModelBase(TestCaseQt):
    def getIndexFromPath(self, model, path):
        """
        :param qt.QAbstractItemModel: model
        """
        index = qt.QModelIndex()
        for name in path:
            for row in range(model.rowCount(index)):
                i = model.index(row, 0, index)
                label = model.data(i)
                if label == name:
                    index = i
                    break
            else:
                raise RuntimeError("Path not found")
        return index

    def getH5ItemFromPath(self, model, path):
        index = self.getIndexFromPath(model, path)
        return model.data(index, hdf5.Hdf5TreeModel.H5PY_ITEM_ROLE)


class TestH5Item(_TestModelBase):
    def testFile(self):
        path = ["base.h5"]
        h5item = self.getH5ItemFromPath(self.model, path)

        self.assertEqual(h5item.dataLink(qt.Qt.DisplayRole), "")

    def testGroup(self):
        path = ["base.h5", "group"]
        h5item = self.getH5ItemFromPath(self.model, path)

        self.assertEqual(h5item.dataLink(qt.Qt.DisplayRole), "")

    def testDataset(self):
        path = ["base.h5", "group", "dataset"]
        h5item = self.getH5ItemFromPath(self.model, path)

        self.assertEqual(h5item.dataLink(qt.Qt.DisplayRole), "")

    def testSoftLink(self):
        path = ["base.h5", "link", "soft_link"]
        h5item = self.getH5ItemFromPath(self.model, path)

        self.assertEqual(h5item.dataLink(qt.Qt.DisplayRole), "Soft")

    def testSoftLinkToLink(self):
        path = ["base.h5", "link", "soft_link_to_soft_link"]
        h5item = self.getH5ItemFromPath(self.model, path)

        self.assertEqual(h5item.dataLink(qt.Qt.DisplayRole), "Soft")

    def testSoftLinkRelative(self):
        path = ["base.h5", "group", "soft_link_relative"]
        h5item = self.getH5ItemFromPath(self.model, path)

        self.assertEqual(h5item.dataLink(qt.Qt.DisplayRole), "Soft")

    def testExternalLink(self):
        path = ["base.h5", "link", "external_link"]
        h5item = self.getH5ItemFromPath(self.model, path)

        self.assertEqual(h5item.dataLink(qt.Qt.DisplayRole), "External")

    def testExternalLinkToLink(self):
        path = ["base.h5", "link", "external_link_to_soft_link"]
        h5item = self.getH5ItemFromPath(self.model, path)

        self.assertEqual(h5item.dataLink(qt.Qt.DisplayRole), "External")

    def testExternalBrokenFile(self):
        path = ["base.h5", "broken_link", "external_broken_file"]
        h5item = self.getH5ItemFromPath(self.model, path)

        self.assertEqual(h5item.dataLink(qt.Qt.DisplayRole), "External")

    def testExternalBrokenLink(self):
        path = ["base.h5", "broken_link", "external_broken_link"]
        h5item = self.getH5ItemFromPath(self.model, path)

        self.assertEqual(h5item.dataLink(qt.Qt.DisplayRole), "External")

    def testSoftBrokenLink(self):
        path = ["base.h5", "broken_link", "soft_broken_link"]
        h5item = self.getH5ItemFromPath(self.model, path)

        self.assertEqual(h5item.dataLink(qt.Qt.DisplayRole), "Soft")

    def testSoftLinkToBrokenLink(self):
        path = ["base.h5", "broken_link", "soft_link_to_broken_link"]
        h5item = self.getH5ItemFromPath(self.model, path)

        self.assertEqual(h5item.dataLink(qt.Qt.DisplayRole), "Soft")

    def testDatasetFromSoftLinkToGroup(self):
        path = ["base.h5", "link", "soft_link_to_group", "dataset"]
        h5item = self.getH5ItemFromPath(self.model, path)

        self.assertEqual(h5item.dataLink(qt.Qt.DisplayRole), "")

    def testDatasetFromSoftLinkToFile(self):
        path = [
            "base.h5",
            "link",
            "soft_link_to_file",
            "link",
            "soft_link_to_group",
            "dataset",
        ]
        h5item = self.getH5ItemFromPath(self.model, path)

        self.assertEqual(h5item.dataLink(qt.Qt.DisplayRole), "")

    @pytest.mark.skipif(not h5py2_9, reason="requires h5py>=2.9")
    def testExternalVirtual(self):
        path = ["base.h5", "ext", "virtual"]
        h5item = self.getH5ItemFromPath(self.model, path)

        self.assertEqual(h5item.dataLink(qt.Qt.DisplayRole), "Virtual")

    @pytest.mark.skipif(not h5py2_9, reason="requires h5py>=2.9")
    def testExternalRaw(self):
        path = ["base.h5", "ext", "raw"]
        h5item = self.getH5ItemFromPath(self.model, path)

        self.assertEqual(h5item.dataLink(qt.Qt.DisplayRole), "ExtRaw")


class TestH5Node(_TestModelBase):
    def getH5NodeFromPath(self, model, path):
        item = self.getH5ItemFromPath(model, path)
        h5node = hdf5.H5Node(item)
        return h5node

    def testFile(self):
        path = ["base.h5"]
        h5node = self.getH5NodeFromPath(self.model, path)

        self.assertEqual(h5node.physical_filename, h5node.local_filename)
        self.assertIn("base.h5", h5node.physical_filename)
        self.assertEqual(h5node.physical_basename, "")
        self.assertEqual(h5node.physical_name, "/")
        self.assertEqual(h5node.local_basename, "")
        self.assertEqual(h5node.local_name, "/")

    def testGroup(self):
        path = ["base.h5", "group"]
        h5node = self.getH5NodeFromPath(self.model, path)

        self.assertEqual(h5node.physical_filename, h5node.local_filename)
        self.assertIn("base.h5", h5node.physical_filename)
        self.assertEqual(h5node.physical_basename, "group")
        self.assertEqual(h5node.physical_name, "/group")
        self.assertEqual(h5node.local_basename, "group")
        self.assertEqual(h5node.local_name, "/group")

    def testDataset(self):
        path = ["base.h5", "group", "dataset"]
        h5node = self.getH5NodeFromPath(self.model, path)

        self.assertEqual(h5node.physical_filename, h5node.local_filename)
        self.assertIn("base.h5", h5node.physical_filename)
        self.assertEqual(h5node.physical_basename, "dataset")
        self.assertEqual(h5node.physical_name, "/group/dataset")
        self.assertEqual(h5node.local_basename, "dataset")
        self.assertEqual(h5node.local_name, "/group/dataset")

    def testSoftLink(self):
        path = ["base.h5", "link", "soft_link"]
        h5node = self.getH5NodeFromPath(self.model, path)

        self.assertEqual(h5node.physical_filename, h5node.local_filename)
        self.assertIn("base.h5", h5node.physical_filename)
        self.assertEqual(h5node.physical_basename, "dataset")
        self.assertEqual(h5node.physical_name, "/group/dataset")
        self.assertEqual(h5node.local_basename, "soft_link")
        self.assertEqual(h5node.local_name, "/link/soft_link")

    def testSoftLinkToSoftLink(self):
        path = ["base.h5", "link", "soft_link_to_soft_link"]
        h5node = self.getH5NodeFromPath(self.model, path)

        self.assertEqual(h5node.physical_filename, h5node.local_filename)
        self.assertIn("base.h5", h5node.physical_filename)
        self.assertEqual(h5node.physical_basename, "dataset")
        self.assertEqual(h5node.physical_name, "/group/dataset")
        self.assertEqual(h5node.local_basename, "soft_link_to_soft_link")
        self.assertEqual(h5node.local_name, "/link/soft_link_to_soft_link")

    def testSoftLinkToExternalLink(self):
        path = ["base.h5", "link", "soft_link_to_external_link"]
        h5node = self.getH5NodeFromPath(self.model, path)

        with self.assertRaises(KeyError):
            # h5py bug: #1706
            self.assertNotEqual(h5node.physical_filename, h5node.local_filename)
            self.assertIn("base.h5", h5node.local_filename)
            self.assertIn("base__external.h5", h5node.physical_filename)
            self.assertEqual(h5node.physical_basename, "dataset")
            self.assertEqual(h5node.physical_name, "/target/dataset")
            self.assertEqual(h5node.local_basename, "soft_link_to_external_link")
            self.assertEqual(h5node.local_name, "/link/soft_link_to_external_link")

    def testSoftLinkRelative(self):
        path = ["base.h5", "group", "soft_link_relative"]
        h5node = self.getH5NodeFromPath(self.model, path)

        self.assertEqual(h5node.physical_filename, h5node.local_filename)
        self.assertIn("base.h5", h5node.physical_filename)
        self.assertEqual(h5node.physical_basename, "dataset")
        self.assertEqual(h5node.physical_name, "/group/dataset")
        self.assertEqual(h5node.local_basename, "soft_link_relative")
        self.assertEqual(h5node.local_name, "/group/soft_link_relative")

    def testExternalLink(self):
        path = ["base.h5", "link", "external_link"]
        h5node = self.getH5NodeFromPath(self.model, path)

        self.assertNotEqual(h5node.physical_filename, h5node.local_filename)
        self.assertIn("base.h5", h5node.local_filename)
        self.assertIn("base__external.h5", h5node.physical_filename)
        self.assertEqual(h5node.physical_basename, "dataset")
        self.assertEqual(h5node.physical_name, "/target/dataset")
        self.assertEqual(h5node.local_basename, "external_link")
        self.assertEqual(h5node.local_name, "/link/external_link")

    def testExternalLinkToSoftLink(self):
        path = ["base.h5", "link", "external_link_to_soft_link"]
        h5node = self.getH5NodeFromPath(self.model, path)

        self.assertNotEqual(h5node.physical_filename, h5node.local_filename)
        self.assertIn("base.h5", h5node.local_filename)
        self.assertIn("base__external.h5", h5node.physical_filename)
        self.assertNotEqual(h5node.physical_filename, h5node.local_filename)
        self.assertEqual(h5node.physical_basename, "dataset")
        self.assertEqual(h5node.physical_name, "/target/dataset")
        self.assertEqual(h5node.local_basename, "external_link_to_soft_link")
        self.assertEqual(h5node.local_name, "/link/external_link_to_soft_link")

    def testExternalBrokenFile(self):
        path = ["base.h5", "broken_link", "external_broken_file"]
        h5node = self.getH5NodeFromPath(self.model, path)

        self.assertNotEqual(h5node.physical_filename, h5node.local_filename)
        self.assertIn("base.h5", h5node.local_filename)
        self.assertIn("not_exists", h5node.physical_filename)
        self.assertEqual(h5node.physical_basename, "link")
        self.assertEqual(h5node.physical_name, "/target/link")
        self.assertEqual(h5node.local_basename, "external_broken_file")
        self.assertEqual(h5node.local_name, "/broken_link/external_broken_file")

    def testExternalBrokenLink(self):
        path = ["base.h5", "broken_link", "external_broken_link"]
        h5node = self.getH5NodeFromPath(self.model, path)

        self.assertNotEqual(h5node.physical_filename, h5node.local_filename)
        self.assertIn("base.h5", h5node.local_filename)
        self.assertIn("__external", h5node.physical_filename)
        self.assertEqual(h5node.physical_basename, "not_exists")
        self.assertEqual(h5node.physical_name, "/target/not_exists")
        self.assertEqual(h5node.local_basename, "external_broken_link")
        self.assertEqual(h5node.local_name, "/broken_link/external_broken_link")

    def testSoftBrokenLink(self):
        path = ["base.h5", "broken_link", "soft_broken_link"]
        h5node = self.getH5NodeFromPath(self.model, path)

        self.assertEqual(h5node.physical_filename, h5node.local_filename)
        self.assertIn("base.h5", h5node.physical_filename)
        self.assertEqual(h5node.physical_basename, "not_exists")
        self.assertEqual(h5node.physical_name, "/group/not_exists")
        self.assertEqual(h5node.local_basename, "soft_broken_link")
        self.assertEqual(h5node.local_name, "/broken_link/soft_broken_link")

    def testSoftLinkToBrokenLink(self):
        path = ["base.h5", "broken_link", "soft_link_to_broken_link"]
        h5node = self.getH5NodeFromPath(self.model, path)

        self.assertEqual(h5node.physical_filename, h5node.local_filename)
        self.assertIn("base.h5", h5node.physical_filename)
        self.assertEqual(h5node.physical_basename, "not_exists")
        self.assertEqual(h5node.physical_name, "/group/not_exists")
        self.assertEqual(h5node.local_basename, "soft_link_to_broken_link")
        self.assertEqual(h5node.local_name, "/broken_link/soft_link_to_broken_link")

    def testDatasetFromSoftLinkToGroup(self):
        path = ["base.h5", "link", "soft_link_to_group", "dataset"]
        h5node = self.getH5NodeFromPath(self.model, path)

        self.assertEqual(h5node.physical_filename, h5node.local_filename)
        self.assertIn("base.h5", h5node.physical_filename)
        self.assertEqual(h5node.physical_basename, "dataset")
        self.assertEqual(h5node.physical_name, "/group/dataset")
        self.assertEqual(h5node.local_basename, "dataset")
        self.assertEqual(h5node.local_name, "/link/soft_link_to_group/dataset")

    def testDatasetFromSoftLinkToFile(self):
        path = [
            "base.h5",
            "link",
            "soft_link_to_file",
            "link",
            "soft_link_to_group",
            "dataset",
        ]
        h5node = self.getH5NodeFromPath(self.model, path)

        self.assertEqual(h5node.physical_filename, h5node.local_filename)
        self.assertIn("base.h5", h5node.physical_filename)
        self.assertEqual(h5node.physical_basename, "dataset")
        self.assertEqual(h5node.physical_name, "/group/dataset")
        self.assertEqual(h5node.local_basename, "dataset")
        self.assertEqual(
            h5node.local_name, "/link/soft_link_to_file/link/soft_link_to_group/dataset"
        )

    @pytest.mark.skipif(not h5py2_9, reason="requires h5py>=2.9")
    def testExternalVirtual(self):
        path = ["base.h5", "ext", "virtual"]
        h5node = self.getH5NodeFromPath(self.model, path)

        self.assertEqual(h5node.physical_filename, h5node.local_filename)
        self.assertIn("base.h5", h5node.physical_filename)
        self.assertEqual(h5node.physical_basename, "virtual")
        self.assertEqual(h5node.physical_name, "/ext/virtual")
        self.assertEqual(h5node.local_basename, "virtual")
        self.assertEqual(h5node.local_name, "/ext/virtual")

    @pytest.mark.skipif(not h5py2_9, reason="requires h5py>=2.9")
    def testExternalRaw(self):
        path = ["base.h5", "ext", "raw"]
        h5node = self.getH5NodeFromPath(self.model, path)

        self.assertEqual(h5node.physical_filename, h5node.local_filename)
        self.assertIn("base.h5", h5node.physical_filename)
        self.assertEqual(h5node.physical_basename, "raw")
        self.assertEqual(h5node.physical_name, "/ext/raw")
        self.assertEqual(h5node.local_basename, "raw")
        self.assertEqual(h5node.local_name, "/ext/raw")


class TestHdf5TreeView(TestCaseQt):
    """Test to check that icons module."""

    def setUp(self):
        super().setUp()

    def testCreate(self):
        view = hdf5.Hdf5TreeView()
        self.assertIsNotNone(view)

    def testContextMenu(self):
        view = hdf5.Hdf5TreeView()
        view._createContextMenu(qt.QPoint(0, 0))

    def testSelection_OriginalModel(self):
        tree = commonh5.File("/foo/bar/1.mock", "w")
        item = tree.create_group("a/b/c/d")
        item.create_group("e").create_group("f")

        view = hdf5.Hdf5TreeView()
        view.findHdf5TreeModel().insertH5pyObject(tree)
        view.setSelectedH5Node(item)

        selected = list(view.selectedH5Nodes())[0]
        self.assertIs(item, selected.h5py_object)

    def testSelection_Simple(self):
        tree = commonh5.File("/foo/bar/1.mock", "w")
        item = tree.create_group("a/b/c/d")
        item.create_group("e").create_group("f")

        model = hdf5.Hdf5TreeModel()
        model.insertH5pyObject(tree)
        view = hdf5.Hdf5TreeView()
        view.setModel(model)
        view.setSelectedH5Node(item)

        selected = list(view.selectedH5Nodes())[0]
        self.assertIs(item, selected.h5py_object)

    def testSelection_NotFound(self):
        tree2 = commonh5.File("/foo/bar/2.mock", "w")
        tree = commonh5.File("/foo/bar/1.mock", "w")
        item = tree.create_group("a/b/c/d")
        item.create_group("e").create_group("f")

        model = hdf5.Hdf5TreeModel()
        model.insertH5pyObject(tree)
        view = hdf5.Hdf5TreeView()
        view.setModel(model)
        view.setSelectedH5Node(tree2)

        selection = list(view.selectedH5Nodes())
        self.assertEqual(len(selection), 0)

    def testSelection_ManyGroupFromSameFile(self):
        tree = commonh5.File("/foo/bar/1.mock", "w")
        group1 = tree.create_group("a1")
        group2 = tree.create_group("a2")
        group3 = tree.create_group("a3")
        group1.create_group("b/c/d")
        item = group2.create_group("b/c/d")
        group3.create_group("b/c/d")

        model = hdf5.Hdf5TreeModel()
        model.insertH5pyObject(group1)
        model.insertH5pyObject(group2)
        model.insertH5pyObject(group3)
        view = hdf5.Hdf5TreeView()
        view.setModel(model)
        view.setSelectedH5Node(item)

        selected = list(view.selectedH5Nodes())[0]
        self.assertIs(item, selected.h5py_object)

    def testSelection_RootFromSubTree(self):
        tree = commonh5.File("/foo/bar/1.mock", "w")
        group = tree.create_group("a1")
        group.create_group("b/c/d")

        model = hdf5.Hdf5TreeModel()
        model.insertH5pyObject(group)
        view = hdf5.Hdf5TreeView()
        view.setModel(model)
        view.setSelectedH5Node(group)

        selected = list(view.selectedH5Nodes())[0]
        self.assertIs(group, selected.h5py_object)

    def testSelection_FileFromSubTree(self):
        tree = commonh5.File("/foo/bar/1.mock", "w")
        group = tree.create_group("a1")
        group.create_group("b").create_group("b").create_group("d")

        model = hdf5.Hdf5TreeModel()
        model.insertH5pyObject(group)
        view = hdf5.Hdf5TreeView()
        view.setModel(model)
        view.setSelectedH5Node(tree)

        selection = list(view.selectedH5Nodes())
        self.assertEqual(len(selection), 0)

    def testSelection_Tree(self):
        tree1 = commonh5.File("/foo/bar/1.mock", "w")
        tree2 = commonh5.File("/foo/bar/2.mock", "w")
        tree3 = commonh5.File("/foo/bar/3.mock", "w")
        tree1.create_group("a/b/c")
        tree2.create_group("a/b/c")
        tree3.create_group("a/b/c")
        item = tree2

        model = hdf5.Hdf5TreeModel()
        model.insertH5pyObject(tree1)
        model.insertH5pyObject(tree2)
        model.insertH5pyObject(tree3)
        view = hdf5.Hdf5TreeView()
        view.setModel(model)
        view.setSelectedH5Node(item)

        selected = list(view.selectedH5Nodes())[0]
        self.assertIs(item, selected.h5py_object)

    def testSelection_RecurssiveLink(self):
        """
        Recurssive link selection

        This example is not really working as expected cause commonh5 do not
        support recurssive links.
        But item.name == "/a/b" and the result is found.
        """
        tree = commonh5.File("/foo/bar/1.mock", "w")
        group = tree.create_group("a")
        group.add_node(commonh5.SoftLink("b", "/"))

        item = tree["/a/b/a/b/a/b/a/b/a/b/a/b/a/b/a/b"]

        model = hdf5.Hdf5TreeModel()
        model.insertH5pyObject(tree)
        view = hdf5.Hdf5TreeView()
        view.setModel(model)
        view.setSelectedH5Node(item)

        selected = list(view.selectedH5Nodes())[0]
        self.assertEqual(item.name, selected.h5py_object.name)

    def testSelection_SelectNone(self):
        tree = commonh5.File("/foo/bar/1.mock", "w")

        model = hdf5.Hdf5TreeModel()
        model.insertH5pyObject(tree)
        view = hdf5.Hdf5TreeView()
        view.setModel(model)
        view.setSelectedH5Node(tree)
        view.setSelectedH5Node(None)

        selection = list(view.selectedH5Nodes())
        self.assertEqual(len(selection), 0)
