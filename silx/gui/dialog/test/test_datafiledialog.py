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
__date__ = "08/03/2019"


import unittest
import tempfile
import numpy
import shutil
import os
import io
import weakref
import fabio
import h5py
import silx.io.url
from silx.gui import qt
from silx.gui.utils import testutils
from ..DataFileDialog import DataFileDialog
from silx.gui.hdf5 import Hdf5TreeModel

_tmpDirectory = None


def setUpModule():
    global _tmpDirectory
    _tmpDirectory = tempfile.mkdtemp(prefix=__name__)

    data = numpy.arange(100 * 100)
    data.shape = 100, 100

    filename = _tmpDirectory + "/singleimage.edf"
    image = fabio.edfimage.EdfImage(data=data)
    image.write(filename)

    filename = _tmpDirectory + "/data.h5"
    f = h5py.File(filename, "w")
    f["scalar"] = 10
    f["image"] = data
    f["cube"] = [data, data + 1, data + 2]
    f["complex_image"] = data * 1j
    f["group/image"] = data
    f["nxdata/foo"] = 10
    f["nxdata"].attrs["NX_class"] = u"NXdata"
    f.close()

    directory = os.path.join(_tmpDirectory, "data")
    os.mkdir(directory)
    filename = os.path.join(directory, "data.h5")
    f = h5py.File(filename, "w")
    f["scalar"] = 10
    f["image"] = data
    f["cube"] = [data, data + 1, data + 2]
    f["complex_image"] = data * 1j
    f["group/image"] = data
    f["nxdata/foo"] = 10
    f["nxdata"].attrs["NX_class"] = u"NXdata"
    f.close()

    filename = _tmpDirectory + "/badformat.h5"
    with io.open(filename, "wb") as f:
        f.write(b"{\nHello Nurse!")


def tearDownModule():
    global _tmpDirectory
    shutil.rmtree(_tmpDirectory)
    _tmpDirectory = None


class _UtilsMixin(object):

    def createDialog(self):
        self._deleteDialog()
        self._dialog = self._createDialog()
        return self._dialog

    def _createDialog(self):
        return DataFileDialog()

    def _deleteDialog(self):
        if not hasattr(self, "_dialog"):
            return
        if self._dialog is not None:
            ref = weakref.ref(self._dialog)
            self._dialog = None
            self.qWaitForDestroy(ref)

    def qWaitForPendingActions(self, dialog):
        for _ in range(20):
            if not dialog.hasPendingEvents():
                return
            self.qWait(10)
        raise RuntimeError("Still have pending actions")

    def assertSamePath(self, path1, path2):
        path1_ = os.path.normcase(path1)
        path2_ = os.path.normcase(path2)
        if path1_ != path2_:
            # Use the unittest API to log and display error
            self.assertEqual(path1, path2)

    def assertNotSamePath(self, path1, path2):
        path1_ = os.path.normcase(path1)
        path2_ = os.path.normcase(path2)
        if path1_ == path2_:
            # Use the unittest API to log and display error
            self.assertNotEqual(path1, path2)


class TestDataFileDialogInteraction(testutils.TestCaseQt, _UtilsMixin):

    def tearDown(self):
        self._deleteDialog()
        testutils.TestCaseQt.tearDown(self)

    def testDisplayAndKeyEscape(self):
        dialog = self.createDialog()
        dialog.show()
        self.qWaitForWindowExposed(dialog)
        self.assertTrue(dialog.isVisible())

        self.keyClick(dialog, qt.Qt.Key_Escape)
        self.assertFalse(dialog.isVisible())
        self.assertEqual(dialog.result(), qt.QDialog.Rejected)

    def testDisplayAndClickCancel(self):
        dialog = self.createDialog()
        dialog.show()
        self.qWaitForWindowExposed(dialog)
        self.assertTrue(dialog.isVisible())

        button = testutils.findChildren(dialog, qt.QPushButton, name="cancel")[0]
        self.mouseClick(button, qt.Qt.LeftButton)
        self.assertFalse(dialog.isVisible())
        self.assertFalse(dialog.isVisible())
        self.assertEqual(dialog.result(), qt.QDialog.Rejected)

    def testDisplayAndClickLockedOpen(self):
        dialog = self.createDialog()
        dialog.show()
        self.qWaitForWindowExposed(dialog)
        self.assertTrue(dialog.isVisible())

        button = testutils.findChildren(dialog, qt.QPushButton, name="open")[0]
        self.mouseClick(button, qt.Qt.LeftButton)
        # open button locked, dialog is not closed
        self.assertTrue(dialog.isVisible())
        self.assertEqual(dialog.result(), qt.QDialog.Rejected)

    def testSelectRoot_Activate(self):
        dialog = self.createDialog()
        browser = testutils.findChildren(dialog, qt.QWidget, name="browser")[0]
        dialog.show()
        self.qWaitForWindowExposed(dialog)
        self.assertTrue(dialog.isVisible())
        filename = _tmpDirectory + "/data.h5"
        dialog.selectFile(os.path.dirname(filename))
        self.qWaitForPendingActions(dialog)

        # select, then double click on the file
        index = browser.rootIndex().model().index(filename)
        browser.selectIndex(index)
        browser.activated.emit(index)
        self.qWaitForPendingActions(dialog)

        button = testutils.findChildren(dialog, qt.QPushButton, name="open")[0]
        self.assertTrue(button.isEnabled())
        self.mouseClick(button, qt.Qt.LeftButton)
        url = silx.io.url.DataUrl(dialog.selectedUrl())
        self.assertTrue(url.data_path() is not None)
        self.assertFalse(dialog.isVisible())
        self.assertEqual(dialog.result(), qt.QDialog.Accepted)

    def testSelectGroup_Activate(self):
        dialog = self.createDialog()
        browser = testutils.findChildren(dialog, qt.QWidget, name="browser")[0]
        dialog.show()
        self.qWaitForWindowExposed(dialog)
        self.assertTrue(dialog.isVisible())
        filename = _tmpDirectory + "/data.h5"
        dialog.selectFile(os.path.dirname(filename))
        self.qWaitForPendingActions(dialog)

        # select, then double click on the file
        index = browser.rootIndex().model().index(filename)
        browser.selectIndex(index)
        browser.activated.emit(index)
        self.qWaitForPendingActions(dialog)

        # select, then double click on the file
        index = browser.rootIndex().model().indexFromH5Object(dialog._AbstractDataFileDialog__h5["/group"])
        browser.selectIndex(index)
        browser.activated.emit(index)
        self.qWaitForPendingActions(dialog)

        button = testutils.findChildren(dialog, qt.QPushButton, name="open")[0]
        self.assertTrue(button.isEnabled())
        self.mouseClick(button, qt.Qt.LeftButton)
        url = silx.io.url.DataUrl(dialog.selectedUrl())
        self.assertEqual(url.data_path(), "/group")
        self.assertFalse(dialog.isVisible())
        self.assertEqual(dialog.result(), qt.QDialog.Accepted)

    def testSelectDataset_Activate(self):
        dialog = self.createDialog()
        browser = testutils.findChildren(dialog, qt.QWidget, name="browser")[0]
        dialog.show()
        self.qWaitForWindowExposed(dialog)
        self.assertTrue(dialog.isVisible())
        filename = _tmpDirectory + "/data.h5"
        dialog.selectFile(os.path.dirname(filename))
        self.qWaitForPendingActions(dialog)

        # select, then double click on the file
        index = browser.rootIndex().model().index(filename)
        browser.selectIndex(index)
        browser.activated.emit(index)
        self.qWaitForPendingActions(dialog)

        # select, then double click on the file
        index = browser.rootIndex().model().indexFromH5Object(dialog._AbstractDataFileDialog__h5["/scalar"])
        browser.selectIndex(index)
        browser.activated.emit(index)
        self.qWaitForPendingActions(dialog)

        button = testutils.findChildren(dialog, qt.QPushButton, name="open")[0]
        self.assertTrue(button.isEnabled())
        self.mouseClick(button, qt.Qt.LeftButton)
        url = silx.io.url.DataUrl(dialog.selectedUrl())
        self.assertEqual(url.data_path(), "/scalar")
        self.assertFalse(dialog.isVisible())
        self.assertEqual(dialog.result(), qt.QDialog.Accepted)

    def testClickOnBackToParentTool(self):
        dialog = self.createDialog()
        dialog.show()
        self.qWaitForWindowExposed(dialog)

        url = testutils.findChildren(dialog, qt.QLineEdit, name="url")[0]
        action = testutils.findChildren(dialog, qt.QAction, name="toParentAction")[0]
        toParentButton = testutils.getQToolButtonFromAction(action)
        filename = _tmpDirectory + "/data/data.h5"

        # init state
        path = silx.io.url.DataUrl(file_path=filename, data_path="/group/image").path()
        dialog.selectUrl(path)
        self.qWaitForPendingActions(dialog)
        path = silx.io.url.DataUrl(scheme="silx", file_path=filename, data_path="/group/image").path()
        self.assertSamePath(url.text(), path)
        # test
        self.mouseClick(toParentButton, qt.Qt.LeftButton)
        self.qWaitForPendingActions(dialog)
        path = silx.io.url.DataUrl(scheme="silx", file_path=filename, data_path="/").path()
        self.assertSamePath(url.text(), path)

        self.mouseClick(toParentButton, qt.Qt.LeftButton)
        self.qWaitForPendingActions(dialog)
        self.assertSamePath(url.text(), _tmpDirectory + "/data")

        self.mouseClick(toParentButton, qt.Qt.LeftButton)
        self.qWaitForPendingActions(dialog)
        self.assertSamePath(url.text(), _tmpDirectory)

    def testClickOnBackToRootTool(self):
        dialog = self.createDialog()
        dialog.show()
        self.qWaitForWindowExposed(dialog)

        url = testutils.findChildren(dialog, qt.QLineEdit, name="url")[0]
        action = testutils.findChildren(dialog, qt.QAction, name="toRootFileAction")[0]
        button = testutils.getQToolButtonFromAction(action)
        filename = _tmpDirectory + "/data.h5"

        # init state
        path = silx.io.url.DataUrl(scheme="silx", file_path=filename, data_path="/group/image").path()
        dialog.selectUrl(path)
        self.qWaitForPendingActions(dialog)
        self.assertSamePath(url.text(), path)
        self.assertTrue(button.isEnabled())
        # test
        self.mouseClick(button, qt.Qt.LeftButton)
        self.qWaitForPendingActions(dialog)
        path = silx.io.url.DataUrl(scheme="silx", file_path=filename, data_path="/").path()
        self.assertSamePath(url.text(), path)
        # self.assertFalse(button.isEnabled())

    def testClickOnBackToDirectoryTool(self):
        dialog = self.createDialog()
        dialog.show()
        self.qWaitForWindowExposed(dialog)

        url = testutils.findChildren(dialog, qt.QLineEdit, name="url")[0]
        action = testutils.findChildren(dialog, qt.QAction, name="toDirectoryAction")[0]
        button = testutils.getQToolButtonFromAction(action)
        filename = _tmpDirectory + "/data.h5"

        # init state
        path = silx.io.url.DataUrl(file_path=filename, data_path="/group/image").path()
        dialog.selectUrl(path)
        self.qWaitForPendingActions(dialog)
        path = silx.io.url.DataUrl(scheme="silx", file_path=filename, data_path="/group/image").path()
        self.assertSamePath(url.text(), path)
        self.assertTrue(button.isEnabled())
        # test
        self.mouseClick(button, qt.Qt.LeftButton)
        self.qWaitForPendingActions(dialog)
        self.assertSamePath(url.text(), _tmpDirectory)
        self.assertFalse(button.isEnabled())

        # FIXME: There is an unreleased qt.QWidget without nameObject
        # No idea where it come from.
        self.allowedLeakingWidgets = 1

    def testClickOnHistoryTools(self):
        dialog = self.createDialog()
        dialog.show()
        self.qWaitForWindowExposed(dialog)

        url = testutils.findChildren(dialog, qt.QLineEdit, name="url")[0]
        forwardAction = testutils.findChildren(dialog, qt.QAction, name="forwardAction")[0]
        backwardAction = testutils.findChildren(dialog, qt.QAction, name="backwardAction")[0]
        filename = _tmpDirectory + "/data.h5"

        dialog.setDirectory(_tmpDirectory)
        self.qWaitForPendingActions(dialog)
        # No way to use QTest.mouseDClick with QListView, QListWidget
        # Then we feed the history using selectPath
        dialog.selectUrl(filename)
        self.qWaitForPendingActions(dialog)
        path2 = silx.io.url.DataUrl(scheme="silx", file_path=filename, data_path="/").path()
        dialog.selectUrl(path2)
        self.qWaitForPendingActions(dialog)
        path3 = silx.io.url.DataUrl(scheme="silx", file_path=filename, data_path="/group").path()
        dialog.selectUrl(path3)
        self.qWaitForPendingActions(dialog)
        self.assertFalse(forwardAction.isEnabled())
        self.assertTrue(backwardAction.isEnabled())

        button = testutils.getQToolButtonFromAction(backwardAction)
        self.mouseClick(button, qt.Qt.LeftButton)
        self.qWaitForPendingActions(dialog)
        self.assertTrue(forwardAction.isEnabled())
        self.assertTrue(backwardAction.isEnabled())
        self.assertSamePath(url.text(), path2)

        button = testutils.getQToolButtonFromAction(forwardAction)
        self.mouseClick(button, qt.Qt.LeftButton)
        self.qWaitForPendingActions(dialog)
        self.assertFalse(forwardAction.isEnabled())
        self.assertTrue(backwardAction.isEnabled())
        self.assertSamePath(url.text(), path3)

    def testSelectImageFromEdf(self):
        dialog = self.createDialog()
        dialog.show()
        self.qWaitForWindowExposed(dialog)

        # init state
        filename = _tmpDirectory + "/singleimage.edf"
        url = silx.io.url.DataUrl(scheme="silx", file_path=filename, data_path="/scan_0/instrument/detector_0/data")
        dialog.selectUrl(url.path())
        self.assertEqual(dialog._selectedData().shape, (100, 100))
        self.assertSamePath(dialog.selectedFile(), filename)
        self.assertSamePath(dialog.selectedUrl(), url.path())

    def testSelectImage(self):
        dialog = self.createDialog()
        dialog.show()
        self.qWaitForWindowExposed(dialog)

        # init state
        filename = _tmpDirectory + "/data.h5"
        path = silx.io.url.DataUrl(scheme="silx", file_path=filename, data_path="/image").path()
        dialog.selectUrl(path)
        # test
        self.assertEqual(dialog._selectedData().shape, (100, 100))
        self.assertSamePath(dialog.selectedFile(), filename)
        self.assertSamePath(dialog.selectedUrl(), path)

    def testSelectScalar(self):
        dialog = self.createDialog()
        dialog.show()
        self.qWaitForWindowExposed(dialog)

        # init state
        filename = _tmpDirectory + "/data.h5"
        path = silx.io.url.DataUrl(scheme="silx", file_path=filename, data_path="/scalar").path()
        dialog.selectUrl(path)
        # test
        self.assertEqual(dialog._selectedData()[()], 10)
        self.assertSamePath(dialog.selectedFile(), filename)
        self.assertSamePath(dialog.selectedUrl(), path)

    def testSelectGroup(self):
        dialog = self.createDialog()
        dialog.show()
        self.qWaitForWindowExposed(dialog)

        # init state
        filename = _tmpDirectory + "/data.h5"
        uri = silx.io.url.DataUrl(scheme="silx", file_path=filename, data_path="/group")
        dialog.selectUrl(uri.path())
        self.qWaitForPendingActions(dialog)
        # test
        self.assertTrue(silx.io.is_group(dialog._selectedData()))
        self.assertSamePath(dialog.selectedFile(), filename)
        uri = silx.io.url.DataUrl(dialog.selectedUrl())
        self.assertSamePath(uri.data_path(), "/group")

    def testSelectRoot(self):
        dialog = self.createDialog()
        dialog.show()
        self.qWaitForWindowExposed(dialog)

        # init state
        filename = _tmpDirectory + "/data.h5"
        uri = silx.io.url.DataUrl(scheme="silx", file_path=filename, data_path="/")
        dialog.selectUrl(uri.path())
        self.qWaitForPendingActions(dialog)
        # test
        self.assertTrue(silx.io.is_file(dialog._selectedData()))
        self.assertSamePath(dialog.selectedFile(), filename)
        uri = silx.io.url.DataUrl(dialog.selectedUrl())
        self.assertSamePath(uri.data_path(), "/")

    def testSelectH5_Activate(self):
        dialog = self.createDialog()
        dialog.show()
        self.qWaitForWindowExposed(dialog)

        # init state
        dialog.selectUrl(_tmpDirectory)
        self.qWaitForPendingActions(dialog)
        browser = testutils.findChildren(dialog, qt.QWidget, name="browser")[0]
        filename = _tmpDirectory + "/data.h5"
        path = silx.io.url.DataUrl(scheme="silx", file_path=filename, data_path="/").path()
        index = browser.rootIndex().model().index(filename)
        # click
        browser.selectIndex(index)
        # double click
        browser.activated.emit(index)
        self.qWaitForPendingActions(dialog)
        # test
        self.assertSamePath(dialog.selectedUrl(), path)

    def testSelectBadFileFormat_Activate(self):
        dialog = self.createDialog()
        dialog.show()
        self.qWaitForWindowExposed(dialog)

        # init state
        dialog.selectUrl(_tmpDirectory)
        self.qWaitForPendingActions(dialog)
        browser = testutils.findChildren(dialog, qt.QWidget, name="browser")[0]
        filename = _tmpDirectory + "/badformat.h5"
        index = browser.model().index(filename)
        browser.selectIndex(index)
        browser.activated.emit(index)
        self.qWaitForPendingActions(dialog)
        # test
        self.assertSamePath(dialog.selectedUrl(), filename)

    def _countSelectableItems(self, model, rootIndex):
        selectable = 0
        for i in range(model.rowCount(rootIndex)):
            index = model.index(i, 0, rootIndex)
            flags = model.flags(index)
            isEnabled = (int(flags) & qt.Qt.ItemIsEnabled) != 0
            if isEnabled:
                selectable += 1
        return selectable

    def testFilterExtensions(self):
        dialog = self.createDialog()
        browser = testutils.findChildren(dialog, qt.QWidget, name="browser")[0]
        dialog.show()
        self.qWaitForWindowExposed(dialog)
        dialog.selectUrl(_tmpDirectory)
        self.qWaitForPendingActions(dialog)
        self.assertEqual(self._countSelectableItems(browser.model(), browser.rootIndex()), 4)


class TestDataFileDialog_FilterDataset(testutils.TestCaseQt, _UtilsMixin):

    def tearDown(self):
        self._deleteDialog()
        testutils.TestCaseQt.tearDown(self)

    def _createDialog(self):
        dialog = DataFileDialog()
        dialog.setFilterMode(DataFileDialog.FilterMode.ExistingDataset)
        return dialog

    def testSelectGroup_Activate(self):
        dialog = self.createDialog()
        browser = testutils.findChildren(dialog, qt.QWidget, name="browser")[0]
        dialog.show()
        self.qWaitForWindowExposed(dialog)
        self.assertTrue(dialog.isVisible())
        filename = _tmpDirectory + "/data.h5"
        dialog.selectFile(os.path.dirname(filename))
        self.qWaitForPendingActions(dialog)

        # select, then double click on the file
        index = browser.rootIndex().model().index(filename)
        browser.selectIndex(index)
        browser.activated.emit(index)
        self.qWaitForPendingActions(dialog)

        # select, then double click on the file
        index = browser.rootIndex().model().indexFromH5Object(dialog._AbstractDataFileDialog__h5["/group"])
        browser.selectIndex(index)
        browser.activated.emit(index)
        self.qWaitForPendingActions(dialog)

        button = testutils.findChildren(dialog, qt.QPushButton, name="open")[0]
        self.assertFalse(button.isEnabled())

    def testSelectDataset_Activate(self):
        dialog = self.createDialog()
        browser = testutils.findChildren(dialog, qt.QWidget, name="browser")[0]
        dialog.show()
        self.qWaitForWindowExposed(dialog)
        self.assertTrue(dialog.isVisible())
        filename = _tmpDirectory + "/data.h5"
        dialog.selectFile(os.path.dirname(filename))
        self.qWaitForPendingActions(dialog)

        # select, then double click on the file
        index = browser.rootIndex().model().index(filename)
        browser.selectIndex(index)
        browser.activated.emit(index)
        self.qWaitForPendingActions(dialog)

        # select, then double click on the file
        index = browser.rootIndex().model().indexFromH5Object(dialog._AbstractDataFileDialog__h5["/scalar"])
        browser.selectIndex(index)
        browser.activated.emit(index)
        self.qWaitForPendingActions(dialog)

        button = testutils.findChildren(dialog, qt.QPushButton, name="open")[0]
        self.assertTrue(button.isEnabled())
        self.mouseClick(button, qt.Qt.LeftButton)
        url = silx.io.url.DataUrl(dialog.selectedUrl())
        self.assertEqual(url.data_path(), "/scalar")
        self.assertFalse(dialog.isVisible())
        self.assertEqual(dialog.result(), qt.QDialog.Accepted)

        data = dialog.selectedData()
        self.assertEqual(data, 10)


class TestDataFileDialog_FilterGroup(testutils.TestCaseQt, _UtilsMixin):

    def tearDown(self):
        self._deleteDialog()
        testutils.TestCaseQt.tearDown(self)

    def _createDialog(self):
        dialog = DataFileDialog()
        dialog.setFilterMode(DataFileDialog.FilterMode.ExistingGroup)
        return dialog

    def testSelectGroup_Activate(self):
        dialog = self.createDialog()
        browser = testutils.findChildren(dialog, qt.QWidget, name="browser")[0]
        dialog.show()
        self.qWaitForWindowExposed(dialog)
        self.assertTrue(dialog.isVisible())
        filename = _tmpDirectory + "/data.h5"
        dialog.selectFile(os.path.dirname(filename))
        self.qWaitForPendingActions(dialog)

        # select, then double click on the file
        index = browser.rootIndex().model().index(filename)
        browser.selectIndex(index)
        browser.activated.emit(index)
        self.qWaitForPendingActions(dialog)

        # select, then double click on the file
        index = browser.rootIndex().model().indexFromH5Object(dialog._AbstractDataFileDialog__h5["/group"])
        browser.selectIndex(index)
        browser.activated.emit(index)
        self.qWaitForPendingActions(dialog)

        button = testutils.findChildren(dialog, qt.QPushButton, name="open")[0]
        self.assertTrue(button.isEnabled())
        self.mouseClick(button, qt.Qt.LeftButton)
        url = silx.io.url.DataUrl(dialog.selectedUrl())
        self.assertEqual(url.data_path(), "/group")
        self.assertFalse(dialog.isVisible())
        self.assertEqual(dialog.result(), qt.QDialog.Accepted)

        self.assertRaises(Exception, dialog.selectedData)

    def testSelectDataset_Activate(self):
        dialog = self.createDialog()
        browser = testutils.findChildren(dialog, qt.QWidget, name="browser")[0]
        dialog.show()
        self.qWaitForWindowExposed(dialog)
        self.assertTrue(dialog.isVisible())
        filename = _tmpDirectory + "/data.h5"
        dialog.selectFile(os.path.dirname(filename))
        self.qWaitForPendingActions(dialog)

        # select, then double click on the file
        index = browser.rootIndex().model().index(filename)
        browser.selectIndex(index)
        browser.activated.emit(index)
        self.qWaitForPendingActions(dialog)

        # select, then double click on the file
        index = browser.rootIndex().model().indexFromH5Object(dialog._AbstractDataFileDialog__h5["/scalar"])
        browser.selectIndex(index)
        browser.activated.emit(index)
        self.qWaitForPendingActions(dialog)

        button = testutils.findChildren(dialog, qt.QPushButton, name="open")[0]
        self.assertFalse(button.isEnabled())


class TestDataFileDialog_FilterNXdata(testutils.TestCaseQt, _UtilsMixin):

    def tearDown(self):
        self._deleteDialog()
        testutils.TestCaseQt.tearDown(self)

    def _createDialog(self):
        def customFilter(obj):
            if "NX_class" in obj.attrs:
                return obj.attrs["NX_class"] == u"NXdata"
            return False

        dialog = DataFileDialog()
        dialog.setFilterMode(DataFileDialog.FilterMode.ExistingGroup)
        dialog.setFilterCallback(customFilter)
        return dialog

    def testSelectGroupRefused_Activate(self):
        dialog = self.createDialog()
        browser = testutils.findChildren(dialog, qt.QWidget, name="browser")[0]
        dialog.show()
        self.qWaitForWindowExposed(dialog)
        self.assertTrue(dialog.isVisible())
        filename = _tmpDirectory + "/data.h5"
        dialog.selectFile(os.path.dirname(filename))
        self.qWaitForPendingActions(dialog)

        # select, then double click on the file
        index = browser.rootIndex().model().index(filename)
        browser.selectIndex(index)
        browser.activated.emit(index)
        self.qWaitForPendingActions(dialog)

        # select, then double click on the file
        index = browser.rootIndex().model().indexFromH5Object(dialog._AbstractDataFileDialog__h5["/group"])
        browser.selectIndex(index)
        browser.activated.emit(index)
        self.qWaitForPendingActions(dialog)

        button = testutils.findChildren(dialog, qt.QPushButton, name="open")[0]
        self.assertFalse(button.isEnabled())

        self.assertRaises(Exception, dialog.selectedData)

    def testSelectNXdataAccepted_Activate(self):
        dialog = self.createDialog()
        browser = testutils.findChildren(dialog, qt.QWidget, name="browser")[0]
        dialog.show()
        self.qWaitForWindowExposed(dialog)
        self.assertTrue(dialog.isVisible())
        filename = _tmpDirectory + "/data.h5"
        dialog.selectFile(os.path.dirname(filename))
        self.qWaitForPendingActions(dialog)

        # select, then double click on the file
        index = browser.rootIndex().model().index(filename)
        browser.selectIndex(index)
        browser.activated.emit(index)
        self.qWaitForPendingActions(dialog)

        # select, then double click on the file
        index = browser.rootIndex().model().indexFromH5Object(dialog._AbstractDataFileDialog__h5["/nxdata"])
        browser.selectIndex(index)
        browser.activated.emit(index)
        self.qWaitForPendingActions(dialog)

        button = testutils.findChildren(dialog, qt.QPushButton, name="open")[0]
        self.assertTrue(button.isEnabled())
        self.mouseClick(button, qt.Qt.LeftButton)
        url = silx.io.url.DataUrl(dialog.selectedUrl())
        self.assertEqual(url.data_path(), "/nxdata")
        self.assertFalse(dialog.isVisible())
        self.assertEqual(dialog.result(), qt.QDialog.Accepted)


class TestDataFileDialogApi(testutils.TestCaseQt, _UtilsMixin):

    def tearDown(self):
        self._deleteDialog()
        testutils.TestCaseQt.tearDown(self)

    def _createDialog(self):
        dialog = DataFileDialog()
        return dialog

    def testSaveRestoreState(self):
        dialog = self.createDialog()
        dialog.setDirectory(_tmpDirectory)
        self.qWaitForPendingActions(dialog)
        state = dialog.saveState()
        dialog = None

        dialog2 = self.createDialog()
        result = dialog2.restoreState(state)
        self.assertTrue(result)
        dialog2 = None

    def printState(self):
        """
        Print state of the ImageFileDialog.

        Can be used to add or regenerate `STATE_VERSION1_QT4` or
        `STATE_VERSION1_QT5`.

        >>> ./run_tests.py -v silx.gui.dialog.test.test_datafiledialog.TestDataFileDialogApi.printState
        """
        dialog = self.createDialog()
        dialog.setDirectory("")
        dialog.setHistory([])
        dialog.setSidebarUrls([])
        state = dialog.saveState()
        string = ""
        strings = []
        for i in range(state.size()):
            d = state.data()[i]
            if not isinstance(d, int):
                d = ord(d)
            if d > 0x20 and d < 0x7F:
                string += chr(d)
            else:
                string += "\\x%02X" % d
            if len(string) > 60:
                strings.append(string)
                string = ""
        strings.append(string)
        strings = ["b'%s'" % s for s in strings]
        print()
        print("\\\n".join(strings))

    STATE_VERSION1_QT4 = b''\
        b'\x00\x00\x00Z\x00s\x00i\x00l\x00x\x00.\x00g\x00u\x00i\x00.\x00'\
        b'd\x00i\x00a\x00l\x00o\x00g\x00.\x00D\x00a\x00t\x00a\x00F\x00i'\
        b'\x00l\x00e\x00D\x00i\x00a\x00l\x00o\x00g\x00.\x00D\x00a\x00t\x00'\
        b'a\x00F\x00i\x00l\x00e\x00D\x00i\x00a\x00l\x00o\x00g\x00\x00\x00'\
        b'\x01\x00\x00\x00\x0C\x00\x00\x00\x00"\x00\x00\x00\xFF\x00\x00'\
        b'\x00\x00\x00\x00\x00\x03\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF'\
        b'\xFF\xFF\x01\x00\x00\x00\x06\x01\x00\x00\x00\x01\x00\x00\x00\x00'\
        b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0C\x00\x00\x00\x00'\
        b'}\x00\x00\x00\x0E\x00B\x00r\x00o\x00w\x00s\x00e\x00r\x00\x00\x00'\
        b'\x01\x00\x00\x00\x0C\x00\x00\x00\x00Z\x00\x00\x00\xFF\x00\x00'\
        b'\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00'\
        b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'\
        b'\x00\x01\x90\x00\x00\x00\x04\x01\x01\x00\x00\x00\x00\x00\x00\x00'\
        b'\x00\x00\x00\x00\x00\x00\x00d\xFF\xFF\xFF\xFF\x00\x00\x00\x81'\
        b'\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x01\x90\x00\x00\x00\x04'\
        b'\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x00'\
        b'\x01\xFF\xFF\xFF\xFF'
    """Serialized state on Qt4. Generated using :meth:`printState`"""

    STATE_VERSION1_QT5 = b''\
        b'\x00\x00\x00Z\x00s\x00i\x00l\x00x\x00.\x00g\x00u\x00i\x00.\x00'\
        b'd\x00i\x00a\x00l\x00o\x00g\x00.\x00D\x00a\x00t\x00a\x00F\x00i'\
        b'\x00l\x00e\x00D\x00i\x00a\x00l\x00o\x00g\x00.\x00D\x00a\x00t\x00'\
        b'a\x00F\x00i\x00l\x00e\x00D\x00i\x00a\x00l\x00o\x00g\x00\x00\x00'\
        b'\x01\x00\x00\x00\x0C\x00\x00\x00\x00#\x00\x00\x00\xFF\x00\x00'\
        b'\x00\x01\x00\x00\x00\x03\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF'\
        b'\xFF\xFF\x01\xFF\xFF\xFF\xFF\x01\x00\x00\x00\x01\x00\x00\x00\x00'\
        b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0C\x00\x00\x00'\
        b'\x00\xAA\x00\x00\x00\x0E\x00B\x00r\x00o\x00w\x00s\x00e\x00r\x00'\
        b'\x00\x00\x01\x00\x00\x00\x0C\x00\x00\x00\x00\x87\x00\x00\x00\xFF'\
        b'\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00'\
        b'\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'\
        b'\x00\x00\x00\x01\x90\x00\x00\x00\x04\x01\x01\x00\x00\x00\x00\x00'\
        b'\x00\x00\x00\x00\x00\x00\x00\x00\x00d\xFF\xFF\xFF\xFF\x00\x00'\
        b'\x00\x81\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00d\x00\x00'\
        b'\x00\x01\x00\x00\x00\x00\x00\x00\x00d\x00\x00\x00\x01\x00\x00'\
        b'\x00\x00\x00\x00\x00d\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00'\
        b'\x00d\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x03\xE8\x00\xFF'\
        b'\xFF\xFF\xFF\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x00\x01'
    """Serialized state on Qt5. Generated using :meth:`printState`"""

    def testAvoidRestoreRegression_Version1(self):
        version = qt.qVersion().split(".")[0]
        if version == "4":
            state = self.STATE_VERSION1_QT4
        elif version == "5":
            state = self.STATE_VERSION1_QT5
        else:
            self.skipTest("Resource not available")

        state = qt.QByteArray(state)
        dialog = self.createDialog()
        result = dialog.restoreState(state)
        self.assertTrue(result)

    def testRestoreRobusness(self):
        """What's happen if you try to open a config file with a different
        binding."""
        state = qt.QByteArray(self.STATE_VERSION1_QT4)
        dialog = self.createDialog()
        dialog.restoreState(state)
        state = qt.QByteArray(self.STATE_VERSION1_QT5)
        dialog = None
        dialog = self.createDialog()
        dialog.restoreState(state)

    def testRestoreNonExistingDirectory(self):
        directory = os.path.join(_tmpDirectory, "dir")
        os.mkdir(directory)
        dialog = self.createDialog()
        dialog.setDirectory(directory)
        self.qWaitForPendingActions(dialog)
        state = dialog.saveState()
        os.rmdir(directory)
        dialog = None

        dialog2 = self.createDialog()
        result = dialog2.restoreState(state)
        self.assertTrue(result)
        self.assertNotEqual(dialog2.directory(), directory)

    def testHistory(self):
        dialog = self.createDialog()
        history = dialog.history()
        dialog.setHistory([])
        self.assertEqual(dialog.history(), [])
        dialog.setHistory(history)
        self.assertEqual(dialog.history(), history)

    def testSidebarUrls(self):
        dialog = self.createDialog()
        urls = dialog.sidebarUrls()
        dialog.setSidebarUrls([])
        self.assertEqual(dialog.sidebarUrls(), [])
        dialog.setSidebarUrls(urls)
        self.assertEqual(dialog.sidebarUrls(), urls)

    def testDirectory(self):
        dialog = self.createDialog()
        self.qWaitForPendingActions(dialog)
        dialog.selectUrl(_tmpDirectory)
        self.qWaitForPendingActions(dialog)
        self.assertSamePath(dialog.directory(), _tmpDirectory)

    def testBadFileFormat(self):
        dialog = self.createDialog()
        dialog.selectUrl(_tmpDirectory + "/badformat.h5")
        self.qWaitForPendingActions(dialog)
        self.assertIsNone(dialog._selectedData())

    def testBadPath(self):
        dialog = self.createDialog()
        dialog.selectUrl("#$%/#$%")
        self.qWaitForPendingActions(dialog)
        self.assertIsNone(dialog._selectedData())

    def testBadSubpath(self):
        dialog = self.createDialog()
        self.qWaitForPendingActions(dialog)

        browser = testutils.findChildren(dialog, qt.QWidget, name="browser")[0]

        filename = _tmpDirectory + "/data.h5"
        url = silx.io.url.DataUrl(scheme="silx", file_path=filename, data_path="/group/foobar")
        dialog.selectUrl(url.path())
        self.qWaitForPendingActions(dialog)
        self.assertIsNotNone(dialog._selectedData())

        # an existing node is browsed, but the wrong path is selected
        index = browser.rootIndex()
        obj = index.model().data(index, role=Hdf5TreeModel.H5PY_OBJECT_ROLE)
        self.assertEqual(obj.name, "/group")
        url = silx.io.url.DataUrl(dialog.selectedUrl())
        self.assertEqual(url.data_path(), "/group")

    def testUnsupportedSlicingPath(self):
        dialog = self.createDialog()
        self.qWaitForPendingActions(dialog)
        dialog.selectUrl(_tmpDirectory + "/data.h5?path=/cube&slice=0")
        self.qWaitForPendingActions(dialog)
        data = dialog._selectedData()
        if data is None:
            # Maybe nothing is selected
            self.assertTrue(True)
        else:
            # Maybe the cube is selected but not sliced
            self.assertEqual(len(data.shape), 3)


def suite():
    test_suite = unittest.TestSuite()
    loadTests = unittest.defaultTestLoader.loadTestsFromTestCase
    test_suite.addTest(loadTests(TestDataFileDialogInteraction))
    test_suite.addTest(loadTests(TestDataFileDialogApi))
    test_suite.addTest(loadTests(TestDataFileDialog_FilterDataset))
    test_suite.addTest(loadTests(TestDataFileDialog_FilterGroup))
    test_suite.addTest(loadTests(TestDataFileDialog_FilterNXdata))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
