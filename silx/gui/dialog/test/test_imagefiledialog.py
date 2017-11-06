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
__date__ = "06/11/2017"


import unittest
import tempfile
import numpy
import shutil
import os
import sys
import io
import fabio
from silx.gui import qt
from silx.gui.test import utils
from ..ImageFileDialog import ImageFileDialog
from silx.gui.plot.Colormap import Colormap
from silx.gui.hdf5 import Hdf5TreeModel

try:
    import h5py
except ImportError:
    h5py = None

_tmpDirectory = None


def setUpModule():
    global _tmpDirectory
    _tmpDirectory = tempfile.mkdtemp(prefix=__name__)

    data = numpy.arange(100 * 100)
    data.shape = 100, 100

    filename = _tmpDirectory + "/singleimage.edf"
    image = fabio.edfimage.EdfImage(data=data)
    image.write(filename)

    filename = _tmpDirectory + "/multiframe.edf"
    image = fabio.edfimage.EdfImage(data=data)
    image.appendFrame(data=data + 1)
    image.appendFrame(data=data + 2)
    image.write(filename)

    filename = _tmpDirectory + "/singleimage.msk"
    image = fabio.fit2dmaskimage.Fit2dMaskImage(data=data % 2 == 1)
    image.write(filename)

    if h5py is not None:
        filename = _tmpDirectory + "/data.h5"
        f = h5py.File(filename, "w")
        f["scalar"] = 10
        f["image"] = data
        f["cube"] = [data, data + 1, data + 2]
        f["complex_image"] = data * 1j
        f["group/image"] = data
        f.close()

    filename = _tmpDirectory + "/badformat.edf"
    with io.open(filename, "wb") as f:
        f.write(b"{\nHello Nurse!")


def tearDownModule():
    global _tmpDirectory
    if sys.platform == "win32" and fabio is not None:
        # gc collect is needed to close a file descriptor
        # opened by fabio and not released.
        # https://github.com/silx-kit/fabio/issues/167
        import gc
        gc.collect()
    shutil.rmtree(_tmpDirectory)
    _tmpDirectory = None


class _UtilsMixin(object):

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
            self.assertEquals(path1, path2)

    def assertNotSamePath(self, path1, path2):
        path1_ = os.path.normcase(path1)
        path2_ = os.path.normcase(path2)
        if path1_ == path2_:
            # Use the unittest API to log and display error
            self.assertNotEquals(path1, path2)


class TestImageFileDialogInteraction(utils.TestCaseQt, _UtilsMixin):

    def setUp(self):
        utils.TestCaseQt.setUp(self)
        self.dialog = None

    def tearDown(self):
        if self.dialog is not None:
            self.dialog.clear()
            self.dialog = None
        utils.TestCaseQt.tearDown(self)

    def createDialog(self):
        self.dialog = ImageFileDialog()
        return self.dialog

    def testDisplayAndKeyEscape(self):
        dialog = self.createDialog()
        dialog.show()
        self.qWaitForWindowExposed(dialog)
        self.assertTrue(dialog.isVisible())

        self.keyClick(dialog, qt.Qt.Key_Escape)
        self.assertFalse(dialog.isVisible())
        self.assertEquals(dialog.result(), qt.QDialog.Rejected)

    def testDisplayAndClickCancel(self):
        dialog = self.createDialog()
        dialog.show()
        self.qWaitForWindowExposed(dialog)
        self.assertTrue(dialog.isVisible())

        button = utils.findChildren(dialog, qt.QPushButton, name="cancel")[0]
        self.mouseClick(button, qt.Qt.LeftButton)
        self.assertFalse(dialog.isVisible())
        self.assertFalse(dialog.isVisible())
        self.assertEquals(dialog.result(), qt.QDialog.Rejected)

    def testDisplayAndClickLockedOpen(self):
        dialog = self.createDialog()
        dialog.show()
        self.qWaitForWindowExposed(dialog)
        self.assertTrue(dialog.isVisible())

        button = utils.findChildren(dialog, qt.QPushButton, name="open")[0]
        self.mouseClick(button, qt.Qt.LeftButton)
        # open button locked, dialog is not closed
        self.assertTrue(dialog.isVisible())
        self.assertEquals(dialog.result(), qt.QDialog.Rejected)

    def testDisplayAndClickOpen(self):
        dialog = self.createDialog()
        dialog.show()
        self.qWaitForWindowExposed(dialog)
        self.assertTrue(dialog.isVisible())
        filename = _tmpDirectory + "/singleimage.edf"
        dialog.selectFile(filename)
        self.qWaitForPendingActions(dialog)

        button = utils.findChildren(dialog, qt.QPushButton, name="open")[0]
        self.assertTrue(button.isEnabled())
        self.mouseClick(button, qt.Qt.LeftButton)
        self.assertFalse(dialog.isVisible())
        self.assertEquals(dialog.result(), qt.QDialog.Accepted)

    def testClickOnShortcut(self):
        dialog = self.createDialog()
        dialog.show()
        self.qWaitForWindowExposed(dialog)

        sidebar = utils.findChildren(dialog, qt.QListView, name="sidebar")[0]
        url = utils.findChildren(dialog, qt.QLineEdit, name="url")[0]
        browser = utils.findChildren(dialog, qt.QWidget, name="browser")[0]
        dialog.setDirectory(_tmpDirectory)
        self.qWaitForPendingActions(dialog)

        self.assertSamePath(url.text(), _tmpDirectory)

        urls = sidebar.urls()
        if len(urls) == 0:
            self.skipTest("No sidebar path")
        path = urls[0].path()
        if path != "" and not os.path.exists(path):
            self.skipTest("Sidebar path do not exists")

        index = sidebar.model().index(0, 0)
        # rect = sidebar.visualRect(index)
        # self.mouseClick(sidebar, qt.Qt.LeftButton, pos=rect.center())
        # Using mouse click is not working, let's use the selection API
        sidebar.selectionModel().select(index, qt.QItemSelectionModel.ClearAndSelect)
        self.qWaitForPendingActions(dialog)

        index = browser.rootIndex()
        if not index.isValid():
            path = ""
        else:
            path = index.model().filePath(index)
        self.assertNotSamePath(_tmpDirectory, path)
        self.assertNotSamePath(url.text(), _tmpDirectory)

    def testClickOnDetailView(self):
        dialog = self.createDialog()
        dialog.show()
        self.qWaitForWindowExposed(dialog)

        action = utils.findChildren(dialog, qt.QAction, name="detailModeAction")[0]
        detailModeButton = utils.getQToolButtonFromAction(action)
        self.mouseClick(detailModeButton, qt.Qt.LeftButton)
        self.assertEqual(dialog.viewMode(), qt.QFileDialog.Detail)

        action = utils.findChildren(dialog, qt.QAction, name="listModeAction")[0]
        listModeButton = utils.getQToolButtonFromAction(action)
        self.mouseClick(listModeButton, qt.Qt.LeftButton)
        self.assertEqual(dialog.viewMode(), qt.QFileDialog.List)

    def testClickOnBackToParentTool(self):
        dialog = self.createDialog()
        dialog.show()
        self.qWaitForWindowExposed(dialog)

        url = utils.findChildren(dialog, qt.QLineEdit, name="url")[0]
        action = utils.findChildren(dialog, qt.QAction, name="toParentAction")[0]
        toParentButton = utils.getQToolButtonFromAction(action)

        # init state
        dialog.selectPath(_tmpDirectory + "/data.h5::/group/image")
        self.qWaitForPendingActions(dialog)
        self.assertSamePath(url.text(), _tmpDirectory + "/data.h5::/group/image")
        # test
        self.mouseClick(toParentButton, qt.Qt.LeftButton)
        self.qWaitForPendingActions(dialog)
        self.assertSamePath(url.text(), _tmpDirectory + "/data.h5::/")

        self.mouseClick(toParentButton, qt.Qt.LeftButton)
        self.qWaitForPendingActions(dialog)
        self.assertSamePath(url.text(), _tmpDirectory)

        self.mouseClick(toParentButton, qt.Qt.LeftButton)
        self.qWaitForPendingActions(dialog)
        self.assertSamePath(url.text(), os.path.dirname(_tmpDirectory))

    def testClickOnBackToRootTool(self):
        dialog = self.createDialog()
        dialog.show()
        self.qWaitForWindowExposed(dialog)

        url = utils.findChildren(dialog, qt.QLineEdit, name="url")[0]
        action = utils.findChildren(dialog, qt.QAction, name="toRootFileAction")[0]
        button = utils.getQToolButtonFromAction(action)

        # init state
        dialog.selectPath(_tmpDirectory + "/data.h5::/group/image")
        self.qWaitForPendingActions(dialog)
        self.assertSamePath(url.text(), _tmpDirectory + "/data.h5::/group/image")
        self.assertTrue(button.isEnabled())
        # test
        self.mouseClick(button, qt.Qt.LeftButton)
        self.qWaitForPendingActions(dialog)
        self.assertSamePath(url.text(), _tmpDirectory + "/data.h5::/")
        # self.assertFalse(button.isEnabled())

    def testClickOnBackToDirectoryTool(self):
        dialog = self.createDialog()
        dialog.show()
        self.qWaitForWindowExposed(dialog)

        url = utils.findChildren(dialog, qt.QLineEdit, name="url")[0]
        action = utils.findChildren(dialog, qt.QAction, name="toDirectoryAction")[0]
        button = utils.getQToolButtonFromAction(action)

        # init state
        dialog.selectPath(_tmpDirectory + "/data.h5::/group/image")
        self.qWaitForPendingActions(dialog)
        self.assertSamePath(url.text(), _tmpDirectory + "/data.h5::/group/image")
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

        url = utils.findChildren(dialog, qt.QLineEdit, name="url")[0]
        forwardAction = utils.findChildren(dialog, qt.QAction, name="forwardAction")[0]
        backwardAction = utils.findChildren(dialog, qt.QAction, name="backwardAction")[0]

        dialog.setDirectory(_tmpDirectory)
        self.qWaitForPendingActions(dialog)
        # No way to use QTest.mouseDClick with QListView, QListWidget
        # Then we feed the history using selectPath
        dialog.selectPath(_tmpDirectory + "/data.h5")
        self.qWaitForPendingActions(dialog)
        dialog.selectPath(_tmpDirectory + "/data.h5::/")
        self.qWaitForPendingActions(dialog)
        dialog.selectPath(_tmpDirectory + "/data.h5::/group")
        self.qWaitForPendingActions(dialog)
        self.assertFalse(forwardAction.isEnabled())
        self.assertTrue(backwardAction.isEnabled())

        button = utils.getQToolButtonFromAction(backwardAction)
        self.mouseClick(button, qt.Qt.LeftButton)
        self.qWaitForPendingActions(dialog)
        self.assertTrue(forwardAction.isEnabled())
        self.assertTrue(backwardAction.isEnabled())
        self.assertSamePath(url.text(), _tmpDirectory + "/data.h5::/")

        button = utils.getQToolButtonFromAction(forwardAction)
        self.mouseClick(button, qt.Qt.LeftButton)
        self.qWaitForPendingActions(dialog)
        self.assertFalse(forwardAction.isEnabled())
        self.assertTrue(backwardAction.isEnabled())
        self.assertSamePath(url.text(), _tmpDirectory + "/data.h5::/group")

    def testSelectImageFromEdf(self):
        dialog = self.createDialog()
        dialog.show()
        self.qWaitForWindowExposed(dialog)

        # init state
        filename = _tmpDirectory + "/singleimage.edf"
        path = filename
        dialog.selectPath(path)
        self.assertTrue(dialog.selectedImage().shape, (100, 100))
        self.assertSamePath(dialog.selectedFile(), filename)
        self.assertSamePath(dialog.selectedPath(), filename)

    def testSelectImageFromEdf_Activate(self):
        dialog = self.createDialog()
        dialog.show()
        self.qWaitForWindowExposed(dialog)

        # init state
        dialog.selectPath(_tmpDirectory)
        self.qWaitForPendingActions(dialog)
        browser = utils.findChildren(dialog, qt.QWidget, name="browser")[0]
        filename = _tmpDirectory + "/singleimage.edf"
        index = browser.rootIndex().model().index(filename)
        # click
        browser.selectIndex(index)
        # double click
        browser.activated.emit(index)
        self.qWaitForPendingActions(dialog)
        # test
        self.assertTrue(dialog.selectedImage().shape, (100, 100))
        self.assertSamePath(dialog.selectedFile(), filename)
        self.assertSamePath(dialog.selectedPath(), filename)

    def testSelectFrameFromEdf(self):
        dialog = self.createDialog()
        dialog.show()
        self.qWaitForWindowExposed(dialog)

        # init state
        filename = _tmpDirectory + "/multiframe.edf"
        path = filename + "::[1]"
        dialog.selectPath(path)
        # test
        self.assertTrue(dialog.selectedImage().shape, (100, 100))
        self.assertTrue(dialog.selectedImage()[0, 0], 1)
        self.assertSamePath(dialog.selectedFile(), filename)
        self.assertSamePath(dialog.selectedPath(), path)

    def testSelectImageFromMsk(self):
        dialog = self.createDialog()
        dialog.show()
        self.qWaitForWindowExposed(dialog)

        # init state
        filename = _tmpDirectory + "/singleimage.msk"
        path = filename
        dialog.selectPath(path)
        # test
        self.assertTrue(dialog.selectedImage().shape, (100, 100))
        self.assertSamePath(dialog.selectedFile(), filename)
        self.assertSamePath(dialog.selectedPath(), filename)

    def testSelectImageFromH5(self):
        dialog = self.createDialog()
        dialog.show()
        self.qWaitForWindowExposed(dialog)

        # init state
        filename = _tmpDirectory + "/data.h5"
        path = filename + "::/image"
        dialog.selectPath(path)
        # test
        self.assertTrue(dialog.selectedImage().shape, (100, 100))
        self.assertSamePath(dialog.selectedFile(), filename)
        self.assertSamePath(dialog.selectedPath(), path)

    def testSelectH5_Activate(self):
        dialog = self.createDialog()
        dialog.show()
        self.qWaitForWindowExposed(dialog)

        # init state
        dialog.selectPath(_tmpDirectory)
        self.qWaitForPendingActions(dialog)
        browser = utils.findChildren(dialog, qt.QWidget, name="browser")[0]
        filename = _tmpDirectory + "/data.h5"
        index = browser.rootIndex().model().index(filename)
        # click
        browser.selectIndex(index)
        # double click
        browser.activated.emit(index)
        self.qWaitForPendingActions(dialog)
        # test
        self.assertSamePath(dialog.selectedPath(), filename)

    def testSelectFrameFromH5(self):
        dialog = self.createDialog()
        dialog.show()
        self.qWaitForWindowExposed(dialog)

        # init state
        filename = _tmpDirectory + "/data.h5"
        path = filename + "::/cube[1]"
        dialog.selectPath(path)
        # test
        self.assertTrue(dialog.selectedImage().shape, (100, 100))
        self.assertTrue(dialog.selectedImage()[0, 0], 1)
        self.assertSamePath(dialog.selectedFile(), filename)
        self.assertSamePath(dialog.selectedPath(), path)

    def testSelectBadFileFormat_Activate(self):
        dialog = self.createDialog()
        dialog.show()
        self.qWaitForWindowExposed(dialog)

        # init state
        dialog.selectPath(_tmpDirectory)
        self.qWaitForPendingActions(dialog)
        browser = utils.findChildren(dialog, qt.QWidget, name="browser")[0]
        filename = _tmpDirectory + "/badformat.edf"
        index = browser.rootIndex().model().index(filename)
        browser.activated.emit(index)
        self.qWaitForPendingActions(dialog)
        # test
        self.assertTrue(dialog.selectedPath(), filename)

    def testFilterExtensions(self):
        dialog = self.createDialog()
        browser = utils.findChildren(dialog, qt.QWidget, name="browser")[0]
        filters = utils.findChildren(dialog, qt.QWidget, name="fileTypeCombo")[0]
        dialog.show()
        self.qWaitForWindowExposed(dialog)
        dialog.selectPath(_tmpDirectory)
        self.qWaitForPendingActions(dialog)
        self.assertEqual(browser.model().rowCount(browser.rootIndex()), 5)

        codec = fabio.edfimage.EdfImage.codec_name()
        index = filters.indexFromFabioCodec(codec)
        filters.setCurrentIndex(index)
        filters.activated[int].emit(index)
        self.qWait(50)
        self.assertEqual(browser.model().rowCount(browser.rootIndex()), 3)

        codec = fabio.fit2dmaskimage.Fit2dMaskImage.codec_name()
        index = filters.indexFromFabioCodec(codec)
        filters.setCurrentIndex(index)
        filters.activated[int].emit(index)
        self.qWait(50)
        self.assertEqual(browser.model().rowCount(browser.rootIndex()), 1)


class TestImageFileDialogApi(utils.TestCaseQt, _UtilsMixin):

    def setUp(self):
        utils.TestCaseQt.setUp(self)
        self.dialog = None

    def tearDown(self):
        if self.dialog is not None:
            self.dialog.clear()
            self.dialog = None
        utils.TestCaseQt.tearDown(self)

    def createDialog(self):
        self.dialog = ImageFileDialog()
        return self.dialog

    def testSaveRestoreState(self):
        dialog = ImageFileDialog()
        dialog.setDirectory(_tmpDirectory)
        colormap = Colormap(normalization=Colormap.LOGARITHM)
        dialog.setColormap(colormap)
        self.qWaitForPendingActions(dialog)

        state = dialog.saveState()
        dialog2 = ImageFileDialog()
        result = dialog2.restoreState(state)
        self.assertTrue(result)
        self.assertTrue(dialog2.colormap().getNormalization(), "log")

    def printState(self):
        dialog = ImageFileDialog()
        colormap = Colormap(normalization=Colormap.LOGARITHM)
        dialog.setDirectory("")
        dialog.setHistory([])
        dialog.setColormap(colormap)
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

    STATE_VERSION1_PYSIDE = b''\
        b'\x00\x00\x00`\x00p\x00y\x00F\x00A\x00I\x00.\x00g\x00u\x00i\x00'\
        b'.\x00d\x00i\x00a\x00l\x00o\x00g\x00.\x00I\x00m\x00a\x00g\x00e'\
        b'\x00F\x00i\x00l\x00e\x00D\x00i\x00a\x00l\x00o\x00g\x00.\x00I\x00'\
        b'm\x00a\x00g\x00e\x00F\x00i\x00l\x00e\x00D\x00i\x00a\x00l\x00o'\
        b'\x00g\x00\x00\x00\x01\x00\x00\x00\x0C\x00\x00\x00\x00"\x00\x00'\
        b'\x00\xFF\x00\x00\x00\x00\x00\x00\x00\x03\xFF\xFF\xFF\xFF\xFF\xFF'\
        b'\xFF\xFF\xFF\xFF\xFF\xFF\x01\x00\x00\x00\x06\x01\x00\x00\x00\x01'\
        b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0C'\
        b'\x00\x00\x00\x00}\x00\x00\x00\x0E\x00B\x00r\x00o\x00w\x00s\x00'\
        b'e\x00r\x00\x00\x00\x01\x00\x00\x00\x0C\x00\x00\x00\x00Z\x00\x00'\
        b'\x00\xFF\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00'\
        b'\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'\
        b'\x00\x00\x00\x00\x00\x01\x90\x00\x00\x00\x04\x01\x01\x00\x00\x00'\
        b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00d\xFF\xFF\xFF\xFF'\
        b'\x00\x00\x00\x81\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x01\x90'\
        b'\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x01'\
        b'\x00\x00\x00\x10\x00C\x00o\x00l\x00o\x00r\x00m\x00a\x00p\x00\x00'\
        b'\x00\x01\x00\x00\x00\x00\x08\x00g\x00r\x00a\x00y\x01\x01\x00\x00'\
        b'\x00\x06\x00l\x00o\x00g'

    STATE_VERSION1_PYQT5 = b''\
        b'\x00\x00\x001silx.gui.dialog.ImageFileDialog.ImageFileDialog'\
        b'\x00\x00\x00\x00\x01\x00\x00\x00\x0C\x00\x00\x00\x00#\x00\x00'\
        b'\x00\xFF\x00\x00\x00\x01\x00\x00\x00\x03\xFF\xFF\xFF\xFF\xFF\xFF'\
        b'\xFF\xFF\xFF\xFF\xFF\xFF\x01\xFF\xFF\xFF\xFF\x01\x00\x00\x00\x01'\
        b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00'\
        b'\x00\x0C\x00\x00\x00\x00\xA4\x00\x00\x00\x08Browser\x00\x00\x00'\
        b'\x00\x01\x00\x00\x00\x0C\x00\x00\x00\x00\x87\x00\x00\x00\xFF\x00'\
        b'\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01'\
        b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'\
        b'\x00\x00\x01\x90\x00\x00\x00\x04\x01\x01\x00\x00\x00\x00\x00\x00'\
        b'\x00\x00\x00\x00\x00\x00\x00\x00d\xFF\xFF\xFF\xFF\x00\x00\x00'\
        b'\x81\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00d\x00\x00\x00'\
        b'\x01\x00\x00\x00\x00\x00\x00\x00d\x00\x00\x00\x01\x00\x00\x00'\
        b'\x00\x00\x00\x00d\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00'\
        b'd\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x03\xE8\x00\xFF\xFF'\
        b'\xFF\xFF\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x09Color'\
        b'map\x00\x00\x00\x00\x01\x00\x00\x00\x00\x05gray\x00\x01\x01\x00'\
        b'\x00\x00\x04log\x00'

    STATE_VERSION1_PYQT4 = b''\
        b'\x00\x00\x001silx.gui.dialog.ImageFileDialog.ImageFileDialog'\
        b'\x00\x00\x00\x00\x01\x00\x00\x00\x0C\x00\x00\x00\x00"\x00\x00'\
        b'\x00\xFF\x00\x00\x00\x00\x00\x00\x00\x03\xFF\xFF\xFF\xFF\xFF\xFF'\
        b'\xFF\xFF\xFF\xFF\xFF\xFF\x01\x00\x00\x00\x06\x01\x00\x00\x00\x01'\
        b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00'\
        b'\x0C\x00\x00\x00\x00w\x00\x00\x00\x08Browser\x00\x00\x00\x00\x01'\
        b'\x00\x00\x00\x0C\x00\x00\x00\x00Z\x00\x00\x00\xFF\x00\x00\x00'\
        b'\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00'\
        b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'\
        b'\x01\x90\x00\x00\x00\x04\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00'\
        b'\x00\x00\x00\x00\x00\x00d\xFF\xFF\xFF\xFF\x00\x00\x00\x81\x00'\
        b'\x00\x00\x00\x00\x00\x00\x01\x00\x00\x01\x90\x00\x00\x00\x04\x00'\
        b'\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x09C'\
        b'olormap\x00\x00\x00\x00\x01\x00\x00\x00\x00\x05gray\x00\x01\x01'\
        b'\x00\x00\x00\x04log\x00'

    def testAvoidRestoreRegression_Version1(self):
        if qt.BINDING == "PySide":
            state = self.STATE_VERSION1_PYSIDE
        elif qt.qVersion().startswith("5."):
            state = self.STATE_VERSION1_PYQT5
        elif qt.qVersion().startswith("4."):
            state = self.STATE_VERSION1_PYQT4
        else:
            self.skipTest("Resource not available")

        state = qt.QByteArray(state)
        dialog = self.createDialog()
        result = dialog.restoreState(state)
        self.assertTrue(result)
        colormap = dialog.colormap()
        self.assertTrue(colormap.getNormalization(), "log")

    def testRestoreRobusness(self):
        """What's happen if you try to open a config file with a different
        binding."""
        state = qt.QByteArray(self.STATE_VERSION1_PYQT4)
        dialog = self.createDialog()
        dialog.restoreState(state)
        state = qt.QByteArray(self.STATE_VERSION1_PYQT5)
        dialog = self.createDialog()
        dialog.restoreState(state)
        state = qt.QByteArray(self.STATE_VERSION1_PYSIDE)
        dialog = self.createDialog()
        dialog.restoreState(state)

    def testRestoreNonExistingDirectory(self):
        directory = os.path.join(_tmpDirectory, "dir")
        os.mkdir(directory)
        dialog = ImageFileDialog()
        dialog.setDirectory(directory)
        self.qWaitForPendingActions(dialog)
        state = dialog.saveState()
        os.rmdir(directory)

        dialog2 = ImageFileDialog()
        result = dialog2.restoreState(state)
        self.assertTrue(result)
        self.assertNotEquals(dialog2.directory(), directory)

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

    def testColomap(self):
        dialog = self.createDialog()
        colormap = dialog.colormap()
        self.assertEqual(colormap.getNormalization(), "linear")
        colormap = Colormap(normalization=Colormap.LOGARITHM)
        dialog.setColormap(colormap)
        self.assertEqual(colormap.getNormalization(), "log")

    def testDirectory(self):
        dialog = self.createDialog()
        self.qWaitForPendingActions(dialog)
        dialog.selectPath(_tmpDirectory)
        self.qWaitForPendingActions(dialog)
        self.assertSamePath(dialog.directory(), _tmpDirectory)

    def testBadDataType(self):
        dialog = self.createDialog()
        dialog.selectPath(_tmpDirectory + "/data.h5::/complex_image")
        self.qWaitForPendingActions(dialog)
        self.assertIsNone(dialog.selectedImage())

    def testBadDataShape(self):
        dialog = self.createDialog()
        dialog.selectPath(_tmpDirectory + "/data.h5::/scalar")
        self.qWaitForPendingActions(dialog)
        self.assertIsNone(dialog.selectedImage())

    def testBadDataFormat(self):
        dialog = self.createDialog()
        dialog.selectPath(_tmpDirectory + "/badformat.edf")
        self.qWaitForPendingActions(dialog)
        self.assertIsNone(dialog.selectedImage())

    def testBadPath(self):
        dialog = self.createDialog()
        dialog.selectPath("#$%/#$%")
        self.qWaitForPendingActions(dialog)
        self.assertIsNone(dialog.selectedImage())

    def testBadSubpath(self):
        dialog = self.createDialog()
        self.qWaitForPendingActions(dialog)

        browser = utils.findChildren(dialog, qt.QWidget, name="browser")[0]

        dialog.selectPath(_tmpDirectory + "/data.h5::/group/foobar")
        self.qWaitForPendingActions(dialog)
        self.assertIsNone(dialog.selectedImage())

        # an existing node is browsed, but the wrong path is selected
        index = browser.rootIndex()
        obj = index.model().data(index, role=Hdf5TreeModel.H5PY_OBJECT_ROLE)
        self.assertEqual(obj.name, "/group")
        self.assertSamePath(dialog.selectedPath(), _tmpDirectory + "/data.h5::/group/foobar")

    def testBadSlicingPath(self):
        dialog = self.createDialog()
        self.qWaitForPendingActions(dialog)
        dialog.selectPath(_tmpDirectory + "/data.h5::/cube[a;45,-90]")
        self.qWaitForPendingActions(dialog)
        self.assertIsNone(dialog.selectedImage())


def suite():
    test_suite = unittest.TestSuite()
    loadTests = unittest.defaultTestLoader.loadTestsFromTestCase
    test_suite.addTest(loadTests(TestImageFileDialogInteraction))
    test_suite.addTest(loadTests(TestImageFileDialogApi))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
