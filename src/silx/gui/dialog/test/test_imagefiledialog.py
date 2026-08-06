from contextlib import contextmanager
import os
import weakref

import fabio
import h5py
import numpy
import pytest

import silx.io.url
from silx.gui import qt
from silx.gui.colors import Colormap
from silx.gui.hdf5 import Hdf5TreeModel
from silx.gui.qt.inspect import isValid
from silx.gui.utils import testutils

from ..ImageFileDialog import ImageFileDialog


@pytest.fixture(scope="module")
def tmp_directory(tmp_path_factory) -> str:
    """Create the files used by every test of this module, once."""
    directory = tmp_path_factory.mktemp("test_imagefiledialog")

    data = numpy.arange(100 * 100)
    data = data.reshape(100, 100)

    filename = directory / "singleimage.edf"
    image = fabio.edfimage.EdfImage(data=data)
    image.write(str(filename))

    filename = directory / "multiframe.edf"
    image = fabio.edfimage.EdfImage(data=data)
    image.append_frame(data=data + 1)
    image.append_frame(data=data + 2)
    image.write(str(filename))

    filename = directory / "singleimage.msk"
    image = fabio.fit2dmaskimage.Fit2dMaskImage(data=data % 2 == 1)
    image.write(str(filename))

    filename = directory / "data.h5"
    with h5py.File(filename, "w") as f:
        f["scalar"] = 10
        f["image"] = data
        f["cube"] = [data, data + 1, data + 2]
        f["single_frame"] = [data + 5]
        f["complex_image"] = data * 1j
        f["group/image"] = data

    sub_directory = directory / "data"
    os.mkdir(sub_directory)
    filename = sub_directory / "data.h5"
    with h5py.File(filename, "w") as f:
        f["scalar"] = 10
        f["image"] = data
        f["cube"] = [data, data + 1, data + 2]
        f["single_frame"] = [data + 5]
        f["complex_image"] = data * 1j
        f["group/image"] = data

    filename = directory / "badformat.edf"
    with open(filename, "wb") as f:
        f.write(b"{\nHello Nurse!")

    return str(directory)


@pytest.fixture
def dialog(qWidgetFactory):
    return qWidgetFactory(ImageFileDialog)


def assert_same_path(path1, path2):
    assert os.path.normcase(os.path.realpath(path1)) == os.path.normcase(
        os.path.realpath(path2)
    ), f"Paths differ: {path1} != {path2}"


def assert_not_same_path(path1, path2):
    assert os.path.normcase(os.path.realpath(path1)) != os.path.normcase(
        os.path.realpath(path2)
    ), f"Paths are equal: {path1} == {path2}"


def assert_same_urls(
    url1: silx.io.url.DataUrl | str,
    url2: silx.io.url.DataUrl | str,
):
    """Check that both DataUrls are equivalent"""
    if isinstance(url1, str):
        url1 = silx.io.url.DataUrl(url1)
    if isinstance(url2, str):
        url2 = silx.io.url.DataUrl(url2)

    assert url1.scheme() == url2.scheme()
    assert_same_path(url1.file_path(), url2.file_path())
    assert url1.data_path() == url2.data_path()
    assert url1.data_slice() == url2.data_slice()


def count_selectable_items(model, root_index):
    selectable = 0
    for i in range(model.rowCount(root_index)):
        index = model.index(i, 0, root_index)
        flags = model.flags(index)
        is_enabled = flags & qt.Qt.ItemIsEnabled == qt.Qt.ItemIsEnabled
        if is_enabled:
            selectable += 1
    return selectable


@contextmanager
def assert_closed_with_result(dialog, expected_result):
    # qWidgetFactory sets WA_DeleteOnClose: the C++ object is gone as
    # soon as the dialog closes, so the result has to be caught on the
    # fly instead of read back from the (now invalid) dialog.
    listener = testutils.SignalListener()
    dialog.finished.connect(listener)
    yield
    assert not isValid(dialog)
    assert listener.arguments(callIndex=0, argumentIndex=0) == expected_result


class TestImageFileDialogInteraction:
    def testDisplayAndKeyEscape(self, dialog, qapp_utils):
        dialog.show()
        qapp_utils.qWaitForWindowExposed(dialog)
        assert dialog.isVisible()

        with assert_closed_with_result(dialog, qt.QDialog.Rejected):
            qapp_utils.keyClick(dialog, qt.Qt.Key_Escape)

    def testDisplayAndClickCancel(self, dialog, qapp_utils):
        dialog.show()
        qapp_utils.qWaitForWindowExposed(dialog)
        assert dialog.isVisible()

        button = testutils.findChildren(dialog, qt.QPushButton, name="cancel")[0]
        with assert_closed_with_result(dialog, qt.QDialog.Rejected):
            qapp_utils.mouseClick(button, qt.Qt.LeftButton)

    def testDisplayAndClickLockedOpen(self, dialog, qapp_utils):
        dialog.show()
        qapp_utils.qWaitForWindowExposed(dialog)
        assert dialog.isVisible()

        button = testutils.findChildren(dialog, qt.QPushButton, name="open")[0]
        qapp_utils.mouseClick(button, qt.Qt.LeftButton)
        # open button locked, dialog is not closed
        assert dialog.isVisible()
        assert dialog.result() == qt.QDialog.Rejected

    def testDisplayAndClickOpen(self, dialog, qapp_utils, tmp_directory):
        dialog.show()
        qapp_utils.qWaitForWindowExposed(dialog)
        assert dialog.isVisible()
        filename = tmp_directory + "/singleimage.edf"
        dialog.selectFile(filename)
        qapp_utils.waitAsLongAs(dialog.hasPendingEvents)

        button = testutils.findChildren(dialog, qt.QPushButton, name="open")[0]
        assert button.isEnabled()
        with assert_closed_with_result(dialog, qt.QDialog.Accepted):
            qapp_utils.mouseClick(button, qt.Qt.LeftButton)

    def testClickOnShortcut(self, dialog, qapp_utils, tmp_directory):
        if qt.BINDING == "PySide6":
            pytest.skip("Avoid segmentation fault with PySide6")

        dialog.show()
        qapp_utils.qWaitForWindowExposed(dialog)

        sidebar = testutils.findChildren(dialog, qt.QListView, name="sidebar")[0]
        url = testutils.findChildren(dialog, qt.QLineEdit, name="url")[0]
        browser = testutils.findChildren(dialog, qt.QWidget, name="browser")[0]
        dialog.setDirectory(tmp_directory)
        qapp_utils.waitAsLongAs(dialog.hasPendingEvents)

        assert_same_path(url.text(), tmp_directory)

        urls = sidebar.urls()
        if len(urls) == 0:
            pytest.skip("No sidebar path")
        path = urls[0].path()
        if path != "" and not os.path.exists(path):
            pytest.skip("Sidebar path do not exists")

        index = sidebar.model().index(0, 0)
        # rect = sidebar.visualRect(index)
        # qapp_utils.mouseClick(sidebar, qt.Qt.LeftButton, pos=rect.center())
        # Using mouse click is not working, let's use the selection API
        sidebar.selectionModel().select(index, qt.QItemSelectionModel.ClearAndSelect)
        qapp_utils.waitAsLongAs(dialog.hasPendingEvents)

        index = browser.rootIndex()
        if not index.isValid():
            path = ""
        else:
            path = index.model().filePath(index)
        assert_not_same_path(tmp_directory, path)
        assert_not_same_path(url.text(), tmp_directory)

    def testClickOnDetailView(self, dialog, qapp_utils):
        dialog.show()
        qapp_utils.qWaitForWindowExposed(dialog)

        action = testutils.findChildren(dialog, qt.QAction, name="detailModeAction")[0]
        detailModeButton = testutils.getQToolButtonFromAction(action)
        qapp_utils.mouseClick(detailModeButton, qt.Qt.LeftButton)
        assert dialog.viewMode() == qt.QFileDialog.Detail

        action = testutils.findChildren(dialog, qt.QAction, name="listModeAction")[0]
        listModeButton = testutils.getQToolButtonFromAction(action)
        qapp_utils.mouseClick(listModeButton, qt.Qt.LeftButton)
        assert dialog.viewMode() == qt.QFileDialog.List

    def testClickOnBackToParentTool(self, dialog, qapp_utils, tmp_directory):
        dialog.show()
        qapp_utils.qWaitForWindowExposed(dialog)

        url = testutils.findChildren(dialog, qt.QLineEdit, name="url")[0]
        action = testutils.findChildren(dialog, qt.QAction, name="toParentAction")[0]
        toParentButton = testutils.getQToolButtonFromAction(action)
        filename = tmp_directory + "/data/data.h5"

        # init state
        path = silx.io.url.DataUrl(file_path=filename, data_path="/group/image").path()
        dialog.selectUrl(path)
        qapp_utils.waitAsLongAs(dialog.hasPendingEvents)
        path = silx.io.url.DataUrl(
            scheme="silx", file_path=filename, data_path="/group/image"
        ).path()
        assert_same_path(url.text(), path)
        # test
        qapp_utils.mouseClick(toParentButton, qt.Qt.LeftButton)
        qapp_utils.waitAsLongAs(dialog.hasPendingEvents)
        path = silx.io.url.DataUrl(
            scheme="silx", file_path=filename, data_path="/"
        ).path()
        assert_same_path(url.text(), path)

        qapp_utils.mouseClick(toParentButton, qt.Qt.LeftButton)
        qapp_utils.waitAsLongAs(dialog.hasPendingEvents)
        assert_same_path(url.text(), tmp_directory + "/data")

        qapp_utils.mouseClick(toParentButton, qt.Qt.LeftButton)
        qapp_utils.waitAsLongAs(dialog.hasPendingEvents)
        assert_same_path(url.text(), tmp_directory)

    def testClickOnBackToRootTool(self, dialog, qapp_utils, tmp_directory):
        dialog.show()
        qapp_utils.qWaitForWindowExposed(dialog)

        url = testutils.findChildren(dialog, qt.QLineEdit, name="url")[0]
        action = testutils.findChildren(dialog, qt.QAction, name="toRootFileAction")[0]
        button = testutils.getQToolButtonFromAction(action)
        filename = tmp_directory + "/data.h5"

        # init state
        path = silx.io.url.DataUrl(
            scheme="silx", file_path=filename, data_path="/group/image"
        ).path()
        dialog.selectUrl(path)
        qapp_utils.waitAsLongAs(dialog.hasPendingEvents)
        assert_same_path(url.text(), path)
        assert button.isEnabled()
        # test
        qapp_utils.mouseClick(button, qt.Qt.LeftButton)
        qapp_utils.waitAsLongAs(dialog.hasPendingEvents)
        path = silx.io.url.DataUrl(
            scheme="silx", file_path=filename, data_path="/"
        ).path()
        assert_same_path(url.text(), path)

    def testClickOnBackToDirectoryTool(self, dialog, qapp_utils, tmp_directory):
        dialog.show()
        qapp_utils.qWaitForWindowExposed(dialog)

        url = testutils.findChildren(dialog, qt.QLineEdit, name="url")[0]
        action = testutils.findChildren(dialog, qt.QAction, name="toDirectoryAction")[0]
        button = testutils.getQToolButtonFromAction(action)
        filename = tmp_directory + "/data.h5"

        # init state
        path = silx.io.url.DataUrl(file_path=filename, data_path="/group/image").path()
        dialog.selectUrl(path)
        qapp_utils.waitAsLongAs(dialog.hasPendingEvents)
        path = silx.io.url.DataUrl(
            scheme="silx", file_path=filename, data_path="/group/image"
        ).path()
        assert_same_path(url.text(), path)
        assert button.isEnabled()
        # test
        qapp_utils.mouseClick(button, qt.Qt.LeftButton)
        qapp_utils.waitAsLongAs(dialog.hasPendingEvents)
        assert_same_path(url.text(), tmp_directory)
        assert not button.isEnabled()

    def testClickOnHistoryTools(self, dialog, qapp_utils, tmp_directory):
        dialog.show()
        qapp_utils.qWaitForWindowExposed(dialog)

        url = testutils.findChildren(dialog, qt.QLineEdit, name="url")[0]
        forwardAction = testutils.findChildren(
            dialog, qt.QAction, name="forwardAction"
        )[0]
        backwardAction = testutils.findChildren(
            dialog, qt.QAction, name="backwardAction"
        )[0]
        filename = tmp_directory + "/data.h5"

        dialog.setDirectory(tmp_directory)
        qapp_utils.waitAsLongAs(dialog.hasPendingEvents)
        # No way to use QTest.mouseDClick with QListView, QListWidget
        # Then we feed the history using selectPath
        dialog.selectUrl(filename)
        qapp_utils.waitAsLongAs(dialog.hasPendingEvents)
        path2 = silx.io.url.DataUrl(
            scheme="silx", file_path=filename, data_path="/"
        ).path()
        dialog.selectUrl(path2)
        qapp_utils.waitAsLongAs(dialog.hasPendingEvents)
        path3 = silx.io.url.DataUrl(
            scheme="silx", file_path=filename, data_path="/group"
        ).path()
        dialog.selectUrl(path3)
        qapp_utils.waitAsLongAs(dialog.hasPendingEvents)
        assert not forwardAction.isEnabled()
        assert backwardAction.isEnabled()

        button = testutils.getQToolButtonFromAction(backwardAction)
        qapp_utils.mouseClick(button, qt.Qt.LeftButton)
        qapp_utils.waitAsLongAs(dialog.hasPendingEvents)
        assert forwardAction.isEnabled()
        assert backwardAction.isEnabled()
        assert_same_path(url.text(), path2)

        button = testutils.getQToolButtonFromAction(forwardAction)
        qapp_utils.mouseClick(button, qt.Qt.LeftButton)
        qapp_utils.waitAsLongAs(dialog.hasPendingEvents)
        assert not forwardAction.isEnabled()
        assert backwardAction.isEnabled()
        assert_same_path(url.text(), path3)

    def testSelectImageFromEdf(self, dialog, qapp_utils, tmp_directory):
        dialog.show()
        qapp_utils.qWaitForWindowExposed(dialog)

        # init state
        filename = tmp_directory + "/singleimage.edf"
        dialog.selectUrl(filename)
        assert dialog.selectedImage().shape == (100, 100)
        assert_same_path(dialog.selectedFile(), filename)
        url = silx.io.url.DataUrl(scheme="fabio", file_path=filename)
        assert_same_urls(dialog.selectedUrl(), url)

    def testSelectImageFromEdf_Activate(self, dialog, qapp_utils, tmp_directory):
        dialog.show()
        qapp_utils.qWaitForWindowExposed(dialog)

        # init state
        dialog.selectUrl(tmp_directory)
        qapp_utils.waitAsLongAs(dialog.hasPendingEvents)
        browser = testutils.findChildren(dialog, qt.QWidget, name="browser")[0]
        filename = tmp_directory + "/singleimage.edf"
        url = silx.io.url.DataUrl(scheme="fabio", file_path=filename).path()
        index = browser.rootIndex().model().index(filename)
        # click
        browser.selectIndex(index)
        # double click
        browser.activated.emit(index)
        qapp_utils.waitAsLongAs(dialog.hasPendingEvents)
        # test
        assert dialog.selectedImage().shape == (100, 100)
        assert_same_path(dialog.selectedFile(), filename)
        assert_same_urls(dialog.selectedUrl(), url)

    def testSelectFrameFromEdf(self, dialog, qapp_utils, tmp_directory):
        dialog.show()
        qapp_utils.qWaitForWindowExposed(dialog)

        # init state
        filename = tmp_directory + "/multiframe.edf"
        url = silx.io.url.DataUrl(scheme="fabio", file_path=filename, data_slice=(1,))
        dialog.selectUrl(url.path())
        # test
        image = dialog.selectedImage()
        assert image.shape == (100, 100)
        assert image[0, 0] == 1
        assert_same_path(dialog.selectedFile(), filename)
        assert_same_urls(dialog.selectedUrl(), url)

    def testSelectImageFromMsk(self, dialog, qapp_utils, tmp_directory):
        dialog.show()
        qapp_utils.qWaitForWindowExposed(dialog)

        # init state
        filename = tmp_directory + "/singleimage.msk"
        url = silx.io.url.DataUrl(scheme="fabio", file_path=filename)
        dialog.selectUrl(url.path())
        # test
        assert dialog.selectedImage().shape == (100, 100)
        assert_same_path(dialog.selectedFile(), filename)
        assert_same_urls(dialog.selectedUrl(), url)

    def testSelectImageFromH5(self, dialog, qapp_utils, tmp_directory):
        dialog.show()
        qapp_utils.qWaitForWindowExposed(dialog)

        # init state
        filename = tmp_directory + "/data.h5"
        url = silx.io.url.DataUrl(scheme="silx", file_path=filename, data_path="/image")
        dialog.selectUrl(url.path())
        # test
        assert dialog.selectedImage().shape == (100, 100)
        assert_same_path(dialog.selectedFile(), filename)
        assert_same_urls(dialog.selectedUrl(), url)

    def testSelectH5_Activate(self, dialog, qapp_utils, tmp_directory):
        dialog.show()
        qapp_utils.qWaitForWindowExposed(dialog)

        # init state
        dialog.selectUrl(tmp_directory)
        qapp_utils.waitAsLongAs(dialog.hasPendingEvents)
        browser = testutils.findChildren(dialog, qt.QWidget, name="browser")[0]
        filename = tmp_directory + "/data.h5"
        url = silx.io.url.DataUrl(scheme="silx", file_path=filename, data_path="/")
        index = browser.rootIndex().model().index(filename)
        # click
        browser.selectIndex(index)
        # double click
        browser.activated.emit(index)
        qapp_utils.waitAsLongAs(dialog.hasPendingEvents)
        # test
        assert_same_urls(dialog.selectedUrl(), url)

    def testSelectFrameFromH5(self, dialog, qapp_utils, tmp_directory):
        dialog.show()
        qapp_utils.qWaitForWindowExposed(dialog)

        # init state
        filename = tmp_directory + "/data.h5"
        url = silx.io.url.DataUrl(
            scheme="silx", file_path=filename, data_path="/cube", data_slice=(1,)
        )
        dialog.selectUrl(url.path())
        # test
        assert dialog.selectedImage().shape == (100, 100)
        assert dialog.selectedImage()[0, 0] == 1
        assert_same_path(dialog.selectedFile(), filename)
        assert_same_urls(dialog.selectedUrl(), url)

    def testSelectSingleFrameFromH5(self, dialog, qapp_utils, tmp_directory):
        dialog.show()
        qapp_utils.qWaitForWindowExposed(dialog)

        # init state
        filename = tmp_directory + "/data.h5"
        url = silx.io.url.DataUrl(
            scheme="silx",
            file_path=filename,
            data_path="/single_frame",
            data_slice=(0,),
        )
        dialog.selectUrl(url.path())
        # test
        assert dialog.selectedImage().shape == (100, 100)
        assert dialog.selectedImage()[0, 0] == 5
        assert_same_path(dialog.selectedFile(), filename)
        assert_same_urls(dialog.selectedUrl(), url)

    def testSelectBadFileFormat_Activate(self, dialog, qapp_utils, tmp_directory):
        dialog.show()
        qapp_utils.qWaitForWindowExposed(dialog)

        # init state
        dialog.selectUrl(tmp_directory)
        qapp_utils.waitAsLongAs(dialog.hasPendingEvents)
        browser = testutils.findChildren(dialog, qt.QWidget, name="browser")[0]
        filename = tmp_directory + "/badformat.edf"
        index = browser.model().index(filename)
        browser.selectIndex(index)
        browser.activated.emit(index)
        qapp_utils.waitAsLongAs(dialog.hasPendingEvents)
        # test
        assert_same_urls(dialog.selectedUrl(), filename)

    def testFilterExtensions(self, dialog, qapp_utils, tmp_directory):
        browser = testutils.findChildren(dialog, qt.QWidget, name="browser")[0]
        filters = testutils.findChildren(dialog, qt.QWidget, name="fileTypeCombo")[0]
        dialog.show()
        qapp_utils.qWaitForWindowExposed(dialog)
        dialog.selectUrl(tmp_directory)
        qapp_utils.waitAsLongAs(dialog.hasPendingEvents)
        assert count_selectable_items(browser.model(), browser.rootIndex()) == 6

        codecName = fabio.edfimage.EdfImage.codec_name()
        index = filters.indexFromCodec(codecName)
        filters.setCurrentIndex(index)
        filters.activated[int].emit(index)
        qapp_utils.qWait(50)
        assert count_selectable_items(browser.model(), browser.rootIndex()) == 4

        codecName = fabio.fit2dmaskimage.Fit2dMaskImage.codec_name()
        index = filters.indexFromCodec(codecName)
        filters.setCurrentIndex(index)
        filters.activated[int].emit(index)
        qapp_utils.qWait(50)
        assert count_selectable_items(browser.model(), browser.rootIndex()) == 2


class TestImageFileDialogApi:
    def testSaveRestoreState(self, qWidgetFactory, qapp_utils, tmp_directory):
        dialog = qWidgetFactory(ImageFileDialog)
        dialog.setDirectory(tmp_directory)
        colormap = Colormap(normalization=Colormap.LOGARITHM)
        dialog.setColormap(colormap)
        qapp_utils.waitAsLongAs(dialog.hasPendingEvents)
        state = dialog.saveState()

        dialog2 = qWidgetFactory(ImageFileDialog)
        result = dialog2.restoreState(state)
        qapp_utils.waitAsLongAs(dialog2.hasPendingEvents)
        assert result
        assert dialog2.colormap().getNormalization() == "log"

    def printState(self, dialog):
        """
        Print state of the ImageFileDialog.

        Can be used to add or regenerate `STATE_VERSION1_QT4` or
        `STATE_VERSION1_QT5`.
        """
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

    STATE_VERSION1_QT4 = (
        b""
        b"\x00\x00\x00^\x00s\x00i\x00l\x00x\x00.\x00g\x00u\x00i\x00.\x00"
        b"d\x00i\x00a\x00l\x00o\x00g\x00.\x00I\x00m\x00a\x00g\x00e\x00F"
        b"\x00i\x00l\x00e\x00D\x00i\x00a\x00l\x00o\x00g\x00.\x00I\x00m\x00"
        b"a\x00g\x00e\x00F\x00i\x00l\x00e\x00D\x00i\x00a\x00l\x00o\x00g"
        b'\x00\x00\x00\x01\x00\x00\x00\x0c\x00\x00\x00\x00"\x00\x00\x00'
        b"\xff\x00\x00\x00\x00\x00\x00\x00\x03\xff\xff\xff\xff\xff\xff\xff"
        b"\xff\xff\xff\xff\xff\x01\x00\x00\x00\x06\x01\x00\x00\x00\x01\x00"
        b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0c\x00"
        b"\x00\x00\x00}\x00\x00\x00\x0e\x00B\x00r\x00o\x00w\x00s\x00e\x00"
        b"r\x00\x00\x00\x01\x00\x00\x00\x0c\x00\x00\x00\x00Z\x00\x00\x00"
        b"\xff\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00"
        b"\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
        b"\x00\x00\x00\x00\x01\x90\x00\x00\x00\x04\x01\x01\x00\x00\x00\x00"
        b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00d\xff\xff\xff\xff\x00"
        b"\x00\x00\x81\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x01\x90\x00"
        b"\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x01\x00"
        b"\x00\x00\x0c\x00\x00\x00\x000\x00\x00\x00\x10\x00C\x00o\x00l\x00"
        b"o\x00r\x00m\x00a\x00p\x00\x00\x00\x01\x00\x00\x00\x08\x00g\x00"
        b"r\x00a\x00y\x01\x01\x00\x00\x00\x06\x00l\x00o\x00g"
    )
    """Serialized state on Qt4. Generated using :meth:`printState`"""

    STATE_VERSION1_QT5 = (
        b""
        b"\x00\x00\x00^\x00s\x00i\x00l\x00x\x00.\x00g\x00u\x00i\x00.\x00"
        b"d\x00i\x00a\x00l\x00o\x00g\x00.\x00I\x00m\x00a\x00g\x00e\x00F"
        b"\x00i\x00l\x00e\x00D\x00i\x00a\x00l\x00o\x00g\x00.\x00I\x00m\x00"
        b"a\x00g\x00e\x00F\x00i\x00l\x00e\x00D\x00i\x00a\x00l\x00o\x00g"
        b"\x00\x00\x00\x01\x00\x00\x00\x0c\x00\x00\x00\x00#\x00\x00\x00"
        b"\xff\x00\x00\x00\x01\x00\x00\x00\x03\xff\xff\xff\xff\xff\xff\xff"
        b"\xff\xff\xff\xff\xff\x01\xff\xff\xff\xff\x01\x00\x00\x00\x01\x00"
        b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0c"
        b"\x00\x00\x00\x00\xaa\x00\x00\x00\x0e\x00B\x00r\x00o\x00w\x00s"
        b"\x00e\x00r\x00\x00\x00\x01\x00\x00\x00\x0c\x00\x00\x00\x00\x87"
        b"\x00\x00\x00\xff\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00"
        b"\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
        b"\x00\x00\x00\x00\x00\x00\x00\x01\x90\x00\x00\x00\x04\x01\x01\x00"
        b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00d\xff\xff"
        b"\xff\xff\x00\x00\x00\x81\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00"
        b"\x00d\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00d\x00\x00\x00"
        b"\x01\x00\x00\x00\x00\x00\x00\x00d\x00\x00\x00\x01\x00\x00\x00"
        b"\x00\x00\x00\x00d\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x03"
        b"\xe8\x00\xff\xff\xff\xff\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00"
        b"\x00\x0c\x00\x00\x00\x000\x00\x00\x00\x10\x00C\x00o\x00l\x00o"
        b"\x00r\x00m\x00a\x00p\x00\x00\x00\x01\x00\x00\x00\x08\x00g\x00"
        b"r\x00a\x00y\x01\x01\x00\x00\x00\x06\x00l\x00o\x00g"
    )
    """Serialized state on Qt5. Generated using :meth:`printState`"""

    def testAvoidRestoreRegression_Version1(self, dialog):
        version = qt.qVersion().split(".")[0]
        if version == "4":
            state = self.STATE_VERSION1_QT4
        elif version == "5":
            state = self.STATE_VERSION1_QT5
        else:
            pytest.skip("Resource not available")

        state = qt.QByteArray(state)
        result = dialog.restoreState(state)
        assert result
        colormap = dialog.colormap()
        assert colormap.getNormalization() == "log"

    def testRestoreRobusness(self, qWidgetFactory):
        """What's happen if you try to open a config file with a different
        binding."""
        state = qt.QByteArray(self.STATE_VERSION1_QT4)
        dialog = qWidgetFactory(ImageFileDialog)
        dialog.restoreState(state)
        state = qt.QByteArray(self.STATE_VERSION1_QT5)
        dialog2 = qWidgetFactory(ImageFileDialog)
        dialog2.restoreState(state)

    def testRestoreNonExistingDirectory(self, qapp_utils, tmp_directory):
        directory = os.path.join(tmp_directory, "dir")
        os.mkdir(directory)
        # We can't use qWidgetFactory here since we need to completely delete
        # the first dialog before the second dialog restores its state,
        # else Windows raises FileNotFoundError.
        dialog = ImageFileDialog()
        dialog.setDirectory(directory)
        qapp_utils.waitAsLongAs(dialog.hasPendingEvents)
        state = dialog.saveState()
        os.rmdir(directory)

        # The first dialog must release its watch on `directory` before
        # the second dialog restores its state, or Windows raises
        # FileNotFoundError.
        ref = weakref.ref(dialog)
        dialog = None
        qapp_utils.qWaitForDestroy(ref)

        dialog2 = ImageFileDialog()
        result = dialog2.restoreState(state)
        assert result
        assert dialog2.directory() != directory

        ref = weakref.ref(dialog2)
        dialog2 = None
        qapp_utils.qWaitForDestroy(ref)

    def testHistory(self, dialog):
        history = dialog.history()
        dialog.setHistory([])
        assert dialog.history() == []
        dialog.setHistory(history)
        assert dialog.history() == history

    def testSidebarUrls(self, dialog):
        urls = dialog.sidebarUrls()
        dialog.setSidebarUrls([])
        assert dialog.sidebarUrls() == []
        dialog.setSidebarUrls(urls)
        assert dialog.sidebarUrls() == urls

    def testColomap(self, dialog):
        colormap = dialog.colormap()
        assert colormap.getNormalization() == "linear"
        colormap = Colormap(normalization=Colormap.LOGARITHM)
        dialog.setColormap(colormap)
        assert colormap.getNormalization() == "log"

    def testDirectory(self, dialog, qapp_utils, tmp_directory):
        qapp_utils.waitAsLongAs(dialog.hasPendingEvents)
        dialog.selectUrl(tmp_directory)
        qapp_utils.waitAsLongAs(dialog.hasPendingEvents)
        assert_same_path(dialog.directory(), tmp_directory)

    def testBadDataType(self, dialog, qapp_utils, tmp_directory):
        dialog.selectUrl(tmp_directory + "/data.h5::/complex_image")
        qapp_utils.waitAsLongAs(dialog.hasPendingEvents)
        assert dialog._selectedData() is None

    def testBadDataShape(self, dialog, qapp_utils, tmp_directory):
        dialog.selectUrl(tmp_directory + "/data.h5::/unknown")
        qapp_utils.waitAsLongAs(dialog.hasPendingEvents)
        assert dialog._selectedData() is None

    def testBadDataFormat(self, dialog, qapp_utils, tmp_directory):
        dialog.selectUrl(tmp_directory + "/badformat.edf")
        qapp_utils.waitAsLongAs(dialog.hasPendingEvents)
        assert dialog._selectedData() is None

    def testBadPath(self, dialog, qapp_utils):
        dialog.selectUrl("#$%/#$%")
        qapp_utils.waitAsLongAs(dialog.hasPendingEvents)
        assert dialog._selectedData() is None

    def testBadSubpath(self, dialog, qapp_utils, tmp_directory):
        qapp_utils.waitAsLongAs(dialog.hasPendingEvents)

        browser = testutils.findChildren(dialog, qt.QWidget, name="browser")[0]

        filename = tmp_directory + "/data.h5"
        url = silx.io.url.DataUrl(
            scheme="silx", file_path=filename, data_path="/group/foobar"
        )
        dialog.selectUrl(url.path())
        qapp_utils.waitAsLongAs(dialog.hasPendingEvents)
        assert dialog._selectedData() is None

        # an existing node is browsed, but the wrong path is selected
        index = browser.rootIndex()
        obj = index.model().data(index, role=Hdf5TreeModel.H5PY_OBJECT_ROLE)
        assert obj.name == "/group"
        url = silx.io.url.DataUrl(dialog.selectedUrl())
        assert url.data_path() == "/group"

    def testBadSlicingPath(self, dialog, qapp_utils, tmp_directory):
        qapp_utils.waitAsLongAs(dialog.hasPendingEvents)
        dialog.selectUrl(tmp_directory + "/data.h5::/cube[a;45,-90]")
        qapp_utils.waitAsLongAs(dialog.hasPendingEvents)
        assert dialog._selectedData() is None
