# /*##########################################################################
# Copyright (C) 2017-2023 European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
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
#############################################################################*/
"""Some widget construction to check if a sample moved"""

__author__ = ["H. Payno"]
__license__ = "MIT"
__date__ = "19/03/2018"

import os
import functools
import logging
from silx.gui import qt
from silx.gui import utils as qtutils
from silx.gui.widgets.TableWidget import TableWidget
from silx.io.url import DataUrl, slice_sequence_to_string
from silx.utils.deprecation import deprecated, deprecated_warning
from silx.gui import constants

logger = logging.getLogger(__name__)


class _IntegratedRadioButton(qt.QWidget):
    """RadioButton integrated in the QTableWidget as a centered widget"""

    toggled = qt.Signal()

    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent=parent)
        self.setContentsMargins(1, 1, 1, 1)
        layout = qt.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)

        self._radio = qt.QRadioButton(parent=self)
        self._radio.setObjectName("radio")
        self._radio.setAutoExclusive(False)
        self._radio.setMinimumSize(self._radio.minimumSizeHint())
        self._radio.setMaximumSize(self._radio.minimumSizeHint())
        self._radio.toggled.connect(self.toggled.emit)
        layout.addWidget(self._radio)
        self.setSizePolicy(qt.QSizePolicy.Fixed, qt.QSizePolicy.Fixed)

    def setChecked(self, checked: bool):
        self._radio.setChecked(checked)

    def isChecked(self) -> bool:
        return self._radio.isChecked()


class _DataUrlItem(qt.QTableWidgetItem):
    FILENAME = 0
    DATAPATH = 1
    SLICE = 2

    def __init__(self, url, display: int):
        qt.QTableWidgetItem.__init__(self)
        self._url = url
        self._display = display

        if self._display == self.FILENAME:
            text = os.path.basename(self._url.file_path())
        elif self._display == self.DATAPATH:
            text = self._url.data_path()
        elif self._display == self.SLICE:
            s = self._url.data_slice()
            if s is not None:
                text = slice_sequence_to_string(self._url.data_slice())
            else:
                text = ""
        else:
            raise RuntimeError(f"Unsupported display node: {self._display}")

        toolTip = self._url.path()

        self.setText(text)
        self.setToolTip(toolTip)

    def dataUrl(self):
        return self._url


class UrlSelectionTable(TableWidget):
    """Table used to select the color channel to be displayed for each"""

    FILENAME_COLUMN = 0
    DATAPATH_COLUMN = 1
    SLICE_COLUMN = 2
    IMG_A_COLUMN = 3
    IMG_B_COLUMN = 4
    NB_COLUMNS = 5

    sigImageAChanged = qt.Signal(str)
    """Signal emitted when the image A change. Param is the image url path"""

    sigImageBChanged = qt.Signal(str)
    """Signal emitted when the image B change. Param is the image url path"""

    def __init__(self, parent=None):
        TableWidget.__init__(self, parent)
        self.clear()

    def clear(self):
        qt.QTableWidget.clear(self)
        self.setRowCount(0)
        self.setColumnCount(self.NB_COLUMNS)
        self.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.setSelectionMode(qt.QAbstractItemView.NoSelection)

        item = qt.QTableWidgetItem()
        item.setText("Filename")
        item.setToolTip("Filename to the data")
        self.setHorizontalHeaderItem(self.FILENAME_COLUMN, item)
        item = qt.QTableWidgetItem()
        item.setText("Datapath")
        item.setToolTip("Data path to the dataset")
        self.setHorizontalHeaderItem(self.DATAPATH_COLUMN, item)
        item = qt.QTableWidgetItem()
        item.setText("Slice")
        item.setToolTip("Slice applied to the dataset")
        self.setHorizontalHeaderItem(self.SLICE_COLUMN, item)
        item = qt.QTableWidgetItem()
        item.setText("A")
        item.setToolTip("Selected image as A")
        self.setHorizontalHeaderItem(self.IMG_A_COLUMN, item)
        item = qt.QTableWidgetItem()
        item.setText("B")
        item.setToolTip("Selected image as B")
        self.setHorizontalHeaderItem(self.IMG_B_COLUMN, item)

        self.verticalHeader().hide()
        setSectionResizeMode = self.horizontalHeader().setSectionResizeMode
        setSectionResizeMode(self.FILENAME_COLUMN, qt.QHeaderView.ResizeToContents)
        setSectionResizeMode(self.DATAPATH_COLUMN, qt.QHeaderView.Stretch)
        setSectionResizeMode(self.SLICE_COLUMN, qt.QHeaderView.ResizeToContents)
        setSectionResizeMode(self.IMG_A_COLUMN, qt.QHeaderView.ResizeToContents)
        setSectionResizeMode(self.IMG_B_COLUMN, qt.QHeaderView.ResizeToContents)
        self.setSortingEnabled(True)
        self._checkBoxes = {}

    def setUrls(self, urls: list) -> None:
        """

        :param urls: urls to be displayed
        """
        for url in urls:
            self.addUrl(url=url)

    def addUrl(self, url: DataUrl, **kwargs):
        """
        Append this DataUrl to the end of the list of URLs.

        :param url:
        :param args:
        :return: index of the created items row
        :rtype int
        """
        assert isinstance(url, DataUrl)
        row = self.rowCount()
        self.setRowCount(row + 1)

        item = _DataUrlItem(url, _DataUrlItem.FILENAME)
        item.setFlags(qt.Qt.ItemIsEnabled | qt.Qt.ItemIsSelectable)
        self.setItem(row, self.FILENAME_COLUMN, item)

        item = _DataUrlItem(url, _DataUrlItem.DATAPATH)
        item.setFlags(qt.Qt.ItemIsEnabled | qt.Qt.ItemIsSelectable)
        self.setItem(row, self.DATAPATH_COLUMN, item)

        item = _DataUrlItem(url, _DataUrlItem.SLICE)
        item.setFlags(qt.Qt.ItemIsEnabled | qt.Qt.ItemIsSelectable)
        self.setItem(row, self.SLICE_COLUMN, item)

        widgetImgA = _IntegratedRadioButton(parent=self)
        self.setCellWidget(row, self.IMG_A_COLUMN, widgetImgA)
        callbackImgA = functools.partial(self._activeImgAChanged, row)
        widgetImgA.toggled.connect(callbackImgA)

        widgetImgB = _IntegratedRadioButton(parent=self)
        self.setCellWidget(row, self.IMG_B_COLUMN, widgetImgB)
        callbackImgB = functools.partial(self._activeImgBChanged, row)
        widgetImgB.toggled.connect(callbackImgB)

        self._checkBoxes[row] = {
            self.IMG_A_COLUMN: widgetImgA,
            self.IMG_B_COLUMN: widgetImgB,
        }
        self.resizeColumnsToContents()
        return row

    def _getItemFromUrlPath(self, urlPath: str) -> _DataUrlItem:
        """Returns the Qt item storing this urlPath, else None"""
        for r in range(self.rowCount()):
            item = self.item(r, self.FILENAME_COLUMN)
            url = item.dataUrl()
            if url.path() == urlPath:
                return item
        return None

    def setError(self, urlPath: str, message: str):
        """Flag this urlPath with an error in the UI."""
        item = self._getItemFromUrlPath(urlPath)
        if item is None:
            return
        if message == "":
            item.setIcon(qt.QIcon())
            item.setToolTip("")
        else:
            style = qt.QApplication.style()
            icon = style.standardIcon(qt.QStyle.SP_MessageBoxCritical)
            item.setIcon(icon)
            item.setToolTip(f"Error: {message}")

    def _activeImgAChanged(self, row):
        if self._checkBoxes[row][self.IMG_A_COLUMN].isChecked():
            self._updateCheckBoxes(self.IMG_A_COLUMN, row)
            url = self.item(row, self.FILENAME_COLUMN).dataUrl()
            self.sigImageAChanged.emit(url.path())
        else:
            self.sigImageAChanged.emit(None)

    def _activeImgBChanged(self, row):
        if self._checkBoxes[row][self.IMG_B_COLUMN].isChecked():
            self._updateCheckBoxes(self.IMG_B_COLUMN, row)
            url = self.item(row, self.FILENAME_COLUMN).dataUrl()
            self.sigImageBChanged.emit(url.path())
        else:
            self.sigImageBChanged.emit(None)

    def _updateCheckBoxes(self, column, row):
        for r in range(self.rowCount()):
            if r == row:
                continue
            c = self._checkBoxes[r][column]
            with qtutils.blockSignals(c):
                c.setChecked(False)

    @deprecated(
        replacement="getUrlSelection",
        since_version="2.0",
        reason="Conflict with Qt API",
    )
    def getSelection(self):
        return self.getUrlSelection()

    def setSelection(self, url_img_a, url_img_b):
        if isinstance(url_img_a, qt.QRect):
            return super().setSelection(url_img_a, url_img_b)
        deprecated_warning(
            "Function",
            "setSelection",
            replacement="setUrlSelection",
            since_version="2.0",
            reason="Conflict with Qt API",
        )
        return self.setUrlSelection(url_img_a, url_img_b)

    def getUrlSelection(self):
        """

        :return: url selected for img A and img B.
        """
        imgA = imgB = None
        for row in range(self.rowCount()):
            url = self.item(row, self.FILENAME_COLUMN).dataUrl()
            if self._checkBoxes[row][self.IMG_A_COLUMN].isChecked():
                imgA = url
            if self._checkBoxes[row][self.IMG_B_COLUMN].isChecked():
                imgB = url
        return imgA, imgB

    def setUrlSelection(self, url_img_a, url_img_b):
        """

        :param ddict: key: image url, values: list of active channels
        """
        rowA = None
        rowB = None
        for row in range(self.rowCount()):
            for img in (self.IMG_A_COLUMN, self.IMG_B_COLUMN):
                c = self._checkBoxes[row][img]
                with qtutils.blockSignals(c):
                    c.setChecked(False)
            url = self.item(row, self.FILENAME_COLUMN).dataUrl()
            if url.path() == url_img_a:
                rowA = row
            if url.path() == url_img_b:
                rowB = row

        if rowA is not None:
            c = self._checkBoxes[rowA][self.IMG_A_COLUMN]
            with qtutils.blockSignals(c):
                c.setChecked(True)

        if rowB is not None:
            c = self._checkBoxes[rowB][self.IMG_B_COLUMN]
            with qtutils.blockSignals(c):
                c.setChecked(True)

        self.sigImageAChanged.emit(url_img_a)
        self.sigImageBChanged.emit(url_img_b)

    def removeUrl(self, url):
        raise NotImplementedError("")

    def supportedDropActions(self):
        """Inherited method to redefine supported drop actions."""
        return qt.Qt.CopyAction | qt.Qt.MoveAction

    def mimeTypes(self):
        """Inherited method to redefine draggable mime types."""
        return [constants.SILX_URI_MIMETYPE]

    def dropMimeData(
        self, row: int, column: int, mimedata: qt.QMimeType, action: qt.Qt.DropAction
    ):
        """Inherited method to handle a drop operation to this model."""
        if action == qt.Qt.IgnoreAction:
            return True
        if mimedata.hasFormat(constants.SILX_URI_MIMETYPE):
            urlText = str(mimedata.data(constants.SILX_URI_MIMETYPE), "utf-8")
            url = DataUrl(urlText)
            self.addUrl(url)
            return True
        return False
