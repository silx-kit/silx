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
from silx.io.url import DataUrl
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
    def __init__(self, url):
        qt.QTableWidgetItem.__init__(self)
        self._url = url

        def slice_to_string(data_slice):
            if data_slice == Ellipsis:
                return "..."
            elif data_slice == slice(None):
                return ":"
            elif isinstance(data_slice, int):
                return str(data_slice)
            else:
                raise TypeError("Unexpected slicing type. Found %s" % type(data_slice))

        text = os.path.basename(url.file_path())
        if url.data_path() is not None:
            text += f" {url.data_path()}"
        if url.data_slice() is not None:
            text += f" [{slice_to_string(url.data_slice())}]"

        self.setText(text)
        self.setToolTip(url.path())

    def dataUrl(self):
        return self._url


class UrlSelectionTable(TableWidget):
    """Table used to select the color channel to be displayed for each"""

    URL_COLUMN = 0
    IMG_A_COLUMN = 1
    IMG_B_COLUMN = 2
    NB_COLUMNS = 3

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
        item.setText("Url")
        item.setToolTip("Silx URL to the data")
        self.setHorizontalHeaderItem(self.URL_COLUMN, item)
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
        setSectionResizeMode(self.URL_COLUMN, qt.QHeaderView.Stretch)
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

    def addUrl(self, url, **kwargs):
        """

        :param url: 
        :param args: 
        :return: index of the created items row
        :rtype int
        """
        assert isinstance(url, DataUrl)
        row = self.rowCount()
        self.setRowCount(row + 1)

        item = _DataUrlItem(url)
        item.setFlags(qt.Qt.ItemIsEnabled | qt.Qt.ItemIsSelectable)
        self.setItem(row, self.URL_COLUMN, item)

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
            self.IMG_B_COLUMN: widgetImgB
        }
        self.resizeColumnsToContents()
        return row

    def _activeImgAChanged(self, row):
        if self._checkBoxes[row][self.IMG_A_COLUMN].isChecked():
            self._updateCheckBoxes(self.IMG_A_COLUMN, row)
            url = self.item(row, self.URL_COLUMN).dataUrl()
            self.sigImageAChanged.emit(url.path())
        else:
            self.sigImageAChanged.emit(None)

    def _activeImgBChanged(self, row):
        if self._checkBoxes[row][self.IMG_B_COLUMN].isChecked():
            self._updateCheckBoxes(self.IMG_B_COLUMN, row)
            url = self.item(row, self.URL_COLUMN).dataUrl()
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

    @deprecated(replacement="getUrlSelection", since_version="2.0", reason="Conflict with Qt API")
    def getSelection(self):
        return self.getUrlSelection()

    def setSelection(self, url_img_a, url_img_b):
        if isinstance(url_img_a, qt.QRect):
            return super().setSelection(url_img_a, url_img_b)
        deprecated_warning(
            'Function',
            'setSelection',
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
            url = self.item(row, self.URL_COLUMN).dataUrl()
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
            url = self.item(row, self.URL_COLUMN).dataUrl()
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

    def dropMimeData(self, row: int, column: int, mimedata: qt.QMimeType, action: qt.Qt.DropAction):
        """Inherited method to handle a drop operation to this model."""
        if action == qt.Qt.IgnoreAction:
            return True
        if mimedata.hasFormat(constants.SILX_URI_MIMETYPE):
            url = DataUrl(mimedata.text())
            self.addUrl(url)
            return True
        return False
