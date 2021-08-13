# /*##########################################################################
# Copyright (C) 2017-2021 European Synchrotron Radiation Facility
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

from silx.gui import qt
from collections import OrderedDict
from silx.gui.widgets.TableWidget import TableWidget
from silx.io.url import DataUrl
import functools
import logging
import os

logger = logging.getLogger(__name__)


class UrlSelectionTable(TableWidget):
    """Table used to select the color channel to be displayed for each"""

    COLUMS_INDEX = OrderedDict([
        ('url', 0),
        ('img A', 1),
        ('img B', 2),
    ])

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
        self.setColumnCount(len(self.COLUMS_INDEX))
        self.setHorizontalHeaderLabels(list(self.COLUMS_INDEX.keys()))
        self.verticalHeader().hide()
        self.horizontalHeader().setSectionResizeMode(0,
                                                     qt.QHeaderView.Stretch)

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

        _item = qt.QTableWidgetItem()
        _item.setText(os.path.basename(url.path()))
        _item.setFlags(qt.Qt.ItemIsEnabled | qt.Qt.ItemIsSelectable)
        self.setItem(row, self.COLUMS_INDEX['url'], _item)

        widgetImgA = qt.QRadioButton(parent=self)
        widgetImgA.setAutoExclusive(False)
        self.setCellWidget(row, self.COLUMS_INDEX['img A'], widgetImgA)
        callbackImgA = functools.partial(self._activeImgAChanged, url.path())
        widgetImgA.toggled.connect(callbackImgA)

        widgetImgB = qt.QRadioButton(parent=self)
        widgetImgA.setAutoExclusive(False)
        self.setCellWidget(row, self.COLUMS_INDEX['img B'], widgetImgB)
        callbackImgB = functools.partial(self._activeImgBChanged, url.path())
        widgetImgB.toggled.connect(callbackImgB)

        self._checkBoxes[url.path()] = {'img A': widgetImgA,
                                        'img B': widgetImgB}
        self.resizeColumnsToContents()
        return row

    def _activeImgAChanged(self, name):
        self._updatecheckBoxes('img A', name)
        self.sigImageAChanged.emit(name)

    def _activeImgBChanged(self, name):
        self._updatecheckBoxes('img B', name)
        self.sigImageBChanged.emit(name)

    def _updatecheckBoxes(self, whichImg, name):
        assert name in self._checkBoxes
        assert whichImg in self._checkBoxes[name]
        if self._checkBoxes[name][whichImg].isChecked():
            for radioUrl in self._checkBoxes:
                if radioUrl != name:
                    self._checkBoxes[radioUrl][whichImg].blockSignals(True)
                    self._checkBoxes[radioUrl][whichImg].setChecked(False)
                    self._checkBoxes[radioUrl][whichImg].blockSignals(False)

    def getSelection(self):
        """

        :return: url selected for img A and img B.
        """
        imgA = imgB = None
        for radioUrl in self._checkBoxes:
            if self._checkBoxes[radioUrl]['img A'].isChecked():
                imgA = radioUrl
            if self._checkBoxes[radioUrl]['img B'].isChecked():
                imgB = radioUrl
        return imgA, imgB

    def setSelection(self, url_img_a, url_img_b):
        """

        :param ddict: key: image url, values: list of active channels
        """
        for radioUrl in self._checkBoxes:
            for img in ('img A', 'img B'):
                self._checkBoxes[radioUrl][img].blockSignals(True)
                self._checkBoxes[radioUrl][img].setChecked(False)
                self._checkBoxes[radioUrl][img].blockSignals(False)

        self._checkBoxes[radioUrl][img].blockSignals(True)
        self._checkBoxes[url_img_a]['img A'].setChecked(True)
        self._checkBoxes[radioUrl][img].blockSignals(False)

        self._checkBoxes[radioUrl][img].blockSignals(True)
        self._checkBoxes[url_img_b]['img B'].setChecked(True)
        self._checkBoxes[radioUrl][img].blockSignals(False)
        self.sigImageAChanged.emit(url_img_a)
        self.sigImageBChanged.emit(url_img_b)

    def removeUrl(self, url):
        raise NotImplementedError("")
