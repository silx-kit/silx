# coding: utf-8
# /*##########################################################################
# Copyright (C) 2018 European Synchrotron Radiation Facility
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
# ############################################################################*/
"""Browse a data file with a GUI"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "29/05/2018"

import logging

from silx.gui import qt
from silx.gui.data.DataViewerFrame import DataViewerFrame


_logger = logging.getLogger(__name__)


class DataPanel(qt.QWidget):

    def __init__(self, parent=None, context=None):
        qt.QWidget.__init__(self, parent)

        self.__customNxdataItem = None

        self.__dataTitle = qt.QLabel(self)

        self.__dataViewer = DataViewerFrame(self)
        self.__dataViewer.setGlobalHooks(context)

        layout = qt.QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.__dataTitle)
        layout.addWidget(self.__dataViewer)

    def getData(self):
        return self.__dataViewer.data()

    def getCustomNxdataItem(self):
        return self.__customNxdataItem

    def setData(self, data):
        self.__customNxdataItem = None
        self.__dataViewer.setData(data)
        if data is None:
            self.__dataTitle.setVisible(False)
        else:
            if hasattr(data, "name"):
                if hasattr(data, "file"):
                    label = str(data.file.filename)
                    label += "::"
                else:
                    label = ""
                label += data.name
            else:
                label = ""
            self.__dataTitle.setText(label)

    def setCustomDataItem(self, item):
        self.__customNxdataItem = item
        data = item.getVirtualGroup()
        self.__dataViewer.setData(data)
        if item is None:
            self.__dataTitle.setVisible(False)
        else:
            text = item.text()
            self.__dataTitle.setText(text)

    def removeDatasetsFrom(self, root):
        data = self.__dataViewer.data()
        if data is not None:
            if data.file is not None:
                # That's an approximation, IS can't be used as h5py generates
                # To objects for each requests to a node
                if data.file.filename == root.file.filename:
                    self.__dataViewer.setData(None)

    def replaceDatasetsFrom(self, removedH5, loadedH5):
        data = self.__dataViewer.data()
        if data is not None:
            if data.file is not None:
                if data.file.filename == removedH5.file.filename:
                    # Try to synchonize the viewed data
                    try:
                        # TODO: It have to update the data without changing the
                        # view which is not so easy
                        newData = loadedH5[data.name]
                        self.__dataViewer.setData(newData)
                    except Exception:
                        _logger.debug("Backtrace", exc_info=True)
