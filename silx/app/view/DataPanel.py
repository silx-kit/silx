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
__date__ = "12/10/2018"

import logging
import os.path

from silx.gui import qt
from silx.gui.data.DataViewerFrame import DataViewerFrame


_logger = logging.getLogger(__name__)


class _HeaderLabel(qt.QLabel):

    def __init__(self, parent=None):
        qt.QLabel.__init__(self, parent=parent)
        self.setFrameShape(qt.QFrame.StyledPanel)

    def sizeHint(self):
        return qt.QSize(10, 30)

    def minimumSizeHint(self):
        return qt.QSize(10, 30)

    def setData(self, filename, path):
        if filename == "" and path == "":
            text = ""
        elif filename == "":
            text = path
        else:
            text = "%s::%s" % (filename, path)
        self.setText(text)
        tooltip = ""
        template = "<li><b>%s</b>: %s</li>"
        tooltip += template % ("Directory", os.path.dirname(filename))
        tooltip += template % ("File name", os.path.basename(filename))
        tooltip += template % ("Data path", path)
        tooltip = "<ul>%s</ul>" % tooltip
        tooltip = "<html>%s</html>" % tooltip
        self.setToolTip(tooltip)

    def paintEvent(self, event):
        painter = qt.QPainter(self)

        opt = qt.QStyleOptionHeader()
        opt.orientation = qt.Qt.Horizontal
        opt.text = self.text()
        opt.textAlignment = self.alignment()
        opt.direction = self.layoutDirection()
        opt.fontMetrics = self.fontMetrics()
        opt.palette = self.palette()
        opt.state = qt.QStyle.State_Active
        opt.position = qt.QStyleOptionHeader.Beginning
        style = self.style()

        # Background
        margin = -1
        opt.rect = self.rect().adjusted(margin, margin, -margin, -margin)
        style.drawControl(qt.QStyle.CE_HeaderSection, opt, painter, None)

        # Frame border and text
        super(_HeaderLabel, self).paintEvent(event)


class DataPanel(qt.QWidget):

    def __init__(self, parent=None, context=None):
        qt.QWidget.__init__(self, parent=parent)

        self.__customNxdataItem = None

        self.__dataTitle = _HeaderLabel(self)
        self.__dataTitle.setVisible(False)

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
        self.__dataTitle.setVisible(data is not None)
        if data is not None:
            self.__dataTitle.setVisible(True)
            if hasattr(data, "name"):
                if hasattr(data, "file"):
                    filename = str(data.file.filename)
                else:
                    filename = ""
                path = data.name
            else:
                filename = ""
                path = ""
            self.__dataTitle.setData(filename, path)

    def setCustomDataItem(self, item):
        self.__customNxdataItem = item
        if item is not None:
            data = item.getVirtualGroup()
        else:
            data = None
        self.__dataViewer.setData(data)
        self.__dataTitle.setVisible(item is not None)
        if item is not None:
            text = item.text()
            self.__dataTitle.setText(text)

    def removeDatasetsFrom(self, root):
        """
        Remove all datasets provided by this root

        .. note:: This function do not update data stored inside
            customNxdataItem cause in the silx-view context this item is
            already updated on his own.

        :param root: The root file of datasets to remove
        """
        data = self.__dataViewer.data()
        if data is not None:
            if data.file is not None:
                # That's an approximation, IS can't be used as h5py generates
                # To objects for each requests to a node
                if data.file.filename == root.file.filename:
                    self.__dataViewer.setData(None)

    def replaceDatasetsFrom(self, removedH5, loadedH5):
        """
        Replace any dataset from any NXdata items using the same dataset name
        from another root.

        Usually used when a file was synchronized.

        .. note:: This function do not update data stored inside
            customNxdataItem cause in the silx-view context this item is
            already updated on his own.

        :param removedRoot: The h5py root file which is replaced
            (which have to be removed)
        :param loadedRoot: The new h5py root file which have to be used
            instread.
        """

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
