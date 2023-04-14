#!/usr/bin/env python
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
"""Main window used to compare images
"""

import sys
import logging
import numpy
import argparse
import os

import silx.io
from silx.gui import qt
import silx.test.utils
from silx.io.url import DataUrl
from silx.gui.plot.CompareImages import CompareImages
from silx.gui.widgets.UrlSelectionTable import UrlSelectionTable

_logger = logging.getLogger(__name__)


class CompareImagesWindow(qt.QMainWindow):
    def __init__(self, backend=None):
        qt.QMainWindow.__init__(self, parent=None)
        self._plot = CompareImages(parent=self, backend=backend)

        self._selectionTable = UrlSelectionTable(parent=self)
        self._selectionTable.setAcceptDrops(True)
        self._dockWidgetMenu = qt.QDockWidget(parent=self)
        self._dockWidgetMenu.layout().setContentsMargins(0, 0, 0, 0)
        self._dockWidgetMenu.setFeatures(qt.QDockWidget.DockWidgetMovable)
        self._dockWidgetMenu.setWidget(self._selectionTable)
        self.addDockWidget(qt.Qt.LeftDockWidgetArea, self._dockWidgetMenu)

        self.setCentralWidget(self._plot)

        self._selectionTable.sigImageAChanged.connect(self._updateImageA)
        self._selectionTable.sigImageBChanged.connect(self._updateImageB)

    def setUrls(self, urls):
        for url in urls:
            self._selectionTable.addUrl(url)

    def setFiles(self, files):
        urls = list()
        for _file in files:
            if os.path.isfile(_file):
                urls.append(DataUrl(file_path=_file, scheme=None))
        urls.sort(key=lambda url: url.path())
        self.setUrls(urls)
        url1 = urls[0].path() if len(urls) >= 1 else None
        url2 = urls[1].path() if len(urls) >= 2 else None
        self._selectionTable.setSelection(
            url_img_a=url1,
            url_img_b=url2
        )

    def clear(self):
        self._plot.clear()
        self._selectionTable.clear()

    def _updateImageA(self, urlpath):
        self._updateImage(urlpath, self._plot.setImage1)

    def _updateImage(self, urlpath, fctptr):
        def getData():
            _url = silx.io.url.DataUrl(path=urlpath)
            for scheme in ('silx', 'fabio'):
                try:
                    dataImg = silx.io.utils.get_data(
                        silx.io.url.DataUrl(file_path=_url.file_path(),
                                            data_slice=_url.data_slice(),
                                            data_path=_url.data_path(),
                                            scheme=scheme))
                except:
                    _logger.debug("Error while loading image with %s" % scheme,
                                  exc_info=True)
                else:
                    # TODO: check is an image
                    return dataImg
            return None

        data = getData()
        if data is not None:
            fctptr(data)

    def _updateImageB(self, urlpath):
        self._updateImage(urlpath, self._plot.setImage2)
