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

import logging
import numpy

import silx.io
from silx.gui import icons
from silx.gui import qt
from silx.io.url import DataUrl
from silx.gui.plot.CompareImages import CompareImages
from silx.gui.widgets.UrlSelectionTable import UrlSelectionTable
from ..utils import parseutils

_logger = logging.getLogger(__name__)


class CompareImagesWindow(qt.QMainWindow):
    def __init__(self, backend=None, settings=None):
        qt.QMainWindow.__init__(self, parent=None)
        self.setWindowTitle("Silx compare")

        silxIcon = icons.getQIcon("silx")
        self.setWindowIcon(silxIcon)

        self.__settings = settings
        if settings:
            self.restoreSettings(settings)

        self._plot = CompareImages(parent=self, backend=backend)
        self._plot.setAutoResetZoom(False)

        self._selectionTable = UrlSelectionTable(parent=self)
        self._selectionTable.setAcceptDrops(True)

        spliter = qt.QSplitter(self)
        spliter.addWidget(self._selectionTable)
        spliter.addWidget(self._plot)
        spliter.setStretchFactor(1, 1)
        spliter.setCollapsible(0, False)
        spliter.setCollapsible(1, False)
        self.__splitter = spliter

        self.setCentralWidget(spliter)

        self._selectionTable.sigImageAChanged.connect(self._updateImageA)
        self._selectionTable.sigImageBChanged.connect(self._updateImageB)

    def setUrls(self, urls):
        self.clear()
        for url in urls:
            self._selectionTable.addUrl(url)
        url1 = urls[0].path() if len(urls) >= 1 else None
        url2 = urls[1].path() if len(urls) >= 2 else None
        self._selectionTable.setSelection(
            url_img_a=url1,
            url_img_b=url2
        )
        self._plot.resetZoom()

    def clear(self):
        self._plot.clear()
        self._selectionTable.clear()

    def _updateImageA(self, urlPath):
        try:
            data = self.readData(urlPath)
        except Exception as e:
            _logger.error("Error while loading URL %s", urlPath, exc_info=True)
            self._selectionTable.setError(urlPath, e.args[0])
            data = None
        self._plot.setImage1(data)

    def _updateImageB(self, urlPath):
        try:
            data = self.readData(urlPath)
        except Exception as e:
            _logger.error("Error while loading URL %s", urlPath, exc_info=True)
            self._selectionTable.setError(urlPath, e.args[0])
            data = None
        self._plot.setImage2(data)

    def readData(self, urlPath: str):
        """Read an URL as an image
        """
        if urlPath in ("", None):
            return None

        url = silx.io.url.DataUrl(path=urlPath)
        if url.scheme() is None:
            for scheme in ('silx', 'fabio'):
                specificUrl = DataUrl(
                    file_path=url.file_path(),
                    data_slice=url.data_slice(),
                    data_path=url.data_path(),
                    scheme=scheme
                )
                try:
                    data = silx.io.utils.get_data(specificUrl)
                except Exception:
                    _logger.debug("Error while trying to loading %s as %s",
                                  urlPath, scheme, exc_info=True)
                else:
                    break
            else:
                raise ValueError(f"Data from '{urlPath}' is not readable as silx nor fabio")
        else:
            try:
                data = silx.io.utils.get_data(url)
            except Exception:
                raise ValueError(f"Data from '{urlPath}' is not readable")

        if not isinstance(data, numpy.ndarray):
            raise ValueError(f"URL '{urlPath}' does not link to a numpy array")
        if data.dtype.kind not in set(["f", "u", "i", "b"]):
            raise ValueError(f"URL '{urlPath}' does not link to a numeric numpy array")

        if data.ndim == 2:
            return data
        if data.ndim == 3 and data.shape[0] in set(3, 4):
            return data

        raise ValueError(f"URL '{urlPath}' does not link to an numpy image")

    def closeEvent(self, event):
        settings = self.__settings
        if settings:
            self.saveSettings(self.__settings)

    def saveSettings(self, settings):
        """Save the window settings to this settings object

        :param qt.QSettings settings: Initialized settings
        """
        isFullScreen = bool(self.windowState() & qt.Qt.WindowFullScreen)
        if isFullScreen:
            # show in normal to catch the normal geometry
            self.showNormal()

        settings.beginGroup("comparewindow")
        settings.setValue("size", self.size())
        settings.setValue("pos", self.pos())
        settings.setValue("full-screen", isFullScreen)
        settings.setValue("spliter", self.__splitter.sizes())
        settings.endGroup()

        if isFullScreen:
            self.showFullScreen()

    def restoreSettings(self, settings):
        """Restore the window settings using this settings object

        :param qt.QSettings settings: Initialized settings
        """
        settings.beginGroup("comparewindow")
        size = settings.value("size", qt.QSize(640, 480))
        pos = settings.value("pos", qt.QPoint())
        isFullScreen = settings.value("full-screen", False)
        isFullScreen = parseutils.to_bool(isFullScreen, False)

        try:
            data = settings.value("spliter")
            data = [int(d) for d in data]
            self.__splitter.setSizes(data)
        except Exception:
            _logger.debug("Backtrace", exc_info=True)
        settings.endGroup()

        if not pos.isNull():
            self.move(pos)
        if not size.isNull():
            self.resize(size)
        if isFullScreen:
            self.showFullScreen()