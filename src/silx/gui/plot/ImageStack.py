# /*##########################################################################
#
# Copyright (c) 2020-2021 European Synchrotron Radiation Facility
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
"""Image stack view with data prefetch capabilty."""

__authors__ = ["H. Payno"]
__license__ = "MIT"
__date__ = "04/03/2019"


from silx.gui import icons, qt
from silx.gui.plot import Plot2D
from silx.gui.utils import concurrent
from silx.io.url import DataUrl
from silx.io.utils import get_data
from collections import OrderedDict
from silx.gui.widgets.FrameBrowser import HorizontalSliderWithBrowser
import time
import threading
import typing
import logging

_logger = logging.getLogger(__name__)


class _PlotWithWaitingLabel(qt.QWidget):
    """Image plot widget with an overlay 'waiting' status.
    """

    class AnimationThread(threading.Thread):
        def __init__(self, label):
            self.running = True
            self._label = label
            self.animated_icon = icons.getWaitIcon()
            self.animated_icon.register(self._label)
            super(_PlotWithWaitingLabel.AnimationThread, self).__init__()

        def run(self):
            while self.running:
                time.sleep(0.05)
                icon = self.animated_icon.currentIcon()
                self.future_result = concurrent.submitToQtMainThread(
                    self._label.setPixmap, icon.pixmap(30, state=qt.QIcon.On))

        def stop(self):
            """Stop the update thread"""
            if self.running:
                self.animated_icon.unregister(self._label)
                self.running = False
                self.join(2)

    def __init__(self, parent):
        super(_PlotWithWaitingLabel, self).__init__(parent=parent)
        self._autoResetZoom = True
        layout = qt.QStackedLayout(self)
        layout.setStackingMode(qt.QStackedLayout.StackAll)

        self._waiting_label = qt.QLabel(parent=self)
        self._waiting_label.setAlignment(qt.Qt.AlignHCenter | qt.Qt.AlignVCenter)
        layout.addWidget(self._waiting_label)

        self._plot = Plot2D(parent=self)
        layout.addWidget(self._plot)

        self.updateThread = _PlotWithWaitingLabel.AnimationThread(self._waiting_label)
        self.updateThread.start()

    def close(self) -> bool:
        super(_PlotWithWaitingLabel, self).close()
        self.stopUpdateThread()

    def stopUpdateThread(self):
        self.updateThread.stop()

    def setAutoResetZoom(self, reset):
        """
        Should we reset the zoom when adding an image (eq. when browsing)

        :param bool reset:
        """
        self._autoResetZoom = reset
        if self._autoResetZoom:
            self._plot.resetZoom()

    def isAutoResetZoom(self):
        """

        :return: True if a reset is done when the image change
        :rtype: bool
        """
        return self._autoResetZoom

    def setWaiting(self, activate=True):
        if activate is True:
            self._plot.clear()
            self._waiting_label.show()
        else:
            self._waiting_label.hide()

    def setData(self, data):
        self.setWaiting(activate=False)
        self._plot.addImage(data=data, resetzoom=self._autoResetZoom)

    def clear(self):
        self._plot.clear()
        self.setWaiting(False)

    def getPlotWidget(self):
        return self._plot


class _HorizontalSlider(HorizontalSliderWithBrowser):

    sigCurrentUrlIndexChanged = qt.Signal(int)

    def __init__(self, parent):
        super(_HorizontalSlider, self).__init__(parent=parent)
        #  connect signal / slot
        self.valueChanged.connect(self._urlChanged)

    def setUrlIndex(self, index):
        self.setValue(index)
        self.sigCurrentUrlIndexChanged.emit(index)

    def _urlChanged(self, value):
        self.sigCurrentUrlIndexChanged.emit(value)


class UrlList(qt.QWidget):
    """List of URLs the user to select an URL"""

    sigCurrentUrlChanged = qt.Signal(str)
    """Signal emitted when the active/current url change"""

    def __init__(self, parent=None):
        super(UrlList, self).__init__(parent)
        self.setLayout(qt.QVBoxLayout())
        self.layout().setSpacing(0)
        self.layout().setContentsMargins(0, 0, 0, 0)
        self._listWidget = qt.QListWidget(parent=self)
        self.layout().addWidget(self._listWidget)

        # connect signal / Slot
        self._listWidget.currentItemChanged.connect(self._notifyCurrentUrlChanged)

        # expose API
        self.currentItem = self._listWidget.currentItem

    def setUrls(self, urls: list) -> None:
        url_names = []
        [url_names.append(url.path()) for url in urls]
        self._listWidget.addItems(url_names)

    def _notifyCurrentUrlChanged(self, current, previous):
        if current is None:
            pass
        else:
            self.sigCurrentUrlChanged.emit(current.text())

    def setUrl(self, url: DataUrl) -> None:
        assert isinstance(url, DataUrl)
        sel_items = self._listWidget.findItems(url.path(), qt.Qt.MatchExactly)
        if sel_items is None:
            _logger.warning(url.path(), ' is not registered in the list.')
        elif len(sel_items) > 0:
            item = sel_items[0]
            self._listWidget.setCurrentItem(item)
            self.sigCurrentUrlChanged.emit(item.text())

    def clear(self):
        self._listWidget.clear()


class _ToggleableUrlSelectionTable(qt.QWidget):

    _BUTTON_ICON = qt.QStyle.SP_ToolBarHorizontalExtensionButton  # noqa

    sigCurrentUrlChanged = qt.Signal(str)
    """Signal emitted when the active/current url change"""

    def __init__(self, parent=None) -> None:
        qt.QWidget.__init__(self, parent)
        self.setLayout(qt.QGridLayout())
        self._toggleButton = qt.QPushButton(parent=self)
        self.layout().addWidget(self._toggleButton, 0, 2, 1, 1)
        self._toggleButton.setSizePolicy(qt.QSizePolicy.Fixed,
                                         qt.QSizePolicy.Fixed)

        self._urlsTable = UrlList(parent=self)
        self.layout().addWidget(self._urlsTable, 1, 1, 1, 2)

        # set up
        self._setButtonIcon(show=True)

        # Signal / slot connection
        self._toggleButton.clicked.connect(self.toggleUrlSelectionTable)
        self._urlsTable.sigCurrentUrlChanged.connect(self._propagateSignal)

        # expose API
        self.setUrls = self._urlsTable.setUrls
        self.setUrl = self._urlsTable.setUrl
        self.currentItem = self._urlsTable.currentItem

    def toggleUrlSelectionTable(self):
        visible = not self.urlSelectionTableIsVisible()
        self._setButtonIcon(show=visible)
        self._urlsTable.setVisible(visible)

    def _setButtonIcon(self, show):
        style = qt.QApplication.instance().style()
        # return a QIcon
        icon = style.standardIcon(self._BUTTON_ICON)
        if show is False:
            pixmap = icon.pixmap(32, 32).transformed(qt.QTransform().scale(-1, 1))
            icon = qt.QIcon(pixmap)
        self._toggleButton.setIcon(icon)

    def urlSelectionTableIsVisible(self):
        return self._urlsTable.isVisible()

    def _propagateSignal(self, url):
        self.sigCurrentUrlChanged.emit(url)

    def clear(self):
        self._urlsTable.clear()


class UrlLoader(qt.QThread):
    """
    Thread use to load DataUrl
    """
    def __init__(self, parent, url):
        super(UrlLoader, self).__init__(parent=parent)
        assert isinstance(url, DataUrl)
        self.url = url
        self.data = None

    def run(self):
        try:
            self.data = get_data(self.url)
        except IOError:
            self.data = None


class ImageStack(qt.QMainWindow):
    """Widget loading on the fly images contained the given urls.

    It prefetches images close to the displayed one.
    """

    N_PRELOAD = 10

    sigLoaded = qt.Signal(str)
    """Signal emitted when new data is available"""

    sigCurrentUrlChanged = qt.Signal(str)
    """Signal emitted when the current url change"""

    def __init__(self, parent=None) -> None:
        super(ImageStack, self).__init__(parent)
        self.__n_prefetch = ImageStack.N_PRELOAD
        self._loadingThreads = []
        self.setWindowFlags(qt.Qt.Widget)
        self._current_url = None
        self._url_loader = UrlLoader
        "class to instantiate for loading urls"

        # main widget
        self._plot = _PlotWithWaitingLabel(parent=self)
        self._plot.setAttribute(qt.Qt.WA_DeleteOnClose, True)
        self.setWindowTitle("Image stack")
        self.setCentralWidget(self._plot)

        # dock widget: url table
        self._tableDockWidget = qt.QDockWidget(parent=self)
        self._urlsTable = _ToggleableUrlSelectionTable(parent=self)
        self._tableDockWidget.setWidget(self._urlsTable)
        self._tableDockWidget.setFeatures(qt.QDockWidget.DockWidgetMovable)
        self.addDockWidget(qt.Qt.RightDockWidgetArea, self._tableDockWidget)
        # dock widget: qslider
        self._sliderDockWidget = qt.QDockWidget(parent=self)
        self._slider = _HorizontalSlider(parent=self)
        self._sliderDockWidget.setWidget(self._slider)
        self.addDockWidget(qt.Qt.BottomDockWidgetArea, self._sliderDockWidget)
        self._sliderDockWidget.setFeatures(qt.QDockWidget.DockWidgetMovable)

        self.reset()

        # connect signal / slot
        self._urlsTable.sigCurrentUrlChanged.connect(self.setCurrentUrl)
        self._slider.sigCurrentUrlIndexChanged.connect(self.setCurrentUrlIndex)

    def close(self) -> bool:
        self._freeLoadingThreads()
        self._plot.close()
        super(ImageStack, self).close()

    def setUrlLoaderClass(self, urlLoader: typing.Type[UrlLoader]) -> None:
        """

        :param urlLoader: define the class to call for loading urls.
                          warning: this should be a class object and not a
                          class instance.
        """
        assert isinstance(urlLoader, type(UrlLoader))
        self._url_loader = urlLoader

    def getUrlLoaderClass(self):
        """

        :return: class to instantiate for loading urls
        :rtype: typing.Type[UrlLoader]
        """
        return self._url_loader

    def _freeLoadingThreads(self):
        for thread in self._loadingThreads:
            thread.blockSignals(True)
            thread.wait(5)
        self._loadingThreads.clear()

    def getPlotWidget(self) -> Plot2D:
        """
        Returns the PlotWidget contained in this window

        :return: PlotWidget contained in this window
        :rtype: Plot2D
        """
        return self._plot.getPlotWidget()

    def reset(self) -> None:
        """Clear the plot and remove any link to url"""
        self._freeLoadingThreads()
        self._urls = None
        self._urlIndexes = None
        self._urlData = OrderedDict({})
        self._current_url = None
        self._plot.clear()
        self._urlsTable.clear()
        self._slider.setMaximum(-1)

    def _preFetch(self, urls: list) -> None:
        """Pre-fetch the given urls if necessary

        :param urls: list of DataUrl to prefetch
        :type: list
        """
        for url in urls:
            if url.path() not in self._urlData:
                self._load(url)

    def _load(self, url):
        """
        Launch background load of a DataUrl

        :param url:
        :type: DataUrl
        """
        assert isinstance(url, DataUrl)
        url_path = url.path()
        assert url_path in self._urlIndexes
        loader = self._url_loader(parent=self, url=url)
        loader.finished.connect(self._urlLoaded, qt.Qt.QueuedConnection)
        self._loadingThreads.append(loader)
        loader.start()

    def _urlLoaded(self) -> None:
        """

        :param url: restul of DataUrl.path() function
        :return:
        """
        sender = self.sender()
        assert isinstance(sender, UrlLoader)
        url = sender.url.path()
        if url in self._urlIndexes:
            self._urlData[url] = sender.data
            if self.getCurrentUrl().path() == url:
                self._plot.setData(self._urlData[url])
            if sender in self._loadingThreads:
                self._loadingThreads.remove(sender)
            self.sigLoaded.emit(url)

    def setNPrefetch(self, n: int) -> None:
        """
        Define the number of url to prefetch around

        :param int n: number of url to prefetch on left and right sides.
                      In total n*2 DataUrl will be prefetch
        """
        self.__n_prefetch = n
        current_url = self.getCurrentUrl()
        if current_url is not None:
            self.setCurrentUrl(current_url)

    def getNPrefetch(self) -> int:
        """

        :return: number of url to prefetch on left and right sides. In total
                 will load 2* NPrefetch DataUrls
        """
        return self.__n_prefetch

    def setUrls(self, urls: list) -> None:
        """list of urls within an index. Warning: urls should contain an image
        compatible with the silx.gui.plot.Plot class

        :param urls: urls we want to set in the stack. Key is the index
                     (position in the stack), value is the DataUrl
        :type: list
        """
        def createUrlIndexes():
            indexes = OrderedDict()
            for index, url in enumerate(urls):
                indexes[index] = url
            return indexes

        urls_with_indexes = createUrlIndexes()
        urlsToIndex = self._urlsToIndex(urls_with_indexes)
        self.reset()
        self._urls = urls_with_indexes
        self._urlIndexes = urlsToIndex

        old_url_table = self._urlsTable.blockSignals(True)
        self._urlsTable.setUrls(urls=list(self._urls.values()))
        self._urlsTable.blockSignals(old_url_table)

        old_slider = self._slider.blockSignals(True)
        self._slider.setMinimum(0)
        self._slider.setMaximum(len(self._urls) - 1)
        self._slider.blockSignals(old_slider)

        if self.getCurrentUrl() in self._urls:
            self.setCurrentUrl(self.getCurrentUrl())
        else:
            if len(self._urls.keys()) > 0:
                first_url = self._urls[list(self._urls.keys())[0]]
                self.setCurrentUrl(first_url)

    def getUrls(self) -> tuple:
        """

        :return: tuple of urls
        :rtype: tuple
        """
        return tuple(self._urlIndexes.keys())

    def _getNextUrl(self, url: DataUrl) -> typing.Union[None, DataUrl]:
        """
        return the next url in the stack

        :param url: url for which we want the next url
        :type: DataUrl
        :return: next url in the stack or None if `url` is the last one
        :rtype: Union[None, DataUrl]
        """
        assert isinstance(url, DataUrl)
        if self._urls is None:
            return None
        else:
            index = self._urlIndexes[url.path()]
            indexes = list(self._urls.keys())
            res = list(filter(lambda x: x > index, indexes))
            if len(res) == 0:
                return None
            else:
                return self._urls[res[0]]

    def _getPreviousUrl(self, url: DataUrl) -> typing.Union[None, DataUrl]:
        """
        return the previous url in the stack

        :param url: url for which we want the previous url
        :type: DataUrl
        :return: next url in the stack or None if `url` is the last one
        :rtype: Union[None, DataUrl]
        """
        if self._urls is None:
            return None
        else:
            index = self._urlIndexes[url.path()]
            indexes = list(self._urls.keys())
            res = list(filter(lambda x: x < index, indexes))
            if len(res) == 0:
                return None
            else:
                return self._urls[res[-1]]

    def _getNNextUrls(self, n: int, url: DataUrl) -> list:
        """
        Deduce the next urls in the stack after `url`

        :param n: the number of url store after `url`
        :type: int
        :param url: url for which we want n next url
        :type: DataUrl
        :return: list of next urls.
        :rtype: list
        """
        res = []
        next_free = self._getNextUrl(url=url)
        while len(res) < n and next_free is not None:
            assert isinstance(next_free, DataUrl)
            res.append(next_free)
            next_free = self._getNextUrl(res[-1])
        return res

    def _getNPreviousUrls(self, n: int, url: DataUrl):
        """
        Deduce the previous urls in the stack after `url`

        :param n: the number of url store after `url`
        :type: int
        :param url: url for which we want n previous url
        :type: DataUrl
        :return: list of previous urls.
        :rtype: list
        """
        res = []
        next_free = self._getPreviousUrl(url=url)
        while len(res) < n and next_free is not None:
            res.insert(0, next_free)
            next_free = self._getPreviousUrl(res[0])
        return res

    def setCurrentUrlIndex(self, index: int):
        """
        Define the url to be displayed

        :param index: url to be displayed
        :type: int
        """
        if index < 0:
            return
        if self._urls is None:
            return
        elif index >= len(self._urls):
            raise ValueError('requested index out of bounds')
        else:
            return self.setCurrentUrl(self._urls[index])

    def setCurrentUrl(self, url: typing.Union[DataUrl, str]) -> None:
        """
        Define the url to be displayed

        :param url: url to be displayed
        :type: DataUrl
        """
        assert isinstance(url, (DataUrl, str))
        if isinstance(url, str):
            url = DataUrl(path=url)
        if url != self._current_url:
            self._current_url = url
            self.sigCurrentUrlChanged.emit(url.path())

        old_url_table = self._urlsTable.blockSignals(True)
        old_slider = self._slider.blockSignals(True)

        self._urlsTable.setUrl(url)
        self._slider.setUrlIndex(self._urlIndexes[url.path()])
        if self._current_url is None:
            self._plot.clear()
        else:
            if self._current_url.path() in self._urlData:
                self._plot.setData(self._urlData[url.path()])
            else:
                self._load(url)
                self._notifyLoading()
            self._preFetch(self._getNNextUrls(self.__n_prefetch, url))
            self._preFetch(self._getNPreviousUrls(self.__n_prefetch, url))
        self._urlsTable.blockSignals(old_url_table)
        self._slider.blockSignals(old_slider)

    def getCurrentUrl(self) -> typing.Union[None, DataUrl]:
        """

        :return: url currently displayed
        :rtype: Union[None, DataUrl]
        """
        return self._current_url

    def getCurrentUrlIndex(self) -> typing.Union[None, int]:
        """

        :return: index of the url currently displayed
        :rtype: Union[None, int]
        """
        if self._current_url is None:
            return None
        else:
            return self._urlIndexes[self._current_url.path()]

    @staticmethod
    def _urlsToIndex(urls):
        """util, return a dictionary with url as key and index as value"""
        res = {}
        for index, url in urls.items():
            res[url.path()] = index
        return res

    def _notifyLoading(self):
        """display a simple image of loading..."""
        self._plot.setWaiting(activate=True)

    def setAutoResetZoom(self, reset):
        """
        Should we reset the zoom when adding an image (eq. when browsing)

        :param bool reset:
        """
        self._plot.setAutoResetZoom(reset)

    def isAutoResetZoom(self) -> bool:
        """

        :return: True if a reset is done when the image change
        :rtype: bool
        """
        return self._plot.isAutoResetZoom()
