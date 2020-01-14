# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2018 European Synchrotron Radiation Facility
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
"""Image stack view with data prefetch
"""

__authors__ = ["H. Payno"]
__license__ = "MIT"
__date__ = "04/03/2019"


from silx.gui import icons, qt
from silx.gui.plot import Plot2D
from silx.gui.widgets.UrlSelectionTable import UrlSelectionTable
from silx.io.url import DataUrl
from silx.io.utils import get_data
import typing
import functools


class _PlotWithWaitingLabel(qt.QWidget):
    """A simple widget that can either display an image or display a
    'processing' or 'waiting' status"""
    def __init__(self, parent):
        super(_PlotWithWaitingLabel, self).__init__(parent=parent)
        self._waiting_label = qt.QLabel(parent=self)
        self.animated_icon = icons.getWaitIcon()
        self.animated_icon.register(self._waiting_label)

        self._plot = Plot2D(parent=self)

    def setWaiting(self, activate=True):
        if activate is True:
            self._plot.hide()
            self._waiting_label.show()
        else:
            self._plot.show()
            self._waiting_label.hide()

    def setImage(self):
        self.setWaiting(activate=False)

    def clear(self):
        self._plot.clear()
        self.setWaiting(False)

    def getPlot(self):
        return self._plot


class ImageStack(qt.QWidget):
    """
    This widget is made to load on the fly image contained the given urls.
    For avoiding lack impression it will prefetch images close to the one
    displayed.
    """

    N_PRELOAD = 10
    """Num"""

    _BUTTON_ICON = qt.QStyle.SP_ToolBarHorizontalExtensionButton  # noqa

    def __init__(self, parent=None) -> None:
        qt.QWidget.__init__(self, parent)

        self.setLayout(qt.QGridLayout())
        self._plot = _PlotWithWaitingLabel(parent=self)
        self.layout().addWidget(self._plot, 0, 0, 2, 1)
        self._toggleButton = qt.QPushButton(parent=self)
        self.layout().addWidget(self._toggleButton, 0, 2, 1, 1)
        self._toggleButton.setSizePolicy(qt.QSizePolicy.Fixed,
                                         qt.QSizePolicy.Fixed)

        self._urlTable = UrlSelectionTable(parent=self)
        self.layout().addWidget(self._urlTable, 1, 1, 1, 2)

        # set up
        self._setButtonIcon(show=True)

        # Signal / slot connection
        self._toggleButton.clicked.connect(self.toggleUrlSelectionTable)

        self.reset()

    def getPlot(self) -> Plot2D:
        """
        Returns the PlotWidget contains in this window

        :return: PlotWidget contains in this window
        :rtype: Plot2D
        """
        return self._plot.getPlot()

    def reset(self) -> None:
        """Clear the plot and remove any link to url"""
        self._urls = None
        self._urlIndexes = None
        self._urlData = {}
        self._current_url = None
        self._plot.clear()
        self.__n_prefetch = ImageStack.N_PRELOAD

    def _preFetch(self, urls: list) -> None:
        """
        Pre-fetch the given urls if necessary
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
        loader = self.getUrlLoader(url_path)
        loader.finished.connect(functools.partial(self._urlLoaded, url_path))
        loader.start()

    def _urlLoaded(self, url: DataUrl) -> None:
        if url in self._urls:
            sender = self.sender()
            assert isinstance(sender, UrlLoader)
            self._urlData[url] = sender.data
            if self.current_url() == url:
                self._plot.setImage(self._urlData[url])

    def setNPrefetch(self, n: int) -> None:
        """
        Define the number of url to prefetch around

        :param int n: number of url to prefetch on the and on the right.
                      In total n*2 DataUrl will be prefetch
        """
        self.__n_prefetch = n
        self.set_current_url(self.getCurrentUrl())

    def setUrls(self, urls: dict) -> None:
        """list of urls within an index. Warning: urls should contain an image
        compatible with the silx.gui.plot.Plot class

        :param urls: urls we want to set in the stack. Key is the index
                     (position in the stack), value is the DataUrl
        :type: dict
        """
        urlsToIndex = self._urlsToIndex(urls)
        if not len(urlsToIndex) == len(urls):
            raise ValueError('each url should be unique')
        self.reset()
        self._urls = urls
        self._urlIndexes = urlsToIndex
        if self.getCurrentUrl() in self._urls:
            self.setCurrentUrl(self.getCurrentUrl())
        else:
            first_url = self._urls[list(self._urls.keys())[0]]
            self.setCurrentUrl(first_url)

    def getNextUrl(self, url: DataUrl) -> typing.Union[None, DataUrl]:
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
            print(index, type(index))
            print(indexes, type(indexes))
            res = list(filter(lambda x: x > index, indexes))
            if len(res) == 0:
                return None
            else:
                return self._urls[res[0]]

    def getPreviousUrl(self, url: DataUrl) -> typing.Union[None, DataUrl]:
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
            index = self._urlIndexes[url]
            res = numpy.where(list(self._urlIndexes.keys())<index)
            return self._urls[res[0]]

    def getNNextUrls(self, n: int, url: DataUrl) -> list:
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
        next_free = self.getNextUrl(url=url)
        while len(res) < n and next_free is not None:
            assert isinstance(next_free, DataUrl)
            res.append(next_free)
            next_free = self.getNextUrl(res[-1])
        return res

    def getNPreviousUrls(self, n: int, url: DataUrl):
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
        next_free = self.getPreviousUrl(url=url)
        while len(res) < n and next_free is not None:
            res.append(next_free)
            next_free = self.getNextUrl(res[-1])
        return res

    def setCurrentUrl(self, url: DataUrl) -> None:
        """
        Define the url to be displayed

        :param url: url to be displayed
        :rtype: DataUrl
        """
        assert isinstance(url, DataUrl)
        self._current_url = url
        if self._current_url is None:
            self._plot.clear()
        else:
            if self._current_url.path() in self._urlData:
                self.plot.add_image(self._urlData[url])
            else:
                self._load(url)
                self._notifyLoading()
            self._preFetch(self.getNNextUrls(self.__n_prefetch, url))
            self._preFetch(self.getNNextUrls(self.__n_prefetch, url))

    def getCurrentUrl(self) -> typing.Union[None, DataUrl]:
        """

        :return: url currently displayed
        :rtype: Union[None, DataUrl]
        """
        return self._current_url

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

    def toggleUrlSelectionTable(self):
        visible = not self.urlSelectionTableIsVisible()
        self._setButtonIcon(show=visible)
        self._urlTable.setVisible(visible)

    def _setButtonIcon(self, show):
        style = qt.QApplication.instance().style()
        # return a QIcon
        icon = style.standardIcon(self._BUTTON_ICON)
        if show is False:
            pixmap = icon.pixmap(32, 32).transformed(qt.QTransform().scale(-1, 1))
            icon = qt.QIcon(pixmap)
        self._toggleButton.setIcon(icon)

    def urlSelectionTableIsVisible(self):
        return self._urlTable.isVisible()

    def getUrlLoader(self, url):
        """
        Might be overwrite if some children class want to redefine ways /
        scheme for loading data
        :return: UrlLoader
        """
        return UrlLoader(parent=self, url=url)


class UrlLoader(qt.QThread):
    """
    Thread use to load DataUrl
    """
    def __init__(self, parent, url):
        super(UrlLoader, self).__init__(parent=parent)
        self.url = url
        self.data = None

    def run(self):
        try:
            self.data = get_data(self.url)
        except IOError:
            self.data = icons.getIcon('data-not-loaded')


if __name__ == '__main__':
    import numpy
    import h5py
    import tempfile

    def create_urls():
        res = {}
        tmp = tempfile.NamedTemporaryFile(prefix="test_image_stack_",
                                          suffix=".h5",
                                          delete=True)
        with h5py.File(tmp.file, 'w') as h5f:
            for i in range(10):
                width = numpy.random.randint(100, 400)
                height = numpy.random.randint(100, 400)
                h5f[str(i)] = numpy.random.random((width, height))
                res[i] = DataUrl(file_path=tmp.name,
                                 data_path=str(i),
                                 # data_slice=(0,),
                                 scheme='silx')
        return res, tmp

    qapp = qt.QApplication([])
    widget = ImageStack()
    urls, file_ = create_urls()
    widget.setUrls(urls=urls)
    widget.show()
    qapp.exec_()
