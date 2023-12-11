# /*##########################################################################
#
# Copyright (c) 2023 European Synchrotron Radiation Facility
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

from __future__ import annotations

import typing
import logging
from collections.abc import Iterable
from silx.io.url import DataUrl
from silx.gui import qt
from silx.utils.deprecation import deprecated

_logger = logging.getLogger(__name__)


class UrlList(qt.QListWidget):
    """List of URLs with user selection"""

    sigCurrentUrlChanged = qt.Signal(str)
    """Signal emitted when the active/current URL has changed.

    This signal emits the empty string when there is no longer an active URL.
    """

    sigUrlRemoved = qt.Signal(str)
    """Signal emit when an url is removed from the URL list.

    Provides the url (DataUrl) as a string
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._editable = False
        # are we in 'editable' mode: for now if true then we can remove some items from the list

        # menu to be triggered when in edition from right-click
        self._menu = qt.QMenu()
        self._removeAction = qt.QAction(text="Remove", parent=self)
        self._removeAction.setShortcuts(
            [
                # qt.Qt.Key_Delete,
                qt.QKeySequence.Delete,
            ]
        )
        self._menu.addAction(self._removeAction)

        # connect signal / Slot
        self.currentItemChanged.connect(self._notifyCurrentUrlChanged)

    def setEditable(self, editable: bool):
        """Toggle whether the user can remove some URLs from the list"""
        if editable != self._editable:
            self._editable = editable
            # discusable choice: should we change the selection mode ? No much meaning
            # to be in ExtendedSelection if we are not in editable mode. But does it has more
            # meaning to change the selection mode ?
            if editable:
                self._removeAction.triggered.connect(self._removeSelectedItems)
                self.addAction(self._removeAction)
            else:
                self._removeAction.triggered.disconnect(self._removeSelectedItems)
                self.removeAction(self._removeAction)

    @deprecated(replacement="addUrls", since_version="2.0")
    def setUrls(self, urls: Iterable[DataUrl]) -> None:
        self.addUrls(urls)

    def addUrls(self, urls: Iterable[DataUrl]) -> None:
        """Append multiple DataUrl to the list"""
        self.addItems([url.path() for url in urls])

    def removeUrl(self, url: str):
        """Remove given URL from the list"""
        sel_items = self.findItems(url, qt.Qt.MatchExactly)
        if len(sel_items) > 0:
            assert len(sel_items) == 0, "at most one item expected"
            self.removeItemWidget(sel_items[0])

    def _notifyCurrentUrlChanged(self, current, previous):
        if current is None:
            self.sigCurrentUrlChanged.emit("")
        else:
            self.sigCurrentUrlChanged.emit(current.text())

    def setUrl(self, url: typing.Optional[DataUrl]) -> None:
        """Set the current URL.

        :param url: The new selected URL. Use `None` to clear the selection.
        """
        if url is None:
            self.clearSelection()
            self.sigCurrentUrlChanged.emit("")
        else:
            assert isinstance(url, DataUrl)
            sel_items = self.findItems(url.path(), qt.Qt.MatchExactly)
            if sel_items is None:
                _logger.warning(url.path(), " is not registered in the list.")
            elif len(sel_items) > 0:
                item = sel_items[0]
                self.setCurrentItem(item)
                self.sigCurrentUrlChanged.emit(item.text())

    def _removeSelectedItems(self):
        if not self._editable:
            raise ValueError("UrlList is not set as 'editable'")
        urls = []
        for item in self.selectedItems():
            url = item.text()
            self.takeItem(self.row(item))
            urls.append(url)
        # as the connected slot of 'sigUrlRemoved' can modify the items, better handling all at the end
        for url in urls:
            self.sigUrlRemoved.emit(url)

    def contextMenuEvent(self, event):
        if self._editable:
            globalPos = self.mapToGlobal(event.pos())
            self._menu.exec_(globalPos)
