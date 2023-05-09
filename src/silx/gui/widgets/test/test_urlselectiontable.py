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
"""Tests for UrlSelectionTable"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "09/05/2023"

import pytest
import weakref
from silx.gui.widgets.UrlSelectionTable import UrlSelectionTable
from silx.gui import qt
from silx.io.url import DataUrl


@pytest.fixture
def urlSelectionTable(qapp, qapp_utils):
    widget = UrlSelectionTable()
    widget.setAttribute(qt.Qt.WA_DeleteOnClose)
    yield widget
    widget.close()
    ref = weakref.ref(widget)
    widget = None
    qapp_utils.qWaitForDestroy(ref)


def test_show(qapp_utils, urlSelectionTable):
    qapp_utils.qWaitForWindowExposed(urlSelectionTable)


def test_add_urls(urlSelectionTable):
    urlSelectionTable.addUrl(DataUrl("aaaa"))
    urlSelectionTable.addUrl(DataUrl("bbbb"))
    urlSelectionTable.addUrl(DataUrl("cccc"))
    assert urlSelectionTable.rowCount() == 3


def test_clear(urlSelectionTable):
    urlSelectionTable.addUrl(DataUrl("aaaa"))
    assert urlSelectionTable.rowCount() == 1
    urlSelectionTable.clear()
    assert urlSelectionTable.rowCount() == 0


def test_set_remove_error(urlSelectionTable):
    urlSelectionTable.addUrl(DataUrl("aaaa"))
    item = urlSelectionTable._getItemFromUrlPath("aaaa")
    urlSelectionTable.setError("aaaa", "Oh... no...")
    assert not item.icon().isNull()
    urlSelectionTable.setError("aaaa", "")
    assert item.icon().isNull()
