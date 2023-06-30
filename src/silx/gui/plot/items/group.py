# /*##########################################################################
#
# Copyright (c) 2018-2023 European Synchrotron Radiation Facility
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
"""This module provides base components to create group item for
the :class:`~silx.gui.plot.PlotWidget`.
"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "30/06/2023"


import logging
import weakref
from typing import List

from ... import qt
from .. import items


logger = logging.getLogger(__name__)


class Group(items.Item):
    """Object describing a group of items.

    A group is only a item which hold a set of items.

    Each child are also part of the plot as flat items.
    """

    sigChildAdded = qt.Signal(object)
    """Signal emitted when a child is added to this group"""

    sigChildRemoved = qt.Signal(object)
    """Signal emitted when a child is added to this group"""

    def __init__(self):
        items.Item.__init__(self)
        self._child = []
        self._childVisibility = weakref.WeakKeyDictionary()

    def _setItemName(self, item):
        """Helper to generate a unique id to a plot item"""
        legend = "__CHILD-%d__%d" % (id(self), id(item))
        item.setName(legend)

    def _setPlot(self, plot):
        """Override `_setPlot` in order to update this children's group."""
        previousPlot = self.getPlot()
        if plot is previousPlot:
            return
        if plot is None:
            for c in self._child:
                previousPlot.removeItem(c)
        else:
            for c in self._child:
                plot.addItem(c)
        items.Item._setPlot(self, plot)

    def addItem(self, item: items.Item):
        """Add an item to this group.

        This item will be added and removed to the plot used by the group.

        If the group is already part of a plot, the item will also be added to
        the plot.

        It the item do not have a name already, a unique one is generated to
        avoid item collision in the plot.

        :param item: A plot item
        """
        assert item is not None
        self._child.append(item)
        self.sigChildAdded.emit(item)
        if item.getName() == '':
            self._setItemName(item)
        itemVisible = item.isVisible()
        if not itemVisible:
            self._childVisibility[item] = False
        plot = self.getPlot()
        if plot is not None:
            if not self.isVisible():
                item.setVisible(False)
            item._parentGroup = True  # FIXME: This can be a normal reference from the API
            plot.addItem(item)

    def removeItem(self, item: items.Item):
        """Remove an item from this group.

        If the item is part of a plot it will be removed too.

        :param item: A plot item
        """
        assert item is not None
        self._child.remove(item)
        self.sigChildRemoved.emit(item)
        plot = item.getPlot()
        if plot is not None:
            del item._parentGroup
            plot.removeItem(item)

    def getItems(self) -> List[items.Item]:
        """Returns the list of PlotWidget items from this group.
        """
        return tuple(self._child)

    def setItemVisible(self, child: items.Item, visible: bool):
        """Set the visibility of this child when this group is set visible.

        :param visible: True to display it, False otherwise
        """
        self._childVisibility[child] = visible
        if self.isVisible():
            child.setVisible(visible)

    def isItemVisible(self, child: items.Item):
        """Define if this child is visible from the group."""
        return self._childVisibility.get(child, True)

    def setVisible(self, visible: bool):
        """Set visibility of item.

        This also affect the child visibility from the description set by
        :meth:`setItemVisible`.

        :param visible: True to display it, False otherwise
        """
        if self.isVisible() == visible:
            return
        for c in self._child:
            if visible:
                c.setVisible(self.isItemVisible(c))
            else:
                c.setVisible(False)
        items.Item.setVisible(self, visible)
