# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017-2020 European Synchrotron Radiation Facility
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
"""This module provides a dialog widget to select plot items.

.. autoclass:: ItemsSelectionDialog

"""

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "28/06/2017"

import logging

from silx.gui import qt
from silx.gui.plot.PlotWidget import PlotWidget

_logger = logging.getLogger(__name__)


class KindsSelector(qt.QListWidget):
    """List widget allowing to select plot item kinds
    ("curve", "scatter", "image"...)
    """
    sigSelectedKindsChanged = qt.Signal(list)

    def __init__(self, parent=None, kinds=None):
        """

        :param parent: Parent QWidget or None
        :param tuple(str) kinds: Sequence of kinds. If None, the default
            behavior is to provide a checkbox for all possible item kinds.
        """
        qt.QListWidget.__init__(self, parent)

        self.plot_item_kinds = []

        self.setAvailableKinds(kinds if kinds is not None else PlotWidget.ITEM_KINDS)

        self.setSelectionMode(qt.QAbstractItemView.ExtendedSelection)
        self.selectAll()

        self.itemSelectionChanged.connect(self.emitSigKindsSelectionChanged)

    def emitSigKindsSelectionChanged(self):
        self.sigSelectedKindsChanged.emit(self.selectedKinds)

    @property
    def selectedKinds(self):
        """Tuple of all selected kinds (as strings)."""
        # check for updates when self.itemSelectionChanged
        return [item.text() for item in self.selectedItems()]

    def setAvailableKinds(self, kinds):
        """Set a list of kinds to be displayed.

        :param list[str] kinds: Sequence of kinds
        """
        self.plot_item_kinds = kinds

        self.clear()
        for kind in self.plot_item_kinds:
            item = qt.QListWidgetItem(self)
            item.setText(kind)
            self.addItem(item)

    def selectAll(self):
        """Select all available kinds."""
        if self.selectionMode() in [qt.QAbstractItemView.SingleSelection,
                                    qt.QAbstractItemView.NoSelection]:
            raise RuntimeError("selectAll requires a multiple selection mode")
        for i in range(self.count()):
            self.item(i).setSelected(True)


class PlotItemsSelector(qt.QTableWidget):
    """Table widget displaying the legend and kind of all
    plot items corresponding to a list of specified kinds.

    Selected plot items are provided as property :attr:`selectedPlotItems`.
    You can be warned of selection changes by listening to signal
    :attr:`itemSelectionChanged`.
    """
    def __init__(self, parent=None, plot=None):
        if plot is None or not isinstance(plot, PlotWidget):
            raise AttributeError("parameter plot is required")
        self.plot = plot
        """:class:`PlotWidget` instance"""

        self.plot_item_kinds = None
        """List of plot item kinds (strings)"""

        qt.QTableWidget.__init__(self, parent)

        self.setColumnCount(2)

        self.setSelectionBehavior(qt.QTableWidget.SelectRows)

    def _clear(self):
        self.clear()
        self.setHorizontalHeaderLabels(["legend", "type"])

    def setAllKindsFilter(self):
        """Display all kinds of plot items."""
        self.setKindsFilter(PlotWidget.ITEM_KINDS)

    def setKindsFilter(self, kinds):
        """Set list of all kinds of plot items to be displayed.

        :param list[str] kinds: Sequence of kinds
        """
        if not set(kinds) <= set(PlotWidget.ITEM_KINDS):
            raise KeyError("Illegal plot item kinds: %s" %
                           set(kinds) - set(PlotWidget.ITEM_KINDS))
        self.plot_item_kinds = kinds

        self.updatePlotItems()

    def updatePlotItems(self):
        self._clear()

        # respect order of kinds as set in method setKindsFilter
        itemsAndKind = []
        for kind in self.plot_item_kinds:
            itemClasses = self.plot._KIND_TO_CLASSES[kind]
            for item in self.plot.getItems():
                if isinstance(item, itemClasses) and item.isVisible():
                    itemsAndKind.append((item, kind))

        self.setRowCount(len(itemsAndKind))

        for index, (item, kind) in enumerate(itemsAndKind):
            legend_twitem = qt.QTableWidgetItem(item.getName())
            self.setItem(index, 0, legend_twitem)

            kind_twitem = qt.QTableWidgetItem(kind)
            self.setItem(index, 1, kind_twitem)

    @property
    def selectedPlotItems(self):
        """List of all selected items"""
        selection_model = self.selectionModel()
        selected_rows_idx = selection_model.selectedRows()
        selected_rows = [idx.row() for idx in selected_rows_idx]

        items = []
        for row in selected_rows:
            legend = self.item(row, 0).text()
            kind = self.item(row, 1).text()
            item = self.plot._getItem(kind, legend)
            if item is not None:
                items.append(item)

        return items


class ItemsSelectionDialog(qt.QDialog):
    """This widget is a modal dialog allowing to select one or more plot
    items, in a table displaying their legend and kind.

    Public methods:

      - :meth:`getSelectedItems`
      - :meth:`setAvailableKinds`
      - :meth:`setItemsSelectionMode`

    This widget inherits QDialog and therefore implements the usual
    dialog methods, e.g. :meth:`exec_`.

    A trivial usage example would be::

        isd = ItemsSelectionDialog(plot=my_plot_widget)
        isd.setItemsSelectionMode(qt.QTableWidget.SingleSelection)
        result = isd.exec_()
        if result:
            for item in isd.getSelectedItems():
                print(item.getName(), type(item))
        else:
            print("Selection cancelled")
    """
    def __init__(self, parent=None, plot=None):
        if plot is None or not isinstance(plot, PlotWidget):
            raise AttributeError("parameter plot is required")
        qt.QDialog.__init__(self, parent)

        self.setWindowTitle("Plot items selector")

        kind_selector_label = qt.QLabel("Filter item kinds:", self)
        item_selector_label = qt.QLabel("Select items:", self)

        self.kind_selector = KindsSelector(self)
        self.kind_selector.setToolTip(
                "select one or more item kinds to show them in the item list")

        self.item_selector = PlotItemsSelector(self, plot)
        self.item_selector.setToolTip("select items")

        self.item_selector.setKindsFilter(self.kind_selector.selectedKinds)
        self.kind_selector.sigSelectedKindsChanged.connect(
            self.item_selector.setKindsFilter
        )

        okb = qt.QPushButton("OK", self)
        okb.clicked.connect(self.accept)

        cancelb = qt.QPushButton("Cancel", self)
        cancelb.clicked.connect(self.reject)

        layout = qt.QGridLayout(self)
        layout.addWidget(kind_selector_label, 0, 0)
        layout.addWidget(item_selector_label, 0, 1)
        layout.addWidget(self.kind_selector, 1, 0)
        layout.addWidget(self.item_selector, 1, 1)
        layout.addWidget(okb, 2, 0)
        layout.addWidget(cancelb, 2, 1)

        self.setLayout(layout)

    def getSelectedItems(self):
        """Return a list of selected plot items

        :return: List of selected plot items
        :rtype: list[silx.gui.plot.items.Item]"""
        return self.item_selector.selectedPlotItems

    def setAvailableKinds(self, kinds):
        """Set a list of kinds to be displayed.

        :param list[str] kinds: Sequence of kinds
        """
        self.kind_selector.setAvailableKinds(kinds)

    def selectAllKinds(self):
        self.kind_selector.selectAll()

    def setItemsSelectionMode(self, mode):
        """Set selection mode for plot item (single item selection,
        multiple...).

        :param mode: One of :class:`QTableWidget` selection modes
        """
        if mode == self.item_selector.SingleSelection:
            self.item_selector.setToolTip(
                    "Select one item by clicking on it.")
        elif mode == self.item_selector.MultiSelection:
            self.item_selector.setToolTip(
                    "Select one or more items by clicking with the left mouse"
                    " button.\nYou can unselect items by clicking them again.\n"
                    "Multiple items can be toggled by dragging the mouse over them.")
        elif mode == self.item_selector.ExtendedSelection:
            self.item_selector.setToolTip(
                    "Select one or more items. You can select multiple items "
                    "by keeping the Ctrl key pushed when clicking.\nYou can "
                    "select a range of items by clicking on the first and "
                    "last while keeping the Shift key pushed.")
        elif mode == self.item_selector.ContiguousSelection:
            self.item_selector.setToolTip(
                    "Select one item by clicking on it. If you press the Shift"
                    " key while clicking on a second item,\nall items between "
                    "the two will be selected.")
        elif mode == self.item_selector.NoSelection:
            raise ValueError("The NoSelection mode is not allowed "
                             "in this context.")
        self.item_selector.setSelectionMode(mode)
