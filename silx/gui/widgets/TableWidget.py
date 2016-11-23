# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2004-2016 European Synchrotron Radiation Facility
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
"""This module provides a QTableWidget handling cut, copy and paste for
multiple cell selections.
"""

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "23/11/2016"


import sys
from .. import qt


if sys.platform.startswith("win"):
    newline = "\r\n"
else:
    newline = "\n"


class CopySelectedCellsAction(qt.QAction):
    """QAction to copy text from selected cells in a :class:`QTableWidget`
    into the clipboard.

    If multiple cells are selected, the copied text will be a concatenation
    of the texts in all selected cells, tabulated with tabulation and
    newline characters.

    If the cells are sparsely selected, the structure is preserved by
    representing the unselected cells as empty strings in between two
    tabulation characters.
    Beware of pasting this data in another table widget, because depending
    on how the paste is implemented, the empty cells may cause data in the
    target table to be deleted, even though you didn't necessarily select the
    corresponding cell in the origin table.

    :param table: :class:`QTableWidget` to which this action belongs.
    """
    def __init__(self, table):
        if not isinstance(table, qt.QTableWidget):
            raise ValueError('CopySelectedCellsAction must be initialised ' +
                             'with a QTableWidget.')
        super(CopySelectedCellsAction, self).__init__(table)
        self.setText("Copy selection")
        self.setShortcut(qt.QKeySequence('Ctrl+C'))
        self.triggered.connect(self.copyCellsToClipboard)
        self.table = table
        self.cut = False

    def copyCellsToClipboard(self):
        """Concatenate the text content of all selected cells into a string
        using tabulations and newlines to keep the table structure.
        Put this text into the clipboard.
        """
        selected_idx = self.table.selectedIndexes()
        selected_idx_tuples = [(idx.row(), idx.column()) for idx in selected_idx]

        selected_rows = [idx[0] for idx in selected_idx_tuples]
        selected_columns = [idx[1] for idx in selected_idx_tuples]

        copied_text = ""
        for row in range(min(selected_rows), max(selected_rows) + 1):
            for col in range(min(selected_columns), max(selected_columns) + 1):
                item = self.table.item(row, col)
                if (row, col) in selected_idx_tuples and item is not None:
                    copied_text += item.text()
                    if self.cut and (item.flags() & qt.Qt.ItemIsEditable):
                        item.setText("")
                copied_text += "\t"
            # remove the right-most tabulation
            copied_text.rstrip("\t")
            # add a newline
            copied_text += newline
        # remove final newline
        copied_text.rstrip(newline)

        # put this text into clipboard
        qapp = qt.QApplication.instance()
        qapp.clipboard().setText(copied_text)


class CopyAllCellsAction(qt.QAction):
    """QAction to copy text from all cells in a :class:`QTableWidget`
    into the clipboard.

    The copied text will be a concatenation
    of the texts in all cells, tabulated with tabulation and
    newline characters.

    :param table: :class:`QTableWidget` to which this action belongs.
    """
    def __init__(self, table):
        if not isinstance(table, qt.QTableWidget):
            raise ValueError('CopyAllCellsAction must be initialised ' +
                             'with a QTableWidget.')
        super(CopyAllCellsAction, self).__init__(table)
        self.setText("Copy all")
        self.triggered.connect(self.copyCellsToClipboard)
        self.table = table
        # self.cut = False

    def copyCellsToClipboard(self):
        """Concatenate the text content of all cells into a string
        using tabulations and newlines to keep the table structure.
        Put this text into the clipboard.
        """
        copied_text = ""
        for row in range(self.table.rowCount()):
            for col in range(self.table.columnCount()):
                item = self.table.item(row, col)
                if item is not None:
                    copied_text += item.text()
                    # if self.cut and (item.flags() & qt.Qt.ItemIsEditable):
                    #     item.setText("")
                copied_text += "\t"
            # remove the right-most tabulation
            copied_text.rstrip("\t")
            # add a newline
            copied_text += newline
        # remove final newline
        copied_text.rstrip(newline)

        # put this text into clipboard
        qapp = qt.QApplication.instance()
        qapp.clipboard().setText(copied_text)


class CutSelectedCellsAction(CopySelectedCellsAction):
    """QAction to cut text from selected cells in a :class:`QTableWidget`
    into the clipboard.

    The text is deleted from the original table widget
    (use :class:`CopySelectedCellsAction` to preserve the original data).

    If multiple cells are selected, the cut text will be a concatenation
    of the texts in all selected cells, tabulated with tabulation and
    newline characters.

    If the cells are sparsely selected, the structure is preserved by
    representing the unselected cells as empty strings in between two
    tabulation characters.
    Beware of pasting this data in another table widget, because depending
    on how the paste is implemented, the empty cells may cause data in the
    target table to be deleted, even though you didn't necessarily select the
    corresponding cell in the origin table.

    :param table: :class:`QTableWidget` to which this action belongs."""
    def __init__(self, table):
        super(CutSelectedCellsAction, self).__init__(table)
        self.setText("Cut")
        self.setShortcut(qt.QKeySequence('Ctrl+X'))
        # cutting is already implemented in CopySelectedCellsAction (but
        # it is disabled), we just need to enable it
        self.cut = True


def _parseTextAsTable(text, record_separator=newline, field_separator="\t"):
    """Parse text into list of lists (2D sequence).

    The input text must be tabulated using tabulation characters and
    newlines to separate columns and rows.

    :param text: text to be parsed
    :param record_separator: String, or single character, to be interpreted
        as a record/row separator.
    :param field_separator: String, or single character, to be interpreted
        as a field/column separator.
    :return: 2D sequence of strings
    """
    rows = text.split(newline)
    table = [row.split("\t") for row in rows]
    return table


class PasteCellsAction(qt.QAction):
    """QAction to paste text from the clipboard into the table.

    If the text contains tabulations and
    newlines, they are interpreted as column and row separators.
    In such a case, the text is split into multiple texts to be pasted
    into multiple cells.

    If a cell content is an empty string in the original text, it is
    ignored: the destination cell's text will not be deleted.

    :param table: :class:`QTableWidget` to which this action belongs.
    """
    def __init__(self, table):
        if not isinstance(table, qt.QTableWidget):
            raise ValueError('PasteCellsAction must be initialised ' +
                             'with a QTableWidget.')
        super(PasteCellsAction, self).__init__(table)
        self.table = table
        self.setText("Paste")
        self.setShortcut(qt.QKeySequence('Ctrl+V'))
        self.triggered.connect(self.pasteCellFromClipboard)

    def pasteCellFromClipboard(self):
        """Paste text from clipboard into the table.

        :return: *True* in case of success, *False* if pasting data failed.
        """
        selected_idx = self.table.selectedIndexes()
        if len(selected_idx) != 1:
            msgBox = qt.QMessageBox(parent=self.table)
            msgBox.setText("A single cell must be selected to paste data")
            msgBox.exec_()
            return False

        selected_row = selected_idx[0].row()
        selected_col = selected_idx[0].column()

        qapp = qt.QApplication.instance()
        clipboard_text = qapp.clipboard().text()
        table_data = _parseTextAsTable(clipboard_text)

        protected_cells = 0
        out_of_range_cells = 0

        # paste table data into cells, using selected cell as origin
        for row_offset in range(len(table_data)):
            for col_offset in range(len(table_data[row_offset])):
                target_row = selected_row + row_offset
                target_col = selected_col + col_offset

                if target_row >= self.table.rowCount() or\
                   target_col >= self.table.columnCount():
                    out_of_range_cells += 1
                    continue

                item = self.table.item(target_row,
                                       target_col)
                # item may not exist for empty cells
                if item is None:
                    item = qt.QTableWidgetItem()
                    self.table.setItem(target_row,
                                       target_col,
                                       item)
                # ignore empty strings
                if table_data[row_offset][col_offset] != "":
                    if not item.flags() & qt.Qt.ItemIsEditable:
                        protected_cells += 1
                        continue
                    item.setText(table_data[row_offset][col_offset])

        if protected_cells or out_of_range_cells:
            msgBox = qt.QMessageBox(parent=self.table)
            msg = "Some data could not be inserted, "
            msg += "due to out-of-range or write-protected cells."
            msgBox.setText(msg)
            msgBox.exec_()
            return False
        return True


class TableWidget(qt.QTableWidget):
    """:class:`QTableWidget` with a context menu displaying 4 actions:

        - :class:`CopySelectedCellsAction`
        - :class:`CopyAllCellsAction`
        - :class:`CutSelectedCellsAction`
        - :class:`PasteCellsAction`

    These actions interact with the clipboard and can be used to copy data
    to or from an external application, or another widget.

    The cut and paste actions are disabled by default, due to the risk of
    overwriting data (no *Undo* action is available). Use :meth:`enablePaste`
    and :meth:`enableCut` to activate them.

    :param parent: Parent QWidget
    :param bool cut: Enable cut action
    :param bool paste: Enable paste action
    """
    def __init__(self, parent=None, cut=False, paste=False):
        super(TableWidget, self).__init__(parent)
        self.addAction(CopySelectedCellsAction(self))
        self.addAction(CopyAllCellsAction(self))
        if cut:
            self.enableCut()
        if paste:
            self.enablePaste()

        self.setContextMenuPolicy(qt.Qt.ActionsContextMenu)

    def enablePaste(self):
        """Enable paste action, to paste data from the clipboard into the
        table.

        This action can be triggered through the context menu (right-click)
        or through the *Ctrl+V* key sequence."""
        self.addAction(PasteCellsAction(self))

    def enableCut(self):
        """Enable cut action.

        This action can be triggered through the context menu (right-click)
        or through the *Ctrl+X* key sequence."""
        self.addAction(CutSelectedCellsAction(self))


if __name__ == "__main__":
    app = qt.QApplication([])

    table = TableWidget()
    table.setColumnCount(10)
    table.setRowCount(7)
    table.enablePaste()
    table.enableCut()
    table.show()
    app.exec_()

