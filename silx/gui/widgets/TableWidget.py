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
__date__ = "07/11/2016"


import sys
from .. import qt


if sys.platform.startswith("win"):
    newline = "\r\n"
else:
    newline = "\n"


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


class TableWidget(qt.QTableWidget):
    def __init__(self, parent=None):
        super(self, TableWidget).__init__(self, parent)

    def _copySelection(self, cut=False):
        """Concatenate the text content of all selected cells into a string
        using tabulations and newlines to keep the table structure.
        Put this text into the clipboard.

        :param boolean cut: If *True*, empty the cells after retrieving the
            text (cut mode). Default is *False* (copy mode)"""
        selected_idx = self.selectedIndexes()
        selected_idx_tuples = [(idx.row(), idx.column()) for idx in selected_idx]

        selected_rows = [idx[0] for idx in selected_idx_tuples]
        selected_columns = [idx[1] for idx in selected_idx_tuples]

        copied_text = ""
        for row in range(min(selected_rows), max(selected_rows) + 1):
            for col in range(min(selected_columns), max(selected_columns) + 1):
                item = self.item(row, col)
                if (row, col) in selected_idx_tuples:
                    copied_text += item.text()
                    if cut and (item.flags & qt.Qt.ItemIsEditable):
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

    def _pasteText(self):
        """Paste text from cipboard into the table.

        If the text contains tabulations and
        newlines, they are interpreted as column and row separators.
        In such a case, the text is split into multiple texts to be paste
        into multiple cells.

        :return: *True* in case of success, *False* if pasting data failed.
        """
        selected_idx = self.selectedIndexes()
        if len(selected_idx) != 1:
            msgBox = qt.QMessageBox(parent=self)
            msgBox.setText("A single cell must be selected to paste data")
            msgBox.exec_()
            return False

        selected_row = selected_idx[0].row()
        selected_col = selected_idx[0].column()

        qapp = qt.QApplication.instance()
        clipboard_text = qapp.clipboard().text()
        table = _parseTextAsTable(clipboard_text)

        protected_cells = 0
        out_of_range_cells = 0

        # paste table data into cells, using selected cell as origin
        for row in range(len(table)):
            for col in range(len(table[row])):
                if selected_row + row >= self.rowCount() or selected_col + col >= self.columnCount():
                    out_of_range_cells += 1
                    continue
                item = self.item(selected_row + row,
                                 selected_col + col)
                # ignore empty strings
                if table[row][col] != "":
                    if not item.flags & qt.Qt.ItemIsEditable:
                        protected_cells += 1
                        continue
                    item.setText(table[row][col])

        if protected_cells or out_of_range_cells:
            msgBox = qt.QMessageBox(parent=self)
            msg = "Some data could not be inserted, "
            msg += "due to out-of-range or write-protected cells."
            msgBox.setText(msg)
            msgBox.exec_()
            return False
        return True

    def keyPressEvent(self, keyEvent):
        if keyEvent.modifiers() & qt.Qt.ControlModifier:
            # control-C event: copy data to the clipboard
            if keyEvent.key() == qt.Qt.Key_C:
                self._copySelection()
            # control-X event: cut data to the clipboard
            elif keyEvent.key() == qt.Qt.Key_X:
                self._copySelection(cut=True)
            # control-V event: copy data from the clipboard
            elif keyEvent.key() == qt.Qt.Key_V:
                self._pasteText()
        super(self, TableWidget).keyPressEvent(self, keyEvent)


# TODO: same for table view
# TODO: move code into a QAction (built-in support for keyboard shortcuts)

