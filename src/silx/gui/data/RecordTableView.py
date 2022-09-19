# /*##########################################################################
#
# Copyright (c) 2017-2021 European Synchrotron Radiation Facility
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
"""
This module  define model and widget to display 1D slices from numpy
array using compound data types or hdf5 databases.
"""

import itertools
import numpy
from silx.gui import qt
import silx.io
from .TextFormatter import TextFormatter
from silx.gui.widgets.TableWidget import CopySelectedCellsAction

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "29/08/2018"


class _MultiLineItem(qt.QItemDelegate):
    """Draw a multiline text without hiding anything.

    The paint method display a cell without any wrap. And an editor is
    available to scroll into the selected cell.
    """

    def __init__(self, parent=None):
        """
        Constructor

        :param qt.QWidget parent: Parent of the widget
        """
        qt.QItemDelegate.__init__(self, parent)
        self.__textOptions = qt.QTextOption()
        self.__textOptions.setFlags(qt.QTextOption.IncludeTrailingSpaces |
                                    qt.QTextOption.ShowTabsAndSpaces)
        self.__textOptions.setWrapMode(qt.QTextOption.NoWrap)
        self.__textOptions.setAlignment(qt.Qt.AlignTop | qt.Qt.AlignLeft)

    def paint(self, painter, option, index):
        """
        Write multiline text without using any wrap or any alignment according
        to the cell size.

        :param qt.QPainter painter: Painter context used to displayed the cell
        :param qt.QStyleOptionViewItem option: Control how the editor is shown
        :param qt.QIndex index: Index of the data to display
        """
        painter.save()

        # set colors
        painter.setPen(qt.QPen(qt.Qt.NoPen))
        if option.state & qt.QStyle.State_Selected:
            brush = option.palette.highlight()
            painter.setBrush(brush)
        else:
            brush = index.data(qt.Qt.BackgroundRole)
            if brush is None:
                # default background color for a cell
                brush = qt.Qt.white
            painter.setBrush(brush)
        painter.drawRect(option.rect)

        if index.isValid():
            if option.state & qt.QStyle.State_Selected:
                brush = option.palette.highlightedText()
            else:
                brush = index.data(qt.Qt.ForegroundRole)
                if brush is None:
                    brush = option.palette.text()
            painter.setPen(qt.QPen(brush.color()))
            text = index.data(qt.Qt.DisplayRole)
            painter.drawText(qt.QRectF(option.rect), text, self.__textOptions)

        painter.restore()

    def createEditor(self, parent, option, index):
        """
        Returns the widget used to edit the item specified by index for editing.

        We use it not to edit the content but to show the content with a
        convenient scroll bar.

        :param qt.QWidget parent: Parent of the widget
        :param qt.QStyleOptionViewItem option: Control how the editor is shown
        :param qt.QIndex index: Index of the data to display
        """
        if not index.isValid():
            return super(_MultiLineItem, self).createEditor(parent, option, index)

        editor = qt.QTextEdit(parent)
        editor.setReadOnly(True)
        return editor

    def setEditorData(self, editor, index):
        """
        Read data from the model and feed the editor.

        :param qt.QWidget editor: Editor widget
        :param qt.QIndex index: Index of the data to display
        """
        text = index.model().data(index, qt.Qt.EditRole)
        editor.setText(text)

    def updateEditorGeometry(self, editor, option, index):
        """
        Update the geometry of the editor according to the changes of the view.

        :param qt.QWidget editor: Editor widget
        :param qt.QStyleOptionViewItem option: Control how the editor is shown
        :param qt.QIndex index: Index of the data to display
        """
        editor.setGeometry(option.rect)


class RecordTableModel(qt.QAbstractTableModel):
    """This data model provides access to 1D slices from numpy array using
    compound data types or hdf5 databases.

    Each entries are displayed in a single row, and each columns contain a
    specific field of the compound type.

    It also allows to display 1D arrays of simple data types.
    array.

    :param qt.QObject parent: Parent object
    :param numpy.ndarray data: A numpy array or a h5py dataset
    """

    MAX_NUMBER_OF_ROWS = 10e6
    """Maximum number of display values of the dataset"""

    def __init__(self, parent=None, data=None):
        qt.QAbstractTableModel.__init__(self, parent)

        self.__data = None
        self.__is_array = False
        self.__fields = None
        self.__formatter = None
        self.__editFormatter = None
        self.setFormatter(TextFormatter(self))

        # set _data
        self.setArrayData(data)

    # Methods to be implemented to subclass QAbstractTableModel
    def rowCount(self, parent_idx=None):
        """Returns number of rows to be displayed in table"""
        if self.__data is None:
            return 0
        elif not self.__is_array:
            return 1
        else:
            return min(len(self.__data), self.MAX_NUMBER_OF_ROWS)

    def columnCount(self, parent_idx=None):
        """Returns number of columns to be displayed in table"""
        if self.__fields is None:
            return 1
        else:
            return len(self.__fields)

    def __clippedData(self, role=qt.Qt.DisplayRole):
        """Return data for cells representing clipped data"""
        if role == qt.Qt.DisplayRole:
            return "..."
        elif role == qt.Qt.ToolTipRole:
            return "Dataset is too large: display is clipped"
        else:
            return None

    def data(self, index, role=qt.Qt.DisplayRole):
        """QAbstractTableModel method to access data values
        in the format ready to be displayed"""
        if not index.isValid():
            return None

        if self.__data is None:
            return None

        # Special display of one before last data for clipped table
        if self.__isClipped() and index.row() == self.rowCount() - 2:
            return self.__clippedData(role)

        if self.__is_array:
            row = index.row()
            if row >= self.rowCount():
                return None
            elif self.__isClipped() and row == self.rowCount() - 1:
                # Clipped array, display last value at the end
                data = self.__data[-1]
            else:
                data = self.__data[row]
        else:
            if index.row() > 0:
                return None
            data = self.__data

        if self.__fields is not None:
            if index.column() >= len(self.__fields):
                return None
            key = self.__fields[index.column()][1]
            data = data[key[0]]
            if len(key) > 1:
                data = data[key[1]]

        # no dtype in case of 1D array of unicode objects (#2093)
        dtype = getattr(data, "dtype", None)

        if role == qt.Qt.DisplayRole:
            return self.__formatter.toString(data, dtype=dtype)
        elif role == qt.Qt.EditRole:
            return self.__editFormatter.toString(data, dtype=dtype)
        return None

    def headerData(self, section, orientation, role=qt.Qt.DisplayRole):
        """Returns the 0-based row or column index, for display in the
        horizontal and vertical headers"""
        if section == -1:
            # PyQt4 send -1 when there is columns but no rows
            return None

        # Handle clipping of huge tables
        if (self.__isClipped() and
                orientation == qt.Qt.Vertical and
                section == self.rowCount() - 2):
            return self.__clippedData(role)

        if role == qt.Qt.DisplayRole:
            if orientation == qt.Qt.Vertical:
                if not self.__is_array:
                    return "Scalar"
                elif section == self.MAX_NUMBER_OF_ROWS - 1:
                    return str(len(self.__data) - 1)
                else:
                    return str(section)
            if orientation == qt.Qt.Horizontal:
                if self.__fields is None:
                    if section == 0:
                        return "Data"
                    else:
                        return None
                else:
                    if section < len(self.__fields):
                        return self.__fields[section][0]
                    else:
                        return None
        return None

    def flags(self, index):
        """QAbstractTableModel method to inform the view whether data
        is editable or not.
        """
        return qt.QAbstractTableModel.flags(self, index)

    def __isClipped(self) -> bool:
        """Returns whether the displayed array is clipped or not"""
        return self.__data is not None and self.__is_array and len(self.__data) > self.MAX_NUMBER_OF_ROWS

    def setArrayData(self, data):
        """Set the data array and the viewing perspective.

        You can set ``copy=False`` if you need more performances, when dealing
        with a large numpy array. In this case, a simple reference to the data
        is used to access the data, rather than a copy of the array.

        .. warning::

            Any change to the data model will affect your original data
            array, when using a reference rather than a copy..

        :param data: 1D numpy array, or any object that can be
            converted to a numpy array using ``numpy.array(data)`` (e.g.
            a nested sequence).
        """
        self.beginResetModel()

        self.__data = data
        if isinstance(data, numpy.ndarray):
            self.__is_array = True
        elif silx.io.is_dataset(data) and data.shape != tuple():
            self.__is_array = True
        else:
            self.__is_array = False

        self.__fields = []
        if data is not None:
            if data.dtype.fields is not None:
                fields = sorted(data.dtype.fields.items(), key=lambda e: e[1][1])
                for name, (dtype, _index) in fields:
                    if dtype.shape != tuple():
                        keys = itertools.product(*[range(x) for x in dtype.shape])
                        for key in keys:
                            label = "%s%s" % (name, list(key))
                            array_key = (name, key)
                            self.__fields.append((label, array_key))
                    else:
                        self.__fields.append((name, (name,)))
            else:
                self.__fields = None

        self.endResetModel()

    def arrayData(self):
        """Returns the internal data.

        :rtype: numpy.ndarray of h5py.Dataset
        """
        return self.__data

    def setFormatter(self, formatter):
        """Set the formatter object to be used to display data from the model

        :param TextFormatter formatter: Formatter to use
        """
        if formatter is self.__formatter:
            return

        self.beginResetModel()

        if self.__formatter is not None:
            self.__formatter.formatChanged.disconnect(self.__formatChanged)

        self.__formatter = formatter
        self.__editFormatter = TextFormatter(formatter)
        self.__editFormatter.setUseQuoteForText(False)

        if self.__formatter is not None:
            self.__formatter.formatChanged.connect(self.__formatChanged)

        self.endResetModel()

    def getFormatter(self):
        """Returns the text formatter used.

        :rtype: TextFormatter
        """
        return self.__formatter

    def __formatChanged(self):
        """Called when the format changed.
        """
        self.__editFormatter = TextFormatter(self, self.getFormatter())
        self.__editFormatter.setUseQuoteForText(False)
        self.reset()


class _ShowEditorProxyModel(qt.QIdentityProxyModel):
    """
    Allow to custom the flag edit of the model
    """

    def __init__(self, parent=None):
        """
        Constructor

        :param qt.QObject arent: parent object
        """
        super(_ShowEditorProxyModel, self).__init__(parent)
        self.__forceEditable = False

    def flags(self, index):
        flag = qt.QIdentityProxyModel.flags(self, index)
        if self.__forceEditable:
            flag = flag | qt.Qt.ItemIsEditable
        return flag

    def forceCellEditor(self, show):
        """
        Enable the editable flag to allow to display cell editor.
        """
        if self.__forceEditable == show:
            return
        self.beginResetModel()
        self.__forceEditable = show
        self.endResetModel()


class RecordTableView(qt.QTableView):
    """TableView using DatabaseTableModel as default model.
    """
    def __init__(self, parent=None):
        """
        Constructor

        :param qt.QWidget parent: parent QWidget
        """
        qt.QTableView.__init__(self, parent)

        model = _ShowEditorProxyModel(self)
        self._model = RecordTableModel()
        model.setSourceModel(self._model)
        self.setModel(model)

        self.__multilineView = _MultiLineItem(self)
        self.setEditTriggers(qt.QAbstractItemView.AllEditTriggers)
        self._copyAction = CopySelectedCellsAction(self)
        self.addAction(self._copyAction)

    def copy(self):
        self._copyAction.trigger()

    def setArrayData(self, data):
        model = self.model()
        sourceModel = model.sourceModel()
        sourceModel.setArrayData(data)

        if data is not None:
            if issubclass(data.dtype.type, (numpy.string_, numpy.unicode_)):
                # TODO it would be nice to also fix fields
                # but using it only for string array is already very useful
                self.setItemDelegateForColumn(0, self.__multilineView)
                model.forceCellEditor(True)
            else:
                self.setItemDelegateForColumn(0, None)
                model.forceCellEditor(False)
