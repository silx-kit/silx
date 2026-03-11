# /*##########################################################################
#
# Copyright (c) 2016-2017 European Synchrotron Radiation Facility
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
This module define a hierarchical table view and model.

It allows to define many headers in the middle of a table.

The implementation hide the default header and allows to custom each cells
to became a header.

Row and column span is a concept of the view in a QTableView.
This implementation also provide a span property as part of the model of the
cell. A role is define to custom this information.
The view is updated everytime the model is reset to take care of the
changes of this information.

A default item delegate is used to redefine the paint of the cells.
"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "07/04/2017"

from silx.gui import qt


class HierarchicalTableModel(qt.QAbstractTableModel):
    """
    Abstract table model to provide more custom on row and column span and
    headers.

    Default headers are ignored and each cells can define IsHeaderRole and
    SpanRole using the `data` function.
    """

    SpanRole = qt.Qt.UserRole + 0
    """Role returning a tuple for number of row span then column span.

    None and (1, 1) are neutral for the rendering.
    """

    IsHeaderRole = qt.Qt.UserRole + 1
    """Role returning True is the identified cell is a header."""

    UserRole = qt.Qt.UserRole + 2
    """First index of user defined roles"""

    def headerData(self, section, orientation, role=qt.Qt.DisplayRole):
        """Returns the 0-based row or column index, for display in the
        horizontal and vertical headers

        In this case the headers are just ignored. Header information is part
        of each cells.
        """
        return None


class _BorderlessItemDelegate(qt.QStyledItemDelegate):
    """
    Delegate item that simply don't display borders of the last cell (expected to be some 'clickable' icons)
    """
    def __init__(self, parent=None, columns_with_borders: tuple[int, ...] = ()):
        self.__columns_with_borders = columns_with_borders
        super().__init__(parent)

    def paint(self, painter, opt, index):
        col = index.column()
        header_style = qt.QStyleOptionHeader()
        palette = header_style.palette
        original_pen = painter.pen()
        # original_pen.setColor(palette.color(qt.QPalette.Pen))
        # painter.setPen(header_style. )

        # Set pen based on column
        rect = opt.rect

        # if col == 0:
        #     pass
        #     # painter.drawRect(rect.right(), rect.bottom(), rect.left(), rect.top())
        # else:
        #     style = qt.QApplication.instance().style()
        #     style.drawControl(qt.QStyle.CE_HeaderSection, opt, painter, None)
            # painter.drawLine(rect.right(), rect.bottom(), rect.left(), rect.bottom())
        # if col in self.__columns_with_borders:
        #     painter.setPen(qt.Qt.NoPen)

        # painter.save()

        # span = index.data(role=HierarchicalTableModel.SpanRole)
        # span = 1 if span is None else span[1]
        # columnCount = index.model().columnCount()
        # if span == columnCount:
        #     mainTitle = True
        #     position = qt.QStyleOptionHeader.OnlyOneSection
        # else:
        #     mainTitle = False
        #     col = index.column()
        #     if col == 0:
        #         position = qt.QStyleOptionHeader.Beginning
        #     elif col < columnCount - 1:
        #         position = qt.QStyleOptionHeader.Middle
        #     else:
        #         position = qt.QStyleOptionHeader.End
        # opt = qt.QStyleOptionHeader()
        # opt.direction = opt.direction
        # opt.text = index.data()
        # opt.textAlignment = qt.Qt.AlignCenter if mainTitle else qt.Qt.AlignVCenter
        # opt.direction = opt.direction
        # opt.fontMetrics = opt.fontMetrics
        # opt.palette = opt.palette
        # opt.rect = opt.rect
        # opt.state = opt.state
        # opt.position = position
        # painter.setPen(qt.QPen(qt.Qt.black, 1))
        # rect = opt.rect

        # painter.drawLine(rect.right(), rect.bottom(), rect.left(), rect.bottom())
        # painter.restore()
        # # Manually draw borders for non-last columns
        # if col not in self.__columns_with_borders:
        #     # Draw vertical line on the right edge
        #     rect = opt.rect
        #     painter.save()
        #     painter.setPen(qt.QPen(qt.Qt.black, 1))
        #     painter.drawLine(rect.right(), rect.bottom(), rect.left(), rect.bottom())
        #     painter.restore()

        #     # style = qt.QApplication.instance().style()
        #     # margin = -1
        #     # opt.rect = opt.rect.adjusted(margin, margin, -margin, -margin)

        #     # style.drawControl(qt.QStyle.CE_HeaderSection, opt, painter, None)

        #     # painter.restore()
        #     # # Restore original pen
        #     # painter.setPen(original_pen)
        #     super().paint(painter, opt, index)

        # else:
        #     painter.setPen(qt.Qt.NoPen)
            # Restore original pen
            # painter.setPen(original_pen)

        # # Paint cell content
        # # painter.setPen(qt.Qt.NoPen)
        super().paint(painter, opt, index)


class HierarchicalItemDelegate(_BorderlessItemDelegate):
    """
    Delegate item to take care of the rendering of the default table cells and
    also the header cells.
    """

    cellClicked = qt.Signal(int, int)
    # row, column

    def __init__(self, parent=None, columns_without_borders: tuple[int, ...] = ()):
        """
        Constructor

        :param qt.QObject parent: Parent of the widget
        """
        super().__init__(parent, columns_with_borders=columns_without_borders)

    def paint(self, painter, option, index):
        """Override the paint function to inject the style of the header.

        :param qt.QPainter painter: Painter context used to displayed the cell
        :param qt.QStyleOptionViewItem option: Control how the editor is shown
        :param qt.QIndex index: Index of the data to display
        """
        isHeader = index.data(role=HierarchicalTableModel.IsHeaderRole)
        if isHeader:
            _BorderlessItemDelegate.paint(self, painter, option, index)

            span = index.data(role=HierarchicalTableModel.SpanRole)
            span = 1 if span is None else span[1]
            columnCount = index.model().columnCount()
            if span == columnCount:
                mainTitle = True
                position = qt.QStyleOptionHeader.OnlyOneSection
            else:
                mainTitle = False
                col = index.column()
                if col == 0:
                    position = qt.QStyleOptionHeader.Beginning
                elif col < columnCount - 1:
                    position = qt.QStyleOptionHeader.Middle
                else:
                    position = qt.QStyleOptionHeader.End
            opt = qt.QStyleOptionHeader()
            opt.direction = option.direction
            opt.text = index.data()
            opt.textAlignment = qt.Qt.AlignCenter if mainTitle else qt.Qt.AlignVCenter
            opt.direction = option.direction
            opt.fontMetrics = option.fontMetrics
            opt.palette = option.palette
            opt.rect = option.rect
            opt.state = option.state
            opt.position = position
            margin = -1
            style = qt.QApplication.instance().style()
            opt.rect = opt.rect.adjusted(margin, margin, -margin, -margin)
            style.drawControl(qt.QStyle.CE_HeaderSection, opt, painter, None)
            margin = 3
            opt.rect = opt.rect.adjusted(margin, margin, -margin, -margin)
            style.drawControl(qt.QStyle.CE_HeaderLabel, opt, painter, None)
        else:
            _BorderlessItemDelegate.paint(self, painter, option, index)


class HierarchicalTableView(qt.QTableView):
    """A TableView which allow to display a `HierarchicalTableModel`."""

    def __init__(self, parent=None):
        """
        Constructor

        :param qt.QWidget parent: Parent of the widget
        """
        super().__init__(parent)
        self.setItemDelegate(HierarchicalItemDelegate(self, columns_without_borders=(0, 1, 2, 3, 4, 5, 6)))
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setVisible(False)
        self.setShowGrid(True)

    def mousePressEvent(self, event):
        if event.button() == qt.Qt.LeftButton:
            index = self.indexAt(event.pos())
            if index.isValid():
                self.model().copyToClipBoard(index.row(), index.column())

        super().mousePressEvent(event)

    def setModel(self, model):
        """Override the default function to connect the model to update
        function"""
        if self.model() is not None:
            model.modelReset.disconnect(self.__modelReset)
        super().setModel(model)
        if self.model() is not None:
            model.modelReset.connect(self.__modelReset)
            self.__modelReset()

    def __modelReset(self):
        """Update the model to take care of the changes of the span
        information"""
        self.clearSpans()
        model = self.model()
        for row in range(model.rowCount()):
            for column in range(model.columnCount()):
                index = model.index(row, column, qt.QModelIndex())
                span = model.data(index, HierarchicalTableModel.SpanRole)
                if span is not None and span != (1, 1):
                    self.setSpan(row, column, span[0], span[1])
