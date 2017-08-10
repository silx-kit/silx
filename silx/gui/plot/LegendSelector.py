# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2004-2017 European Synchrotron Radiation Facility
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
"""Widget displaying curves legends and allowing to operate on curves.

This widget is meant to work with :class:`PlotWindow`.
"""

__authors__ = ["V.A. Sole", "T. Rueter", "T. Vincent"]
__license__ = "MIT"
__data__ = "08/08/2016"


import logging
import weakref

from .. import qt


_logger = logging.getLogger(__name__)

# Build all symbols
# Courtesy of the pyqtgraph project
Symbols = dict([(name, qt.QPainterPath())
                for name in ['o', 's', 't', 'd', '+', 'x', '.', ',']])
Symbols['o'].addEllipse(qt.QRectF(.1, .1, .8, .8))
Symbols['.'].addEllipse(qt.QRectF(.3, .3, .4, .4))
Symbols[','].addEllipse(qt.QRectF(.4, .4, .2, .2))
Symbols['s'].addRect(qt.QRectF(.1, .1, .8, .8))

coords = {
    't': [(0.5, 0.), (.1, .8), (.9, .8)],
    'd': [(0.1, 0.5), (0.5, 0.), (0.9, 0.5), (0.5, 1.)],
    '+': [(0.0, 0.40), (0.40, 0.40), (0.40, 0.), (0.60, 0.),
          (0.60, 0.40), (1., 0.40), (1., 0.60), (0.60, 0.60),
          (0.60, 1.), (0.40, 1.), (0.40, 0.60), (0., 0.60)],
    'x': [(0.0, 0.40), (0.40, 0.40), (0.40, 0.), (0.60, 0.),
          (0.60, 0.40), (1., 0.40), (1., 0.60), (0.60, 0.60),
          (0.60, 1.), (0.40, 1.), (0.40, 0.60), (0., 0.60)]
}
for s, c in coords.items():
    Symbols[s].moveTo(*c[0])
    for x, y in c[1:]:
        Symbols[s].lineTo(x, y)
    Symbols[s].closeSubpath()
tr = qt.QTransform()
tr.rotate(45)
Symbols['x'].translate(qt.QPointF(-0.5, -0.5))
Symbols['x'] = tr.map(Symbols['x'])
Symbols['x'].translate(qt.QPointF(0.5, 0.5))

NoSymbols = (None, 'None', 'none', '', ' ')
"""List of values resulting in no symbol being displayed for a curve"""


LineStyles = {
    None: qt.Qt.NoPen,
    'None': qt.Qt.NoPen,
    'none': qt.Qt.NoPen,
    '': qt.Qt.NoPen,
    ' ': qt.Qt.NoPen,
    '-': qt.Qt.SolidLine,
    '--': qt.Qt.DashLine,
    ':': qt.Qt.DotLine,
    '-.': qt.Qt.DashDotLine
}
"""Conversion from matplotlib-like linestyle to Qt"""

NoLineStyle = (None, 'None', 'none', '', ' ')
"""List of style values resulting in no line being displayed for a curve"""


class LegendIcon(qt.QWidget):
    """Object displaying a curve linestyle and symbol."""

    def __init__(self, parent=None):
        super(LegendIcon, self).__init__(parent)

        # Visibilities
        self.showLine = True
        self.showSymbol = True

        # Line attributes
        self.lineStyle = qt.Qt.NoPen
        self.lineWidth = 1.
        self.lineColor = qt.Qt.green

        self.symbol = ''
        # Symbol attributes
        self.symbolStyle = qt.Qt.SolidPattern
        self.symbolColor = qt.Qt.green
        self.symbolOutlineBrush = qt.QBrush(qt.Qt.white)

        # Control widget size: sizeHint "is the only acceptable
        # alternative, so the widget can never grow or shrink"
        # (c.f. Qt Doc, enum QSizePolicy::Policy)
        self.setSizePolicy(qt.QSizePolicy.Fixed,
                           qt.QSizePolicy.Fixed)

    def sizeHint(self):
        return qt.QSize(50, 15)

    # Modify Symbol
    def setSymbol(self, symbol):
        symbol = str(symbol)
        if symbol not in NoSymbols:
            if symbol not in Symbols:
                raise ValueError("Unknown symbol: <%s>" % symbol)
        self.symbol = symbol
        # self.update() after set...?
        # Does not seem necessary

    def setSymbolColor(self, color):
        """
        :param color: determines the symbol color
        :type style: qt.QColor
        """
        self.symbolColor = qt.QColor(color)

    # Modify Line

    def setLineColor(self, color):
        self.lineColor = qt.QColor(color)

    def setLineWidth(self, width):
        self.lineWidth = float(width)

    def setLineStyle(self, style):
        """Set the linestyle.

        Possible line styles:

        - '', ' ', 'None': No line
        - '-': solid
        - '--': dashed
        - ':': dotted
        - '-.': dash and dot

        :param str style: The linestyle to use
        """
        if style not in LineStyles:
            raise ValueError('Unknown style: %s', style)
        self.lineStyle = LineStyles[style]

    # Paint

    def paintEvent(self, event):
        """
        :param event: event
        :type event: QPaintEvent
        """
        painter = qt.QPainter(self)
        self.paint(painter, event.rect(), self.palette())

    def paint(self, painter, rect, palette):
        painter.save()
        painter.setRenderHint(qt.QPainter.Antialiasing)
        # Scale painter to the icon height
        # current -> width = 2.5, height = 1.0
        scale = float(self.height())
        ratio = float(self.width()) / scale
        painter.scale(scale,
                      scale)
        symbolOffset = qt.QPointF(.5 * (ratio - 1.), 0.)
        # Determine and scale offset
        offset = qt.QPointF(float(rect.left()) / scale, float(rect.top()) / scale)
        # Draw BG rectangle (for debugging)
        # bottomRight = qt.QPointF(
        #    float(rect.right())/scale,
        #    float(rect.bottom())/scale)
        # painter.fillRect(qt.QRectF(offset, bottomRight),
        #                 qt.QBrush(qt.Qt.green))
        llist = []
        if self.showLine:
            linePath = qt.QPainterPath()
            linePath.moveTo(0., 0.5)
            linePath.lineTo(ratio, 0.5)
            # linePath.lineTo(2.5, 0.5)
            linePen = qt.QPen(
                qt.QBrush(self.lineColor),
                (self.lineWidth / self.height()),
                self.lineStyle,
                qt.Qt.FlatCap
            )
            llist.append((linePath,
                          linePen,
                          qt.QBrush(self.lineColor)))
        if (self.showSymbol and len(self.symbol) and
                self.symbol not in NoSymbols):
            # PITFALL ahead: Let this be a warning to others
            # symbolPath = Symbols[self.symbol]
            # Copy before translate! Dict is a mutable type
            symbolPath = qt.QPainterPath(Symbols[self.symbol])
            symbolPath.translate(symbolOffset)
            symbolBrush = qt.QBrush(
                self.symbolColor,
                self.symbolStyle
            )
            symbolPen = qt.QPen(
                self.symbolOutlineBrush,  # Brush
                1. / self.height(),       # Width
                qt.Qt.SolidLine           # Style
            )
            llist.append((symbolPath,
                          symbolPen,
                          symbolBrush))
        # Draw
        for path, pen, brush in llist:
            path.translate(offset)
            painter.setPen(pen)
            painter.setBrush(brush)
            painter.drawPath(path)
        painter.restore()


class LegendModel(qt.QAbstractListModel):
    """Data model of curve legends.

    It holds the information of the curve:

    - color
    - line width
    - line style
    - visibility of the lines
    - symbol
    - visibility of the symbols
    """
    iconColorRole = qt.Qt.UserRole + 0
    iconLineWidthRole = qt.Qt.UserRole + 1
    iconLineStyleRole = qt.Qt.UserRole + 2
    showLineRole = qt.Qt.UserRole + 3
    iconSymbolRole = qt.Qt.UserRole + 4
    showSymbolRole = qt.Qt.UserRole + 5

    def __init__(self, legendList=None, parent=None):
        super(LegendModel, self).__init__(parent)
        if legendList is None:
            legendList = []
        self.legendList = []
        self.insertLegendList(0, legendList)
        self._palette = qt.QPalette(self)

    def __getitem__(self, idx):
        if idx >= len(self.legendList):
            raise IndexError('list index out of range')
        return self.legendList[idx]

    def rowCount(self, modelIndex=None):
        return len(self.legendList)

    def flags(self, index):
        return (qt.Qt.ItemIsEditable |
                qt.Qt.ItemIsEnabled |
                qt.Qt.ItemIsSelectable)

    def data(self, modelIndex, role):
        if modelIndex.isValid:
            idx = modelIndex.row()
        else:
            return None
        if idx >= len(self.legendList):
            raise IndexError('list index out of range')

        item = self.legendList[idx]
        isActive = item[1].get("active", False)
        if role == qt.Qt.DisplayRole:
            # Data to be rendered in the form of text
            legend = str(item[0])
            return legend
        elif role == qt.Qt.SizeHintRole:
            # size = qt.QSize(200,50)
            _logger.warning('LegendModel -- size hint role not implemented')
            return qt.QSize()
        elif role == qt.Qt.TextAlignmentRole:
            alignment = qt.Qt.AlignVCenter | qt.Qt.AlignLeft
            return alignment
        elif role == qt.Qt.BackgroundRole:
            # Background color, must be QBrush
            if isActive:
                brush = self._palette.brush(qt.QPalette.Normal, qt.QPalette.Highlight)
            elif idx % 2:
                brush = qt.QBrush(qt.QColor(240, 240, 240))
            else:
                brush = qt.QBrush(qt.Qt.white)
            return brush
        elif role == qt.Qt.ForegroundRole:
            # ForegroundRole color, must be QBrush
            if isActive:
                brush = self._palette.brush(qt.QPalette.Normal, qt.QPalette.HighlightedText)
            else:
                brush = self._palette.brush(qt.QPalette.Normal, qt.QPalette.WindowText)
            return brush
        elif role == qt.Qt.CheckStateRole:
            return bool(item[2])  # item[2] == True
        elif role == qt.Qt.ToolTipRole or role == qt.Qt.StatusTipRole:
            return ''
        elif role == self.iconColorRole:
            return item[1]['color']
        elif role == self.iconLineWidthRole:
            return item[1]['linewidth']
        elif role == self.iconLineStyleRole:
            return item[1]['linestyle']
        elif role == self.iconSymbolRole:
            return item[1]['symbol']
        elif role == self.showLineRole:
            return item[3]
        elif role == self.showSymbolRole:
            return item[4]
        else:
            _logger.info('Unkown role requested: %s', str(role))
            return None

    def setData(self, modelIndex, value, role):
        if modelIndex.isValid:
            idx = modelIndex.row()
        else:
            return None
        if idx >= len(self.legendList):
            # raise IndexError('list index out of range')
            _logger.warning(
                'setData -- List index out of range, idx: %d', idx)
            return None

        item = self.legendList[idx]
        try:
            if role == qt.Qt.DisplayRole:
                # Set legend
                item[0] = str(value)
            elif role == self.iconColorRole:
                item[1]['color'] = qt.QColor(value)
            elif role == self.iconLineWidthRole:
                item[1]['linewidth'] = int(value)
            elif role == self.iconLineStyleRole:
                item[1]['linestyle'] = str(value)
            elif role == self.iconSymbolRole:
                item[1]['symbol'] = str(value)
            elif role == qt.Qt.CheckStateRole:
                item[2] = value
            elif role == self.showLineRole:
                item[3] = value
            elif role == self.showSymbolRole:
                item[4] = value
        except ValueError:
            _logger.warning('Conversion failed:\n\tvalue: %s\n\trole: %s',
                            str(value), str(role))
        # Can that be right? Read docs again..
        self.dataChanged.emit(modelIndex, modelIndex)
        return True

    def insertLegendList(self, row, llist):
        """
        :param int row: Determines after which row the items are inserted
        :param llist: Carries the new legend information
        :type llist: List
        """
        modelIndex = self.createIndex(row, 0)
        count = len(llist)
        super(LegendModel, self).beginInsertRows(modelIndex,
                                                 row,
                                                 row + count)
        head = self.legendList[0:row]
        tail = self.legendList[row:]
        new = []
        for (legend, icon) in llist:
            linestyle = icon.get('linestyle', None)
            if linestyle in NoLineStyle:
                # Curve had no line, give it one and hide it
                # So when toggle line, it will display a solid line
                showLine = False
                icon['linestyle'] = '-'
            else:
                showLine = True

            symbol = icon.get('symbol', None)
            if symbol in NoSymbols:
                # Curve had no symbol, give it one and hide it
                # So when toggle symbol, it will display 'o'
                showSymbol = False
                icon['symbol'] = 'o'
            else:
                showSymbol = True

            selected = icon.get('selected', True)
            item = [legend,
                    icon,
                    selected,
                    showLine,
                    showSymbol]
            new.append(item)
        self.legendList = head + new + tail
        super(LegendModel, self).endInsertRows()
        return True

    def insertRows(self, row, count, modelIndex=qt.QModelIndex()):
        raise NotImplementedError('Use LegendModel.insertLegendList instead')

    def removeRow(self, row):
        return self.removeRows(row, 1)

    def removeRows(self, row, count, modelIndex=qt.QModelIndex()):
        length = len(self.legendList)
        if length == 0:
            # Nothing to do..
            return True
        if row < 0 or row >= length:
            raise IndexError('Index out of range -- ' +
                             'idx: %d, len: %d' % (row, length))
        if count == 0:
            return False
        super(LegendModel, self).beginRemoveRows(modelIndex,
                                                 row,
                                                 row + count)
        del(self.legendList[row:row + count])
        super(LegendModel, self).endRemoveRows()
        return True

    def setEditor(self, event, editor):
        """
        :param str event: String that identifies the editor
        :param editor: Widget used to change data in the underlying model
        :type editor: QWidget
        """
        if event not in self.eventList:
            raise ValueError('setEditor -- Event must be in %s' %
                             str(self.eventList))
        self.editorDict[event] = editor


class LegendListItemWidget(qt.QItemDelegate):
    """Object displaying a single item (i.e., a row) in the list."""

    # Notice: LegendListItem does NOT inherit
    # from QObject, it cannot emit signals!

    def __init__(self, parent=None, itemType=0):
        super(LegendListItemWidget, self).__init__(parent)

        # Dictionary to render checkboxes
        self.cbDict = {}
        self.labelDict = {}
        self.iconDict = {}

        # Keep checkbox and legend to get sizeHint
        self.checkbox = qt.QCheckBox()
        self.legend = qt.QLabel()
        self.icon = LegendIcon()

        # Context Menu and Editors
        self.contextMenu = None

    def paint(self, painter, option, modelIndex):
        """
        Here be docs..

        :param QPainter painter:
        :param QStyleOptionViewItem option:
        :param QModelIndex modelIndex:
        """
        painter.save()
        rect = option.rect

        # Calculate the icon rectangle
        iconSize = self.icon.sizeHint()
        # Calculate icon position
        x = rect.left() + 2
        y = rect.top() + int(.5 * (rect.height() - iconSize.height()))
        iconRect = qt.QRect(qt.QPoint(x, y), iconSize)

        # Calculate label rectangle
        legendSize = qt.QSize(rect.width() - iconSize.width() - 30,
                              rect.height())
        # Calculate label position
        x = rect.left() + iconRect.width()
        y = rect.top()
        labelRect = qt.QRect(qt.QPoint(x, y), legendSize)
        labelRect.translate(qt.QPoint(10, 0))

        # Calculate the checkbox rectangle
        x = rect.right() - 30
        y = rect.top()
        chBoxRect = qt.QRect(qt.QPoint(x, y), rect.bottomRight())

        # Remember the rectangles
        idx = modelIndex.row()
        self.cbDict[idx] = chBoxRect
        self.iconDict[idx] = iconRect
        self.labelDict[idx] = labelRect

        # Draw background first!
        if option.state & qt.QStyle.State_MouseOver:
            backgroundBrush = option.palette.highlight()
        else:
            backgroundBrush = modelIndex.data(qt.Qt.BackgroundRole)
        painter.fillRect(rect, backgroundBrush)

        # Draw label
        legendText = modelIndex.data(qt.Qt.DisplayRole)
        textBrush = modelIndex.data(qt.Qt.ForegroundRole)
        textAlign = modelIndex.data(qt.Qt.TextAlignmentRole)
        painter.setBrush(textBrush)
        painter.setFont(self.legend.font())
        painter.setPen(textBrush.color())
        painter.drawText(labelRect, textAlign, legendText)

        # Draw icon
        iconColor = modelIndex.data(LegendModel.iconColorRole)
        iconLineWidth = modelIndex.data(LegendModel.iconLineWidthRole)
        iconLineStyle = modelIndex.data(LegendModel.iconLineStyleRole)
        iconSymbol = modelIndex.data(LegendModel.iconSymbolRole)
        icon = LegendIcon()
        icon.resize(iconRect.size())
        icon.move(iconRect.topRight())
        icon.showSymbol = modelIndex.data(LegendModel.showSymbolRole)
        icon.showLine = modelIndex.data(LegendModel.showLineRole)
        icon.setSymbolColor(iconColor)
        icon.setLineColor(iconColor)
        icon.setLineWidth(iconLineWidth)
        icon.setLineStyle(iconLineStyle)
        icon.setSymbol(iconSymbol)
        icon.symbolOutlineBrush = backgroundBrush
        icon.paint(painter, iconRect, option.palette)

        # Draw the checkbox
        if modelIndex.data(qt.Qt.CheckStateRole):
            checkState = qt.Qt.Checked
        else:
            checkState = qt.Qt.Unchecked

        self.drawCheck(
            painter, qt.QStyleOptionViewItem(), chBoxRect, checkState)

        painter.restore()

    def editorEvent(self, event, model, option, modelIndex):
        # From the docs:
        # Mouse events are sent to editorEvent()
        # even if they don't start editing of the item.
        if event.button() == qt.Qt.RightButton and self.contextMenu:
            self.contextMenu.exec_(event.globalPos(), modelIndex)
            return True
        elif event.button() == qt.Qt.LeftButton:
            # Check if checkbox was clicked
            idx = modelIndex.row()
            cbRect = self.cbDict[idx]
            if cbRect.contains(event.pos()):
                # Toggle checkbox
                model.setData(modelIndex,
                              not modelIndex.data(qt.Qt.CheckStateRole),
                              qt.Qt.CheckStateRole)
            event.ignore()
            return True
        else:
            return super(LegendListItemWidget, self).editorEvent(
                event, model, option, modelIndex)

    def createEditor(self, parent, option, idx):
        _logger.info('### Editor request ###')

    def sizeHint(self, option, idx):
        # return qt.QSize(68,24)
        iconSize = self.icon.sizeHint()
        legendSize = self.legend.sizeHint()
        checkboxSize = self.checkbox.sizeHint()
        height = max([iconSize.height(),
                      legendSize.height(),
                      checkboxSize.height()]) + 4
        width = iconSize.width() + legendSize.width() + checkboxSize.width()
        return qt.QSize(width, height)


class LegendListView(qt.QListView):
    """Widget displaying a list of curve legends, line style and symbol."""

    sigLegendSignal = qt.Signal(object)
    """Signal emitting a dict when an action is triggered by the user."""

    __mouseClickedEvent = 'mouseClicked'
    __checkBoxClickedEvent = 'checkBoxClicked'
    __legendClickedEvent = 'legendClicked'

    def __init__(self, parent=None, model=None, contextMenu=None):
        super(LegendListView, self).__init__(parent)
        self.__lastButton = None
        self.__lastClickPos = None
        self.__lastModelIdx = None
        # Set default delegate
        self.setItemDelegate(LegendListItemWidget())
        # Set default editors
        # self.setSizePolicy(qt.QSizePolicy.MinimumExpanding,
        #                    qt.QSizePolicy.MinimumExpanding)
        # Set edit triggers by hand using self.edit(QModelIndex)
        # in mousePressEvent (better to control than signals)
        self.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)

        # Control layout
        # self.setBatchSize(2)
        # self.setLayoutMode(qt.QListView.Batched)
        # self.setFlow(qt.QListView.LeftToRight)

        # Control selection
        self.setSelectionMode(qt.QAbstractItemView.NoSelection)

        if model is None:
            model = LegendModel(parent=self)
        self.setModel(model)
        self.setContextMenu(contextMenu)

    def setLegendList(self, legendList, row=None):
        self.clear()
        if row is None:
            row = 0
        model = self.model()
        model.insertLegendList(row, legendList)
        _logger.debug('LegendListView.setLegendList(legendList) finished')

    def clear(self):
        model = self.model()
        model.removeRows(0, model.rowCount())
        _logger.debug('LegendListView.clear() finished')

    def setContextMenu(self, contextMenu=None):
        delegate = self.itemDelegate()
        if isinstance(delegate, LegendListItemWidget) and self.model():
            if contextMenu is None:
                delegate.contextMenu = LegendListContextMenu(self.model())
                delegate.contextMenu.sigContextMenu.connect(
                    self._contextMenuSlot)
            else:
                delegate.contextMenu = contextMenu

    def __getitem__(self, idx):
        model = self.model()
        try:
            item = model[idx]
        except ValueError:
            item = None
        return item

    def _contextMenuSlot(self, ddict):
        self.sigLegendSignal.emit(ddict)

    def mousePressEvent(self, event):
        self.__lastButton = event.button()
        self.__lastPosition = event.pos()
        super(LegendListView, self).mousePressEvent(event)
        # call _handleMouseClick after editing was handled
        # If right click (context menu) is aborted, no
        # signal is emitted..
        self._handleMouseClick(self.indexAt(self.__lastPosition))

    def mouseDoubleClickEvent(self, event):
        self.__lastButton = event.button()
        self.__lastPosition = event.pos()
        super(LegendListView, self).mouseDoubleClickEvent(event)
        # call _handleMouseClick after editing was handled
        # If right click (context menu) is aborted, no
        # signal is emitted..
        self._handleMouseClick(self.indexAt(self.__lastPosition))

    def mouseMoveEvent(self, event):
        # LegendListView.mouseMoveEvent is overwritten
        # to suppress unwanted behavior in the delegate.
        pass

    def mouseReleaseEvent(self, event):
        # LegendListView.mouseReleaseEvent is overwritten
        # to subpress unwanted behavior in the delegate.
        pass

    def _handleMouseClick(self, modelIndex):
        """
        Distinguish between mouse click on Legend
        and mouse click on CheckBox by setting the
        currentCheckState attribute in LegendListItem.

        Emits signal sigLegendSignal(ddict)

        :param QModelIndex modelIndex: index of the clicked item
        """
        _logger.debug('self._handleMouseClick called')
        if self.__lastButton not in [qt.Qt.LeftButton,
                                     qt.Qt.RightButton]:
            return
        if not modelIndex.isValid():
            _logger.debug('_handleMouseClick -- Invalid QModelIndex')
            return
        # model = self.model()
        idx = modelIndex.row()

        delegate = self.itemDelegate()
        cbClicked = False
        if isinstance(delegate, LegendListItemWidget):
            for cbRect in delegate.cbDict.values():
                if cbRect.contains(self.__lastPosition):
                    cbClicked = True
                    break

        # TODO: Check for doubleclicks on legend/icon and spawn editors

        ddict = {
            'legend': str(modelIndex.data(qt.Qt.DisplayRole)),
            'icon': {
                'linewidth': str(modelIndex.data(
                    LegendModel.iconLineWidthRole)),
                'linestyle': str(modelIndex.data(
                    LegendModel.iconLineStyleRole)),
                'symbol': str(modelIndex.data(LegendModel.iconSymbolRole))
            },
            'selected': modelIndex.data(qt.Qt.CheckStateRole),
            'type': str(modelIndex.data())
        }
        if self.__lastButton == qt.Qt.RightButton:
            _logger.debug('Right clicked')
            ddict['button'] = "right"
            ddict['event'] = self.__mouseClickedEvent
        elif cbClicked:
            _logger.debug('CheckBox clicked')
            ddict['button'] = "left"
            ddict['event'] = self.__checkBoxClickedEvent
        else:
            _logger.debug('Legend clicked')
            ddict['button'] = "left"
            ddict['event'] = self.__legendClickedEvent
        _logger.debug('  idx: %d\n  ddict: %s', idx, str(ddict))
        self.sigLegendSignal.emit(ddict)


class LegendListContextMenu(qt.QMenu):
    """Contextual menu associated to items in a :class:`LegendListView`."""

    sigContextMenu = qt.Signal(object)
    """Signal emitting a dict upon contextual menu actions."""

    def __init__(self, model):
        super(LegendListContextMenu, self).__init__(parent=None)
        self.model = model

        self.addAction('Set Active', self.setActiveAction)
        self.addAction('Map to left', self.mapToLeftAction)
        self.addAction('Map to right', self.mapToRightAction)

        self._pointsAction = self.addAction(
            'Points', self.togglePointsAction)
        self._pointsAction.setCheckable(True)

        self._linesAction = self.addAction('Lines', self.toggleLinesAction)
        self._linesAction.setCheckable(True)

        self.addAction('Remove curve', self.removeItemAction)
        self.addAction('Rename curve', self.renameItemAction)

    def exec_(self, pos, idx):
        self.__currentIdx = idx

        # Set checkable action state
        modelIndex = self.currentIdx()
        self._pointsAction.setChecked(
            modelIndex.data(LegendModel.showSymbolRole))
        self._linesAction.setChecked(
            modelIndex.data(LegendModel.showLineRole))

        super(LegendListContextMenu, self).popup(pos)

    def currentIdx(self):
        return self.__currentIdx

    def mapToLeftAction(self):
        _logger.debug('LegendListContextMenu.mapToLeftAction called')
        modelIndex = self.currentIdx()
        legend = str(modelIndex.data(qt.Qt.DisplayRole))
        ddict = {
            'legend': legend,
            'label': legend,
            'selected': modelIndex.data(qt.Qt.CheckStateRole),
            'type': str(modelIndex.data()),
            'event': "mapToLeft"
        }
        self.sigContextMenu.emit(ddict)

    def mapToRightAction(self):
        _logger.debug('LegendListContextMenu.mapToRightAction called')
        modelIndex = self.currentIdx()
        legend = str(modelIndex.data(qt.Qt.DisplayRole))
        ddict = {
            'legend': legend,
            'label': legend,
            'selected': modelIndex.data(qt.Qt.CheckStateRole),
            'type': str(modelIndex.data()),
            'event': "mapToRight"
        }
        self.sigContextMenu.emit(ddict)

    def removeItemAction(self):
        _logger.debug('LegendListContextMenu.removeCurveAction called')
        modelIndex = self.currentIdx()
        legend = str(modelIndex.data(qt.Qt.DisplayRole))
        ddict = {
            'legend': legend,
            'label': legend,
            'selected': modelIndex.data(qt.Qt.CheckStateRole),
            'type': str(modelIndex.data()),
            'event': "removeCurve"
        }
        self.model.removeRow(modelIndex.row())
        self.sigContextMenu.emit(ddict)

    def renameItemAction(self):
        _logger.debug('LegendListContextMenu.renameCurveAction called')
        modelIndex = self.currentIdx()
        legend = str(modelIndex.data(qt.Qt.DisplayRole))
        ddict = {
            'legend': legend,
            'label': legend,
            'selected': modelIndex.data(qt.Qt.CheckStateRole),
            'type': str(modelIndex.data()),
            'event': "renameCurve"
        }
        self.sigContextMenu.emit(ddict)

    def toggleLinesAction(self):
        modelIndex = self.currentIdx()
        legend = str(modelIndex.data(qt.Qt.DisplayRole))
        ddict = {
            'legend': legend,
            'label': legend,
            'selected': modelIndex.data(qt.Qt.CheckStateRole),
            'type': str(modelIndex.data()),
        }
        linestyle = modelIndex.data(LegendModel.iconLineStyleRole)
        visible = not modelIndex.data(LegendModel.showLineRole)
        _logger.debug('toggleLinesAction -- lines visible: %s', str(visible))
        ddict['event'] = "toggleLine"
        ddict['line'] = visible
        ddict['linestyle'] = linestyle if visible else ''
        self.model.setData(modelIndex, visible, LegendModel.showLineRole)
        self.sigContextMenu.emit(ddict)

    def togglePointsAction(self):
        modelIndex = self.currentIdx()
        legend = str(modelIndex.data(qt.Qt.DisplayRole))
        ddict = {
            'legend': legend,
            'label': legend,
            'selected': modelIndex.data(qt.Qt.CheckStateRole),
            'type': str(modelIndex.data()),
        }
        flag = modelIndex.data(LegendModel.showSymbolRole)
        symbol = modelIndex.data(LegendModel.iconSymbolRole)
        visible = not flag or symbol in NoSymbols
        _logger.debug(
            'togglePointsAction -- Symbols visible: %s', str(visible))

        ddict['event'] = "togglePoints"
        ddict['points'] = visible
        ddict['symbol'] = symbol if visible else ''
        self.model.setData(modelIndex, visible, LegendModel.showSymbolRole)
        self.sigContextMenu.emit(ddict)

    def setActiveAction(self):
        modelIndex = self.currentIdx()
        legend = str(modelIndex.data(qt.Qt.DisplayRole))
        _logger.debug('setActiveAction -- active curve: %s', legend)
        ddict = {
            'legend': legend,
            'label': legend,
            'selected': modelIndex.data(qt.Qt.CheckStateRole),
            'type': str(modelIndex.data()),
            'event': "setActiveCurve",
        }
        self.sigContextMenu.emit(ddict)


class RenameCurveDialog(qt.QDialog):
    """Dialog box to input the name of a curve."""

    def __init__(self, parent=None, current="", curves=()):
        super(RenameCurveDialog, self).__init__(parent)
        self.setWindowTitle("Rename Curve %s" % current)
        self.curves = curves
        layout = qt.QVBoxLayout(self)
        self.lineEdit = qt.QLineEdit(self)
        self.lineEdit.setText(current)
        self.hbox = qt.QWidget(self)
        self.hboxLayout = qt.QHBoxLayout(self.hbox)
        self.hboxLayout.addStretch(1)
        self.okButton = qt.QPushButton(self.hbox)
        self.okButton.setText('OK')
        self.hboxLayout.addWidget(self.okButton)
        self.cancelButton = qt.QPushButton(self.hbox)
        self.cancelButton.setText('Cancel')
        self.hboxLayout.addWidget(self.cancelButton)
        self.hboxLayout.addStretch(1)
        layout.addWidget(self.lineEdit)
        layout.addWidget(self.hbox)
        self.okButton.clicked.connect(self.preAccept)
        self.cancelButton.clicked.connect(self.reject)

    def preAccept(self):
        text = str(self.lineEdit.text())
        addedText = ""
        if len(text):
            if text not in self.curves:
                self.accept()
                return
            else:
                addedText = "Curve already exists."
        text = "Invalid Curve Name"
        msg = qt.QMessageBox(self)
        msg.setIcon(qt.QMessageBox.Critical)
        msg.setWindowTitle(text)
        text += "\n%s" % addedText
        msg.setText(text)
        msg.exec_()

    def getText(self):
        return str(self.lineEdit.text())


class LegendsDockWidget(qt.QDockWidget):
    """QDockWidget with a :class:`LegendSelector` connected to a PlotWindow.

    It makes the link between the LegendListView widget and the PlotWindow.

    :param parent: See :class:`QDockWidget`
    :param plot: :class:`.PlotWindow` instance on which to operate
    """

    def __init__(self, parent=None, plot=None):
        assert plot is not None
        self._plotRef = weakref.ref(plot)
        self._isConnected = False  # True if widget connected to plot signals

        super(LegendsDockWidget, self).__init__("Legends", parent)

        self._legendWidget = LegendListView()

        self.layout().setContentsMargins(0, 0, 0, 0)
        self.setWidget(self._legendWidget)

        self.visibilityChanged.connect(
            self._visibilityChangedHandler)

        self._legendWidget.sigLegendSignal.connect(self._legendSignalHandler)

    @property
    def plot(self):
        """The :class:`.PlotWindow` this widget is attached to."""
        return self._plotRef()

    def renameCurve(self, oldLegend, newLegend):
        """Change the name of a curve using remove and addCurve

        :param str oldLegend: The legend of the curve to be change
        :param str newLegend: The new legend of the curve
        """
        curve = self.plot.getCurve(oldLegend)
        self.plot.remove(oldLegend, kind='curve')
        self.plot.addCurve(curve.getXData(copy=False),
                           curve.getYData(copy=False),
                           legend=newLegend,
                           info=curve.getInfo(),
                           color=curve.getColor(),
                           symbol=curve.getSymbol(),
                           linewidth=curve.getLineWidth(),
                           linestyle=curve.getLineStyle(),
                           xlabel=curve.getXLabel(),
                           ylabel=curve.getYLabel(),
                           xerror=curve.getXErrorData(copy=False),
                           yerror=curve.getYErrorData(copy=False),
                           z=curve.getZValue(),
                           selectable=curve.isSelectable(),
                           fill=curve.isFill(),
                           resetzoom=False)

    def _legendSignalHandler(self, ddict):
        """Handles events from the LegendListView signal"""
        _logger.debug("Legend signal ddict = %s", str(ddict))

        if ddict['event'] == "legendClicked":
            if ddict['button'] == "left":
                self.plot.setActiveCurve(ddict['legend'])

        elif ddict['event'] == "removeCurve":
            self.plot.removeCurve(ddict['legend'])

        elif ddict['event'] == "renameCurve":
            curveList = self.plot.getAllCurves(just_legend=True)
            oldLegend = ddict['legend']
            dialog = RenameCurveDialog(self.plot, oldLegend, curveList)
            ret = dialog.exec_()
            if ret:
                newLegend = dialog.getText()
                self.renameCurve(oldLegend, newLegend)

        elif ddict['event'] == "setActiveCurve":
            self.plot.setActiveCurve(ddict['legend'])

        elif ddict['event'] == "checkBoxClicked":
            self.plot.hideCurve(ddict['legend'], not ddict['selected'])

        elif ddict['event'] in ["mapToRight", "mapToLeft"]:
            legend = ddict['legend']
            curve = self.plot.getCurve(legend)
            yaxis = 'right' if ddict['event'] == 'mapToRight' else 'left'
            self.plot.addCurve(x=curve.getXData(copy=False),
                               y=curve.getYData(copy=False),
                               legend=curve.getLegend(),
                               info=curve.getInfo(),
                               yaxis=yaxis)

        elif ddict['event'] == "togglePoints":
            legend = ddict['legend']
            curve = self.plot.getCurve(legend)
            symbol = ddict['symbol'] if ddict['points'] else ''
            self.plot.addCurve(x=curve.getXData(copy=False),
                               y=curve.getYData(copy=False),
                               legend=curve.getLegend(),
                               info=curve.getInfo(),
                               symbol=symbol)

        elif ddict['event'] == "toggleLine":
            legend = ddict['legend']
            curve = self.plot.getCurve(legend)
            linestyle = ddict['linestyle'] if ddict['line'] else ''
            self.plot.addCurve(x=curve.getXData(copy=False),
                               y=curve.getYData(copy=False),
                               legend=curve.getLegend(),
                               info=curve.getInfo(),
                               linestyle=linestyle)

        else:
            _logger.debug("unhandled event %s", str(ddict['event']))

    def updateLegends(self, *args):
        """Sync the LegendSelector widget displayed info with the plot.
        """
        legendList = []
        for curve in self.plot.getAllCurves(withhidden=True):
            legend = curve.getLegend()
            # Use active color if curve is active
            if legend == self.plot.getActiveCurve(just_legend=True):
                color = qt.QColor(self.plot.getActiveCurveColor())
                isActive = True
            else:
                color = qt.QColor.fromRgbF(*curve.getColor())
                isActive = False

            curveInfo = {
                'color': color,
                'linewidth': curve.getLineWidth(),
                'linestyle': curve.getLineStyle(),
                'symbol': curve.getSymbol(),
                'selected': not self.plot.isCurveHidden(legend),
                'active': isActive}
            legendList.append((legend, curveInfo))

        self._legendWidget.setLegendList(legendList)

    def _visibilityChangedHandler(self, visible):
        if visible:
            self.updateLegends()
            if not self._isConnected:
                self.plot.sigContentChanged.connect(self.updateLegends)
                self.plot.sigActiveCurveChanged.connect(self.updateLegends)
                self._isConnected = True
        else:
            if self._isConnected:
                self.plot.sigContentChanged.disconnect(self.updateLegends)
                self.plot.sigActiveCurveChanged.disconnect(self.updateLegends)
                self._isConnected = False

    def showEvent(self, event):
        """Make sure this widget is raised when it is shown
        (when it is first created as a tab in PlotWindow or when it is shown
        again after hiding).
        """
        self.raise_()
