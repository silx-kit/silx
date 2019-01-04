# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2004-2018 European Synchrotron Radiation Facility
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
"""This module implements a print preview dialog.

The dialog provides methods to send images, pixmaps and SVG
items to the page to be printed.

The user can interactively move and resize the items.
"""
import sys
import logging
from silx.gui import qt, printer


__authors__ = ["V.A. Sole", "P. Knobel"]
__license__ = "MIT"
__date__ = "11/07/2017"


_logger = logging.getLogger(__name__)


class PrintPreviewDialog(qt.QDialog):
    """Print preview dialog widget.
    """
    def __init__(self, parent=None, printer=None):

        qt.QDialog.__init__(self, parent)
        self.setWindowTitle("Print Preview")
        self.setModal(False)
        self.resize(400, 500)

        self.mainLayout = qt.QVBoxLayout(self)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.mainLayout.setSpacing(0)

        self._buildToolbar()

        self.printer = printer
        # :class:`QPrinter` (paint device that paints on a printer).
        # :meth:`showEvent` has been reimplemented to enforce printer
        # setup.

        self.printDialog = None
        # :class:`QPrintDialog` (dialog for specifying the printer's
        # configuration)

        self.scene = None
        # :class:`QGraphicsScene` (surface for managing
        # 2D graphical items)

        self.page = None
        # :class:`QGraphicsRectItem` used as white background page on which
        # to display the print preview.

        self.view = None
        # :class:`QGraphicsView` widget for displaying :attr:`scene`

        self._svgItems = []
        # List storing :class:`QSvgRenderer` items to be printed, added in
        # :meth:`addSvgItem`, cleared in :meth:`_clearAll`.
        # This ensures that there is a reference pointing to the items,
        # which ensures they are not destroyed before being printed.

        self._viewScale = 1.0
        # Zoom level (1.0 is 100%)

        self._toBeCleared = False
        # Flag indicating that all items must be removed from :attr:`scene`
        # and from :attr:`_svgItems`.
        # Set to True after a successful printing. The widget is then hidden,
        # and it will be cleared the next time it is shown.
        # Reset to False after :meth:`_clearAll` has done its job.

    def _buildToolbar(self):
        toolBar = qt.QWidget(self)
        # a layout for the toolbar
        toolsLayout = qt.QHBoxLayout(toolBar)
        toolsLayout.setContentsMargins(0, 0, 0, 0)
        toolsLayout.setSpacing(0)

        hideBut = qt.QPushButton("Hide", toolBar)
        hideBut.setToolTip("Hide print preview dialog")
        hideBut.clicked.connect(self.hide)

        cancelBut = qt.QPushButton("Clear All", toolBar)
        cancelBut.setToolTip("Remove all items")
        cancelBut.clicked.connect(self._clearAll)

        removeBut = qt.QPushButton("Remove",
                                   toolBar)
        removeBut.setToolTip("Remove selected item (use left click to select)")
        removeBut.clicked.connect(self._remove)

        setupBut = qt.QPushButton("Setup", toolBar)
        setupBut.setToolTip("Select and configure a printer")
        setupBut.clicked.connect(self.setup)

        printBut = qt.QPushButton("Print", toolBar)
        printBut.setToolTip("Print page and close print preview")
        printBut.clicked.connect(self._print)

        zoomPlusBut = qt.QPushButton("Zoom +", toolBar)
        zoomPlusBut.clicked.connect(self._zoomPlus)

        zoomMinusBut = qt.QPushButton("Zoom -", toolBar)
        zoomMinusBut.clicked.connect(self._zoomMinus)

        toolsLayout.addWidget(hideBut)
        toolsLayout.addWidget(printBut)
        toolsLayout.addWidget(cancelBut)
        toolsLayout.addWidget(removeBut)
        toolsLayout.addWidget(setupBut)
        # toolsLayout.addStretch()
        # toolsLayout.addWidget(marginLabel)
        # toolsLayout.addWidget(self.marginSpin)
        toolsLayout.addStretch()
        # toolsLayout.addWidget(scaleLabel)
        # toolsLayout.addWidget(self.scaleCombo)
        toolsLayout.addWidget(zoomPlusBut)
        toolsLayout.addWidget(zoomMinusBut)
        # toolsLayout.addStretch()
        self.toolBar = toolBar
        self.mainLayout.addWidget(self.toolBar)

    def _buildStatusBar(self):
        """Create the status bar used to display the printer name
        or output file name."""
        # status bar
        statusBar = qt.QStatusBar(self)
        self.targetLabel = qt.QLabel(statusBar)
        self._updateTargetLabel()
        statusBar.addWidget(self.targetLabel)
        self.mainLayout.addWidget(statusBar)

    def _updateTargetLabel(self):
        """Update printer name or file name shown in the status bar."""
        if self.printer is None:
            self.targetLabel.setText("Undefined printer")
            return
        if self.printer.outputFileName():
            self.targetLabel.setText("File:" +
                                     self.printer.outputFileName())
        else:
            self.targetLabel.setText("Printer:" +
                                     self.printer.printerName())

    def _updatePrinter(self):
        """Resize :attr:`page`, :attr:`scene` and :attr:`view` to :attr:`printer`
        width and height."""
        printer = self.printer
        assert printer is not None, \
            "_updatePrinter should not be called unless a printer is defined"
        if self.scene is None:
            self.scene = qt.QGraphicsScene()
            self.scene.setBackgroundBrush(qt.QColor(qt.Qt.lightGray))
            self.scene.setSceneRect(qt.QRectF(0, 0, printer.width(), printer.height()))

        if self.page is None:
            self.page = qt.QGraphicsRectItem(0, 0, printer.width(), printer.height())
            self.page.setBrush(qt.QColor(qt.Qt.white))
            self.scene.addItem(self.page)

        self.scene.setSceneRect(qt.QRectF(0, 0, printer.width(), printer.height()))
        self.page.setPos(qt.QPointF(0.0, 0.0))
        self.page.setRect(qt.QRectF(0, 0, printer.width(), printer.height()))

        if self.view is None:
            self.view = qt.QGraphicsView(self.scene)
            self.mainLayout.addWidget(self.view)
            self._buildStatusBar()
        # self.view.scale(1./self._viewScale, 1./self._viewScale)
        self.view.fitInView(self.page.rect(), qt.Qt.KeepAspectRatio)
        self._viewScale = 1.00
        self._updateTargetLabel()

    # Public methods
    def addImage(self, image, title=None, comment=None, commentPosition=None):
        """Add an image to the print preview scene.

        :param QImage image: Image to be added to the scene
        :param str title: Title shown above (centered) the image
        :param str comment: Comment displayed below the image
        :param commentPosition: "CENTER" or "LEFT"
        """
        self.addPixmap(qt.QPixmap.fromImage(image),
                       title=title, comment=comment,
                       commentPosition=commentPosition)

    def addPixmap(self, pixmap, title=None, comment=None, commentPosition=None):
        """Add a pixmap to the print preview scene

        :param QPixmap pixmap: Pixmap to be added to the scene
        :param str title: Title shown above (centered) the pixmap
        :param str comment: Comment displayed below the pixmap
        :param commentPosition: "CENTER" or "LEFT"
        """
        if self._toBeCleared:
            self._clearAll()
        self.ensurePrinterIsSet()
        if self.printer is None:
            _logger.error("printer is not set, cannot add pixmap to page")
            return
        if title is None:
            title = ' ' * 88
        if comment is None:
            comment = ' ' * 88
        if commentPosition is None:
            commentPosition = "CENTER"
        if qt.qVersion() < "5.0":
            rectItem = qt.QGraphicsRectItem(self.page, self.scene)
        else:
            rectItem = qt.QGraphicsRectItem(self.page)

        rectItem.setRect(qt.QRectF(1, 1,
                                   pixmap.width(), pixmap.height()))

        pen = rectItem.pen()
        color = qt.QColor(qt.Qt.red)
        color.setAlpha(1)
        pen.setColor(color)
        rectItem.setPen(pen)
        rectItem.setZValue(1)
        rectItem.setFlag(qt.QGraphicsItem.ItemIsSelectable, True)
        rectItem.setFlag(qt.QGraphicsItem.ItemIsMovable, True)
        rectItem.setFlag(qt.QGraphicsItem.ItemIsFocusable, False)

        rectItemResizeRect = _GraphicsResizeRectItem(rectItem, self.scene)
        rectItemResizeRect.setZValue(2)

        if qt.qVersion() < "5.0":
            pixmapItem = qt.QGraphicsPixmapItem(rectItem, self.scene)
        else:
            pixmapItem = qt.QGraphicsPixmapItem(rectItem)
        pixmapItem.setPixmap(pixmap)
        pixmapItem.setZValue(0)

        # I add the title
        if qt.qVersion() < "5.0":
            textItem = qt.QGraphicsTextItem(title, rectItem, self.scene)
        else:
            textItem = qt.QGraphicsTextItem(title, rectItem)
        textItem.setTextInteractionFlags(qt.Qt.TextEditorInteraction)
        offset = 0.5 * textItem.boundingRect().width()
        textItem.moveBy(0.5 * pixmap.width() - offset, -20)
        textItem.setZValue(2)

        # I add the comment
        if qt.qVersion() < "5.0":
            commentItem = qt.QGraphicsTextItem(comment, rectItem, self.scene)
        else:
            commentItem = qt.QGraphicsTextItem(comment, rectItem)
        commentItem.setTextInteractionFlags(qt.Qt.TextEditorInteraction)
        offset = 0.5 * commentItem.boundingRect().width()
        if commentPosition.upper() == "LEFT":
            x = 1
        else:
            x = 0.5 * pixmap.width() - offset
        commentItem.moveBy(x, pixmap.height() + 20)
        commentItem.setZValue(2)

        rectItem.moveBy(20, 40)

    def addSvgItem(self, item, title=None,
                   comment=None, commentPosition=None,
                   viewBox=None, keepRatio=True):
        """Add a SVG item to the scene.

        :param QSvgRenderer item: SVG item to be added to the scene.
        :param str title: Title shown above (centered) the SVG item.
        :param str comment: Comment displayed below the SVG item.
        :param str commentPosition: "CENTER" or "LEFT"
        :param QRectF viewBox: Bounding box for the item on the print page
            (xOffset, yOffset, width, height). If None, use original
            item size.
        :param bool keepRatio: If True, resizing the item will preserve its
            original aspect ratio.
        """
        if not qt.HAS_SVG:
            raise RuntimeError("Missing QtSvg library.")
        if not isinstance(item, qt.QSvgRenderer):
            raise TypeError("addSvgItem: QSvgRenderer expected")
        if self._toBeCleared:
            self._clearAll()
        self.ensurePrinterIsSet()
        if self.printer is None:
            _logger.error("printer is not set, cannot add SvgItem to page")
            return

        if title is None:
            title = 50 * ' '
        if comment is None:
            comment = 80 * ' '
        if commentPosition is None:
            commentPosition = "CENTER"

        if viewBox is None:
            if hasattr(item, "_viewBox"):
                # PyMca compatibility: viewbox attached to item
                viewBox = item._viewBox
            else:
                # try the original item viewbox
                viewBox = item.viewBoxF()

        svgItem = _GraphicsSvgRectItem(viewBox, self.page)
        svgItem.setSvgRenderer(item)

        svgItem.setCacheMode(qt.QGraphicsItem.NoCache)
        svgItem.setZValue(0)
        svgItem.setFlag(qt.QGraphicsItem.ItemIsSelectable, True)
        svgItem.setFlag(qt.QGraphicsItem.ItemIsMovable, True)
        svgItem.setFlag(qt.QGraphicsItem.ItemIsFocusable, False)

        rectItemResizeRect = _GraphicsResizeRectItem(svgItem, self.scene,
                                                     keepratio=keepRatio)
        rectItemResizeRect.setZValue(2)

        self._svgItems.append(item)

        # Comment / legend
        dummyComment = 80 * "1"
        if qt.qVersion() < '5.0':
            commentItem = qt.QGraphicsTextItem(dummyComment, svgItem, self.scene)
        else:
            commentItem = qt.QGraphicsTextItem(dummyComment, svgItem)
        commentItem.setTextInteractionFlags(qt.Qt.TextEditorInteraction)
        # we scale the text to have the legend  box have the same width as the graph
        scaleCalculationRect = qt.QRectF(commentItem.boundingRect())
        scale = svgItem.boundingRect().width() / scaleCalculationRect.width()

        commentItem.setPlainText(comment)
        commentItem.setZValue(1)

        commentItem.setFlag(qt.QGraphicsItem.ItemIsMovable, True)
        if qt.qVersion() < "5.0":
            commentItem.scale(scale, scale)
        else:
            commentItem.setScale(scale)

        # align
        if commentPosition.upper() == "CENTER":
            alignment = qt.Qt.AlignCenter
        elif commentPosition.upper() == "RIGHT":
            alignment = qt.Qt.AlignRight
        else:
            alignment = qt.Qt.AlignLeft
        commentItem.setTextWidth(commentItem.boundingRect().width())
        center_format = qt.QTextBlockFormat()
        center_format.setAlignment(alignment)
        cursor = commentItem.textCursor()
        cursor.select(qt.QTextCursor.Document)
        cursor.mergeBlockFormat(center_format)
        cursor.clearSelection()
        commentItem.setTextCursor(cursor)
        if alignment == qt.Qt.AlignLeft:
            deltax = 0
        else:
            deltax = (svgItem.boundingRect().width() - commentItem.boundingRect().width()) / 2.
        commentItem.moveBy(svgItem.boundingRect().x() + deltax,
                           svgItem.boundingRect().y() + svgItem.boundingRect().height())

        # Title
        if qt.qVersion() < '5.0':
            textItem = qt.QGraphicsTextItem(title, svgItem, self.scene)
        else:
            textItem = qt.QGraphicsTextItem(title, svgItem)
        textItem.setTextInteractionFlags(qt.Qt.TextEditorInteraction)
        textItem.setZValue(1)
        textItem.setFlag(qt.QGraphicsItem.ItemIsMovable, True)

        title_offset = 0.5 * textItem.boundingRect().width()
        textItem.moveBy(svgItem.boundingRect().x() +
                        0.5 * svgItem.boundingRect().width() - title_offset * scale,
                        svgItem.boundingRect().y())
        if qt.qVersion() < "5.0":
            textItem.scale(scale, scale)
        else:
            textItem.setScale(scale)

    def setup(self):
        """Open a print dialog to ensure the :attr:`printer` is set.

        If the setting fails or is cancelled, :attr:`printer` is reset to
        *None*.
        """
        if self.printer is None:
            self.printer = printer.getDefaultPrinter()
        if self.printDialog is None:
            self.printDialog = qt.QPrintDialog(self.printer, self)
        if self.printDialog.exec_():
            if self.printer.width() <= 0 or self.printer.height() <= 0:
                self.message = qt.QMessageBox(self)
                self.message.setIcon(qt.QMessageBox.Critical)
                self.message.setText("Unknown library error \non printer initialization")
                self.message.setWindowTitle("Library Error")
                self.message.setModal(0)
                self.printer = None
                return
            self.printer.setFullPage(True)
            self._updatePrinter()
        else:
            # printer setup cancelled, check for a possible previous configuration
            if self.page is None:
                # not initialized
                self.printer = None

    def ensurePrinterIsSet(self):
        """If the printer is not already set, try to interactively
        setup the printer using a QPrintDialog.
        In case of failure, hide widget and log a warning.

        :return: True if printer was set. False if it failed or if the
            selection dialog was canceled.
        """
        if self.printer is None:
            self.setup()
        if self.printer is None:
            self.hide()
            _logger.warning("Printer setup failed or was cancelled, " +
                            "but printer is required.")
        return self.printer is not None

    def setOutputFileName(self, name):
        """Set output filename.

        Setting a non-empty name enables printing to file.

        :param str name: File name (path)"""
        self.printer.setOutputFileName(name)

    # overloaded methods
    def exec_(self):
        if self._toBeCleared:
            self._clearAll()
        return qt.QDialog.exec_(self)

    def raise_(self):
        if self._toBeCleared:
            self._clearAll()
        return qt.QDialog.raise_(self)

    def showEvent(self, event):
        """Reimplemented to force printer setup.
        In case of failure, hide the widget."""
        if self._toBeCleared:
            self._clearAll()
        self.ensurePrinterIsSet()

        return super(PrintPreviewDialog, self).showEvent(event)

    # button callbacks
    def _print(self):
        """Do the printing, hide the print preview dialog,
        set :attr:`_toBeCleared` flag to True to trigger clearing the
        next time the dialog is shown.

        If the printer is not setup, do it first."""
        printer = self.printer

        painter = qt.QPainter()
        if not painter.begin(printer) or printer is None:
            _logger.error("Cannot initialize printer")
            return
        try:
            self.scene.render(painter, qt.QRectF(0, 0, printer.width(), printer.height()),
                              qt.QRectF(self.page.rect().x(), self.page.rect().y(),
                                        self.page.rect().width(), self.page.rect().height()),
                              qt.Qt.KeepAspectRatio)
            painter.end()
            self.hide()
            self.accept()
            self._toBeCleared = True
        except:              # FIXME
            painter.end()
            qt.QMessageBox.critical(self, "ERROR",
                                    'Printing problem:\n %s' % sys.exc_info()[1])
            _logger.error('printing problem:\n %s' % sys.exc_info()[1])
            return

    def _zoomPlus(self):
        self._viewScale *= 1.20
        self.view.scale(1.20, 1.20)

    def _zoomMinus(self):
        self._viewScale *= 0.80
        self.view.scale(0.80, 0.80)

    def _clearAll(self):
        """
        Clear the print preview window, remove all items
        but keep the page.
        """
        itemlist = self.scene.items()
        keep = self.page
        while len(itemlist) != 1:
            if itemlist.index(keep) == 0:
                self.scene.removeItem(itemlist[1])
            else:
                self.scene.removeItem(itemlist[0])
            itemlist = self.scene.items()
        self._svgItems = []
        self._toBeCleared = False

    def _remove(self):
        """Remove selected item in :attr:`scene`.
        """
        itemlist = self.scene.items()

        # this loop is not efficient if there are many items ...
        for item in itemlist:
            if item.isSelected():
                self.scene.removeItem(item)


class SingletonPrintPreviewDialog(PrintPreviewDialog):
    """Singleton print preview dialog.

    All widgets in a program that instantiate this class will share
    a single print preview dialog. This enables sending
    multiple images to a single page to be printed.
    """
    _instance = None

    def __new__(self, *var, **kw):
        if self._instance is None:
            self._instance = PrintPreviewDialog(*var, **kw)
        return self._instance


class _GraphicsSvgRectItem(qt.QGraphicsRectItem):
    """:class:`qt.QGraphicsRectItem` with an attached
    :class:`qt.QSvgRenderer`, and with a painter redefined to render
    the SVG item."""
    def setSvgRenderer(self, renderer):
        """

        :param QSvgRenderer renderer: svg renderer
        """
        self._renderer = renderer

    def paint(self, painter, *var, **kw):
        self._renderer.render(painter, self.boundingRect())


class _GraphicsResizeRectItem(qt.QGraphicsRectItem):
    """Resizable QGraphicsRectItem."""
    def __init__(self, parent=None, scene=None, keepratio=True):
        if qt.qVersion() < '5.0':
            qt.QGraphicsRectItem.__init__(self, parent, scene)
        else:
            qt.QGraphicsRectItem.__init__(self, parent)
        rect = parent.boundingRect()
        x = rect.x()
        y = rect.y()
        w = rect.width()
        h = rect.height()
        self._newRect = None
        self.keepRatio = keepratio
        self.setRect(qt.QRectF(x + w - 40, y + h - 40, 40, 40))
        self.setAcceptHoverEvents(True)
        pen = qt.QPen()
        color = qt.QColor(qt.Qt.white)
        color.setAlpha(0)
        pen.setColor(color)
        pen.setStyle(qt.Qt.NoPen)
        self.setPen(pen)
        self.setBrush(color)
        self.setFlag(self.ItemIsMovable, True)
        self.show()

    def hoverEnterEvent(self, event):
        if self.parentItem().isSelected():
            self.parentItem().setSelected(False)
        if self.keepRatio:
            self.setCursor(qt.QCursor(qt.Qt.SizeFDiagCursor))
        else:
            self.setCursor(qt.QCursor(qt.Qt.SizeAllCursor))
        self.setBrush(qt.QBrush(qt.Qt.yellow, qt.Qt.SolidPattern))
        return qt.QGraphicsRectItem.hoverEnterEvent(self, event)

    def hoverLeaveEvent(self, event):
        self.setCursor(qt.QCursor(qt.Qt.ArrowCursor))
        pen = qt.QPen()
        color = qt.QColor(qt.Qt.white)
        color.setAlpha(0)
        pen.setColor(color)
        pen.setStyle(qt.Qt.NoPen)
        self.setPen(pen)
        self.setBrush(color)
        return qt.QGraphicsRectItem.hoverLeaveEvent(self, event)

    def mousePressEvent(self, event):
        if self._newRect is not None:
            self._newRect = None
        self._point0 = self.pos()
        parent = self.parentItem()
        scene = self.scene()
        # following line prevents dragging along the previously selected
        # item when resizing another one
        scene.clearSelection()

        rect = parent.boundingRect()
        self._x = rect.x()
        self._y = rect.y()
        self._w = rect.width()
        self._h = rect.height()
        self._ratio = self._w / self._h
        if qt.qVersion() < "5.0":
            self._newRect = qt.QGraphicsRectItem(parent, scene)
        else:
            self._newRect = qt.QGraphicsRectItem(parent)
        self._newRect.setRect(qt.QRectF(self._x,
                                        self._y,
                                        self._w,
                                        self._h))
        qt.QGraphicsRectItem.mousePressEvent(self, event)

    def mouseMoveEvent(self, event):
        point1 = self.pos()
        deltax = point1.x() - self._point0.x()
        deltay = point1.y() - self._point0.y()
        if self.keepRatio:
            r1 = (self._w + deltax) / self._w
            r2 = (self._h + deltay) / self._h
            if r1 < r2:
                self._newRect.setRect(qt.QRectF(self._x,
                                                self._y,
                                                self._w + deltax,
                                                (self._w + deltax) / self._ratio))
            else:
                self._newRect.setRect(qt.QRectF(self._x,
                                                self._y,
                                                (self._h + deltay) * self._ratio,
                                                self._h + deltay))
        else:
            self._newRect.setRect(qt.QRectF(self._x,
                                            self._y,
                                            self._w + deltax,
                                            self._h + deltay))
        qt.QGraphicsRectItem.mouseMoveEvent(self, event)

    def mouseReleaseEvent(self, event):
        point1 = self.pos()
        deltax = point1.x() - self._point0.x()
        deltay = point1.y() - self._point0.y()
        self.moveBy(-deltax, -deltay)
        parent = self.parentItem()

        # deduce scale from rectangle
        if (qt.qVersion() < "5.0") or self.keepRatio:
            scalex = self._newRect.rect().width() / self._w
            scaley = scalex
        else:
            scalex = self._newRect.rect().width() / self._w
            scaley = self._newRect.rect().height() / self._h

        if qt.qVersion() < "5.0":
            parent.scale(scalex, scaley)
        else:
            # apply the scale to the previous transformation matrix
            previousTransform = parent.transform()
            parent.setTransform(
                    previousTransform.scale(scalex, scaley))

        self.scene().removeItem(self._newRect)
        self._newRect = None
        qt.QGraphicsRectItem.mouseReleaseEvent(self, event)


def main():
    """
    """
    if len(sys.argv) < 2:
        print("give an image file as parameter please.")
        sys.exit(1)

    if len(sys.argv) > 2:
        print("only one parameter please.")
        sys.exit(1)

    filename = sys.argv[1]
    w = PrintPreviewDialog()
    w.resize(400, 500)

    comment = ""
    for i in range(20):
        comment += "Line number %d: En un lugar de La Mancha de cuyo nombre ...\n" % i

    if filename[-3:] == "svg":
        item = qt.QSvgRenderer(filename, w.page)
        w.addSvgItem(item, title=filename,
                     comment=comment, commentPosition="CENTER")
    else:
        w.addPixmap(qt.QPixmap.fromImage(qt.QImage(filename)),
                    title=filename,
                    comment=comment,
                    commentPosition="CENTER")
        w.addImage(qt.QImage(filename), comment=comment, commentPosition="LEFT")

    sys.exit(w.exec_())


if __name__ == '__main__':
    a = qt.QApplication(sys.argv)
    main()
    a.exec_()
