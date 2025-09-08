from silx.gui import qt
from silx.gui.qt import inspect as qt_inspect
from silx.gui.plot import PlotWidget


class OverlayMixIn:
    """
    MixIn class for overlay widget.

    For usage examples refer to :class:`WaitingOverlay`, :class:`LabelOverlay` and :class:`ButtonOverlay`

    .. warning:: Any class inheriting from this mixin must also inherit from a QWidget.
    """

    def __init__(
        self,
        parent: qt.QWidget | None = None,
    ):
        self._alignment: qt.Qt.AlignmentFlag = qt.Qt.AlignCenter
        self._alignmentOffsets: tuple[int, int] = (0, 0)
        self._registerParent(parent=parent)

    def getAlignment(self) -> qt.Qt.AlignmentFlag:
        return self._alignment

    def setAlignment(self, alignment: qt.Qt.AlignmentFlag):
        self._alignment = alignment
        self._resize()
        self.update()

    def getAlignmentOffsets(self) -> tuple[int, int]:
        return self._alignmentOffsets

    def setAlignmentOffsets(self, offsets: tuple[int, int]):
        self._alignmentOffsets = offsets
        self._resize()
        self.update()

    def _listenedWidget(self, parent: qt.QWidget) -> qt.QWidget:
        """Returns widget to register event filter to according to parent"""
        if isinstance(parent, PlotWidget):
            return parent.getWidgetHandle()
        return parent

    def _backendChanged(self):
        self._listenedWidget(self.parent()).installEventFilter(self)
        self._resizeLater()

    def _registerParent(self, parent: qt.QWidget | None):
        if parent is None:
            return
        self._listenedWidget(parent).installEventFilter(self)
        if isinstance(parent, PlotWidget):
            parent.sigBackendChanged.connect(self._backendChanged)
        self._resize()

    def _unregisterParent(self, parent: qt.QWidget | None):
        if parent is None:
            return
        if isinstance(parent, PlotWidget):
            parent.sigBackendChanged.disconnect(self._backendChanged)
        self._listenedWidget(parent).removeEventFilter(self)

    def setParent(self, parent: qt.QWidget):
        self._unregisterParent(self.parent())
        super().setParent(parent)
        self._registerParent(parent)

    def _getGeometry(self) -> qt.QRect | None:

        parent = self.parent()
        if parent is None:
            return None

        overlaySize = self.sizeHint()
        if isinstance(parent, PlotWidget):
            offset = parent.getWidgetHandle().mapTo(parent, qt.QPoint(0, 0))
            canvasLeft, canvasTop, canvasWidth, canvasHeight = (
                parent.getPlotBoundsInPixels()
            )
            canvasLeft += offset.x()
            canvasTop += offset.y()
        else:
            canvasWidth = parent.size().width()
            canvasHeight = parent.size().height()
            canvasLeft = 0
            canvasTop = 0

        # calculate left position
        if self._alignment & qt.Qt.AlignTop:
            top = canvasTop
        elif self._alignment & qt.Qt.AlignBottom:
            top = canvasTop + canvasHeight - overlaySize.height()
        else:
            top = canvasTop + (canvasHeight - overlaySize.height()) / 2

        # calculate top position
        if self._alignment & qt.Qt.AlignLeft:
            left = canvasLeft
        elif self._alignment & qt.Qt.AlignRight:
            left = canvasLeft + canvasWidth - overlaySize.width()
        else:
            left = canvasLeft + (canvasWidth - overlaySize.width()) / 2

        topLeft = qt.QPoint(
            int(left + self._alignmentOffsets[0]),
            int(top + self._alignmentOffsets[1]),
        )
        return qt.QRect(
            topLeft,
            overlaySize,
        )

    def _resize(self):
        if not qt_inspect.isValid(self):
            return  # For _resizeLater in case the widget has been deleted

        rect = self._getGeometry()
        if rect is None:
            return

        self.setGeometry(rect)
        self.raise_()

    def _resizeLater(self):
        qt.QTimer.singleShot(0, self._resize)

    def eventFilter(self, watched: qt.QWidget, event: qt.QEvent):
        if event.type() == qt.QEvent.Resize:
            self._resize()
            self._resizeLater()  # Defer resize for the receiver to have handled it
        return super().eventFilter(watched, event)
