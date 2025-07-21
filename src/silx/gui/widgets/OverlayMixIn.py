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
        parent,
        alignment: qt.Qt.AlignmentFlag = qt.Qt.AlignCenter,
        alignment_offsets: tuple[int, int] = (0, 0),
    ):
        """
        :param parent: parent widget
        :param alignment: alignment of the overlay.
        :param alignment_offsets: alignment offset as (horizontal offset, vertical offset). Values can be positive or negative. It will offset the alignment of this value
        """
        self._alignment: qt.Qt.AlignmentFlag = alignment
        self._alignment_offsets: tuple[int, int] = alignment_offsets
        self._registerParent(parent=parent)

    def getAlignment(self) -> qt.Qt.AlignmentFlag:
        return self._alignment

    def setAlignment(self, alignment: qt.Qt.AlignmentFlag):
        self._alignment = alignment
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
        """Return the top left corner of the geometry to set up the geometry"""

        parent = self.parent()
        if parent is None:
            return None

        overlay_size: qt.QSize = self.sizeHint()
        if isinstance(parent, PlotWidget):
            offset = parent.getWidgetHandle().mapTo(parent, qt.QPoint(0, 0))
            canvas_left, canvas_top, canvas_width, canvas_height = (
                parent.getPlotBoundsInPixels()
            )
            canvas_left += offset.x()
            canvas_top += offset.y()
        else:
            canvas_width = parent.size().width()
            canvas_height = parent.size().height()
            canvas_left = 0
            canvas_top = 0

        # calculate left position
        if self._alignment & qt.Qt.AlignTop:
            top = canvas_top
        elif self._alignment & qt.Qt.AlignBottom:
            top = canvas_top + canvas_height - overlay_size.height()
        else:
            top = canvas_top + (canvas_height - overlay_size.height()) / 2

        # calculate top position
        if self._alignment & qt.Qt.AlignLeft:
            left = canvas_left
        elif self._alignment & qt.Qt.AlignRight:
            left = canvas_left + canvas_width - overlay_size.width()
        else:
            left = canvas_left + (canvas_width - overlay_size.width()) / 2

        topLeft = qt.QPoint(
            int(left + self._alignment_offsets[0]),
            int(top + self._alignment_offsets[1]),
        )
        return qt.QRect(
            topLeft,
            overlay_size,
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
