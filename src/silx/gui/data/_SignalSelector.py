"""This module provides a :class:`SignalSelector` widget."""

from silx.gui import qt


class SignalSelector(qt.QWidget):
    selectionChanged = qt.Signal(int)

    def __init__(self, parent: qt.QWidget | None = None):
        super().__init__(parent)

        self._layout = qt.QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(2)

        self._label = qt.QLabel("Select signal:")
        self._layout.addWidget(self._label)

        self._combobox = qt.QComboBox()
        self._layout.addWidget(self._combobox)
        self._layout.addStretch(1)

        self._combobox.currentIndexChanged.connect(self.selectionChanged)

    def setSignalNames(self, names: list[str]) -> None:
        self._combobox.clear()
        self._combobox.addItems(names)

    def getSignalNames(self) -> list[str]:
        return [self._combobox.itemText(i) for i in range(self._combobox.count())]

    def setSignalName(self, index, name) -> None:
        if 0 <= index < self._combobox.count():
            self._combobox.setItemText(index, name)

    def getSignalName(self, index) -> str | None:
        if 0 <= index < self._combobox.count():
            return self._combobox.itemText(index)
        return None

    def setSignalIndex(self, index) -> None:
        self._combobox.setCurrentIndex(index)

    def getSignalIndex(self) -> int:
        return self._combobox.currentIndex()
