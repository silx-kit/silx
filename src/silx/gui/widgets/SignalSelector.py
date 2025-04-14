"""This module provides a :class:`SignalSelector` widget.

.. image:: img/SignalSelector.png
   :align: center
"""

from silx.gui import qt


class SignalSelector(qt.QWidget):
    selectionChanged = qt.Signal(int)

    def __init__(self, label_text: str | None = None, parent: qt.QWidget | None = None):
        super().__init__(parent)

        self._layout = qt.QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(2)

        self._label = None
        if label_text:
            self._label = qt.QLabel(label_text)
            self._layout.addWidget(self._label)

        self._combobox = qt.QComboBox()
        self._layout.addWidget(self._combobox)

        self._combobox.currentIndexChanged.connect(self._comboboxSlot)

    def _comboboxSlot(self, index: int) -> None:
        self.selectionChanged.emit(index)

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

    def getSignalIndex(self) -> None:
        return self._combobox.currentIndex()
