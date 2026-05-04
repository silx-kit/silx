import numpy

from silx.gui import qt


class IntEdit(qt.QLineEdit):
    """QLineEdit for integers with a default value and update on validation.

    :param QWidget parent:
    """

    sigValueChanged = qt.Signal(int)
    """Signal emitted when the value has changed (on editing finished)"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.__value = None
        self.setAlignment(qt.Qt.AlignRight)
        validator = qt.QIntValidator()
        self.setValidator(validator)
        validator.bottomChanged.connect(self.__updateSize)
        validator.topChanged.connect(self.__updateSize)
        self.__updateSize()

        self.textEdited.connect(self.__textEdited)

    def __updateSize(self, *args):
        """Update widget's maximum size according to bounds"""
        bottom, top = self.getRange()
        nbchar = max(len(str(bottom)), len(str(top)))
        font = self.font()
        font.setStyle(qt.QFont.StyleItalic)
        fontMetrics = qt.QFontMetrics(font)
        self.setMaximumWidth(fontMetrics.boundingRect("0" * (nbchar + 1)).width())
        self.setMaxLength(nbchar)

    def __textEdited(self, _):
        if self.font().style() != qt.QFont.StyleItalic:
            font = self.font()
            font.setStyle(qt.QFont.StyleItalic)
            self.setFont(font)

    # Use events rather than editingFinished to also trigger with empty text

    def focusOutEvent(self, event):
        self.__commitValue()
        return super().focusOutEvent(event)

    def keyPressEvent(self, event):
        if event.key() in (qt.Qt.Key_Enter, qt.Qt.Key_Return):
            self.__commitValue()
        return super().keyPressEvent(event)

    def __commitValue(self):
        """Update the value returned by :meth:`getValue`"""
        value = self.getCurrentValue()
        if value is None:
            value = self.getDefaultValue()
            if value is None:
                return  # No value, keep previous one

        if self.font().style() != qt.QFont.StyleNormal:
            font = self.font()
            font.setStyle(qt.QFont.StyleNormal)
            self.setFont(font)

        if value != self.__value:
            self.__value = value
            self.sigValueChanged.emit(value)

    def getValue(self) -> int | None:
        """Return current value (None if never set)."""
        return self.__value

    def setRange(self, bottom: int, top: int):
        """Set the range of valid values"""
        self.validator().setRange(bottom, top)

    def getRange(self) -> tuple[int, int]:
        """Returns the current range of valid values

        :returns: (bottom, top)
        """
        return self.validator().bottom(), self.validator().top()

    def __validate(self, value: int, extend_range: bool):
        """Ensure value is in range

        :param int value:
        :param bool extend_range:
            True to extend range if needed.
            False to clip value if needed.
        """
        if extend_range:
            bottom, top = self.getRange()
            self.setRange(min(value, bottom), max(value, top))
        return numpy.clip(value, *self.getRange())

    def setDefaultValue(self, value: int, extend_range: bool = False):
        """Set default value when QLineEdit is empty

        :param int value:
        :param bool extend_range:
            True to extend range if needed.
            False to clip value if needed
        """
        self.setPlaceholderText(str(self.__validate(value, extend_range)))
        if self.getCurrentValue() is None:
            self.__commitValue()

    def getDefaultValue(self) -> int | None:
        """Return the default value or the bottom one if not set"""
        try:
            return int(self.placeholderText())
        except ValueError:
            return None

    def setCurrentValue(self, value: int, extend_range: bool = False):
        """Set the currently displayed value

        :param int value:
        :param bool extend_range:
            True to extend range if needed.
            False to clip value if needed
        """
        self.setText(str(self.__validate(value, extend_range)))
        self.__commitValue()

    def getCurrentValue(self) -> int | None:
        """Returns the displayed value or None if not correct"""
        try:
            return int(self.text())
        except ValueError:
            return None
