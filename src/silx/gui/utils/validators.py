# /*##########################################################################
#
# Copyright (c) 2016-2022 European Synchrotron Radiation Facility
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
"""Validator class to write Qt widget."""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "21/06/2020"


from typing import Tuple, Optional
from silx.gui import qt


class CustomValidator:
    """Extra API for a QValidator in order to put more logic inside."""

    def toValue(self, text: str) -> Tuple[object, bool]:
        """Convert the input string into an interpreted value.

        :param text: Input string
        :returns: A tuple containing the resulting object and True if the
            string is valid
        """
        raise NotImplementedError()

    def toText(self, value: object) -> str:
        """Convert an interpreted value into a string representation.

        :param object value: Input object
        """
        raise NotImplementedError()


class DoubleValidator(qt.QDoubleValidator, CustomValidator):
    """
    Double validator with extra feature.

    The default locale used is not the default one. It uses locale C with
    RejectGroupSeparator option. This allows to have consistent rendering of
    double using dot separator without any comma.

    QLocale provides an API to support or not groups on numbers. Unfortunately
    the default Qt QDoubleValidator do not filter out the group character in
    case the locale rejected it. This implementation reject the group character
    from the validation, and remove it from the fixup. Only if the locale is
    defined to reject it.

    This validator also allow to type a dot anywhere in the text. The last dot
    replace the previous one. In this way, it became convenient to fix the
    location of the dot, without complex manual manipulation of the text.
    """
    def __init__(self, parent: qt.QObject=None):
        qt.QDoubleValidator.__init__(self, parent)
        CustomValidator.__init__(self)
        locale = qt.QLocale(qt.QLocale.C)
        locale.setNumberOptions(qt.QLocale.RejectGroupSeparator)
        self.setLocale(locale)

    def validate(self, inputText: str, pos: int):
        """
        Reimplemented from `QDoubleValidator.validate`.

        :param inputText: Text to validate
        :param pos: Position of the cursor
        """
        if pos > 0:
            locale = self.locale()

            # If the typed character is a dot, move the dot instead of ignoring it
            if inputText[pos - 1] == locale.decimalPoint():
                beforeDot = inputText[0:pos].count(locale.decimalPoint())
                pos -= beforeDot
                inputText = inputText.replace(locale.decimalPoint(), "")
                inputText = inputText[0:pos] + locale.decimalPoint() + inputText[pos:]
                pos = pos + 1

            print(pos, inputText, inputText[pos - 1])
            if locale.numberOptions() == qt.QLocale.RejectGroupSeparator:
                if inputText[pos - 1] == locale.groupSeparator():
                    # filter the group separator
                    inputText = inputText[:pos - 1] + inputText[pos:]
                    pos = pos - 1

        # QDoubleValidator is buggy with ".000" when a range is set
        if len(inputText) > 0 and inputText[0] == locale.decimalPoint():
            if self.bottom() != float("-inf") or self.top() != float("inf"):
                result = super(DoubleValidator, self).validate("0" + inputText, pos+1)
                return result[0], inputText, pos

        result = super(DoubleValidator, self).validate(inputText, pos)
        return result

    def fixup(self, inputText: str):
        """
        Remove group characters from the input text if the locale is defined to
        do so.

        :param inputText: Text to validate
        """
        locale = self.locale()
        if locale.numberOptions() == qt.QLocale.RejectGroupSeparator:
            inputText = inputText.replace(locale.groupSeparator(), "")
        return inputText

    def toValue(self, text: str) -> Tuple[float, bool]:
        """Convert the input string into an interpreted value.

        :param text: Input string
        :returns: A tuple containing the resulting object and True if the
            string is valid
        """
        value, validated = self.locale().toDouble(text)
        return value, validated

    def toText(self, value: float) -> str:
        """Convert the input string into an interpreted value

        :param value: Input object
        """
        return str(value)


class AdvancedDoubleValidator(DoubleValidator):
    """
    Validate double values and provides features to allow or disable other
    things.
    """
    def __init__(self, parent=None):
        super(AdvancedDoubleValidator, self).__init__(parent=parent)
        self.__allowEmpty: bool = False
        self.__boundIncluded: Tuple[bool, bool] = (True, True)

    def setAllowEmpty(self, allow: bool):
        """
        Allow the field to be empty. Default is false.

        An empty field is represented as a `None` value.

        :param allow: New state.
        """
        self.__allowEmpty = allow

    def setIncludedBound(self, minBoundIncluded: bool, maxBoundIncluded: bool):
        """
        Allow the include or exclude boundary ranges. Default including both
        boundaries.
        """
        self.__boundIncluded = minBoundIncluded, maxBoundIncluded

    def validate(self, inputText: str, pos: int) -> Tuple[qt.QValidator.State, str, int]:
        """
        Reimplemented from `QDoubleValidator.validate`.

        Allow to provide an empty value.

        :param inputText: Text to validate
        :param pos: Position of the cursor
        """
        if self.__allowEmpty:
            if inputText.strip() == "":
                # python API is not the same as C++ one
                return qt.QValidator.Acceptable, inputText, pos

        acceptable, inputText, pos = super(AdvancedDoubleValidator, self).validate(inputText, pos)

        if acceptable == qt.QValidator.Acceptable:
            # Check boundaries
            if self.__boundIncluded != (True, True):
                _value, isValid = self.toValue(inputText)
                if not isValid:
                    acceptable = qt.QValidator.Intermediate

        return acceptable, inputText, pos

    def toValue(self, text: str) -> Tuple[Optional[float], bool]:
        """Convert the input string into an interpreted value

        :param text: Input string
        :returns: A tuple containing the resulting object and True if the
            string is valid
        """
        if self.__allowEmpty:
            if text.strip() == "":
                return None, True

        value, isValid = super(AdvancedDoubleValidator, self).toValue(text)

        if isValid:
            # Check boundaries
            if self.__boundIncluded != (True, True):
                if not self.__boundIncluded[0]:
                    if value == self.bottom():
                        isValid = False
                if not self.__boundIncluded[1]:
                    if value == self.top():
                        isValid = False

        return value, isValid

    def toText(self, value: Optional[float]) -> str:
        """Convert the input string into an interpreted value

        :param value: Input object
        """
        if self.__allowEmpty:
            if value is None:
                return ""
        return super(AdvancedDoubleValidator, self).toText(value)


class DoublePintValidator(qt.QDoubleValidator, CustomValidator):
    """
    Double validator with extra feature.

    The default locale used is not the default one. It uses locale C with
    RejectGroupSeparator option. This allows to have consistent rendering of
    double using dot separator without any comma.

    QLocale provides an API to support or not groups on numbers. Unfortunately
    the default qt QDoubleValidator do not filter out the group character in
    case the locale rejected it. This implementation reject the group character
    from the validation, and remove it from the fixup. Only if the locale is
    defined to reject it.

    This validator also allow to type a dot anywhere in the text. The last dot
    replace the previous one. In this way, it became convenient to fix the
    location of the dot, without complex manual manipulation of the text.
    """
    def __init__(self, parent: qt.QObject=None):
        qt.QDoubleValidator.__init__(self, parent)
        CustomValidator.__init__(self)
        locale = qt.QLocale(qt.QLocale.C)
        locale.setNumberOptions(qt.QLocale.RejectGroupSeparator)
        self.setLocale(locale)

    def validate(self, inputText: str, pos: int) -> Tuple[qt.QValidator.State, str, int]:
        """
        Reimplemented from `QDoubleValidator.validate`.

        :param inputText: Text to validate
        :param pos: Position of the cursor
        """
        suffix = None
        elements = inputText.rsplit(" ", 1)
        if len(elements) > 1:
            inputText = elements[0]
            suffix = elements[1]
        acceptable, inputText, pos = self.validateDouble(inputText, pos)

        if suffix is not None:
            inputText = inputText + " " + suffix
        return acceptable, inputText, pos

    def validateDouble(self, inputText: str, pos: int) -> Tuple[qt.QValidator.State, str, int]:
        locale = self.locale()

        # If the typed character is a dot, move the dot instead of ignoring it
        if pos > 0:
            if len(inputText) > pos - 1 and inputText[pos - 1] == locale.decimalPoint():
                beforeDot = inputText[0:pos].count(locale.decimalPoint())
                pos -= beforeDot
                inputText = inputText.replace(locale.decimalPoint(), "")
                inputText = inputText[0:pos] + locale.decimalPoint() + inputText[pos:]
                pos = pos + 1

            if locale.decimalPoint() in inputText:
                num = inputText.split(locale.decimalPoint(), 1)
                decimals = min(len(num[1]), self.decimals())
                inputText = num[0] + locale.decimalPoint() + num[1][0:decimals]

            if locale.numberOptions() == qt.QLocale.RejectGroupSeparator:
                if len(inputText) > pos - 1 and inputText[pos - 1] == locale.groupSeparator():
                    # filter the group separator
                    inputText = inputText[:pos - 1] + inputText[pos:]
                    pos = pos - 1

        # QDoubleValidator is buggy with ".000" when a range is set
        if len(inputText) > 0 and inputText[0] == locale.decimalPoint():
            return qt.QValidator.Intermediate, inputText, pos

        acceptable, inputText, pos = super(DoublePintValidator, self).validate(inputText, pos)
        return acceptable, inputText, pos

    def fixup(self, inputText: str) -> str:
        """
        Remove group characters from the input text if the locale is defined to
        do so.

        :param inputText: Text to validate
        """
        locale = self.locale()
        if locale.numberOptions() == qt.QLocale.RejectGroupSeparator:
            inputText = inputText.replace(locale.groupSeparator(), "")

        if len(inputText) > 0 and inputText[0] == locale.decimalPoint():
            inputText = "0" + inputText

        return inputText

    def toValue(self, text: str) -> Tuple[Tuple[float, str], bool]:
        """Convert the input string into an interpreted value

        :param text: Input string
        :returns: A tuple containing the resulting object and True if the
            string is valid
        """
        elements = text.rsplit(" ", 1)
        if len(elements) > 1:
            text = elements[0]
            unit = elements[1]
        else:
            unit = ""
        value, validated = self.locale().toDouble(text)
        return (value, unit), validated

    def toText(self, value: Tuple[float, str]) -> str:
        """Convert the input string into an interpreted value

        :param value: Input object
        """
        return f"{value[0]} {value[1]}"


class AdvancedDoublePintValidator(DoublePintValidator):
    """
    Validate double values and provides features to allow or disable other
    things.
    """
    def __init__(self, parent=None):
        super(AdvancedDoublePintValidator, self).__init__(parent=parent)
        self.__allowEmpty = False
        self.__boundIncluded = True, True

    def setAllowEmpty(self, allow: bool):
        """
        Allow the field to be empty. Default is false.

        An empty field is represented as a `None` value.

        :param allow: New state.
        """
        self.__allowEmpty = allow

    def setIncludedBound(self, minBoundIncluded: bool, maxBoundIncluded: bool):
        """
        Allow the include or exclude boundary ranges. Default including both
        boundaries.
        """
        self.__boundIncluded = minBoundIncluded, maxBoundIncluded

    def validateDouble(self, inputText: str, pos: int) -> Tuple[qt.QValidator.State, str, int]:
        """
        Reimplemented from `QDoubleValidator.validate`.

        Allow to provide an empty value.

        :param inputText: Text to validate
        :param pos: Position of the cursor
        """
        if self.__allowEmpty:
            if inputText.strip() == "":
                # python API is not the same as C++ one
                return qt.QValidator.Acceptable, inputText, pos

        acceptable, resultText, pos = super(AdvancedDoublePintValidator, self).validateDouble(inputText, pos)

        if acceptable == qt.QValidator.Acceptable:
            # Check boundaries
            if self.__boundIncluded != (True, True):
                _value, isValid = self.toValue(resultText)
                if not isValid:
                    acceptable = qt.QValidator.Intermediate

        return acceptable, resultText, pos

    def toValue(self, text: str) -> Tuple[Optional[Tuple[float, str]], bool]:
        """Convert the input string into an interpreted value

        :param text: Input string
        :returns: A tuple containing the resulting object and True if the
            string is valid
        """
        if self.__allowEmpty:
            if text.strip() == "":
                return None, True

        value, isValid = super(AdvancedDoublePintValidator, self).toValue(text)
        (value, unit) = value

        if isValid:
            # Check boundaries
            if self.__boundIncluded != (True, True):
                if not self.__boundIncluded[0]:
                    if value == self.bottom():
                        isValid = False
                if not self.__boundIncluded[1]:
                    if value == self.top():
                        isValid = False

        return (value, unit), isValid

    def toText(self, value: Tuple[Optional[float], str]) -> str:
        """Convert the input string into an interpreted value

        :param value: Input object
        """
        if value is None:
            return ""
        if value[0] is None:
            return f" {value[1]}"
        return f"{value[0]} {value[1]}"
