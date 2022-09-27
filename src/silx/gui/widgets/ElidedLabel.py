# /*##########################################################################
#
# Copyright (c) 2004-2021 European Synchrotron Radiation Facility
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
"""Module contains an elidable label
"""

__license__ = "MIT"
__date__ = "07/12/2018"

from ...utils.deprecation import deprecated
from silx.gui import qt


class ElidedLabel(qt.QLabel):
    """QLabel with an elide property.

    By default if the text is too long, it is elided on the right.

    This mode can be changed with :func:`setElideMode`.

    In this case the text is elided, the full content is displayed as part of the
    tool tip. This behavior can be disabled with :func:`setTextAsToolTip`.
    """

    def __init__(self, parent=None):
        super(ElidedLabel, self).__init__(parent)
        self.__text = ""
        self.__toolTip = ""
        self.__textAsToolTip = True
        self.__textIsElided = False
        self.__elideMode = qt.Qt.ElideRight
        self.__updateMinimumSize()

    def resizeEvent(self, event):
        self.__updateText()
        return qt.QLabel.resizeEvent(self, event)

    def setFont(self, font):
        qt.QLabel.setFont(self, font)
        self.__updateMinimumSize()
        self.__updateText()

    def __updateMinimumSize(self):
        metrics = self.fontMetrics()
        if qt.BINDING in ('PySide2', 'PyQt5'):
            width = metrics.width("...")
        else:  # Qt6
            width = metrics.horizontalAdvance("...")
        self.setMinimumWidth(width)

    def __updateText(self):
        metrics = self.fontMetrics()
        elidedText = metrics.elidedText(self.__text, self.__elideMode, self.width())
        qt.QLabel.setText(self, elidedText)
        wasElided = self.__textIsElided
        self.__textIsElided = elidedText != self.__text
        if self.__textIsElided or wasElided != self.__textIsElided:
            self.__updateToolTip()

    def __updateToolTip(self):
        if self.__textIsElided and self.__textAsToolTip:
            qt.QLabel.setToolTip(self, self.__text + "<br/>" + self.__toolTip)
        else:
            qt.QLabel.setToolTip(self, self.__toolTip)

    # Inherited properties

    def text(self):
        """Returns the text defined by the user.

        It can be different from the one really displayed, depending on the
        `elideMode` defined for this widget.
        """
        return self.__text

    @deprecated(replacement='text', since_version='1.1.0')
    def getText(self):
        return self.text()

    def setText(self, text):
        self.__text = text
        self.__updateText()

    def toolTip(self):
        """Returns the tooltip defined by the user.

        It can be different from the one really displayed, if `textAsToolTip` was
        set to true.
        """
        return self.__toolTip

    @deprecated(replacement='toolTip', since_version='1.1.0')
    def getToolTip(self):
        return self.toolTip()

    def setToolTip(self, toolTip):
        self.__toolTip = toolTip
        self.__updateToolTip()

    # New properties

    def setElideMode(self, elideMode):
        """Set the elide mode.

        :param qt.Qt.TextElideMode elidMode: Elide mode to use
        """
        self.__elideMode = elideMode
        self.__updateText()

    def getElideMode(self):
        """Returns the used elide mode.

        :rtype: qt.Qt.TextElideMode
        """
        return self.__elideMode

    elideMode = qt.Property(qt.Qt.TextElideMode, getElideMode, setElideMode)

    def setTextAsToolTip(self, enabled):
        """Enable displaying text as part of the tooltip if it is elided.

        :param bool enabled: Enable the behavior
        """
        if self.__textAsToolTip == enabled:
            return
        self.__textAsToolTip = enabled
        self.__updateToolTip()

    def getTextAsToolTip(self):
        """True if an elided text is displayed as part of the tooltip.

        :rtype: bool
        """
        return self.__textAsToolTip

    textAsToolTip = qt.Property(bool, getTextAsToolTip, setTextAsToolTip)
