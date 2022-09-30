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
"""WaitingPushButton module
"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "26/04/2017"

from .. import qt
from .. import icons


class WaitingPushButton(qt.QPushButton):
    """Button which allows to display a waiting status when, for example,
    something is still computing.

    The component is graphically disabled when it is in waiting. Then we
    overwrite the enabled method to dissociate the 2 concepts:
    graphically enabled/disabled, and enabled/disabled

    .. image:: img/WaitingPushButton.png
    """

    def __init__(self, parent=None, text=None, icon=None):
        """Constructor

        :param str text: Text displayed on the button
        :param qt.QIcon icon: Icon displayed on the button
        :param qt.QWidget parent: Parent of the widget
        """
        if icon is not None:
            qt.QPushButton.__init__(self, icon, text, parent)
        elif text is not None:
            qt.QPushButton.__init__(self, text, parent)
        else:
            qt.QPushButton.__init__(self, parent)

        self.__waiting = False
        self.__enabled = True
        self.__icon = icon
        self.__disabled_when_waiting = True
        self.__waitingIcon = icons.getWaitIcon()

    def sizeHint(self):
        """Returns the recommended size for the widget.

        This implementation of the recommended size always consider there is an
        icon. In this way it avoid to update the layout when the waiting icon
        is displayed.
        """
        self.ensurePolished()

        w = 0
        h = 0

        opt = qt.QStyleOptionButton()
        self.initStyleOption(opt)

        # Content with icon
        # no condition, assume that there is an icon to avoid blinking
        # when the widget switch to waiting state
        ih = opt.iconSize.height()
        iw = opt.iconSize.width() + 4
        w += iw
        h = max(h, ih)

        # Content with text
        text = self.text()
        isEmpty = text == ""
        if isEmpty:
            text = "XXXX"
        fm = self.fontMetrics()
        textSize = fm.size(qt.Qt.TextShowMnemonic, text)
        if not isEmpty or w == 0:
            w += textSize.width()
        if not isEmpty or h == 0:
            h = max(h, textSize.height())

        # Content with menu indicator
        opt.rect.setSize(qt.QSize(w, h))  # PM_MenuButtonIndicator depends on the height
        if self.menu() is not None:
            w += self.style().pixelMetric(qt.QStyle.PM_MenuButtonIndicator, opt, self)

        contentSize = qt.QSize(w, h)
        sizeHint = self.style().sizeFromContents(qt.QStyle.CT_PushButton, opt, contentSize, self)
        if qt.BINDING in ('PySide2', 'PyQt5'):  # Qt6: globalStrut not available
            sizeHint = sizeHint.expandedTo(qt.QApplication.globalStrut())
        return sizeHint

    def setDisabledWhenWaiting(self, isDisabled):
        """Enable or disable the auto disable behaviour when the button is waiting.

        :param bool isDisabled: Enable the auto-disable behaviour
        """
        if self.__disabled_when_waiting == isDisabled:
            return
        self.__disabled_when_waiting = isDisabled
        self.__updateVisibleEnabled()

    def isDisabledWhenWaiting(self):
        """Returns true if the button is auto disabled when it is waiting.

        :rtype: bool
        """
        return self.__disabled_when_waiting

    disabledWhenWaiting = qt.Property(bool, isDisabledWhenWaiting, setDisabledWhenWaiting)
    """Property to enable/disable the auto disabled state when the button is waiting."""

    def __setWaitingIcon(self, icon):
        """Called when the waiting icon is updated. It is called every frames
        of the animation.

        :param qt.QIcon icon: The new waiting icon
        """
        qt.QPushButton.setIcon(self, icon)

    def setIcon(self, icon):
        """Set the button icon. If the button is waiting, the icon is not
        visible directly, but will be visible when the waiting state will be
        removed.

        :param qt.QIcon icon: An icon
        """
        self.__icon = icon
        self.__updateVisibleIcon()

    def getIcon(self):
        """Returns the icon set to the button. If the widget is waiting
        it is not returning the visible icon, but the one requested by
        the application (the one displayed when the widget is not in
        waiting state).

        :rtype: qt.QIcon
        """
        return self.__icon

    icon = qt.Property(qt.QIcon, getIcon, setIcon)
    """Property providing access to the icon."""

    def __updateVisibleIcon(self):
        """Update the visible icon according to the state of the widget."""
        if not self.isWaiting():
            icon = self.__icon
        else:
            icon = self.__waitingIcon.currentIcon()
        if icon is None:
            icon = qt.QIcon()
        qt.QPushButton.setIcon(self, icon)

    def setEnabled(self, enabled):
        """Set the enabled state of the widget.

        :param bool enabled: The enabled state
        """
        if self.__enabled == enabled:
            return
        self.__enabled = enabled
        self.__updateVisibleEnabled()

    def isEnabled(self):
        """Returns the enabled state of the widget.

        :rtype: bool
        """
        return self.__enabled

    enabled = qt.Property(bool, isEnabled, setEnabled)
    """Property providing access to the enabled state of the widget"""

    def __updateVisibleEnabled(self):
        """Update the visible enabled state according to the state of the
        widget."""
        if self.__disabled_when_waiting:
            enabled = not self.isWaiting() and self.__enabled
        else:
            enabled = self.__enabled
        qt.QPushButton.setEnabled(self, enabled)

    def setWaiting(self, waiting):
        """Set the waiting state of the widget.

        :param bool waiting: Requested state"""
        if self.__waiting == waiting:
            return
        self.__waiting = waiting

        if self.__waiting:
            self.__waitingIcon.register(self)
            self.__waitingIcon.iconChanged.connect(self.__setWaitingIcon)
        else:
            # unregister only if the object is registred
            self.__waitingIcon.unregister(self)
            self.__waitingIcon.iconChanged.disconnect(self.__setWaitingIcon)

        self.__updateVisibleEnabled()
        self.__updateVisibleIcon()

    def isWaiting(self):
        """Returns true if the widget is in waiting state.

        :rtype: bool"""
        return self.__waiting

    @qt.Slot()
    def wait(self):
        """Enable the waiting state."""
        self.setWaiting(True)

    @qt.Slot()
    def stopWaiting(self):
        """Disable the waiting state."""
        self.setWaiting(False)

    @qt.Slot()
    def swapWaiting(self):
        """Swap the waiting state."""
        self.setWaiting(not self.isWaiting())
