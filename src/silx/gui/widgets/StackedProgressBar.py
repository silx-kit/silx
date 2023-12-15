# /*##########################################################################
#
# Copyright (c) 2023 European Synchrotron Radiation Facility
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

from __future__ import annotations

from typing import NamedTuple, Any, ValuesView
from silx.gui import qt


class ProgressItem(NamedTuple):
    """Item storing the state of a stacked progress item"""

    value: int
    """Progression of the item"""

    visible: bool
    """Is the item displayed"""

    color: qt.QColor
    """Color of the progress"""

    striped: bool
    """If true, apply a stripe color to the gradiant"""

    animated: bool
    """If true, the stripe is animated"""

    toolTip: str
    """Tool tip of this item"""

    userData: Any
    """Any user data"""


class _UndefinedType:
    pass


_Undefined = _UndefinedType()


class StackedProgressBar(qt.QProgressBar):
    """
    Multiple stacked progress bar in single component
    """

    def __init__(self, parent: qt.Qwidget | None = None):
        super().__init__(parent=parent)
        self.__stacks: dict[str, ProgressItem] = {}
        self._animated: int = 0
        self._timer = qt.QTimer(self)
        self._timer.setInterval(80)
        self._timer.timeout.connect(self._tick)
        self._spacing: int = 0
        self._spacingCollapsible: bool = True

    def _tick(self):
        self._animated += 2
        self.update()

    def setSpacing(self, spacing: int):
        """Spacing between items, in pixels"""
        if self._spacing == spacing:
            return
        self._spacing = spacing
        self.update()

    def spacing(self) -> int:
        return self._spacing

    def setSpacingCollapsible(self, collapse: bool):
        """
        Set whether consecutive spacing should be collapsed.

        It can be useful to disable that to ensure pixel perfect
        rendering is some use cases.


        By default, this property is true.
        """
        if self._spacingCollapsible == collapse:
            return
        self._spacingCollapsible = collapse
        self.update()

    def spacingCollapsible(self) -> bool:
        return self._spacingCollapsible

    def clear(self):
        """Remove every stacked items from the widget"""
        if len(self.__stacks) == 0:
            return
        self.__stacks.clear()
        self.update()

    def setProgressItem(
        self,
        name: str,
        value: int | None | _UndefinedType = _Undefined,
        visible: bool | _UndefinedType = _Undefined,
        color: qt.QColor | None | _UndefinedType = _Undefined,
        striped: bool | _UndefinedType = _Undefined,
        animated: bool | _UndefinedType = _Undefined,
        toolTip: str | None | _UndefinedType = _Undefined,
        userData: Any = _Undefined,
    ):
        """Add or update a stacked items by its name"""

        previousItem = self.__stacks.get(name)

        if previousItem is not None:
            if value is _Undefined:
                value = previousItem.value
            if visible is _Undefined:
                visible = previousItem.visible
            if striped is _Undefined:
                striped = previousItem.striped
            if color is _Undefined:
                color = previousItem.color
            if toolTip is _Undefined:
                toolTip = previousItem.toolTip
            if animated is _Undefined:
                animated = previousItem.animated
            if userData is _Undefined:
                userData = previousItem.userData
        else:
            if value is _Undefined:
                value = 0
            if visible is _Undefined:
                visible = True
            if striped is _Undefined:
                striped = False
            if color is _Undefined:
                color = qt.QColor()
            if toolTip is _Undefined:
                toolTip = ""
            if animated is _Undefined:
                animated = False
            if userData is _Undefined:
                userData = None

        newItem = ProgressItem(
            value=value,
            visible=visible,
            color=color,
            striped=striped,
            animated=animated,
            toolTip=toolTip,
            userData=userData,
        )
        if previousItem == newItem:
            return
        self.__stacks[name] = newItem
        animated = any([s.animated for s in self.__stacks.values()])
        self._setAnimated(animated)
        self.update()

    def _setAnimated(self, animated: bool):
        if animated == self._timer.isActive():
            return
        if animated:
            self._timer.start()
        else:
            self._timer.stop()

    def removeProgressItem(self, name: str):
        """Remove a stacked item by its name"""
        s = self.__stacks.pop(name, None)
        if s is None:
            return
        self.update()

    def _brushFromProgressItem(self, item: ProgressItem) -> qt.QPalette | None:
        if item.color is None:
            return None

        palette = qt.QPalette()
        color = qt.QColor(item.color)

        if item.striped:
            if item.animated:
                delta = self._animated
            else:
                delta = 0
            color2 = color.lighter(120)
            shadowGradient = qt.QLinearGradient()
            shadowGradient.setSpread(qt.QGradient.RepeatSpread)
            shadowGradient.setStart(-delta, 0)
            shadowGradient.setFinalStop(8 - delta, -8)
            shadowGradient.setColorAt(0.0, color)
            shadowGradient.setColorAt(0.5, color)
            shadowGradient.setColorAt(0.50001, color2)
            shadowGradient.setColorAt(1.0, color2)
            brush = qt.QBrush(shadowGradient)
            palette.setBrush(qt.QPalette.Highlight, brush)
            palette.setBrush(qt.QPalette.Window, color2)
        else:
            palette.setColor(qt.QPalette.Highlight, color)

        return palette

    def paintEvent(self, event):
        painter = qt.QStylePainter(self)
        opt = qt.QStyleOptionProgressBar()
        self.initStyleOption(opt)
        painter.drawControl(qt.QStyle.CE_ProgressBarGroove, opt)
        self._drawProgressItems(painter, self.__stacks.values())

    def _drawProgressItems(self, painter: qt.QPainter, items: ValuesView[ProgressItem]):
        opt = qt.QStyleOptionProgressBar()
        self.initStyleOption(opt)

        visibleItems = [i for i in items if i.value and i.visible]
        xpos: int = 0
        w = opt.rect.width()
        if self._spacingCollapsible:
            cumspacing = max(0, len(visibleItems) - 1) * self._spacing
            w -= cumspacing
        vw = opt.maximum - opt.minimum
        opt.minimum = 0
        opt.maximum = w

        for item in visibleItems:
            xwidth = int(item.value * w / vw)
            opt.progress = xwidth * 2
            palette = self._brushFromProgressItem(item)
            if palette is not None:
                opt.palette = palette
            self._drawProgressItem(painter, opt, xpos, xwidth)
            xpos += xwidth + self._spacing

    def _drawProgressItem(
        self,
        painter: qt.QPainter,
        option: qt.QStyleOptionProgressBar,
        xpos: int,
        xwidth: int,
    ):
        if xwidth == 0:
            return
        rect: qt.QRect = option.rect
        style = self.style()

        if option.minimum == 0 and option.maximum == 0:
            return
        x0 = rect.x() + 3
        y0 = rect.y()

        h = rect.height()
        w = rect.width()
        xmaxwith = min(x0 + xpos + xwidth, w - 1) - x0 - xpos
        if xmaxwith < 0:
            return
        rect = qt.QRect(x0 + xpos, y0, xmaxwith, h)
        opt = qt.QStyleOptionProgressBar()
        opt.state = qt.QStyle.State_None
        margin = 1
        opt.rect = rect.marginsAdded(qt.QMargins(margin, margin, margin, margin))
        opt.palette = option.palette
        style.drawPrimitive(qt.QStyle.PE_IndicatorProgressChunk, opt, painter, self)

    def getProgressItemByPosition(self, pos: qt.QPoint) -> ProgressItem | None:
        """Returns the stacked item at a position of the component."""
        minimum = self.minimum()
        maximum = self.maximum()
        vRange = maximum - minimum
        w = self.width()
        v = pos.x() * vRange / w
        current = 0
        for item in self.__stacks.values():
            if not item.visible:
                continue
            current += item.value
            if v < current:
                return item
        return None

    def tooltipFromProgressItem(self, item: ProgressItem) -> str | None:
        """Returns the tooltip to display over an item.

        It is triggered when the tooltip have to be displayed.
        """
        return item.toolTip

    def event(self, event: qt.QEvent):
        if event.type() == qt.QEvent.ToolTip:
            item = self.getProgressItemByPosition(event.pos())
            if item is not None:
                toolTip = self.tooltipFromProgressItem(item)
                if toolTip:
                    qt.QToolTip.showText(event.globalPos(), toolTip, self)
            return True
        return super().event(event)
