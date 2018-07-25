# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2015-2018 European Synchrotron Radiation Facility
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

from __future__ import absolute_import, division

__authors__ = ["D. Naudet", "T. Vincent"]
__license__ = "MIT"
__date__ = "25/07/2018"


import numpy as numpy

from silx.gui import qt, icons, colors
from silx.gui.utils._image import convertArrayToQImage


class RangeSlider(qt.QWidget):
    """Range slider with 2 thumbs and an optional colored groove.

    :param QWidget parent: See QWidget
    """

    _SLIDER_WIDTH = 10
    """Width of the slider rectangle"""

    _PIXMAP_VOFFSET = 7
    """Vertical groove pixmap offset"""

    sigNumberOfStepsChanged = qt.Signal(int)
    """This signal is emitted when the number of steps has changed.

    It provides the new number of steps.
    """

    sigPositionChanged = qt.Signal(int, int)
    """Signal emitted when the position of the sliders has changed.

    It provides the slider positions (first, second).
    """

    sigRangeChanged = qt.Signal(float, float)
    """Signal emitted when the value range has changed.

    It provides the new range (min, max).
    """

    sigValueChanged = qt.Signal(float, float)
    """Signal emitted when the value of the sliders has changed.

    It provides the slider values (first, second).
    """

    def __init__(self, parent=None):
        self.__pixmap = None
        self.__steps = 100
        self.__firstPosition = 0
        self.__secondPosition = self.__steps - 1
        self.__minValue = 0.
        self.__maxValue = 1.

        self.__focus = None
        self.__moving = None

        self.__icons = {
            'first': icons.getQIcon('previous'),
            'second': icons.getQIcon('next')
        }

        # call the super constructor AFTER defining all members that
        # are used in the "paint" method
        super(RangeSlider, self).__init__(parent)

        self.setFocusPolicy(qt.Qt.ClickFocus)

        self.setMinimumSize(qt.QSize(50, 20))
        self.setMaximumHeight(20)

        # Broadcast value changed signal
        self.sigPositionChanged.connect(self.__emitValueChanged)
        self.sigRangeChanged.connect(self.__emitValueChanged)

    # Position <-> Value conversion

    def __positionToValue(self, position):
        """Returns value corresponding to position

        :param int position:
        :rtype: float
        """
        min_, max_ = self.getMinimum(), self.getMaximum()
        maxPos = self.getNumberOfSteps() - 1
        return min_ + (max_ - min_) * int(position) / maxPos

    def __valueToPosition(self, value):
        """Returns closest position corresponding to value

        :param float value:
        :rtype: int
        """
        min_, max_ = self.getMinimum(), self.getMaximum()
        maxPos = self.getNumberOfSteps() - 1
        return int(0.5 + maxPos * (float(value) - min_) / (max_ - min_))

    # Position (int) API

    def getNumberOfSteps(self):
        """Returns the number of steps.

        :rtype: int"""
        return self.__steps

    def setNumberOfSteps(self, steps):
        """Set the number of steps.

        Slider positions are eventually adjusted.

        :param int steps:
        :raise ValueError: If steps is negative or null
        """
        steps = int(steps)
        if steps != self.getNumberOfSteps():
            if steps <= 0:
                raise ValueError("Number of steps must be strictly positive")
            previousPositions = self.getPositions()
            self.__steps = steps
            self.sigNumberOfStepsChanged.emit(steps)
            self.__setPositions(*previousPositions)

    def getFirstPosition(self):
        """Returns first slider position

        :rtype: int
        """
        return self.__firstPosition

    def setFirstPosition(self, position):
        """Set the position of the first slider

        The position is adjusted to valid values

        :param int position:
        """
        position = int(position)
        if position != self.getFirstPosition():
            self.__firstPosition = min(max(0, position),
                                       self.getSecondPosition())
            self.update()
            self.sigPositionChanged.emit(*self.getPositions())

    def getSecondPosition(self):
        """Returns second slider position

        :rtype: int
        """
        return self.__secondPosition

    def setSecondPosition(self, position):
        """Set the position of the second slider

        The position is adjusted to valid values

        :param int position:
        """
        position = int(position)
        if position != self.getSecondPosition():
            self.__secondPosition = min(max(self.getFirstPosition(), position),
                                        self.getNumberOfSteps() - 1)
            self.update()
            self.sigPositionChanged.emit(*self.getPositions())

    def getPositions(self):
        """Returns slider positions (first, second)

        :rtype: List[int]
        """
        return self.getFirstPosition(), self.getSecondPosition()

    def __setPositions(self, first, second):
        """Set slider positions.

        This method does not check if slider positions are already set

        :param int first:
        :param int second:
        """
        maxPos = self.getNumberOfSteps() - 1
        first, second = int(first), int(second)
        self.__firstPosition = min(max(0, first), maxPos)
        self.__secondPosition = min(max(first, second), maxPos)
        self.update()
        self.sigPositionChanged.emit(*self.getPositions())

    def setPositions(self, first, second):
        """Set the position of both sliders at once

        First is clipped to the slider range: [0, number of steps].
        Second is clipped to valid values: [first, number of steps]

        :param int first:
        :param int second:
        """
        if (first != self.getFirstPosition() or
                second != self.getSecondPosition()):
            self.__setPositions(first, second)

    # Value (float) API

    def __emitValueChanged(self, *args, **kwargs):
        self.sigValueChanged.emit(*self.getValues())

    def getMinimum(self):
        """Returns the minimum value of the slider range

        :rtype: float
        """
        return self.__minValue

    def setMinimum(self, minimum):
        """Set the minimum value of the slider range.

        It eventually adjusts maximum.
        Slider positions remains unchanged and slider values are modified.

        :param float minimum:
        """
        minimum = float(minimum)
        if minimum != self.getMinimum():
            if minimum > self.getMaximum():
                self.__maxValue = minimum
            self.__minValue = minimum
            self.sigRangeChanged.emit(*self.getRange())

    def getMaximum(self):
        """Returns the maximum value of the slider range

        :rtype: float
        """
        return self.__maxValue

    def setMaximum(self, maximum):
        """Set the maximum value of the slider range

        It eventually adjusts minimum.
        Slider positions remains unchanged and slider values are modified.

        :param float maximum:
        """
        maximum = float(maximum)
        if maximum != self.getMaximum():
            if maximum < self.getMinimum():
                self.__minValue = maximum
            self.__maxValue = maximum
            self.sigRangeChanged.emit(*self.getRange())

    def getRange(self):
        """Returns the range of values (min, max)

        :rtype: List[float]
        """
        return self.getMinimum(), self.getMaximum()

    def setRange(self, minimum, maximum):
        """Set the range of values.

        If maximum is lower than minimum, minimum is the only valid value.
        Slider positions remains unchanged and slider values are modified.

        :param float minimum:
        :param float maximum:
        """
        minimum, maximum = float(minimum), float(maximum)
        if minimum != self.getMinimum() or maximum != self.getMaximum():
            self.__minValue = minimum
            self.__maxValue = max(maximum, minimum)
            self.sigRangeChanged.emit(*self.getRange())

    def getFirstValue(self):
        """Returns the value of the first slider

        :rtype: float
        """
        return self.__positionToValue(self.getFirstPosition())

    def setFirstValue(self, value):
        """Set the value of the first slider

        Value is clipped to valid values.

        :param float value:
        """
        self.setFirstPosition(self.__valueToPosition(value))

    def getSecondValue(self):
        """Returns the value of the second slider

        :rtype: float
        """
        return self.__positionToValue(self.getSecondPosition())

    def setSecondValue(self, value):
        """Set the value of the second slider

        Value is clipped to valid values.

        :param float value:
        """
        self.setSecondPosition(self.__positionToValue(value))

    def getValues(self):
        """Returns value of both sliders at once

        :return: (first value, second value)
        :rtype: List[float]
        """
        return self.getFirstValue(), self.getSecondValue()

    def setValues(self, first, second):
        """Set values for both sliders at once

        First is clipped to the slider range: [minimum, maximum].
        Second is clipped to valid values: [first, maximum]

        :param float first:
        :param float second:
        """
        self.setPositions(self.__valueToPosition(first),
                          self.__valueToPosition(second))

    # Groove API

    def getGroovePixmap(self):
        """Returns the pixmap displayed in the slider groove if any.

        :rtype: Union[QPixmap,None]
        """
        return self.__pixmap

    def setGroovePixmap(self, pixmap):
        """Set the pixmap displayed in the slider groove.

        :param Union[QPixmap,None] pixmap: The QPixmap to use or None to unset.
        """
        assert pixmap is None or isinstance(pixmap, qt.QPixmap)
        self.__pixmap = pixmap
        self.update()

    def setGroovePixmapFromProfile(self, profile, colormap=None):
        """Set the pixmap displayed in the slider groove from histogram values.

        :param Union[numpy.ndarray,None] profile:
            1D array of values to display
        :param Union[Colormap,str] colormap:
            The colormap name or object to convert profile values to colors
        """
        if profile is None:
            self.setSliderPixmap(None)
            return

        profile = numpy.array(profile, copy=False)

        if profile.size == 0:
            self.setSliderPixmap(None)
            return

        if colormap is None:
            colormap = colors.Colormap()
        elif isinstance(colormap, str):
            colormap = colors.Colormap(name=colormap)
        assert isinstance(colormap, colors.Colormap)

        rgbImage = colormap.applyToData(profile.reshape(1, -1))[:, :, :3]
        qimage = convertArrayToQImage(rgbImage)
        qpixmap = qt.QPixmap.fromImage(qimage)
        self.setGroovePixmap(qpixmap)

    # Handle interaction

    def mousePressEvent(self, event):
        super(RangeSlider, self).mousePressEvent(event)

        if event.buttons() == qt.Qt.LeftButton:
            picked = None
            for name in ('first', 'second'):
                area = self.__sliderRect(name)
                if area.contains(event.pos()):
                    picked = name
                    break

            self.__moving = picked
            self.__focus = picked
            self.update()

    def mouseMoveEvent(self, event):
        super(RangeSlider, self).mouseMoveEvent(event)

        if self.__moving is not None:
            position = self.__xPixelToPosition(event.pos().x(), self.__moving)
            if self.__moving == 'first':
                self.setFirstPosition(position)
            else:
                self.setSecondPosition(position)

    def mouseReleaseEvent(self, event):
        super(RangeSlider, self).mouseReleaseEvent(event)

        if event.button() == qt.Qt.LeftButton and self.__moving is not None:
            self.__moving = None
            self.update()

    def focusOutEvent(self, event):
        if self.__focus is not None:
            self.__focus = None
            self.update()
        super(RangeSlider, self).focusOutEvent(event)

    def keyPressEvent(self, event):
        key = event.key()
        if event.modifiers() == qt.Qt.NoModifier and self.__focus is not None:
            if key in (qt.Qt.Key_Left, qt.Qt.Key_Down):
                if self.__focus == 'first':
                    self.setFirstPosition(self.getFirstPosition() - 1)
                else:
                    self.setSecondPosition(self.getSecondPosition() - 1)
                return  # accept event
            elif key in (qt.Qt.Key_Right, qt.Qt.Key_Up):
                if self.__focus == 'first':
                    self.setFirstPosition(self.getFirstPosition() + 1)
                else:
                    self.setSecondPosition(self.getSecondPosition() + 1)
                return  # accept event

        super(RangeSlider, self).keyPressEvent(event)

    # Handle repaint

    def __xPixelToPosition(self, x, name):
        """Convert position in pixel to slider position

        :param int x: X in pixel coordinates
        :rtype: int
        """
        sliderArea = self.__sliderAreaRect()
        maxPos = self.getNumberOfSteps() - 1
        position = maxPos * (x - sliderArea.left()) / (sliderArea.width() - 1)
        if name == 'first':
            return int(position + 0.5)
        else:
            return int(position)

    def __sliderRect(self, name):
        """Returns rectangle corresponding to slider in pixels

        :param str name: 'first' or 'second'
        :rtype: QRect
        :raise ValueError: If wrong name
        """
        assert name in ('first', 'second')
        if name == 'first':
            offset = - self._SLIDER_WIDTH
            position = self.getFirstPosition()
        elif name == 'second':
            offset = 0
            position = self.getSecondPosition()
        else:
            raise ValueError('Unknown name')

        sliderArea = self.__sliderAreaRect()

        maxPos = self.getNumberOfSteps() - 1
        xOffset = int((sliderArea.width() - 1) * position / maxPos)
        xPos = sliderArea.left() + xOffset + offset

        return qt.QRect(xPos,
                        sliderArea.top(),
                        self._SLIDER_WIDTH,
                        sliderArea.height())

    def __drawArea(self):
        return self.rect().adjusted(self._SLIDER_WIDTH, 0,
                                    -self._SLIDER_WIDTH, 0)

    def __sliderAreaRect(self):
        return self.__drawArea().adjusted(self._SLIDER_WIDTH / 2.,
                                          0,
                                          -self._SLIDER_WIDTH / 2.,
                                          0)

    def __pixMapRect(self):
        return self.__sliderAreaRect().adjusted(0,
                                                self._PIXMAP_VOFFSET,
                                                0,
                                                -self._PIXMAP_VOFFSET)

    def paintEvent(self, event):
        painter = qt.QPainter(self)

        style = qt.QApplication.style()

        area = self.__drawArea()
        pixmapRect = self.__pixMapRect()

        option = qt.QStyleOptionProgressBar()
        option.initFrom(self)
        option.rect = area
        option.state = ((self.isEnabled() and qt.QStyle.State_Enabled)
                        or qt.QStyle.State_None)
        style.drawControl(qt.QStyle.CE_ProgressBarGroove,
                          option,
                          painter,
                          self)

        painter.save()
        pen = painter.pen()
        pen.setWidth(1)
        pen.setColor(qt.Qt.black if self.isEnabled() else qt.Qt.gray)
        painter.setPen(pen)
        painter.drawRect(pixmapRect.adjusted(-1, -1, 1, 1))
        painter.restore()

        if self.isEnabled() and self.__pixmap is not None:
            painter.drawPixmap(area.adjusted(self._SLIDER_WIDTH / 2,
                                             self._PIXMAP_VOFFSET,
                                             -self._SLIDER_WIDTH / 2,
                                             -self._PIXMAP_VOFFSET + 1),
                               self.__pixmap,
                               self.__pixmap.rect())

        for name in ('first', 'second'):
            rect = self.__sliderRect(name)
            option = qt.QStyleOptionButton()
            option.initFrom(self)
            option.icon = self.__icons[name]
            option.iconSize = rect.size() * 0.7
            if option.state & qt.QStyle.State_MouseOver:
               option.state ^= qt.QStyle.State_MouseOver
            if self.__focus == name:
                option.state |= qt.QStyle.State_HasFocus
            elif option.state & qt.QStyle.State_HasFocus:
                option.state ^= qt.QStyle.State_HasFocus
            option.rect = rect
            style.drawControl(
                qt.QStyle.CE_PushButton, option, painter, self)

    def sizeHint(self):
        return qt.QSize(200, self.minimumHeight())
