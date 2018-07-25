# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2015-2016 European Synchrotron Radiation Facility
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

from __future__ import absolute_import

__authors__ = ["D. Naudet"]
__license__ = "MIT"
__date__ = "15/09/2016"


from collections import namedtuple, OrderedDict

import numpy as np

from silx.gui import qt as Qt

from ...gui.icons import getQIcon as getKmapIcon


RangeSliderState = namedtuple('RangeSliderState', ['left', 'right',
                                                   'leftIndex', 'rightIndex'])


class RangeSlider(Qt.QWidget):
    _defaultMinimumHeight = 20
    _defaultMinimumSize = (50, 20)
    _defaultMaximumHeight = 20.

    _defaultKeybordValueMove = 1
    """ Number of steps moved when using the arrow keys """

    _defaultNSteps = 1000

    _sliderWidth = 10
    _pixmapVOffset = 7

    sigSliderMoved = Qt.Signal(object)

    # TODO : add center slider

    def __init__(self, *args, **kwargs):
        self.__pixmap = None

        self.__moving = None
        self.__hover = None
        self.__focus = None
        self.__range = None
        self.__nSteps = self._defaultNSteps
        self.__snap = False

        self.__sliderIcons = {'left': getKmapIcon('right_arrow'),
                              'right': getKmapIcon('left_arrow')}

        self.__sliders = OrderedDict([('left', Qt.QRect()),
                                      ('right', Qt.QRect())])
        self.__values = {'left': None, 'right': None}
        self.__showRangeBackground = None

        # call the super constructor AFTER defining all members that
        # are used in the "paint" method
        super(RangeSlider, self).__init__(*args, **kwargs)

        self.setFocusPolicy(Qt.Qt.ClickFocus)

        self.__setupSliders()

        self.setMouseTracking(True)
        self.setMinimumSize(Qt.QSize(*self._defaultMinimumSize))
        self.setMaximumHeight(self._defaultMaximumHeight)

        self.setSliderValues(None, None)

    def setSliderResolution(self, nSteps):
        """
        Sets the slider resolution. Set to None to reset it to its default
         value (1000).
        :param nSteps:
        :return:
        """
        if nSteps is None:
            if self.__pixmap is None:
                nSteps = self._defaultNSteps
            else:
                nSteps = self.__pixmap.width()
        self.__nSteps = nSteps
        self.update()

    def getSliderResolution(self):
        """
        Returns the slider resolution.
        :return:
        """
        return self.__nSteps

    def setSnap(self, snap):
        """
        Tells the slider to snap to the grid.
        See also : RangeSlider.setSliderResolution.
        :param snap:
        :return:
        """
        self.update()
        self.__snap = snap

    def isSnap(self):
        """
        Returns True if the slider snaps to the grid.
        :return:
        """
        return self.__snap

    def getRange(self):
        if self.__range is None:
            if self.__pixmap is None:
                sliderRange = [0., self.__nSteps]
            else:
                sliderRange = [0., self.__pixmap.width() - 1.]
        else:
            sliderRange = self.__range
        return sliderRange

    def setRange(self, sliderRng):
        """
        Set to None to reset (range will be 0 -> _defaultNSteps) or
            (0 -> profile length)
        :param sliderRng:
        :return:
        """
        if sliderRng is not None:
            reset = self.__range is None
            if len(sliderRng) != 2:
                raise ValueError('The slider range must be a 2-elements '
                                 'array, or None.')
            if sliderRng[0] >= sliderRng[1]:
                raise ValueError('Min range must be sctrictly lower than'
                                 'max range.')
            self.__range = np.array(sliderRng, dtype=np.float)
        else:
            reset = True
            self.__range = None

        if reset:
            self.resetSliderValues()

    def resetSliderValues(self):
        """
        Sets the left and right slider values to the min and max of the range.
        :return:
        """
        self.setSliderValues(None, None)

    def setShowRangeBackground(self, show):
        """
        Set to True to color the area between the two slider.
        Set to False to hide it.
        Set to None to revert back to the default behaviour : shown when no
            pixmap is set, and hidden otherwise
        :param show:
        :return:
        """
        self.__showRangeBackground = show
        self.update()

    def __drawArea(self):
        return self.rect().adjusted(self._sliderWidth, 0,
                                    -self._sliderWidth, 0)

    def __sliderRect(self):
        return self.__drawArea().adjusted(self._sliderWidth / 2.,
                                          0,
                                          -self._sliderWidth / 2.,
                                          0)

    def __pixMapRect(self):
        return self.__sliderRect().adjusted(0,
                                            self._pixmapVOffset,
                                            0,
                                            -self._pixmapVOffset)

    def __setupSliders(self):
        area = self.__sliderRect()
        height = area.height()
        width = self._sliderWidth

        template = Qt.QRect(area.left(),
                            area.top(),
                            width,
                            height)

        self.__sliders['left'] = template.translated(width/-2, 0)
        self.__sliders['right'] = template.translated(width/2, 0)

    def __valueToIndex(self, value):
        if self.__pixmap:
            sliderRange = self.getRange()
            pixLength = self.__pixmap.width()
            sliderWidth = sliderRange[1] - sliderRange[0]
            ratio = (pixLength - 1) / sliderWidth
            index = int(np.floor(0.5 + ratio * (value - sliderRange[0])))
        else:
            index = None
        return index

    def __setSliderValue(self, side, value):
        slider = self.__sliders[side]
        values = self.__values
        sliderRange = self.getRange()

        if side == 'left':
            moveMeth = slider.moveRight
            minValue = sliderRange[0]
            maxValue = values['right']
            if maxValue is None:
                maxValue = sliderRange[1]
            default = minValue
        else:
            moveMeth = slider.moveLeft
            minValue = values['left']
            if minValue is None:
                minValue = sliderRange[0]
            maxValue = sliderRange[1]
            default = maxValue

        if value is not None:
            if value < minValue:
                value = minValue
            elif value > maxValue:
                value = maxValue
        else:
            value = default

        if self.__snap:
            # snapping to the grid
            sRange = sliderRange[1] - sliderRange[0]
            step = np.floor(0.5 +
                            (value - sliderRange[0]) *
                            (self.__nSteps - 1) /
                            sRange)
            value = sliderRange[0] + step * (sRange / (self.__nSteps - 1))

        x = self.__valueToPos(value)

        moveMeth(x)

        if values[side] != value:
            notify = True
            values[side] = value
        else:
            notify = False

        self.update()

        if notify:
            left = values['left']
            if left is None:
                left = sliderRange[0]
            right = values['right']
            if right is None:
                right = sliderRange[1]

            leftIndex = self.__valueToIndex(left)
            rightIndex = self.__valueToIndex(right)

            event = RangeSliderState(left=left,
                                     right=right,
                                     leftIndex=leftIndex,
                                     rightIndex=rightIndex)

            self.sigSliderMoved.emit(event)

        return value

    def getSliderState(self):
        """
        Returns the state of the slider : values and indices.
        :return:
        """
        indices = self.getSliderIndices()
        values = self.getSliderValues()
        state = RangeSliderState(left=values[0],
                                 right=values[1],
                                 leftIndex=indices[0],
                                 rightIndex=indices[1])
        return state

    def __posToValue(self, x):
        """
        Returns the value corresponding the the given slider position.
        :param x:
        :return:
        """
        sliderArea = self.__sliderRect()
        sliderRange = self.getRange()

        value = (sliderRange[0] +
                 (x - sliderArea.left()) * (sliderRange[1] - sliderRange[0])
                 / (sliderArea.width() - 1))

        return value

    def __valueToPos(self, value):
        """
        Returns the slider position corresponding to the given value.
        :param value:
        :return:
        """
        sliderArea = self.__sliderRect()
        sliderRange = self.getRange()

        x = (sliderArea.left() +
             (sliderArea.width() - 1.) * (value - sliderRange[0])
             / (sliderRange[1] - sliderRange[0]))

        return x

    def getSliderValue(self, side):
        """
        Returns the slider value.
        :param side:
        :return:
        """
        value = self.__values[side]
        if value is None:
            sliderRange = self.getRange()
            if side == 'left':
                value = sliderRange[0]
            else:
                value = sliderRange[1]
        return value

    def getSliderValues(self):
        """
        Returns the left and right slider values.
        :return:
        """
        return (self.getSliderValue('left'),
                self.getSliderValue('right'))

    def getSliderIndex(self, side):
        """
        Returns the slider index
            (i.e : the step the sliders is closest to,
            between 0 and resolution - 1).
            See also : RangeSlider.setResolution.
        :param side:
        :return:
        """
        return self.__valueToIndex(self.getSliderValue(side))

    def getSliderIndices(self):
        """
        Returns the left and right slider index
            (i.e : the step the slider is closest to,
            between 0 and resolution - 1).
            See also : RangeSlider.setResolution.
        :return:
        """
        return (self.getSliderIndex('left'),
                self.getSliderIndex('right'))

    def setSliderValue(self, side, value):
        """

        :param side: 'left' or 'right'
        :param value: float between range min and range max.
            (leftmost to rightmost)
        :return:
        """
        assert side in ('left', 'right')

        self.__setSliderValue(side, value)

    def setSliderValues(self, leftValue, rightValue):
        self.setSliderValue('left', leftValue)
        self.setSliderValue('right', rightValue)

    def resizeEvent(self, event):
        super(RangeSlider, self).resizeEvent(event)
        self.__setupSliders()
        for side, value in self.__values.items():
            self.setSliderValue(side, value)

    def showEvent(self, event):
        super(RangeSlider, self).showEvent(event)

    def __mouseOnItem(self, pos):
        """
        Returns True if the given pos intersects whith one of the shapes
        :param pos:
        :type pos: QPoint
        :return:
        """
        for side, area in self.__sliders.items():
            if area.contains(pos):
                return side
        return None

    def mouseMoveEvent(self, event):
        super(RangeSlider, self).mouseMoveEvent(event)

        pos = event.pos()
        update = False

        if not self.__moving:
            side = self.__mouseOnItem(pos)
            if side != self.__hover:
                update = True
            self.__hover = side
        else:
            self.__values[self.__moving] =\
                self.__setSliderValue(self.__moving,
                                      self.__posToValue(pos.x()))
            update = True
        if update:
            self.update()

    def mousePressEvent(self, event):
        super(RangeSlider, self).mousePressEvent(event)

        if not self.__moving and event.buttons() == Qt.Qt.LeftButton:
            self.__moving = self.__mouseOnItem(event.pos())
            self.__focus = self.__moving
            self.update()

    def focusOutEvent(self, event):
        self.__focus = None
        self.__hover = None
        super(RangeSlider, self).focusOutEvent(event)

    def mouseReleaseEvent(self, event):
        super(RangeSlider, self).mouseReleaseEvent(event)
        if event.button() == Qt.Qt.LeftButton:
            if self.__moving:
                self.__moving = None
                self.update()

    def keyPressEvent(self, event):
        accepted = (Qt.Qt.Key_Left, Qt.Qt.Key_Right)
        key = event.key()
        sliderRange = self.getRange()
        if (self.__focus and
                    event.modifiers() == Qt.Qt.NoModifier and
                    key in accepted):
            disp = (self._defaultKeybordValueMove *
                    (sliderRange[1] - sliderRange[0]) / (self.__nSteps - 1))
            move = ((key == Qt.Qt.Key_Left and
                     -1.0 * disp) or
                    disp)
            self.__setSliderValue(self.__focus,
                                  self.__values[self.__focus] + move)

        super(RangeSlider, self).keyPressEvent(event)

    def setSliderPixmap(self, pixmap, resetSliders=None, snap=True):
        """
        Sets the pixmap displayed in the slider groove.
        None to unset.
        See also : RangeSlider.setSliderProfile
        :param pixmap:
        :param resetSliders: True to reset the slider positions to their
            extremes. If None the values positions will be reset if there
            was no previous pixmap.
        :param snap: set to True to set the number of steps equal to the length
            of the pixmap (equivalent to calling setNSteps).
            If false, the number of steps will be unchanged, and snap will be
            set to False.
        :return:
        """
        if pixmap is not None and pixmap.width() <= 1:
            raise ValueError('Pixmap must have a width > 1.')
        self.__pixmap = pixmap
        if resetSliders:
            self.setSliderValues(None, None)
        if snap and pixmap is not None:
            self.setSliderResolution(pixmap.width())
        self.setSnap(snap)
        self.update()

    def setSliderProfile(self,
                         profile,
                         colormap=None,
                         resetSliders=None,
                         snap=True):
        """
        Use the profile array to create a pixmap displayed in the slider
        groove.
        See also : RangeSlider.setSliderPixmap
        :param profile: 1D array
        :param colormap: a list of QRgb values (see : QImage.setColorTable)
        :param resetSliders: True to reset the slider positions to their
            extremes. If None the values positions will be reset if there
            was no previous profile.
        :param snap: set to True to set the number of steps equal to the length
            of the profile (equivalent to calling setNSteps and setSnap).
            If false, the number of steps will be unchanged, and snap will be
            set to False.
        :return:
        """
        if profile is None:
            self.setSliderPixmap(None)
            return

        if profile.ndim != 1:
            raise ValueError('Profile must be a 1D array.')

        if profile.shape[0] <= 1:
            raise ValueError('Profile must be have a length > 1.')

        if colormap is not None:
            nColors = len(colormap)
            if nColors > 255:
                raise ValueError('Max 256 indexed colors supported'
                                 ' at the moment')
        else:
            nColors = 255

        _min = profile.min()
        _max = profile.max()
        indices = np.int8(nColors * (profile.astype(np.float64) - _min)
                          / (_max - _min))
        qimage = Qt.QImage(indices.data,
                           indices.shape[0],
                           1,
                           Qt.QImage.Format_Indexed8)

        if colormap is not None:
            qimage.setColorTable(colormap)

        qpixmap = Qt.QPixmap.fromImage(qimage)
        self.setSliderPixmap(qpixmap,
                             resetSliders=resetSliders,
                             snap=snap)

    def paintEvent(self, event):
        painter = Qt.QPainter(self)

        style = Qt.QApplication.style()

        area = self.__drawArea()
        pixmapRect = self.__pixMapRect()

        sliders = self.__sliders

        option = Qt.QStyleOptionProgressBar()
        option.initFrom(self)
        option.rect = area
        option.state = ((self.isEnabled() and Qt.QStyle.State_Enabled)
                        or Qt.QStyle.State_None)
        style.drawControl(Qt.QStyle.CE_ProgressBarGroove,
                          option,
                          painter,
                          self)

        # showing interval rect only if show is forced or if there is not
        # background and show is True or None
        showRngBckgrnd = (self.__showRangeBackground
                          or (self.__pixmap and self.__showRangeBackground)
                          or (self.__pixmap is None
                              and self.__showRangeBackground is None)
                          or self.__showRangeBackground)

        alpha = (self.isEnabled() and 255) or 100

        if showRngBckgrnd:
            painter.save()
            rect = Qt.QRect(area)
            rect.setLeft(sliders['left'].center().x())
            rect.setRight(sliders['right'].center().x())
            gradient = Qt.QLinearGradient(area.topLeft(), area.bottomLeft())
            color = Qt.QColor(Qt.Qt.cyan)
            color.setAlpha(alpha)
            gradient.setColorAt(0., color)
            color = Qt.QColor(Qt.Qt.blue)
            color.setAlpha(alpha)
            gradient.setColorAt(1., color)
            brush = Qt.QBrush(gradient)
            painter.setBrush(brush)
            painter.drawRect(rect)
            painter.restore()

        if self.__pixmap and alpha == 255:
            painter.save()
            pen = painter.pen()
            pen.setWidth(2)
            pen.setColor(Qt.Qt.black)
            painter.setPen(pen)
            painter.drawRect(pixmapRect.adjusted(-1, -1, 1, 1))
            painter.restore()

            painter.drawPixmap(area.adjusted(self._sliderWidth/2,
                                             self._pixmapVOffset,
                                             -self._sliderWidth/2,
                                             -self._pixmapVOffset),
                               self.__pixmap,
                               self.__pixmap.rect())

        option = Qt.QStyleOptionButton()
        option.initFrom(self)

        for side, slider in sliders.items():
            option.icon = self.__sliderIcons[side]
            option.iconSize = slider.size() * 0.7
            if self.__hover == side:
                option.state |= Qt.QStyle.State_MouseOver
            elif option.state & Qt.QStyle.State_MouseOver:
                option.state ^= Qt.QStyle.State_MouseOver
            if self.__focus == side:
                option.state |= Qt.QStyle.State_HasFocus
            elif option.state & Qt.QStyle.State_HasFocus:
                option.state ^= Qt.QStyle.State_HasFocus
            option.rect = slider
            style.drawControl(Qt.QStyle.CE_PushButton,
                              option,
                              painter,
                              self)

    def sizeHint(self):
        return Qt.QSize(*self._defaultMinimumSize)
