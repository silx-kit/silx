# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016 European Synchrotron Radiation Facility
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
from .. import qt

__authors__ = ["V.A. Sole"]
__license__ = "MIT"
__date__ = "17/10/2016"


# TODO: Use png icons in silx.resources
icon_first = ["22 22 2 1",
              ". c None",
              "# c #000000",
              "......................",
              "......................",
              ".#.................##.",
              ".#...............####.",
              ".#.............######.",
              ".#...........########.",
              ".#.........##########.",
              ".#.......############.",
              ".#.....##############.",
              ".#...################.",
              ".#.##################.",
              ".#.##################.",
              ".#...################.",
              ".#.....##############.",
              ".#.......############.",
              ".#.........##########.",
              ".#...........########.",
              ".#.............######.",
              ".#...............####.",
              ".#.................##.",
              "......................",
              "......................"]

icon_previous = ["22 22 2 1",
                 ". c None",
                 "# c #000000",
                 "......................",
                 "......................",
                 "...................##.",
                 ".................####.",
                 "...............######.",
                 ".............########.",
                 "...........##########.",
                 ".........############.",
                 ".......##############.",
                 ".....################.",
                 "...##################.",
                 "...##################.",
                 ".....################.",
                 ".......##############.",
                 ".........############.",
                 "...........##########.",
                 ".............########.",
                 "...............######.",
                 ".................####.",
                 "...................##.",
                 "......................",
                 "......................"]

icon_next = ["22 22 2 1",
             ". c None",
             "# c #000000",
             "......................",
             "......................",
             ".##...................",
             ".####.................",
             ".######...............",
             ".########.............",
             ".##########...........",
             ".############.........",
             ".##############.......",
             ".################.....",
             ".##################...",
             ".##################...",
             ".################.....",
             ".##############.......",
             ".############.........",
             ".##########...........",
             ".########.............",
             ".######...............",
             ".####.................",
             ".##...................",
             "......................",
             "......................"]

icon_last = ["22 22 2 1",
             ". c None",
             "# c #000000",
             "......................",
             "......................",
             ".##.................#.",
             ".####...............#.",
             ".######.............#.",
             ".########...........#.",
             ".##########.........#.",
             ".############.......#.",
             ".##############.....#.",
             ".################...#.",
             ".##################.#.",
             ".##################.#.",
             ".################...#.",
             ".##############.....#.",
             ".############.......#.",
             ".##########.........#.",
             ".########...........#.",
             ".######.............#.",
             ".####...............#.",
             ".##.................#.",
             "......................",
             "......................"]


class HorizontalSpacer(qt.QWidget):
    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)
        self.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Expanding,
                                          qt.QSizePolicy.Fixed))


class FrameBrowser(qt.QWidget):
    sigIndexChanged = qt.pyqtSignal(object)

    def __init__(self, parent=None, n=1):
        qt.QWidget.__init__(self, parent)
        self.mainLayout = qt.QHBoxLayout(self)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.mainLayout.setSpacing(0)
        self.firstButton = qt.QPushButton(self)
        self.firstButton.setIcon(qt.QIcon(qt.QPixmap(icon_first)))
        self.previousButton = qt.QPushButton(self)
        self.previousButton.setIcon(qt.QIcon(qt.QPixmap(icon_previous)))
        self.lineEdit = qt.QLineEdit(self)
        self.lineEdit.setFixedWidth(self.lineEdit.fontMetrics().width('%05d' % n))
        validator = qt.QIntValidator(1, n, self.lineEdit)
        self.lineEdit.setText("1")
        self._oldIndex = 0    # what is oldIndex
        self.lineEdit.setValidator(validator)
        self.label = qt.QLabel(self)
        self.label.setText("of %d" % n)
        self.nextButton = qt.QPushButton(self)
        self.nextButton.setIcon(qt.QIcon(qt.QPixmap(icon_next)))
        self.lastButton = qt.QPushButton(self)
        self.lastButton.setIcon(qt.QIcon(qt.QPixmap(icon_last)))

        self.mainLayout.addWidget(HorizontalSpacer(self))
        self.mainLayout.addWidget(self.firstButton)
        self.mainLayout.addWidget(self.previousButton)
        self.mainLayout.addWidget(self.lineEdit)
        self.mainLayout.addWidget(self.label)
        self.mainLayout.addWidget(self.nextButton)
        self.mainLayout.addWidget(self.lastButton)
        self.mainLayout.addWidget(HorizontalSpacer(self))

        self.firstButton.clicked.connect(self._firstClicked)
        self.previousButton.clicked.connect(self._previousClicked)
        self.nextButton.clicked.connect(self._nextClicked)
        self.lastButton.clicked.connect(self._lastClicked)
        self.lineEdit.editingFinished.connect(self._textChangedSlot)

    def _firstClicked(self):
        self.lineEdit.setText("%d" % self.lineEdit.validator().bottom())
        self._textChangedSlot()

    def _previousClicked(self):
        if self._oldIndex >= self.lineEdit.validator().bottom():
            self.lineEdit.setText("%d" % self._oldIndex)
            self._textChangedSlot()

    def _nextClicked(self):
        if self._oldIndex < (self.lineEdit.validator().top() - 1):
            self.lineEdit.setText("%d" % (self._oldIndex + 2))     # why +2?
            self._textChangedSlot()

    def _lastClicked(self):
        self.lineEdit.setText("%d" % self.lineEdit.validator().top())
        self._textChangedSlot()

    def _textChangedSlot(self):
        txt = self.lineEdit.text()
        if not len(txt):
            self.lineEdit.setText("%d" % (self._oldIndex + 1))
            return
        newValue = int(txt) - 1
        if newValue == self._oldIndex:
            return
        ddict = {
            "event": "indexChanged",
            "old": self._oldIndex + 1,
            "new": newValue + 1,
            "id": id(self)
        }
        self._oldIndex = newValue
        self.sigIndexChanged.emit(ddict)

    def setRange(self, first, last):
        return self.setLimits(first, last)

    def setLimits(self, first, last):
        bottom = min(first, last)
        top = max(first, last)
        self.lineEdit.validator().setTop(top)
        self.lineEdit.validator().setBottom(bottom)
        self._oldIndex = bottom - 1
        self.lineEdit.setText("%d" % (self._oldIndex + 1))
        self.label.setText(" limits = %d, %d" % (bottom, top))

    def setNFrames(self, nframes):
        bottom = 1
        top = nframes
        self.lineEdit.validator().setTop(top)
        self.lineEdit.validator().setBottom(bottom)
        self._oldIndex = bottom - 1
        self.lineEdit.setText("%d" % (self._oldIndex + 1))
        self.label.setText(" of %d" % top)

    def getCurrentIndex(self):
        return self._oldIndex + 1

    def setValue(self, value):
        self.lineEdit.setText("%d" % value)
        self._textChangedSlot()


class HorizontalSliderWithBrowser(qt.QAbstractSlider):
    sigIndexChanged = qt.pyqtSignal(object)

    def __init__(self, parent=None):
        qt.QAbstractSlider.__init__(self, parent)
        self.setOrientation(qt.Qt.Horizontal)

        self.mainLayout = qt.QHBoxLayout(self)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.mainLayout.setSpacing(2)

        self._slider = qt.QSlider(self)
        self._slider.setOrientation(qt.Qt.Horizontal)

        self._browser = FrameBrowser(self)

        self.mainLayout.addWidget(self._slider)
        self.mainLayout.addWidget(self._browser)

        self._slider.valueChanged[int].connect(self._sliderSlot)
        self._browser.sigIndexChanged.connect(self._browserSlot)

    def setMinimum(self, value):
        self._slider.setMinimum(value)
        maximum = self._slider.maximum()
        if value == 1:
            self._browser.setNFrames(maximum)
        else:
            self._browser.setRange(value, maximum)

    def setMaximum(self, value):
        self._slider.setMaximum(value)
        minimum = self._slider.minimum()
        if minimum == 1:
            self._browser.setNFrames(value)
        else:
            self._browser.setRange(minimum, value)

    def setRange(self, first, last):
        self._slider.setRange(first, last)
        self._browser.setRange(first, last)

    def _sliderSlot(self, value):
        self._browser.setValue(value)
        self.valueChanged.emit(value)

    def _browserSlot(self, ddict):
        self._slider.setValue(ddict['new'])

    def setValue(self, value):
        self._slider.setValue(value)
        self._browser.setValue(value)

    def value(self):
        return self._slider.value()


def test1(args):
    app = qt.QApplication(args)
    w = HorizontalSliderWithBrowser()

    def slot(ddict):
        print(ddict)

    w.valueChanged[int].connect(slot)
    w.setRange(8, 20)
    w.show()
    app.exec_()


def test2(args):
    app = qt.QApplication(args)
    w = FrameBrowser()

    def slot(ddict):
        print(ddict)

    w.sigIndexChanged.connect(slot)
    if len(args) > 1:
        w.setLimits(8, 20)
    w.show()
    app.exec_()


if __name__ == "__main__":
    import sys
    test1(sys.argv)

