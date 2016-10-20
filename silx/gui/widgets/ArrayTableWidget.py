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
"""This module defines a widget designed to display 2D frames (images, slices)
in a numpy array: class:`ArrayTableWidget`.
"""
from silx.gui import icons
from silx.gui import qt
from silx.gui.widgets import ArrayTableModel

__authors__ = ["V.A. Sole", "P. Knobel"]
__license__ = "MIT"
__date__ = "18/10/2016"


icon_first = icons.getQIcon("first")
icon_previous = icons.getQIcon("previous")
icon_next = icons.getQIcon("next")
icon_last = icons.getQIcon("last")


class HorizontalSpacer(qt.QWidget):
    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)
        self.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Expanding,
                                          qt.QSizePolicy.Fixed))


class FrameBrowser(qt.QWidget):
    """Frame browser widget, with 4 buttons/icons and a line edit to select
    a frame number in a stack of images."""
    sigIndexChanged = qt.pyqtSignal(object)

    def __init__(self, parent=None, n=1):
        qt.QWidget.__init__(self, parent)
        self.mainLayout = qt.QHBoxLayout(self)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.mainLayout.setSpacing(0)
        self.firstButton = qt.QPushButton(self)
        self.firstButton.setIcon(icon_first)
        self.previousButton = qt.QPushButton(self)
        self.previousButton.setIcon(icon_previous)
        self.lineEdit = qt.QLineEdit(self)
        self.lineEdit.setFixedWidth(self.lineEdit.fontMetrics().width('%05d' % n))
        validator = qt.QIntValidator(1, n, self.lineEdit)
        self.lineEdit.setText("1")
        self._oldIndex = 0
        """0-based index"""
        self.lineEdit.setValidator(validator)
        self.label = qt.QLabel(self)
        self.label.setText("of %d" % n)
        self.nextButton = qt.QPushButton(self)
        self.nextButton.setIcon(icon_next)
        self.lastButton = qt.QPushButton(self)
        self.lastButton.setIcon(icon_last)

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
        """Select first/lowest frame number"""
        self.lineEdit.setText("%d" % self.lineEdit.validator().bottom())
        self._textChangedSlot()

    def _previousClicked(self):
        """Select previous frame number"""
        if self._oldIndex >= self.lineEdit.validator().bottom():
            self.lineEdit.setText("%d" % self._oldIndex)
            self._textChangedSlot()

    def _nextClicked(self):
        """Select next frame number"""
        if self._oldIndex < (self.lineEdit.validator().top() - 1):
            self.lineEdit.setText("%d" % (self._oldIndex + 2))     # why +2?
            self._textChangedSlot()

    def _lastClicked(self):
        """Select last/highest frame number"""
        self.lineEdit.setText("%d" % self.lineEdit.validator().top())
        self._textChangedSlot()

    def _textChangedSlot(self):
        """Select frame number typed in the line edit widget"""
        txt = self.lineEdit.text()
        if not len(txt):
            self.lineEdit.setText("%d" % (self._oldIndex + 1))
            return
        new_value = int(txt) - 1
        if new_value == self._oldIndex:
            return
        ddict = {
            "event": "indexChanged",
            "old": self._oldIndex + 1,
            "new": new_value + 1,
            "id": id(self)
        }
        self._oldIndex = new_value
        self.sigIndexChanged.emit(ddict)

    def setRange(self, first, last):
        """Set minimum and maximum frame numbers"""
        return self.setLimits(first, last)

    def setLimits(self, first, last):
        """Set minimum and maximum frame numbers"""
        bottom = min(first, last)
        top = max(first, last)
        self.lineEdit.validator().setTop(top)
        self.lineEdit.validator().setBottom(bottom)
        self._oldIndex = bottom - 1
        self.lineEdit.setText("%d" % (self._oldIndex + 1))
        self.label.setText(" limits = %d, %d" % (bottom, top))

    def setNFrames(self, nframes):
        """Set minimum=1 and maximum frame numbers"""
        bottom = 1
        top = nframes
        self.lineEdit.validator().setTop(top)
        self.lineEdit.validator().setBottom(bottom)
        self._oldIndex = bottom - 1
        self.lineEdit.setText("%d" % (self._oldIndex + 1))
        self.label.setText(" of %d" % top)

    def getCurrentIndex(self):
        """Get 1-based index"""
        return self._oldIndex + 1

    def setValue(self, value):
        """Set 1-based frame index

        :param int value: Frame number"""
        self.lineEdit.setText("%d" % value)
        self._textChangedSlot()


class HorizontalSliderWithBrowser(qt.QAbstractSlider):
    """Frame browser widget, a :class:`FrameBrowser` widget and a slider,
    to select a frame in a stack of images."""
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
        """Set minimum frame number"""
        self._slider.setMinimum(value)
        maximum = self._slider.maximum()
        if value == 1:
            self._browser.setNFrames(maximum)
        else:
            self._browser.setRange(value, maximum)

    def setMaximum(self, value):
        """Set maximum frame number"""
        self._slider.setMaximum(value)
        minimum = self._slider.minimum()
        if minimum == 1:
            self._browser.setNFrames(value)
        else:
            self._browser.setRange(minimum, value)

    def setRange(self, first, last):
        """Set minimum/maximum frame numbers"""
        self._slider.setRange(first, last)
        self._browser.setRange(first, last)

    def _sliderSlot(self, value):
        """Emit selected frame number when slider is activated"""
        self._browser.setValue(value)
        self.valueChanged.emit(value)

    def _browserSlot(self, ddict):
        """Emit selected frame number when browser state is changed"""
        self._slider.setValue(ddict['new'])

    def setValue(self, value):
        """Set frame number

        :param int value: Frame number"""
        self._slider.setValue(value)
        self._browser.setValue(value)

    def value(self):
        """Get selected frame number"""
        return self._slider.value()


# TODO: color the cells according to the value?
# (subclass QItemDelegate, overload its paint method, then
# table.setItemDelegate(...))
class ArrayTableView(qt.QTableView):
    """QTableView with an additional methods to load numpy arrays
    into the model :class:`ArrayTableModel`:

     - :meth:`setArrayData`: fill data model and adjust its display format
       based on the data type
     - :meth:`setCurrentArrayIndex`: select index of slice (image) to be
       viewed """
    def __init__(self, parent=None):
        qt.QTableView.__init__(self, parent)
        self._model = ArrayTableModel.ArrayTableModel(self)
        self.setModel(self._model)

    def setArrayData(self, data):
        """Fill data model and adjust its display format
        based on the data type

        :param data: Numpy array
        """
        t = "%s" % data.dtype
        if '|' in t:
            fmt = "%s"
        else:
            fmt = "%g"
        self._model.setFormat(fmt)
        self._model.setArrayData(data)
        # some linux distributions need this call
        self.setModel(self._model)

    def setCurrentArrayIndex(self, index):
        """Set the active slice/image index in the n-dimensional array

        :param index: Sequence of indices defining the active data slice in
            a n-dimensional array. The sequence length is :mat:`n-2`
        :raise IndexError: If any index in the index sequence is out of bound
            on its respective axis.
        """
        self._model.setCurrentArrayIndex(index)


class ArrayTableWidget(qt.QWidget):
    """This widget is designed to display data of 2D frames (images, slices)
    in a table view. The widget can load any n-dimensional array, and display
    images whose axes are the last two dimensions of the array.

    The index in all the other dimensions can be set with mouse controls
    (sliders and buttons) and a text entry.

    To set the data, use :meth:`setArrayData`. """
    def __init__(self, parent=None):
        qt.QTableWidget.__init__(self, parent)
        self.mainLayout = qt.QVBoxLayout(self)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.mainLayout.setSpacing(0)

        self.browserContainer = qt.QWidget(self)
        self.browserLayout = qt.QVBoxLayout(self.browserContainer)
        self.browserLayout.setContentsMargins(0, 0, 0, 0)
        self.browserLayout.setSpacing(0)

        self._widgetList = []
        for i in range(4):
            browser = HorizontalSliderWithBrowser(self.browserContainer)
            self.browserLayout.addWidget(browser)
            self._widgetList.append(browser)
            browser.valueChanged.connect(self.browserSlot)
            if i == 0:
                browser.setEnabled(False)
                browser.hide()
        self.view = ArrayTableView(self)
        self.mainLayout.addWidget(self.browserContainer)
        self.mainLayout.addWidget(self.view)

    def setArrayData(self, data):
        """Set the data array

        :param data: Numpy array
        """
        self._array = data
        n_widgets = len(self._widgetList)
        n_dimensions = len(self._array.shape)
        if n_widgets > (n_dimensions - 2):
            for i in range((n_dimensions - 2), n_widgets):
                self._widgetList[i].setEnabled(False)
                self._widgetList[i].hide()
        else:
            for i in range(n_widgets, n_dimensions - 2):
                browser = HorizontalSliderWithBrowser(self.browserContainer)
                self.browserLayout.addWidget(browser)
                self._widgetList.append(browser)
                browser.valueChanged.connect(self.browserSlot)
                browser.setEnabled(False)
                browser.hide()
        for i in range(n_widgets):
            browser = self._widgetList[i]
            if (i + 2) < n_dimensions:
                browser.setEnabled(True)
                if browser.isHidden():
                    browser.show()
                browser.setRange(1, self._array.shape[i])
            else:
                browser.setEnabled(False)
                browser.hide()
        self.view.setArrayData(self._array)

    def browserSlot(self, value):
        if len(self._array.shape) == 3:
            self.view.setCurrentArrayIndex(value - 1)
            self.view.reset()
        else:
            index = []
            for browser in self._widgetList:
                if browser.isEnabled():
                    index.append(browser.value() - 1)
            self.view.setCurrentArrayIndex(index)
            self.view.reset()


def main():
    import numpy
    import sys
    a = qt.QApplication([])
    d = numpy.random.normal(0, 1, (4, 5, 1000, 1000))
    for j in range(4):
        for i in range(5):
            d[j, i, :, :] += i + 10 * j
    w = ArrayTableWidget()
    if "2" in sys.argv:
        print("sending a single image")
        w.setArrayData(d[0, 0])
    elif "3" in sys.argv:
        print("sending 5 images")
        w.setArrayData(d[0])
    else:
        print("sending 4 * 5 images ")
        w.setArrayData(d)
    w.show()
    a.exec_()

if __name__ == "__main__":
    main()
