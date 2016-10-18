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
in a numpy array.

"""

from .. import qt
from . import FrameBrowser
from . import ArrayTableView

__authors__ = ["V.A. Sole", "P. Knobel"]
__license__ = "MIT"
__date__ = "18/10/2016"


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
            browser = FrameBrowser.HorizontalSliderWithBrowser(self.browserContainer)
            self.browserLayout.addWidget(browser)
            self._widgetList.append(browser)
            browser.valueChanged.connect(self.browserSlot)
            if i == 0:
                browser.setEnabled(False)
                browser.hide()
        self.view = ArrayTableView.ArrayTableView(self)
        self.mainLayout.addWidget(self.browserContainer)
        self.mainLayout.addWidget(self.view)

    def setArrayData(self, data):
        """Set the data array

        :param data: Numpy array
        """
        self._array = data
        nWidgets = len(self._widgetList)
        nDimensions = len(self._array.shape) 
        if nWidgets > (nDimensions - 2):
            for i in range((nDimensions - 2), nWidgets):
                self._widgetList[i].setEnabled(False)
                self._widgetList[i].hide()
        else:
            for i in range(nWidgets, nDimensions - 2):
                browser = FrameBrowser.HorizontalSliderWithBrowser(self.browserContainer)
                self.browserLayout.addWidget(browser)
                self._widgetList.append(browser)
                browser.valueChanged.connect(self.browserSlot)
                browser.setEnabled(False)
                browser.hide()
        for i in range(nWidgets):
            browser = self._widgetList[i]
            if (i + 2) < nDimensions:
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
        print("sending 5 images ")
        w.setArrayData(d[0])
    else:
        print("sending a 4 * 5 images ")
        w.setArrayData(d)
    w.show()
    a.exec_()

if __name__ == "__main__":
    main()
