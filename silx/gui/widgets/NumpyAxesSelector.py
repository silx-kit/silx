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
"""This module defines a widget designed to display data using to most adapted
view from available ones from silx.
"""
from __future__ import division

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "08/12/2016"

import numpy
import functools
from silx.gui.widgets.FrameBrowser import HorizontalSliderWithBrowser

try:
    import h5py
except ImportError:
    h5py = None


from silx.gui import qt


class _Axis(qt.QWidget):

    valueChanged = qt.Signal(int)

    axisNameChanged = qt.Signal(object)

    def __init__(self, parent):
        """Constructor"""
        super(_Axis, self).__init__(parent)
        self.__axisNumber = None
        self.__label = qt.QLabel(self)
        self.__axes = qt.QComboBox(self)
        self.__axes.currentIndexChanged[int].connect(self.__axisMappingChanged)
        self.__slider = HorizontalSliderWithBrowser(self)
        self.__slider.valueChanged[int].connect(self.__sliderValueChanged)

        self.setLayout(qt.QHBoxLayout(self))
        self.layout().setMargin(0)
        self.layout().addWidget(self.__label)
        self.layout().addWidget(self.__axes)
        self.layout().addWidget(self.__slider, 10000)
        self.layout().addStretch(1)

    def setAxis(self, number, position, size):
        self.__label.setText("Dimension %s" % number)
        self.__axisNumber = number
        self.__slider.setMaximum(size - 1)

    def axisNumber(self):
        return self.__axisNumber

    def setAxisName(self, axisName):
        if axisName == "" and self.__axes.count() == 0:
            self.__axes.setCurrentIndex(-1)
            self.__slider.setVisible(True)
        for index in range(self.__axes.count()):
            name = self.__axes.itemData(index)
            if name == axisName:
                self.__slider.setVisible(name == "")
                self.__axes.setCurrentIndex(index)
                return
        raise Exception("Axis name '%s' not found", axisName)

    def axisName(self):
        index = self.__axes.currentIndex()
        if index == -1:
            return ""
        return self.__axes.itemData(index)

    def setAxisNames(self, axesNames):
        self.__axes.clear()
        previous = self.__axes.blockSignals(True)
        self.__axes.addItem(" ", "")
        for axis in axesNames:
            self.__axes.addItem(axis, axis)
        self.__axes.blockSignals(previous)
        self.__slider.setVisible(True)

    def __axisMappingChanged(self, index):
        name = self.axisName()
        self.__slider.setVisible(name == "")
        self.axisNameChanged.emit(name)

    def value(self):
        return self.__slider.value()

    def __sliderValueChanged(self, value):
        self.valueChanged.emit(value)


class NumpyAxesSelector(qt.QWidget):
    """Widget to select a view from a numpy array"""

    dataChanged = qt.Signal()
    """Emitted when the data change"""

    selectionChanged = qt.Signal()
    """Emitted when the selected data change"""

    def __init__(self, parent=None):
        """Constructor"""
        super(NumpyAxesSelector, self).__init__(parent)

        self.__data = None
        self.__selectedData = None
        self.__axis = []
        self.__axisNames = []
        self.setLayout(qt.QVBoxLayout())
        self.layout().setMargin(0)

    def clear(self):
        self.setData(None)

    def setAxisNames(self, axesNames):
        """Set the axis names of the output selected data.

        The size of the list will constrain the dimension of the resulting
        array.

        :param list axesNames: List of string identifying axis names
        """
        self.__axisNames = list(axesNames)
        for index, axis in enumerate(self.__axis):
            previous = axis.blockSignals(True)
            axis.setAxisNames(self.__axisNames)
            if index < len(self.__axisNames):
                axis.setAxisName(self.__axisNames[index])
            else:
                axis.setAxisName("")
            axis.blockSignals(previous)

    def setData(self, data):
        if data is not None:
            # clean up
            for widget in self.__axis:
                self.layout().removeWidget(widget)
                widget.deleteLater()
            self.__axis = []

        self.__data = data

        if data is not None:
            # create expected axes
            dimensionNumber = len(data.shape)
            for number in range(dimensionNumber):
                axis = _Axis(self)
                axis.setAxis(number, 0, data.shape[number])
                axis.setAxisNames(self.__axisNames)
                if number < len(self.__axisNames):
                    axis.setAxisName(self.__axisNames[number])
                axis.valueChanged.connect(self.__updateSelectedData)
                axis.axisNameChanged.connect(functools.partial(self.__axisNameChanged, axis))
                self.layout().addWidget(axis)
                self.__axis.append(axis)

        self.dataChanged.emit()
        self.__updateSelectedData()

    def __axisNameChanged(self, axis, name):
        names = [x.axisName() for x in self.__axis]
        missingName = set(self.__axisNames) - set(names) - set("")
        if len(missingName) == 0:
            missingName = None
        elif len(missingName) == 1:
            missingName = list(missingName)[0]
        else:
            raise Exception("Unexpected state")

        if axis.axisName() == "":
            # set the removed label to another widget if it is possible
            availableWidget = None
            for widget in self.__axis:
                if widget is axis:
                    continue
                if widget.axisName() == "":
                    availableWidget = widget
                    break
            if availableWidget is None:
                # If there is no other solution we set the name at the same place
                availableWidget = axis
            previous = availableWidget.blockSignals(True)
            availableWidget.setAxisName(missingName)
            availableWidget.blockSignals(previous)
        else:
            # there is a duplicated name somewhere
            # we swap it with the missing name or with nothing
            dupWidget = None
            for widget in self.__axis:
                if widget is axis:
                    continue
                if widget.axisName() == axis.axisName():
                    dupWidget = widget
                    break
            if missingName is None:
                missingName = ""
            previous = dupWidget.blockSignals(True)
            dupWidget.setAxisName(missingName)
            dupWidget.blockSignals(previous)

        self.__updateSelectedData()

    def __updateSelectedData(self):
        if self.__data is None:
            return

        selection = []
        axisNames = []
        for slider in self.__axis:
            name = slider.axisName()
            if name == "":
                selection.append(slider.value())
            else:
                selection.append(slice(None))
                axisNames.append(name)

        # get a view with few fixed dimensions
        view = self.__data[tuple(selection)]

        # order axis as expected
        source = []
        destination = []
        order = []
        for index, name in enumerate(self.__axisNames):
            destination.append(index)
            source.append(axisNames.index(name))
        for _, s in sorted(zip(destination, source)):
            order.append(s)
        view = numpy.transpose(view, order)

        self.__selectedData = view
        self.selectionChanged.emit()

    def data(self):
        return self.__data

    def selectedData(self):
        return self.__selectedData
