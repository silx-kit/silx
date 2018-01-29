# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2018 European Synchrotron Radiation Facility
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
"""This module defines a widget able to convert a numpy array from n-dimensions
to a numpy array with less dimensions.
"""
from __future__ import division

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "29/01/2018"

import numpy
import functools
from silx.gui.widgets.FrameBrowser import HorizontalSliderWithBrowser
from silx.gui import qt
import silx.utils.weakref


class _Axis(qt.QWidget):
    """Widget displaying an axis.

    It allows to display and scroll in the axis, and provide a widget to
    map the axis with a named axis (the one from the view).
    """

    valueChanged = qt.Signal(int)
    """Emitted when the location on the axis change."""

    axisNameChanged = qt.Signal(object)
    """Emitted when the user change the name of the axis."""

    def __init__(self, parent=None):
        """Constructor

        :param parent: Parent of the widget
        """
        super(_Axis, self).__init__(parent)
        self.__axisNumber = None
        self.__customAxisNames = set([])
        self.__label = qt.QLabel(self)
        self.__axes = qt.QComboBox(self)
        self.__axes.currentIndexChanged[int].connect(self.__axisMappingChanged)
        self.__slider = HorizontalSliderWithBrowser(self)
        self.__slider.valueChanged[int].connect(self.__sliderValueChanged)
        layout = qt.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.__label)
        layout.addWidget(self.__axes)
        layout.addWidget(self.__slider, 10000)
        layout.addStretch(1)
        self.setLayout(layout)

    def slider(self):
        """Returns the slider used to display axes location.

        :rtype: HorizontalSliderWithBrowser
        """
        return self.__slider

    def setAxis(self, number, position, size):
        """Set axis information.

        :param int number: The number of the axis (from the original numpy
            array)
        :param int position: The current position in the axis (for a slicing)
        :param int size: The size of this axis (0..n)
        """
        self.__label.setText("Dimension %s" % number)
        self.__axisNumber = number
        self.__slider.setMaximum(size - 1)

    def axisNumber(self):
        """Returns the axis number.

        :rtype: int
        """
        return self.__axisNumber

    def setAxisName(self, axisName):
        """Set the current used axis name.

        If this name is not available an exception is raised. An empty string
        means that no name is selected.

        :param str axisName: The new name of the axis
        :raise ValueError: When the name is not available
        """
        if axisName == "" and self.__axes.count() == 0:
            self.__axes.setCurrentIndex(-1)
            self.__updateSliderVisibility()
        for index in range(self.__axes.count()):
            name = self.__axes.itemData(index)
            if name == axisName:
                self.__axes.setCurrentIndex(index)
                self.__updateSliderVisibility()
                return
        raise ValueError("Axis name '%s' not found", axisName)

    def axisName(self):
        """Returns the selected axis name.

        If no names are selected, an empty string is retruned.

        :rtype: str
        """
        index = self.__axes.currentIndex()
        if index == -1:
            return ""
        return self.__axes.itemData(index)

    def setAxisNames(self, axesNames):
        """Set the available list of names for the axis.

        :param List[str] axesNames: List of available names
        """
        self.__axes.clear()
        previous = self.__axes.blockSignals(True)
        self.__axes.addItem(" ", "")
        for axis in axesNames:
            self.__axes.addItem(axis, axis)
        self.__axes.blockSignals(previous)
        self.__updateSliderVisibility()

    def setCustomAxis(self, axesNames):
        """Set the available list of named axis which can be set to a value.

        :param List[str] axesNames: List of customable axis names
        """
        self.__customAxisNames = set(axesNames)
        self.__updateSliderVisibility()

    def __axisMappingChanged(self, index):
        """Called when the selected name change.

        :param int index: Selected index
        """
        self.__updateSliderVisibility()
        name = self.axisName()
        self.axisNameChanged.emit(name)

    def __updateSliderVisibility(self):
        """Update the visibility of the slider according to axis names and
        customable axis names."""
        name = self.axisName()
        isVisible = name == "" or name in self.__customAxisNames
        self.__slider.setVisible(isVisible)

    def value(self):
        """Returns the current selected position in the axis.

        :rtype: int
        """
        return self.__slider.value()

    def __sliderValueChanged(self, value):
        """Called when the selected position in the axis change.

        :param int value: Position of the axis
        """
        self.valueChanged.emit(value)

    def setNamedAxisSelectorVisibility(self, visible):
        """Hide or show the named axis combobox.
        If both the selector and the slider are hidden,
        hide the entire widget.

        :param visible: boolean
        """
        self.__axes.setVisible(visible)
        name = self.axisName()

        if not visible and name != "":
            self.setVisible(False)
        else:
            self.setVisible(True)


class NumpyAxesSelector(qt.QWidget):
    """Widget to select a view from a numpy array.

    .. image:: img/NumpyAxesSelector.png

    The widget is set with an input data using :meth:`setData`, and a requested
    output dimension using :meth:`setAxisNames`.

    Widgets are provided to selected expected input axis, and a slice on the
    non-selected axis.

    The final selected array can be reached using the getter
    :meth:`selectedData`, and the event `selectionChanged`.

    If the input data is a HDF5 Dataset, the selected output data will be a
    new numpy array.
    """

    dataChanged = qt.Signal()
    """Emitted when the input data change"""

    selectedAxisChanged = qt.Signal()
    """Emitted when the selected axis change"""

    selectionChanged = qt.Signal()
    """Emitted when the selected data change"""

    customAxisChanged = qt.Signal(str, int)
    """Emitted when a custom axis change"""

    def __init__(self, parent=None):
        """Constructor

        :param parent: Parent of the widget
        """
        super(NumpyAxesSelector, self).__init__(parent)

        self.__data = None
        self.__selectedData = None
        self.__selection = tuple()
        self.__axis = []
        self.__axisNames = []
        self.__customAxisNames = set([])
        self.__namedAxesVisibility = True
        layout = qt.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSizeConstraint(qt.QLayout.SetMinAndMaxSize)
        self.setLayout(layout)

    def clear(self):
        """Clear the widget."""
        self.setData(None)

    def setAxisNames(self, axesNames):
        """Set the axis names of the output selected data.

        Axis names are defined from slower to faster axis.

        The size of the list will constrain the dimension of the resulting
        array.

        :param List[str] axesNames: List of distinct strings identifying axis names
        """
        self.__axisNames = list(axesNames)
        assert len(set(self.__axisNames)) == len(self.__axisNames),\
            "Non-unique axes names: %s" % self.__axisNames

        delta = len(self.__axis) - len(self.__axisNames)
        if delta < 0:
            delta = 0
        for index, axis in enumerate(self.__axis):
            previous = axis.blockSignals(True)
            axis.setAxisNames(self.__axisNames)
            if index >= delta and index - delta < len(self.__axisNames):
                axis.setAxisName(self.__axisNames[index - delta])
            else:
                axis.setAxisName("")
            axis.blockSignals(previous)
        self.__updateSelectedData()

    def setCustomAxis(self, axesNames):
        """Set the available list of named axis which can be set to a value.

        :param List[str] axesNames: List of customable axis names
        """
        self.__customAxisNames = set(axesNames)
        for axis in self.__axis:
            axis.setCustomAxis(self.__customAxisNames)

    def setData(self, data):
        """Set the input data unsed by the widget.

        :param numpy.ndarray data: The input data
        """
        if self.__data is not None:
            # clean up
            for widget in self.__axis:
                self.layout().removeWidget(widget)
                widget.deleteLater()
            self.__axis = []

        self.__data = data

        if data is not None:
            # create expected axes
            dimensionNumber = len(data.shape)
            delta = dimensionNumber - len(self.__axisNames)
            for index in range(dimensionNumber):
                axis = _Axis(self)
                axis.setAxis(index, 0, data.shape[index])
                axis.setAxisNames(self.__axisNames)
                axis.setCustomAxis(self.__customAxisNames)
                if index >= delta and index - delta < len(self.__axisNames):
                    axis.setAxisName(self.__axisNames[index - delta])
                # this weak method was expected to be able to delete sub widget
                callback = functools.partial(silx.utils.weakref.WeakMethodProxy(self.__axisValueChanged), axis)
                axis.valueChanged.connect(callback)
                # this weak method was expected to be able to delete sub widget
                callback = functools.partial(silx.utils.weakref.WeakMethodProxy(self.__axisNameChanged), axis)
                axis.axisNameChanged.connect(callback)
                axis.setNamedAxisSelectorVisibility(self.__namedAxesVisibility)
                self.layout().addWidget(axis)
                self.__axis.append(axis)
        self.__normalizeAxisGeometry()

        self.dataChanged.emit()
        self.__updateSelectedData()

    def __normalizeAxisGeometry(self):
        """Update axes geometry to align all axes components together."""
        if len(self.__axis) <= 0:
            return
        lineEditWidth = max([a.slider().lineEdit().minimumSize().width() for a in self.__axis])
        limitWidth = max([a.slider().limitWidget().minimumSizeHint().width() for a in self.__axis])
        for a in self.__axis:
            a.slider().lineEdit().setFixedWidth(lineEditWidth)
            a.slider().limitWidget().setFixedWidth(limitWidth)

    def __axisValueChanged(self, axis, value):
        name = axis.axisName()
        if name in self.__customAxisNames:
            self.customAxisChanged.emit(name, value)
        else:
            self.__updateSelectedData()

    def __axisNameChanged(self, axis, name):
        """Called when an axis name change.

        :param _Axis axis: The changed axis
        :param str name: The new name of the axis
        """
        names = [x.axisName() for x in self.__axis]
        missingName = set(self.__axisNames) - set(names) - set("")
        if len(missingName) == 0:
            missingName = None
        elif len(missingName) == 1:
            missingName = list(missingName)[0]
        else:
            raise Exception("Unexpected state")

        axisChanged = True

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
                axisChanged = False
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

        if self.__data is None:
            return
        if axisChanged:
            self.selectedAxisChanged.emit()
        self.__updateSelectedData()

    def __updateSelectedData(self):
        """Update the selected data according to the state of the widget.

        It fires a `selectionChanged` event.
        """
        if self.__data is None:
            if self.__selectedData is not None:
                self.__selectedData = None
                self.__selection = tuple()
                self.selectionChanged.emit()
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
        self.__selection = tuple(selection)
        # get a view with few fixed dimensions
        # with a h5py dataset, it create a copy
        # TODO we can reuse the same memory in case of a copy
        view = self.__data[self.__selection]

        if set(self.__axisNames) - set(axisNames) != set([]):
            # Not all the expected axis are there
            if self.__selectedData is not None:
                self.__selectedData = None
                self.__selection = tuple()
                self.selectionChanged.emit()
            return

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
        """Returns the input data.

        :rtype: numpy.ndarray
        """
        return self.__data

    def selectedData(self):
        """Returns the output data.

        :rtype: numpy.ndarray
        """
        return self.__selectedData

    def selection(self):
        """Returns the selection tuple used to slice the data.

        :rtype: tuple
        """
        return self.__selection

    def setNamedAxesSelectorVisibility(self, visible):
        """Show or hide the combo-boxes allowing to map the plot axes
        to the data dimension.

        :param visible: Boolean
        """
        self.__namedAxesVisibility = visible
        for axis in self.__axis:
            axis.setNamedAxisSelectorVisibility(visible)
