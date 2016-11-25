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
"""This module defines a widget designed to display data arrays with any
number of dimensions as 2D frames (images, slices) in a table view.
The dimensions not displayed in the table can be browsed using improved
sliders.

The widget uses a standard QTableView that relies on a custom abstract table
model: :class:`silx.gui.widgets.ArrayTableModel`.
"""
from __future__ import division
import sys

from silx.gui import qt
from .ArrayTableModel import ArrayTableModel
from .FrameBrowser import HorizontalSliderWithBrowser

__authors__ = ["V.A. Sole", "P. Knobel"]
__license__ = "MIT"
__date__ = "25/11/2016"


class HorizontalSpacer(qt.QWidget):
    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)
        self.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Expanding,
                                          qt.QSizePolicy.Fixed))


def _get_shape(array_like):
    """Return shape of an array like object.

    In case the object is a nested sequence (list of lists, tuples...),
    the size of each dimension is assumed to be uniform, and is deduced from
    the length of the first sequence.

    :param array_like: Array like object: numpy array, hdf5 dataset,
        multi-dimensional sequence
    :return: Shape of array, as a tuple of integers
    """
    if hasattr(array_like, "shape"):
        return array_like.shape

    shape = []
    subsequence = array_like
    while hasattr(subsequence, "__len__"):
        shape.append(len(subsequence))
        subsequence = subsequence[0]

    return tuple(shape)


def _data_is_text(array_like):
    """Return True if data in array like object is text.

    :param array_like: Array like object: numpy array, hdf5 dataset,
        multi-dimensional sequence
    :return: True if array contains string, False otherwise.
    """
    if hasattr(array_like, "dtype"):
        t = "%s" % array_like.dtype
        if '|' in t:
            return True
        else:
            return False

    subsequence = array_like
    while hasattr(subsequence, "__len__"):
        subsequence = subsequence[0]
    else:
        first_element = subsequence

    if type(first_element) in [str, bytes]:
        return True
    if not sys.version_info[0] == 3:
        if type(first_element) == unicode:
            return True
    return False


class ArrayTableWidget(qt.QWidget):
    """This widget is designed to display data of 2D frames (images, slices)
    in a table view. The widget can load any n-dimensional array, and display
    any 2-D frame/slice in the array.

    The index of the dimensions orthogonal to the displayed frame can be set
    interactively using a browser widget (sliders, buttons and text entries).

    To set the data, use :meth:`setArrayData`.
    To select the perspective, use :meth:`setPerspective` or
    use :meth:`setFrameAxes`.
    To select the frame, use :meth:`setFrameIndex`.
    """
    def __init__(self, parent=None):
        """

        :param parent: parent QWidget
        :param labels: list of labels for each dimension of the array
        """
        qt.QWidget.__init__(self, parent)
        self.mainLayout = qt.QVBoxLayout(self)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.mainLayout.setSpacing(0)

        self.browserContainer = qt.QWidget(self)
        self.browserLayout = qt.QGridLayout(self.browserContainer)
        self.browserLayout.setContentsMargins(0, 0, 0, 0)
        self.browserLayout.setSpacing(0)

        self._dimensionLabelsText = []
        """List of text labels sorted in the increasing order of the dimension
        they apply to."""
        self._browserLabels = []
        """List of QLabel widgets."""
        self._browserWidgets = []
        """List of HorizontalSliderWithBrowser widgets."""

        self.view = qt.QTableView(self)
        self.mainLayout.addWidget(self.browserContainer)
        self.mainLayout.addWidget(self.view)

        self.model = ArrayTableModel(self)
        self.view.setModel(self.model)

    def setArrayData(self, data, labels=None, copy=True, editable=True):
        """Set the data array. Update frame browsers and labels.

        :param data: Numpy array or similar object (e.g. nested sequence,
            h5py dataset...)
        :param labels: list of labels for each dimension of the array, or
            boolean ``True`` to use default labels ("dimension 0",
            "dimension 1", ...). `None` to disable labels (default).
        :param bool copy: If *True*, store a copy of *data* in the model. If
            *False*, store a reference to *data* if possible (only possible if
            *data* is a proper numpy array or an object that implements the
            same methods).
        :param bool editable: Flag to enable editing data. Default *True*.
        """
        self._data_shape = _get_shape(data)

        n_widgets = len(self._browserWidgets)
        n_dimensions = len(self._data_shape)

        # Reset text of labels
        self._dimensionLabelsText = []
        for i in range(n_dimensions):
            if labels in [True, 1]:
                label_text = "Dimension %d" % i
            elif labels is None or i >= len(labels):
                label_text = ""
            else:
                label_text = labels[i]
            self._dimensionLabelsText.append(label_text)

        # not enough widgets, create new ones (we need n_dim - 2)
        for i in range(n_widgets, n_dimensions - 2):
            browser = HorizontalSliderWithBrowser(self.browserContainer)
            self.browserLayout.addWidget(browser, i, 1)
            self._browserWidgets.append(browser)
            browser.valueChanged.connect(self._browserSlot)
            browser.setEnabled(False)
            browser.hide()

            label = qt.QLabel(self.browserContainer)
            self._browserLabels.append(label)
            self.browserLayout.addWidget(label, i, 0)
            label.hide()

        n_widgets = len(self._browserWidgets)
        for i in range(n_widgets):
            label = self._browserLabels[i]
            browser = self._browserWidgets[i]

            if (i + 2) < n_dimensions:
                label.setText(self._dimensionLabelsText[i])
                browser.setRange(0, self._data_shape[i] - 1)
                browser.setEnabled(True)
                browser.show()
                if labels is not None:
                    label.show()
                else:
                    label.hide()
            else:
                browser.setEnabled(False)
                browser.hide()
                label.hide()

        # set model
        if _data_is_text(data):
            fmt = "%s"
        else:
            fmt = "%g"

        self.model.setFormat(fmt)
        self.model.setArrayData(data, copy=copy, editable=editable)
        # some linux distributions need this call
        self.view.setModel(self.model)

    def setFrameIndex(self, index):
        """Set the active slice/image index in the n-dimensional array.

        A frame is a 2D array extracted from an array. This frame is
        necessarily parallel to 2 axes, and orthogonal to all other axes.

        The index of a frame is a sequence of indices along the orthogonal
        axes, where the frame intersects the respective axis. The indices
        are listed in the same order as the corresponding dimensions of the
        data array.

        For example, it the data array has 5 dimensions, and we are
        considering frames whose parallel axes are the 2nd and 4th dimensions
        of the array, the frame index will be a sequence of length 3
        corresponding to the indices where the frame intersects the 1st, 3rd
        and 5th axes.

        :param index: Sequence of indices defining the active data slice in
            a n-dimensional array. The sequence length is :math:`n-2`
        :raise: IndexError if any index in the index sequence is out of bound
            on its respective axis.
        """
        self.model.setFrameIndex(index)

    def _resetBrowsers(self, perspective):
        """Adjust limits for browsers based on the perspective and the
        size of the corresponding dimensions. Reset the index to 0.
        Update the dimension in the labels.

        :param perspective: Sequence of axes/dimensions numbers (0-based)
            defining the axes orthogonal to the frame.
        """
        # for 3D arrays we can accept an int rather than a 1-tuple
        if not hasattr(perspective, "__len__"):
            perspective = [perspective]

        # perspective must be sorted
        perspective = sorted(perspective)

        n_dimensions = len(self._data_shape)
        for i in range(n_dimensions - 2):
            browser = self._browserWidgets[i]
            label = self._browserLabels[i]
            browser.setRange(0, self._data_shape[perspective[i]] - 1)
            browser.setValue(0)
            label.setText(self._dimensionLabelsText[perspective[i]])

    def setPerspective(self, perspective):
        """Set the *perspective* by specifying which axes are orthogonal
        to the frame.

        For the opposite approach (defining parallel axes), use
        :meth:`setFrameAxes` instead.

        :param perspective: Sequence of unique axes numbers (0-based) defining
            the orthogonal axes. For a n-dimensional array, the sequence
            length is :math:`n-2`. The order is of the sequence is not taken
            into account (the dimensions are displayed in increasing order
            in the widget).
        """
        self.model.setPerspective(perspective)
        self._resetBrowsers(perspective)

    def setFrameAxes(self, row_axis, col_axis):
        """Set the *perspective* by specifying which axes are parallel
        to the frame.

        For the opposite approach (defining orthogonal axes), use
        :meth:`setPerspective` instead.

        :param int row_axis: Index (0-based) of the first dimension used as a frame
            axis
        :param int col_axis: Index (0-based) of the 2nd dimension used as a frame
            axis
        """
        self.model.setFrameAxes(row_axis, col_axis)
        n_dimensions = len(self._data_shape)
        perspective = tuple(set(range(0, n_dimensions)) - {row_axis, col_axis})
        self._resetBrowsers(perspective)

    def _browserSlot(self, value):
        index = []
        for browser in self._browserWidgets:
            if browser.isEnabled():
                index.append(browser.value())
        self.setFrameIndex(index)
        self.view.reset()

    def getData(self, copy=True):
        """Return a copy of the data array, or a reference to it if
        *copy=False* is passed as parameter.

        :param bool copy: If *True* (default), return a copy of the data. If
            *False*, return a reference.
        :return: Numpy array of data, or reference to original data object
            if *copy=False*
        """
        return self.model.getData(copy=copy)


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
        w.setArrayData(d, labels=True)
    w.show()
    a.exec_()

if __name__ == "__main__":
    main()
