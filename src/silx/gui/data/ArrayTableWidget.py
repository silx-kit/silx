# /*##########################################################################
#
# Copyright (c) 2016-2021 European Synchrotron Radiation Facility
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

The widget uses a TableView that relies on a custom abstract item
model: :class:`silx.gui.data.ArrayTableModel`.
"""
import sys

from silx.gui import qt
from silx.gui.widgets.TableWidget import TableView
from .ArrayTableModel import ArrayTableModel
from silx.gui.widgets.FrameBrowser import HorizontalSliderWithBrowser

__authors__ = ["V.A. Sole", "P. Knobel"]
__license__ = "MIT"
__date__ = "24/01/2017"


class AxesSelector(qt.QWidget):
    """Widget with two combo-boxes to select two dimensions among
    all possible dimensions of an n-dimensional array.

    The first combobox contains values from :math:`0` to :math:`n-2`.

    The choices in the 2nd CB depend on the value selected in the first one.
    If the value selected in the first CB is :math:`m`, the second one lets you
    select values from :math:`m+1` to :math:`n-1`.

    The two axes can be used to select the row axis and the column axis t
    display a slice of the array data in a table view.
    """
    sigDimensionsChanged = qt.Signal(int, int)
    """Signal emitted whenever one of the comboboxes is changed.
    The signal carries the two selected dimensions."""

    def __init__(self, parent=None, n=None):
        qt.QWidget.__init__(self, parent)
        self.layout = qt.QHBoxLayout(self)
        self.layout.setContentsMargins(0, 2, 0, 2)
        self.layout.setSpacing(10)

        self.rowsCB = qt.QComboBox(self)
        self.columnsCB = qt.QComboBox(self)

        self.layout.addWidget(qt.QLabel("Rows dimension", self))
        self.layout.addWidget(self.rowsCB)
        self.layout.addWidget(qt.QLabel("    ", self))
        self.layout.addWidget(qt.QLabel("Columns dimension", self))
        self.layout.addWidget(self.columnsCB)
        self.layout.addStretch(1)

        self._slotsAreConnected = False
        if n is not None:
            self.setNDimensions(n)

    def setNDimensions(self, n):
        """Initialize combo-boxes depending on number of dimensions of array.
        Initially, the rows dimension is the second-to-last one, and the
        columns dimension is the last one.

        Link the CBs together. MAke them emit a signal when their value is
        changed.

        :param int n: Number of dimensions of array
        """
        # remember the number of dimensions and the rows dimension
        self.n = n
        self._rowsDim = n - 2

        # ensure slots are disconnected before (re)initializing widget
        if self._slotsAreConnected:
            self.rowsCB.currentIndexChanged.disconnect(self._rowDimChanged)
            self.columnsCB.currentIndexChanged.disconnect(self._colDimChanged)

        self._clear()
        self.rowsCB.addItems([str(i) for i in range(n - 1)])
        self.rowsCB.setCurrentIndex(n - 2)
        if n >= 1:
            self.columnsCB.addItem(str(n - 1))
            self.columnsCB.setCurrentIndex(0)

        # reconnect slots
        self.rowsCB.currentIndexChanged.connect(self._rowDimChanged)
        self.columnsCB.currentIndexChanged.connect(self._colDimChanged)
        self._slotsAreConnected = True

        # emit new dimensions
        if n > 2:
            self.sigDimensionsChanged.emit(n - 2, n - 1)

    def setDimensions(self, row_dim, col_dim):
        """Set the rows and columns dimensions.

        The rows dimension must be lower than the columns dimension.

        :param int row_dim: Rows dimension
        :param int col_dim: Columns dimension
        """
        if row_dim >= col_dim:
            raise IndexError("Row dimension must be lower than column dimension")
        if not (0 <= row_dim < self.n - 1):
            raise IndexError("Row dimension must be between 0 and %d" % (self.n - 2))
        if not (row_dim < col_dim <= self.n - 1):
            raise IndexError("Col dimension must be between %d and %d" % (row_dim + 1, self.n - 1))

        # set the rows dimension; this triggers an update of columnsCB
        self.rowsCB.setCurrentIndex(row_dim)
        # columnsCB first item is "row_dim + 1". So index of "col_dim" is
        # col_dim - (row_dim + 1)
        self.columnsCB.setCurrentIndex(col_dim - row_dim - 1)

    def getDimensions(self):
        """Return a 2-tuple of the rows dimension and the columns dimension.

        :return: 2-tuple of axes numbers (row_dimension, col_dimension)
        """
        return self._getRowDim(), self._getColDim()

    def _clear(self):
        """Empty the combo-boxes"""
        self.rowsCB.clear()
        self.columnsCB.clear()

    def _getRowDim(self):
        """Get rows dimension, selected in :attr:`rowsCB`
        """
        # rows combobox contains elements "0", ..."n-2",
        # so the selected dim is always equal to the index
        return self.rowsCB.currentIndex()

    def _getColDim(self):
        """Get columns dimension, selected in :attr:`columnsCB`"""
        # columns combobox contains elements "row_dim+1", "row_dim+2", ..., "n-1"
        # so the selected dim is equal to row_dim + 1 + index
        return self._rowsDim + 1 + self.columnsCB.currentIndex()

    def _rowDimChanged(self):
        """Update columns combobox when the rows dimension is changed.

        Emit :attr:`sigDimensionsChanged`"""
        old_col_dim = self._getColDim()
        new_row_dim = self._getRowDim()

        # clear cols CB
        self.columnsCB.currentIndexChanged.disconnect(self._colDimChanged)
        self.columnsCB.clear()
        # refill cols CB
        for i in range(new_row_dim + 1, self.n):
            self.columnsCB.addItem(str(i))

        # keep previous col dimension if possible
        new_col_cb_idx = old_col_dim - (new_row_dim + 1)
        if new_col_cb_idx < 0:
            # if row_dim is now greater than the previous col_dim,
            # we select a new col_dim = row_dim + 1 (first element in cols CB)
            new_col_cb_idx = 0
        self.columnsCB.setCurrentIndex(new_col_cb_idx)

        # reconnect slot
        self.columnsCB.currentIndexChanged.connect(self._colDimChanged)

        self._rowsDim = new_row_dim

        self.sigDimensionsChanged.emit(self._getRowDim(), self._getColDim())

    def _colDimChanged(self):
        """Emit :attr:`sigDimensionsChanged`"""
        self.sigDimensionsChanged.emit(self._getRowDim(), self._getColDim())


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

    .. image:: img/ArrayTableWidget.png
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

        self.axesSelector = AxesSelector(self)

        self.view = TableView(self)

        self.mainLayout.addWidget(self.browserContainer)
        self.mainLayout.addWidget(self.axesSelector)
        self.mainLayout.addWidget(self.view)

        self.model = ArrayTableModel(self)
        self.view.setModel(self.model)

    def setArrayData(self, data, labels=None, copy=True, editable=False):
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
        :param bool editable: Flag to enable editing data. Default is *False*
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
        self.model.setArrayData(data, copy=copy, editable=editable)
        # some linux distributions need this call
        self.view.setModel(self.model)
        if editable:
            self.view.enableCut()
            self.view.enablePaste()

        # initialize & connect axesSelector
        self.axesSelector.setNDimensions(n_dimensions)
        self.axesSelector.sigDimensionsChanged.connect(self.setFrameAxes)

    def setArrayColors(self, bgcolors=None, fgcolors=None):
        """Set the colors for all table cells by passing an array
        of RGB or RGBA values (integers between 0 and 255).

        The shape of the colors array must be consistent with the data shape.

        If the data array is n-dimensional, the colors array must be
        (n+1)-dimensional, with the first n-dimensions identical to the data
        array dimensions, and the last dimension length-3 (RGB) or
        length-4 (RGBA).

        :param bgcolors: RGB or RGBA colors array, defining the background color
            for each cell in the table.
        :param fgcolors: RGB or RGBA colors array, defining the foreground color
            (text color) for each cell in the table.
        """
        self.model.setArrayColors(bgcolors, fgcolors)

    def displayAxesSelector(self, isVisible):
        """Allow to display or hide the axes selector.

        :param bool isVisible: True to display the axes selector.
        """
        self.axesSelector.setVisible(isVisible)

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
    a.exec()

if __name__ == "__main__":
    main()
