# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2017 European Synchrotron Radiation Facility
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
"""
This module defines a data model for displaying and editing arrays of any
number of dimensions in a table view.
"""
from __future__ import division
import numpy
import logging
from silx.gui import qt
from silx.gui.data.TextFormatter import TextFormatter

__authors__ = ["V.A. Sole"]
__license__ = "MIT"
__date__ = "27/09/2017"


_logger = logging.getLogger(__name__)


def _is_array(data):
    """Return True if object implements all necessary attributes to be used
    as a numpy array.

    :param object data: Array-like object (numpy array, h5py dataset...)
    :return: boolean
    """
    # add more required attribute if necessary
    for attr in ("shape", "dtype"):
        if not hasattr(data, attr):
            return False
    return True


class ArrayTableModel(qt.QAbstractTableModel):
    """This data model provides access to 2D slices in a N-dimensional
    array.

    A slice for a 3-D array is characterized by a perspective (the number of
    the axis orthogonal to the slice) and an index at which the slice
    intersects the orthogonal axis.

    In the n-D case, only slices parallel to the last two axes are handled. A
    slice is therefore characterized by a list of indices locating the
    slice on all the :math:`n - 2` orthogonal axes.

    :param parent: Parent QObject
    :param data: Numpy array, or object implementing a similar interface
        (e.g. h5py dataset)
    :param str fmt: Format string for representing numerical values.
        Default is ``"%g"``.
    :param sequence[int] perspective: See documentation
        of :meth:`setPerspective`.
    """
    def __init__(self, parent=None, data=None, perspective=None):
        qt.QAbstractTableModel.__init__(self, parent)

        self._array = None
        """n-dimensional numpy array"""

        self._bgcolors = None
        """(n+1)-dimensional numpy array containing RGB(A) color data
        for the background color
        """

        self._fgcolors = None
        """(n+1)-dimensional numpy array containing RGB(A) color data
        for the foreground color
        """

        self._formatter = None
        """Formatter for text representation of data"""

        formatter = TextFormatter(self)
        formatter.setUseQuoteForText(False)
        self.setFormatter(formatter)

        self._index = None
        """This attribute stores the slice index, as a list of indices
        where the frame intersects orthogonal axis."""

        self._perspective = None
        """Sequence of dimensions orthogonal to the frame to be viewed.
        For an array with ``n`` dimensions, this is a sequence of ``n-2``
        integers. the first dimension is numbered ``0``.
        By default, the data frames use the last two dimensions as their axes
        and therefore the perspective is a sequence of the first ``n-2``
        dimensions.
        For example, for a 5-D array, the default perspective is ``(0, 1, 2)``
        and the default frames axes are ``(3, 4)``."""

        # set _data and _perspective
        self.setArrayData(data, perspective=perspective)

    def _getRowDim(self):
        """The row axis is the first axis parallel to the frames
        (lowest dimension number)

        Return None for 0-D (scalar) or 1-D arrays
        """
        n_dimensions = len(self._array.shape)
        if n_dimensions < 2:
            # scalar or 1D array: no row index
            return None
        # take all dimensions and remove the orthogonal ones
        frame_axes = set(range(0, n_dimensions)) - set(self._perspective)
        # sanity check
        assert len(frame_axes) == 2
        return min(frame_axes)

    def _getColumnDim(self):
        """The column axis is the second (highest dimension) axis parallel
        to the frames

        Return None for 0-D (scalar)
        """
        n_dimensions = len(self._array.shape)
        if n_dimensions < 1:
            # scalar: no column index
            return None
        frame_axes = set(range(0, n_dimensions)) - set(self._perspective)
        # sanity check
        assert (len(frame_axes) == 2) if n_dimensions > 1 else (len(frame_axes) == 1)
        return max(frame_axes)

    def _getIndexTuple(self, table_row, table_col):
        """Return the n-dimensional index of a value in the original array,
        based on its row and column indices in the table view

        :param table_row: Row index (0-based) of a table cell
        :param table_col: Column index (0-based) of a table cell
        :return: Tuple of indices of the element in the numpy array
        """
        row_dim = self._getRowDim()
        col_dim = self._getColumnDim()

        # get indices on all orthogonal axes
        selection = list(self._index)
        # insert indices on parallel axes
        if row_dim is not None:
            selection.insert(row_dim, table_row)
        if col_dim is not None:
            selection.insert(col_dim, table_col)
        return tuple(selection)

    # Methods to be implemented to subclass QAbstractTableModel
    def rowCount(self, parent_idx=None):
        """QAbstractTableModel method
        Return number of rows to be displayed in table"""
        row_dim = self._getRowDim()
        if row_dim is None:
            # 0-D and 1-D arrays
            return 1
        return self._array.shape[row_dim]

    def columnCount(self, parent_idx=None):
        """QAbstractTableModel method
        Return number of columns to be displayed in table"""
        col_dim = self._getColumnDim()
        if col_dim is None:
            # 0-D array
            return 1
        return self._array.shape[col_dim]

    def data(self, index, role=qt.Qt.DisplayRole):
        """QAbstractTableModel method to access data values
        in the format ready to be displayed"""
        if index.isValid():
            selection = self._getIndexTuple(index.row(),
                                            index.column())
            if role == qt.Qt.DisplayRole:
                return self._formatter.toString(self._array[selection], self._array.dtype)

            if role == qt.Qt.BackgroundRole and self._bgcolors is not None:
                r, g, b = self._bgcolors[selection][0:3]
                if self._bgcolors.shape[-1] == 3:
                    return qt.QColor(r, g, b)
                if self._bgcolors.shape[-1] == 4:
                    a = self._bgcolors[selection][3]
                    return qt.QColor(r, g, b, a)

            if role == qt.Qt.ForegroundRole:
                if self._fgcolors is not None:
                    r, g, b = self._fgcolors[selection][0:3]
                    if self._fgcolors.shape[-1] == 3:
                        return qt.QColor(r, g, b)
                    if self._fgcolors.shape[-1] == 4:
                        a = self._fgcolors[selection][3]
                        return qt.QColor(r, g, b, a)

                # no fg color given, use black or white
                # based on luminosity threshold
                elif self._bgcolors is not None:
                    r, g, b = self._bgcolors[selection][0:3]
                    lum = 0.21 * r + 0.72 * g + 0.07 * b
                    if lum < 128:
                        return qt.QColor(qt.Qt.white)
                    else:
                        return qt.QColor(qt.Qt.black)

    def headerData(self, section, orientation, role=qt.Qt.DisplayRole):
        """QAbstractTableModel method
        Return the 0-based row or column index, for display in the
        horizontal and vertical headers"""
        if role == qt.Qt.DisplayRole:
            if orientation == qt.Qt.Vertical:
                return "%d" % section
            if orientation == qt.Qt.Horizontal:
                return "%d" % section
        return None

    def flags(self, index):
        """QAbstractTableModel method to inform the view whether data
        is editable or not."""
        if not self._editable:
            return qt.QAbstractTableModel.flags(self, index)
        return qt.QAbstractTableModel.flags(self, index) | qt.Qt.ItemIsEditable

    def setData(self, index, value, role=None):
        """QAbstractTableModel method to handle editing data.
        Cast the new value into the same format as the array before editing
        the array value."""
        if index.isValid() and role == qt.Qt.EditRole:
            try:
                # cast value to same type as array
                v = numpy.asscalar(
                        numpy.array(value, dtype=self._array.dtype))
            except ValueError:
                return False

            selection = self._getIndexTuple(index.row(),
                                            index.column())
            self._array[selection] = v
            self.dataChanged.emit(index, index)
            return True
        else:
            return False

    # Public methods
    def setArrayData(self, data, copy=True,
                     perspective=None, editable=False):
        """Set the data array and the viewing perspective.

        You can set ``copy=False`` if you need more performances, when dealing
        with a large numpy array. In this case, a simple reference to the data
        is used to access the data, rather than a copy of the array.

        .. warning::

            Any change to the data model will affect your original data
            array, when using a reference rather than a copy..

        :param data: n-dimensional numpy array, or any object that can be
            converted to a numpy array using ``numpy.array(data)`` (e.g.
            a nested sequence).
        :param bool copy: If *True* (default), a copy of the array is stored
            and the original array is not modified if the table is edited.
            If *False*, then the behavior depends on the data type:
            if possible (if the original array is a proper numpy array)
            a reference to the original array is used.
        :param perspective: See documentation of :meth:`setPerspective`.
            If None, the default perspective is the list of the first ``n-2``
            dimensions, to view frames parallel to the last two axes.
        :param bool editable: Flag to enable editing data. Default *False*.
        """
        if qt.qVersion() > "4.6":
            self.beginResetModel()
        else:
            self.reset()

        if data is None:
            # empty array
            self._array = numpy.array([])
        elif copy:
            # copy requested (default)
            self._array = numpy.array(data, copy=True)
            if hasattr(data, "dtype"):
                # Avoid to lose the monkey-patched h5py dtype
                self._array.dtype = data.dtype
        elif not _is_array(data):
            raise TypeError("data is not a proper array. Try setting" +
                            " copy=True to convert it into a numpy array" +
                            " (this will cause the data to be copied!)")
            # # copy not requested, but necessary
            # _logger.warning(
            #         "data is not an array-like object. " +
            #         "Data must be copied.")
            # self._array = numpy.array(data, copy=True)
        else:
            # Copy explicitly disabled & data implements required attributes.
            # We can use a reference.
            self._array = data

        # reset colors to None if new data shape is inconsistent
        valid_color_shapes = (self._array.shape + (3,),
                              self._array.shape + (4,))
        if self._bgcolors is not None:
            if self._bgcolors.shape not in valid_color_shapes:
                self._bgcolors = None
        if self._fgcolors is not None:
            if self._fgcolors.shape not in valid_color_shapes:
                self._fgcolors = None

        self.setEditable(editable)

        self._index = [0 for _i in range((len(self._array.shape) - 2))]
        self._perspective = tuple(perspective) if perspective is not None else\
            tuple(range(0, len(self._array.shape) - 2))

        if qt.qVersion() > "4.6":
            self.endResetModel()

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
        # array must be RGB or RGBA
        valid_shapes = (self._array.shape + (3,), self._array.shape + (4,))
        errmsg = "Inconsistent shape for color array, should be %s or %s" % valid_shapes

        if bgcolors is not None:
            if not _is_array(bgcolors):
                bgcolors = numpy.array(bgcolors)
            assert bgcolors.shape in valid_shapes, errmsg

        self._bgcolors = bgcolors

        if fgcolors is not None:
            if not _is_array(fgcolors):
                fgcolors = numpy.array(fgcolors)
            assert fgcolors.shape in valid_shapes, errmsg

        self._fgcolors = fgcolors

    def setEditable(self, editable):
        """Set flags to make the data editable.

        .. warning::

            If the data is a reference to a h5py dataset open in read-only
            mode, setting *editable=True* will fail and print a warning.

        .. warning::

            Making the data editable means that the underlying data structure
            in this data model will be modified.
            If the data is a reference to a public object (open with
            ``copy=False``), this could have side effects. If it is a
            reference to an HDF5 dataset, this means the file will be
            modified.

        :param bool editable: Flag to enable editing data.
        :return: True if setting desired flag succeeded, False if it failed.
        """
        self._editable = editable
        if hasattr(self._array, "file"):
            if hasattr(self._array.file, "mode"):
                if editable and self._array.file.mode == "r":
                    _logger.warning(
                            "Data is a HDF5 dataset open in read-only " +
                            "mode. Editing must be disabled.")
                    self._editable = False
                    return False
        return True

    def getData(self, copy=True):
        """Return a copy of the data array, or a reference to it
        if *copy=False* is passed as parameter.

        In case the shape was modified, to convert 0-D or 1-D data
        into 2-D data, the original shape is restored in the returned data.

        :param bool copy: If *True* (default), return a copy of the data. If
            *False*, return a reference.
        :return: numpy array of data, or reference to original data object
            if *copy=False*
        """
        data = self._array if not copy else numpy.array(self._array, copy=True)
        return data

    def setFrameIndex(self, index):
        """Set the active slice index.

        This method is only relevant to arrays with at least 3 dimensions.

        :param index: Index of the active slice in the array.
            In the general n-D case, this is a sequence of :math:`n - 2`
            indices where the slice intersects the respective orthogonal axes.
        :raise IndexError: If any index in the index sequence is out of bound
            on its respective axis.
        """
        shape = self._array.shape
        if len(shape) < 3:
            # index is ignored
            return

        if qt.qVersion() > "4.6":
            self.beginResetModel()
        else:
            self.reset()

        if len(shape) == 3:
            len_ = shape[self._perspective[0]]
            # accept integers as index in the case of 3-D arrays
            if not hasattr(index, "__len__"):
                self._index = [index]
            else:
                self._index = index
            if not 0 <= self._index[0] < len_:
                raise ValueError("Index must be a positive integer " +
                                 "lower than %d" % len_)
        else:
            # general n-D case
            for i_, idx in enumerate(index):
                if not 0 <= idx < shape[self._perspective[i_]]:
                    raise IndexError("Invalid index %d " % idx +
                                     "not in range 0-%d" % (shape[i_] - 1))
            self._index = index

        if qt.qVersion() > "4.6":
            self.endResetModel()

    def setFormatter(self, formatter):
        """Set the formatter object to be used to display data from the model

        :param TextFormatter formatter: Formatter to use
        """
        if formatter is self._formatter:
            return

        if qt.qVersion() > "4.6":
            self.beginResetModel()

        if self._formatter is not None:
            self._formatter.formatChanged.disconnect(self.__formatChanged)

        self._formatter = formatter
        if self._formatter is not None:
            self._formatter.formatChanged.connect(self.__formatChanged)

        if qt.qVersion() > "4.6":
            self.endResetModel()
        else:
            self.reset()

    def getFormatter(self):
        """Returns the text formatter used.

        :rtype: TextFormatter
        """
        return self._formatter

    def __formatChanged(self):
        """Called when the format changed.
        """
        self.reset()

    def setPerspective(self, perspective):
        """Set the perspective by defining a sequence listing all axes
        orthogonal to the frame or 2-D slice to be visualized.

        Alternatively, you can use :meth:`setFrameAxes` for the complementary
        approach of specifying the two axes parallel  to the frame.

        In the 1-D or 2-D case, this parameter is irrelevant.

        In the 3-D case, if the unit vectors describing
        your axes are :math:`\vec{x}, \vec{y}, \vec{z}`, a perspective of 0
        means you slices are parallel to :math:`\vec{y}\vec{z}`, 1 means they
        are parallel to :math:`\vec{x}\vec{z}` and 2 means they
        are parallel to :math:`\vec{x}\vec{y}`.

        In the n-D case, this parameter is a sequence of :math:`n-2` axes
        numbers.
        For instance if you want to display 2-D frames whose axes are the
        second and third dimensions of a 5-D array, set the perspective to
        ``(0, 3, 4)``.

        :param perspective: Sequence of dimensions/axes orthogonal to the
            frames.
        :raise: IndexError if any value in perspective is higher than the
            number of dimensions minus one (first dimension is 0), or
            if the number of values is different from the number of dimensions
            minus two.
        """
        n_dimensions = len(self._array.shape)
        if n_dimensions < 3:
            _logger.warning(
                    "perspective is not relevant for 1D and 2D arrays")
            return

        if not hasattr(perspective, "__len__"):
            # we can tolerate an integer for 3-D array
            if n_dimensions == 3:
                perspective = [perspective]
            else:
                raise ValueError("perspective must be a sequence of integers")

        # ensure unicity of dimensions in perspective
        perspective = tuple(set(perspective))

        if len(perspective) != n_dimensions - 2 or\
                min(perspective) < 0 or max(perspective) >= n_dimensions:
            raise IndexError(
                    "Invalid perspective " + str(perspective) +
                    " for %d-D array " % n_dimensions +
                    "with shape " + str(self._array.shape))

        if qt.qVersion() > "4.6":
            self.beginResetModel()
        else:
            self.reset()

        self._perspective = perspective

        # reset index
        self._index = [0 for _i in range(n_dimensions - 2)]

        if qt.qVersion() > "4.6":
            self.endResetModel()

    def setFrameAxes(self, row_axis, col_axis):
        """Set the perspective by specifying the two axes parallel to the frame
        to be visualised.

        The complementary approach of defining the orthogonal axes can be used
        with :meth:`setPerspective`.

        :param int row_axis: Index (0-based) of the first dimension used as a frame
            axis
        :param int col_axis: Index (0-based) of the 2nd dimension used as a frame
            axis
        :raise: IndexError if axes are invalid
        """
        if row_axis > col_axis:
            _logger.warning("The dimension of the row axis must be lower " +
                            "than the dimension of the column axis. Swapping.")
            row_axis, col_axis = min(row_axis, col_axis), max(row_axis, col_axis)

        n_dimensions = len(self._array.shape)
        if n_dimensions < 3:
            _logger.warning(
                    "Frame axes cannot be changed for 1D and 2D arrays")
            return

        perspective = tuple(set(range(0, n_dimensions)) - {row_axis, col_axis})

        if len(perspective) != n_dimensions - 2 or\
                min(perspective) < 0 or max(perspective) >= n_dimensions:
            raise IndexError(
                    "Invalid perspective " + str(perspective) +
                    " for %d-D array " % n_dimensions +
                    "with shape " + str(self._array.shape))

        if qt.qVersion() > "4.6":
            self.beginResetModel()
        else:
            self.reset()

        self._perspective = perspective
        # reset index
        self._index = [0 for _i in range(n_dimensions - 2)]

        if qt.qVersion() > "4.6":
            self.endResetModel()


if __name__ == "__main__":
    app = qt.QApplication([])
    w = qt.QTableView()
    d = numpy.random.normal(0, 1, (5, 1000, 1000))
    for i in range(5):
        d[i, :, :] += i * 10
    m = ArrayTableModel(data=d)
    w.setModel(m)
    m.setFrameIndex(3)
    # m.setArrayData(numpy.ones((100,)))
    w.show()
    app.exec_()
