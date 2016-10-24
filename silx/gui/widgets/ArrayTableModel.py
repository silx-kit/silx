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
import numpy
import logging
from silx.gui import qt


__authors__ = ["V.A. Sole"]
__license__ = "MIT"
__date__ = "17/10/2016"


_logger = logging.getLogger(__name__)


class NumpyArrayTableModel(qt.QAbstractTableModel):
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
    def __init__(self, parent=None, data=None, fmt="%g", perspective=None):
        qt.QAbstractTableModel.__init__(self, parent)

        self._array = None
        """n-dimensional numpy array"""

        self._format = fmt
        """Format string (default "%g")"""

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
        (lowest dimension number)"""
        n_dimensions = len(self._array.shape)
        # take all dimensions and remove the orthogonal ones
        frame_axes = set(range(0, n_dimensions)) - set(self._perspective)
        assert len(frame_axes) == 2
        return min(frame_axes)

    def _getColumnDim(self):
        """The column axis is the second (highest number) axis parallel
        to the frames"""
        n_dimensions = len(self._array.shape)
        frame_axes = set(range(0, n_dimensions)) - set(self._perspective)
        assert len(frame_axes) == 2
        return max(frame_axes)

    def _getIndexTuple(self, table_row, table_col):
        """Return the n-dimensional index of a value in the original array,
        based on its row and column indices in the table view

        :param table_row: Row index (0-based) of a table cell
        :param table_col: Column index (0-based) of a table cell
        :return: Tuple of indices of the element in the numpy array
        """
        # get indices on all orthogonal axes
        selection = list(self._index)
        # insert indices on parallel axes
        selection.insert(self._getRowDim(), table_row)
        selection.insert(self._getColumnDim(), table_col)
        return tuple(selection)

    # Methods to be implemented to subclass QAbstractTableModel
    def rowCount(self, parent_idx=None):
        return self._array.shape[self._getRowDim()]

    def columnCount(self, parent_idx=None):
        return self._array.shape[self._getColumnDim()]

    def data(self, index, role=qt.Qt.DisplayRole):
        if index.isValid() and role == qt.Qt.DisplayRole:
            selection = self._getIndexTuple(index.row(),
                                            index.column())
            return self._format % self._array[selection]
        return None

    def headerData(self, section, orientation, role=qt.Qt.DisplayRole):
        """Return the 0-based row or column index, for display in the
        horizontal and vertical headers"""
        if role == qt.Qt.DisplayRole:
            if orientation == qt.Qt.Vertical:
                return "%d" % section
            if orientation == qt.Qt.Horizontal:
                return "%d" % section
        return None

    def flags(self, index):
        """All cells are editable, if possible"""
        if not self._editable:
            # case of h5py dataset open in read-only mode
            return qt.QAbstractTableModel.flags(self, index)
        return qt.QAbstractTableModel.flags(self, index) | qt.Qt.ItemIsEditable

    def setData(self, index, value, role=None):
        """When a cell is changed, modify the corresponding value in the array"""
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
    def setArrayData(self, data, fmt=None, perspective=None, copy=True):
        """Set the data array and the viewing perspective.

        You can set ``copy=False`` if you need more performances, when dealing
        with a large numpy array. In this case, a simple reference to the data
        is used to access the data, rather than a copy of the array. Any
        change to the data model may affect your original data array.

        .. warning::

            If ``copy=False`` is used with a 0-D (scalar) or 1-D array,
            the array shape will be modified to change it into a 2D array.

        :param data: n-dimensional numpy array, or any object that can be
            converted to a numpy array using ``numpy.array(data)`` (e.g.
            a nested sequence).
        :param fmt: Format string for representing numerical values.
            By default, use the format set when initializing this model.
        :param perspective: See documentation of :meth:`setPerspective`.
            If None, the default perspective is the list of the first ``n-2``
            dimensions, to view frames parallel to the last two axes.
        :param bool copy: If *True* (default), a copy of the array is stored
            and the original array is not modified if the table is edited.
            If *False*, then the behavior depends on the data type:
            if possible (if the original array is a proper numpy array)
            a reference to the original array is used.
        """
        if qt.qVersion() > "4.6":
            self.beginResetModel()
        else:
            self.reset()

        self._editable = True
        if hasattr(data, "file"):
            if hasattr(data.file, "mode"):
                self._is_h5py_dataset = True
                if data.file.mode == "r" and not copy:
                    _logger.warning(
                            "Data is an h5py dataset open in read-only " +
                            "mode. Editing is disabled.")
                    self._editable = False

        if fmt is not None:
            self._format = fmt

        # ensure data is a numpy array
        if data is None:
            self._array = numpy.array([])
        else:
            if not isinstance(data, numpy.ndarray) and not copy:
                _logger.warning(
                        "data is not a numpy array. " +
                        "Param copy=False might have no effect.")
            self._array = numpy.array(data, copy=copy)

        # remember original shape, for :meth:`getData`
        self._original_shape = self._array.shape

        # ensure data is at least 2-dimensional
        if len(self._array.shape) < 1:
            self._array.shape = (1, 1)
            _logger.warning("modifying shape of 0-D array")
        elif len(self._array.shape) < 2:
            self._array.shape = (1, self._array.shape[0])
            _logger.warning("modifying shape of 1-D array")

        self._index = [0 for _i in range((len(self._array.shape) - 2))]
        self._perspective = tuple(perspective) if perspective is not None else\
            tuple(range(0, len(self._array.shape) - 2))

        if qt.qVersion() > "4.6":
            self.endResetModel()

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
        data = numpy.array(self._array, copy=copy)
        if not self._array.shape == self._original_shape:
            data = data.reshape(self._original_shape)
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
                                     "higher than %d" % shape[i_])
            self._index = index

        if qt.qVersion() > "4.6":
            self.endResetModel()

    def setFormat(self, fmt):
        """Set format string controlling how the values are represented in
        the table view.

        :param str fmt: Format string (e.g. "%.3f", "%d", "%-10.2f", "%10.3e")
            This is the C-style format string used by python when formatting
            strings with the modulus operator.
        """
        self._format = fmt

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
        numbers. The first axis is numbered :math:`0` and the last axis is
        numbered :math:`n-2`.
        So for instance if you want to display 2-D frames whose axes are the
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
    a = qt.QApplication([])
    w = qt.QTableView()
    d = numpy.random.normal(0, 1, (5, 1000, 1000))
    for i in range(5):
        d[i, :, :] += i * 10
    # m = NumpyArrayTableModel(fmt="%.5f")
    # m = NumpyArrayTableModel(None, numpy.arange(100.), fmt="%.5f")
    # m = NumpyArrayTableModel(None, numpy.ones((100,20)), fmt="%.5f")
    m = NumpyArrayTableModel(data=d, fmt="%.5f")
    w.setModel(m)
    m.setFrameIndex(3)
    # m.setArrayData(numpy.ones((100,)))
    w.show()
    a.exec_()
