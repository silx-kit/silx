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


# TODO: generalise perspective to n-D arrays
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
    :param data: Numpy array
    :param str fmt: Format string for representing numerical values.
        Default is ``"%g"``.
    :param sequence[int] perspective: See documentation
        of :meth:`setPerspective`.
    """
    def __init__(self, parent=None, data=None, fmt="%g", perspective=None):
        qt.QAbstractTableModel.__init__(self, parent)

        # make sure data is an array (not a scalar, list, or list of lists)
        if data is None:
            data = numpy.array([])
        elif not isinstance(data, numpy.ndarray):
            data = numpy.array(data)
        # ensure data is at least 2-dimensional
        if len(data.shape) < 1:
            data.shape = (1, 1)
        elif len(data.shape) < 2:
            data.shape = (1, data.shape[0])

        self._array = data
        """n-dimensional numpy array"""

        self._format = fmt
        """Format string (default "%g")"""

        self._index = [0 for _i in range((len(data.shape) - 2))]
        """This attribute stores the slice index. In case _array is a 3D
        array, it is an integer.  If the number of dimensions is greater
        than 3, it is a list of indices."""

        self._perspective = tuple(perspective) if perspective is not None else\
            tuple(range(0, len(self._array.shape) - 2))
        """Sequence of dimensions orthogonal to the frame to be viewed.
        For an array with ``n`` dimensions, this is a sequence of ``n-2``
        integers. the first dimension is numbered ``0``.
        By default, the data frames use the last two dimensions as their axes
        and therefore the perspective is a sequence of the first ``n-2``
        dimensions.
        For example, for a 5-D array, the default perspective is ``(0, 1, 2)``
        and the default frames axes are ``(3, 4)``."""

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

    # Methods to be implemented to subclass QAbstractTableModel
    def rowCount(self, parent_idx=None):
        return self._array.shape[self._getRowDim()]
        # return self._array.shape[-2]

    def columnCount(self, parent_idx=None):
        return self._array.shape[self._getColumnDim()]
        # return self._array.shape[-1]

    def data(self, index, role=qt.Qt.DisplayRole):
        if index.isValid() and role == qt.Qt.DisplayRole:
            row = index.row()
            col = index.column()
            selection = list(self._index)
            selection.insert(self._getRowDim(), row)
            selection.insert(self._getColumnDim(), col)

            return self._format % self._array[tuple(selection)]
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

    # Public methods
    def setArrayData(self, data, perspective=None):
        """Set the data array

        :param data: Numpy array
        :param int perspective: For a 3-D array, the array dimension acting
            as index of images, orthogonal to the image
        """
        if qt.qVersion() > "4.6":
            self.beginResetModel()
        else:
            self.reset()

        # ensure data is a numpy array
        if data is None:
            data = numpy.array([])
        elif not isinstance(data, numpy.ndarray):
            data = numpy.array(data)
        # ensure data is at least 2-dimensional
        if len(data.shape) < 1:
            data.shape = (1, 1)
        elif len(data.shape) < 2:
            data.shape = (1, data.shape[0])

        self._array = data
        self._index = [0 for _i in range((len(data.shape) - 2))]
        self._perspective = tuple(perspective) if perspective is not None else\
            tuple(range(0, len(self._array.shape) - 2))

        if qt.qVersion() > "4.6":
            self.endResetModel()

    def setCurrentArrayIndex(self, index):
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
            for i, idx in enumerate(index):
                if not 0 <= idx < shape[i]:
                    raise IndexError("Invalid index %d " % idx +
                                     "higher than %d" % shape[i])
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
                        "perspective is not relevant for 1D and 2D array")

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
                    "Invalid perspective %s for %d-D array" % (perspective, n_dimensions))

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
        perspective = tuple(set(range(0, n_dimensions)) - {row_axis, col_axis})

        if len(perspective) != n_dimensions - 2 or\
                min(perspective) < 0 or max(perspective) >= n_dimensions:
            raise IndexError(
                    "Invalid perspective %s for %d-D array" % (perspective, n_dimensions))

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
    # m = ArrayTableModel(fmt="%.5f")
    # m = ArrayTableModel(None, numpy.arange(100.), fmt="%.5f")
    # m = ArrayTableModel(None, numpy.ones((100,20)), fmt="%.5f")
    m = ArrayTableModel(data=d, fmt="%.5f")
    w.setModel(m)
    m.setCurrentArrayIndex(3)
    # m.setArrayData(numpy.ones((100,)))
    w.show()
    a.exec_()
