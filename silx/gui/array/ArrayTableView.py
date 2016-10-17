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
from . import ArrayTableModel


__authors__ = ["V.A. Sole"]
__license__ = "MIT"
__date__ = "17/10/2016"


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
        #some linux distributions need this call
        self.setModel(self._model)

    def setCurrentArrayIndex(self, index):
        """Set the active slice/image index in the n-dimensional array

        :param index: Sequence of indices defining the active data slice in
            a n-dimensional array. The sequence length is :mat:`n-2`
        :raise IndexError: If any index in the index sequence is out of bound
            on its respective axis.
        """
        self._model.setCurrentArrayIndex(index)


if __name__ == "__main__":
    # display the 4-th (1000, 1000) image in an array of 5 images
    import numpy
    a = qt.QApplication([])
    d = numpy.random.normal(0, 1, (5, 1000, 1000))
    for i in range(5):
        d[i, :, :] += i
    w = ArrayTableView()
    w.setArrayData(d)
    w.setCurrentArrayIndex([3])
    w.show()
    a.exec_()
