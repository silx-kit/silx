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

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "06/10/2016"


import logging
import re
import numpy
from .. import qt
from .Hdf5TreeModel import Hdf5TreeModel

_logger = logging.getLogger(__name__)

try:
    import h5py
except ImportError as e:
    _logger.error("Module %s requires h5py", __name__)
    raise e

_logger = logging.getLogger(__name__)


class NexusSortFilterProxyModel(qt.QSortFilterProxyModel):
    """Try to sort items according to Nexus structure. Else sort by name."""

    def __init__(self, parent=None):
        qt.QSortFilterProxyModel.__init__(self, parent)
        self.__split = re.compile("(\\d+|\\D+)")

    def lessThan(self, sourceLeft, sourceRight):
        """Returns True if the value of the item referred to by the given
        index `sourceLeft` is less than the value of the item referred to by
        the given index `sourceRight`, otherwise returns false.

        :param qt.QModelIndex sourceLeft:
        :param qt.QModelIndex sourceRight:
        :rtype: bool
        """
        if sourceLeft.column() != Hdf5TreeModel.NAME_COLUMN:
            return super(NexusSortFilterProxyModel, self).lessThan(
                sourceLeft, sourceRight)

        # Do not sort child of root (files)
        if sourceLeft.parent() == qt.QModelIndex():
            return sourceLeft.row() < sourceRight.row()

        left = self.sourceModel().data(sourceLeft, Hdf5TreeModel.H5PY_ITEM_ROLE)
        right = self.sourceModel().data(sourceRight, Hdf5TreeModel.H5PY_ITEM_ROLE)

        if issubclass(left.h5pyClass, h5py.Group) and issubclass(right.h5pyClass, h5py.Group):
            less = self.childDatasetLessThan(left, right, "start_time")
            if less is not None:
                return less
            less = self.childDatasetLessThan(left, right, "end_time")
            if less is not None:
                return less

        left = self.sourceModel().data(sourceLeft, qt.Qt.DisplayRole)
        right = self.sourceModel().data(sourceRight, qt.Qt.DisplayRole)
        return self.nameLessThan(left, right)

    def getWordsAndNumbers(self, name):
        """
        Returns a list of words and integers composing the name.

        An input `"aaa10bbb50.30"` will return
        `["aaa", 10, "bbb", 50, ".", 30]`.

        :param str name: A name
        :rtype: list
        """
        words = self.__split.findall(name)
        result = []
        for i in words:
            if i[0].isdigit():
                i = int(i)
            result.append(i)
        return result

    def nameLessThan(self, left, right):
        """Returns True if the left string is less than the right string.

        Number composing the names are compared as integers, as result "name2"
        is smaller than "name10".

        :param str left: A string
        :param str right: A string
        :rtype: bool
        """
        leftList = self.getWordsAndNumbers(left)
        rightList = self.getWordsAndNumbers(right)
        try:
            return leftList < rightList
        except TypeError:
            # Back to string comparison if list are not type consistent
            return left < right

    def childDatasetLessThan(self, left, right, childName):
        """
        Reach the same children name of two items and compare their values.

        Returns True if the left one is smaller than the right one.

        :param Hdf5Item left: An item
        :param Hdf5Item right: An item
        :param str childName: Name of the children to search. Returns None if
            the children is not found.
        :rtype: bool
        """
        try:
            left_time = left.obj[childName].value
            right_time = right.obj[childName].value
            if isinstance(left_time, numpy.ndarray):
                return left_time[0] < right_time[0]
            return left_time < right_time
        except KeyboardInterrupt:
            raise
        except Exception as e:
            _logger.debug("Exception occurred", exc_info=True)
        return None
