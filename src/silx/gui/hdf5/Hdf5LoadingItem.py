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
__date__ = "06/07/2018"

from typing import Optional

from .. import qt
from .Hdf5Node import Hdf5Node
import silx.io.utils


class Hdf5LoadingItem(Hdf5Node):
    """Item displayed when an Hdf5Node is loading.

    At the end of the loading this item is replaced by the loaded one.
    """

    def __init__(
        self,
        text,
        parent,
        animatedIcon,
        openedPath: Optional[str] = None,
    ):
        """Constructor"""
        Hdf5Node.__init__(self, parent, openedPath=openedPath)
        self.__text = text
        self.__animatedIcon = animatedIcon
        self.__animatedIcon.register(self)

    @property
    def obj(self):
        return None

    @property
    def h5Class(self):
        """Returns the class of the stored object.

        :rtype: silx.io.utils.H5Type
        """
        return silx.io.utils.H5Type.FILE

    def dataName(self, role):
        if role == qt.Qt.DecorationRole:
            return self.__animatedIcon.currentIcon()
        if role == qt.Qt.TextAlignmentRole:
            return qt.Qt.AlignTop | qt.Qt.AlignLeft
        if role == qt.Qt.DisplayRole:
            return self.__text
        return None

    def dataDescription(self, role):
        if role == qt.Qt.DecorationRole:
            return None
        if role == qt.Qt.TextAlignmentRole:
            return qt.Qt.AlignTop | qt.Qt.AlignLeft
        if role == qt.Qt.DisplayRole:
            return "Loading..."
        return None
