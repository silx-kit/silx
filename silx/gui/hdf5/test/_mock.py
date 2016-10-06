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
"""Mock for silx.gui.hdf5 module"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "29/09/2016"


try:
    import h5py
except ImportError:
    h5py = None


class Node(object):

    def __init__(self, basename, parent, h5py_class):
        self.basename = basename
        self.h5py_class = h5py_class
        self.attrs = {}
        self.parent = parent
        if parent is not None:
            self.parent._add(self)

    @property
    def name(self):
        if self.parent is None:
            return self.basename
        if self.parent.name == "":
            return self.basename
        return self.parent.name + "/" + self.basename

    @property
    def file(self):
        if self.parent is None:
            return self
        return self.parent.file


class Group(Node):
    """Mock an h5py Group"""

    def __init__(self, name, parent, h5py_class=h5py.Group):
        super(Group, self).__init__(name, parent, h5py_class)
        self.__items = {}

    def _add(self, node):
        self.__items[node.basename] = node

    def __getitem__(self, key):
        return self.__items[key]

    def __iter__(self):
        for k in self.__items:
            yield k

    def __len__(self):
        return len(self.__items)

    def get(self, name, getclass=False, getlink=False):
        result = self.__items[name]
        if getclass:
            return result.h5py_class
        return result

    def create_dataset(self, name, data):
        return Dataset(name, self, data)

    def create_group(self, name):
        return Group(name, self)


class File(Group):
    """Mock an h5py File"""

    def __init__(self, filename):
        super(File, self).__init__("", None, h5py.File)
        self.filename = filename


class Dataset(Node):
    """Mock an h5py Dataset"""

    def __init__(self, name, parent, value):
        super(Dataset, self).__init__(name, parent, h5py.Dataset)
        self.value = value
        self.shape = value.shape
        self.dtype = value.dtype
