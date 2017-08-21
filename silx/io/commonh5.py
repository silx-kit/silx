# coding: utf-8
# /*##########################################################################
# Copyright (C) 2016-2017 European Synchrotron Radiation Facility
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
# ############################################################################*/
"""
This module contains generic objects, emulating *h5py* groups, datasets and
files. They are used in :mod:`spech5` and :mod:`fabioh5`.

.. note:: This module has a dependency on the `h5py <http://www.h5py.org/>`_
    library, which is not a mandatory dependency for `silx`.
"""

__authors__ = ["V. Valls", "P. Knobel"]
__license__ = "MIT"
__date__ = "18/08/2017"

import collections
import h5py

import numpy


class Node(object):
    """This is the base class for all :mod:`spech5` and :mod:`fabioh5`
    classes. It represents a tree node, and knows its parent node
    (:attr:`parent`).
    The API mimics a *h5py* node, with following attributes: :attr:`file`,
    :attr:`attrs`, :attr:`name`, and :attr:`basename`

    """

    def __init__(self, name, parent=None):
        self.__parent = parent
        self.__basename = name

    @property
    def h5py_class(self):
        """Returns the h5py classes which is mimicked by this class. It can be
        one of `h5py.File, h5py.Group` or `h5py.Dataset`

        :rtype: Class
        """
        raise NotImplementedError()

    @property
    def parent(self):
        """Returns the parent of the node.

        :rtype: Node
        """
        return self.__parent

    @property
    def file(self):
        """Returns the file node of this node.

        :rtype: Node
        """
        node = self
        while node.__parent is not None:
            node = node.__parent
        if isinstance(node, File):
            return node
        else:
            return None

    def _set_parent(self, parent):
        """Set the parent of this node.

        It do not update the parent object.

        :param Node parent: New parent for this node
        """
        self.__parent = parent

    @property
    def attrs(self):
        """Returns HDF5 attributes of this node.

        :rtype: dict
        """
        return {}

    @property
    def name(self):
        """Returns the HDF5 name of this node.
        """
        if self.__parent is None:
            return "/"
        if self.__parent.name == "/":
            return "/" + self.basename
        return self.__parent.name + "/" + self.basename

    @property
    def basename(self):
        """Returns the HDF5 basename of this node.
        """
        return self.__basename


class Dataset(Node):
    """This class handles a numpy data object, as a mimicry of a
    *h5py.Dataset*.
    """

    def __init__(self, name, data, parent=None, attrs=None):
        self.__data = data
        Node.__init__(self, name, parent)
        if attrs is None:
            self.__attrs = {}
        else:
            self.__attrs = attrs

    def _set_data(self, data):
        """Set the data exposed by the dataset.

        It have to be called only one time before the data is used. It should
        not be edited after use.

        :param numpy.ndarray data: Data associated to the dataset
        """
        self.__data = data

    def _get_data(self):
        """Returns the exposed data

        :rtype: numpy.ndarray
        """
        return self.__data

    @property
    def attrs(self):
        """Returns HDF5 attributes of this node.

        :rtype: dict
        """
        return self.__attrs

    @property
    def h5py_class(self):
        """Returns the h5py classes which is mimicked by this class. It can be
        one of `h5py.File, h5py.Group` or `h5py.Dataset`

        :rtype: Class
        """
        return h5py.Dataset

    @property
    def dtype(self):
        """Returns the numpy datatype exposed by this dataset.

        :rtype: numpy.dtype
        """
        return self._get_data().dtype

    @property
    def shape(self):
        """Returns the shape of the data exposed by this dataset.

        :rtype: tuple
        """
        if isinstance(self._get_data(), numpy.ndarray):
            return self._get_data().shape
        else:
            return tuple()

    @property
    def size(self):
        """Returns the size of the data exposed by this dataset.

        :rtype: int
        """
        if isinstance(self._get_data(), numpy.ndarray):
            return self._get_data().size
        else:
            # It is returned as float64 1.0 by h5py
            return numpy.float64(1.0)

    def __len__(self):
        """Returns the size of the data exposed by this dataset.

        :rtype: int
        """
        if isinstance(self._get_data(), numpy.ndarray):
            return len(self._get_data())
        else:
            # It is returned as float64 1.0 by h5py
            raise TypeError("Attempt to take len() of scalar dataset")

    def __getitem__(self, item):
        """Returns the slice of the data exposed by this dataset.

        :rtype: numpy.ndarray
        """
        if not isinstance(self._get_data(), numpy.ndarray):
            if item == Ellipsis:
                return numpy.array(self._get_data())
            elif item == tuple():
                return self._get_data()
            else:
                raise ValueError("Scalar can only be reached with an ellipsis or an empty tuple")
        return self._get_data().__getitem__(item)

    def __str__(self):
        basename = self.name.split("/")[-1]
        return '<HDF5-like dataset "%s": shape %s, type "%s">' % \
               (basename, self.shape, self.dtype.str)

    def __getslice__(self, i, j):
        """Returns the slice of the data exposed by this dataset.

        Deprecated but still in use for python 2.7

        :rtype: numpy.ndarray
        """
        return self.__getitem__(slice(i, j, None))

    @property
    def value(self):
        """Returns the data exposed by this dataset.

        Deprecated by h5py. It is prefered to use indexing `[()]`.

        :rtype: numpy.ndarray
        """
        return self._get_data()

    @property
    def compression(self):
        """Returns compression as provided by `h5py.Dataset`.

        There is no compression."""
        return None

    @property
    def compression_opts(self):
        """Returns compression options as provided by `h5py.Dataset`.

        There is no compression."""
        return None

    @property
    def chunks(self):
        """Returns chunks as provided by `h5py.Dataset`.

        There is no chunks."""
        return None

    def __array__(self, dtype=None):
        # Special case for (0,)*-shape datasets
        if numpy.product(self.shape) == 0:
            return self[()]
        else:
            return numpy.array(self[...], dtype=self.dtype if dtype is None else dtype)


class LazyLoadableDataset(Dataset):
    """Abstract dataset which provides a lazy loading of the data.

    The class has to be inherited and the :meth:`_create_data` method has to be
    implemented to return the numpy data exposed by the dataset. This factory
    method is only called once, when the data is needed.
    """

    def __init__(self, name, parent=None, attrs=None):
        super(LazyLoadableDataset, self).__init__(name, None, parent, attrs=attrs)
        self.__is_initialized = False

    def _create_data(self):
        """
        Factory to create the data exposed by the dataset when it is needed.

        It have to be implemented to work.

        :rtype: numpy.ndarray
        """
        raise NotImplementedError()

    def _get_data(self):
        """Returns the data exposed by the dataset.

        Overwrite Dataset method :meth:`_get_data` to implement the lazy
        loading feature.

        :rtype: numpy.ndarray
        """
        if not self.__is_initialized:
            data = self._create_data()
            self._set_data(data)
            self.__is_initialized = True
        return super(LazyLoadableDataset, self)._get_data()


class SoftLink(Node):
    """This class is a tree node that mimics a
    *h5py.Softlink*.
    """
    def __init__(self, name, target, parent):
        Node.__init__(self, name, parent)

        self.target = target

        self._target_node = None

    @property
    def h5py_class(self):
        """Returns the h5py classes which is mimicked by this class. It can be
        one of `h5py.File, h5py.Group` or `h5py.Dataset`

        :rtype: Class
        """
        return h5py.SoftLink

    @property
    def target_node(self):
        """Returns the h5py classes which is mimicked by this class. It can be
        one of `h5py.File, h5py.Group` or `h5py.Dataset`

        :rtype: Class
        """
        if self._target_node is None:
            self._target_node = self.file[self.target]
        return self._target_node

    def __getattr__(self, item):
        """Proxy for target nodes attributes.
        Only called if there *isn't* an attribute with this name.
        By calling setattr, we make sure the attribute is set,
        so __getattr__ is only called the first time for each attribute.
        """
        try:
            value = getattr(self.target_node, item)
        except AttributeError:
            raise AttributeError("Soft link target %s has no attribute %s" %
                                 (self.target, item))
        setattr(self, item, value)
        return value


class Group(Node):
    """This class mimics a `h5py.Group`."""

    def __init__(self, name, parent=None, attrs=None):
        Node.__init__(self, name, parent)
        self.__items = collections.OrderedDict()
        if attrs is None:
            attrs = {}
        self.__attrs = attrs

    def _get_items(self):
        """Returns the child items as a name-node dictionary.

        :rtype: dict
        """
        return self.__items

    def add_node(self, node):
        """Add a child to this group.

        :param Node node: Child to add to this group
        """
        self._get_items()[node.basename] = node
        node._set_parent(self)

    @property
    def h5py_class(self):
        """Returns the h5py classes which is mimicked by this class.

        It returns `h5py.Group`

        :rtype: Class
        """
        return h5py.Group

    @property
    def attrs(self):
        """Returns HDF5 attributes of this node.

        :rtype: dict
        """
        return self.__attrs

    def items(self):
        """Returns items iterator containing name-node mapping.

        :rtype: iterator
        """
        return self._get_items().items()

    def get(self, name, default=None, getclass=False, getlink=False):
        """Retrieve an item or other information.

        If getlink only is true, the returned value is always `h5py.HardLink`,
        because this implementation do not use links. Like the original
        implementation.

        :param str name: name of the item
        :param object default: default value returned if the name is not found
        :param bool getclass: if true, the returned object is the class of the object found
        :param bool getlink: if true, links object are returned instead of the target
        :return: An object, else None
        :rtype: object
        """
        if name not in self._get_items():
            return default

        if getlink:
            node = h5py.HardLink()
            # TODO?: softlink
        else:
            node = self._get_items()[name]

        if getclass:
            if hasattr(node, "h5py_class"):
                obj = node.h5py_class
            else:
                obj = node.__class__
        else:
            obj = node
        return obj

    def __len__(self):
        """Returns the number of child contained in this group.

        :rtype: int
        """
        return len(self._get_items())

    def __iter__(self):
        """Iterate over member names"""
        for x in self._get_items().__iter__():
            yield x

    def __getitem__(self, name):
        """Return a child from is name.

        :param name str: name of a member or a path throug members using '/'
            separator. A '/' as a prefix access to the root item of the tree.
        :rtype: Node
        """

        if name is None or name == "":
            raise ValueError("No name")

        if "/" not in name:
            return self._get_items()[name]

        if name.startswith("/"):
            root = self
            while root.parent is not None:
                root = root.parent
            if name == "/":
                return root
            return root[name[1:]]

        path = name.split("/")
        result = self
        for item_name in path:
            if not isinstance(result, Group):
                raise KeyError("Unable to open object (Component not found)")
            result = result._get_items()[item_name]

        return result

    def __contains__(self, name):
        """Returns true if name is an existing child of this group.

        :rtype: bool
        """
        return name in self._get_items()

    def keys(self):
        return self._get_items().keys()

    def values(self):
        """
        .. versionadded:: 0.6
        """
        return self._get_items().values()


class LazyLoadableGroup(Group):
    """Abstract group which provides a lazy loading of the child.

    The class has to be inherited and the :meth:`_create_child` method has
    to be implemented to add (:meth:`_add_node`) all children. This factory
    is only called once, when children are needed.
    """

    def __init__(self, name, parent=None, attrs=None):
        Group.__init__(self, name, parent, attrs)
        self.__is_initialized = False

    def _get_items(self):
        """Returns the internal structure which contains the children.

        It overwrite method :meth:`_get_items` to implement the lazy
        loading feature.

        :rtype: dict
        """
        if not self.__is_initialized:
            self.__is_initialized = True
            self._create_child()
        return Group._get_items(self)

    def _create_child(self):
        """
        Factory to create the child contained by the group when it is needed.

        It has to be implemented to work.
        """
        raise NotImplementedError()


class File(Group):
    """This class is the special :class:`Group` that is the root node
    of the tree structure. It mimics `h5py.File`."""
    def __init__(self, file_name=None):
        Group.__init__(self, name="", parent=None,
                       attrs={"NX_class": "NXroot"})
        self._file_name = file_name

    @property
    def filename(self):
        return self._file_name

    @property
    def h5py_class(self):
        return h5py.File

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close the object, and free up associated resources.
        """
        # should be implemented in subclass
        pass
