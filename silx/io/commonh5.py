# coding: utf-8
# /*##########################################################################
# Copyright (C) 2016-2019 European Synchrotron Radiation Facility
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
import collections
try:
    from collections import abc
except ImportError:  # Python2 support
    import collections as abc
import weakref

import h5py
import numpy
import six

from . import utils

__authors__ = ["V. Valls", "P. Knobel"]
__license__ = "MIT"
__date__ = "02/07/2018"


class _MappingProxyType(abc.MutableMapping):
    """Read-only dictionary

    This class is available since Python 3.3, but not on earlyer Python
    versions.
    """

    def __init__(self, data):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def get(self, key, default=None):
        return self._data.get(key, default)

    def __setitem__(self, key, value):
        raise RuntimeError("Cannot modify read-only dictionary")

    def __delitem__(self, key):
        raise RuntimeError("Cannot modify read-only dictionary")

    def pop(self, key):
        raise RuntimeError("Cannot modify read-only dictionary")

    def clear(self):
        raise RuntimeError("Cannot modify read-only dictionary")

    def update(self, key, value):
        raise RuntimeError("Cannot modify read-only dictionary")

    def setdefault(self, key):
        raise RuntimeError("Cannot modify read-only dictionary")


class Node(object):
    """This is the base class for all :mod:`spech5` and :mod:`fabioh5`
    classes. It represents a tree node, and knows its parent node
    (:attr:`parent`).
    The API mimics a *h5py* node, with following attributes: :attr:`file`,
    :attr:`attrs`, :attr:`name`, and :attr:`basename`.
    """

    def __init__(self, name, parent=None, attrs=None):
        self._set_parent(parent)
        self.__basename = name
        self.__attrs = {}
        if attrs is not None:
            self.__attrs.update(attrs)

    def _set_basename(self, name):
        self.__basename = name

    @property
    def h5_class(self):
        """Returns the HDF5 class which is mimicked by this class.

        :rtype: H5Type
        """
        raise NotImplementedError()

    @property
    def h5py_class(self):
        """Returns the h5py classes which is mimicked by this class. It can be
        one of `h5py.File, h5py.Group` or `h5py.Dataset`

        This should not be used anymore. Prefer using `h5_class`

        :rtype: Class
        """
        h5_class = self.h5_class
        if h5_class == utils.H5Type.FILE:
            return h5py.File
        elif h5_class == utils.H5Type.GROUP:
            return h5py.Group
        elif h5_class == utils.H5Type.DATASET:
            return h5py.Dataset
        elif h5_class == utils.H5Type.SOFT_LINK:
            return h5py.SoftLink
        raise NotImplementedError()

    @property
    def parent(self):
        """Returns the parent of the node.

        :rtype: Node
        """
        if self.__parent is None:
            parent = None
        else:
            parent = self.__parent()
            if parent is None:
                self.__parent = None
        return parent

    def _set_parent(self, parent):
        """Set the parent of this node.

        It do not update the parent object.

        :param Node parent: New parent for this node
        """
        if parent is not None:
            self.__parent = weakref.ref(parent)
        else:
            self.__parent = None

    @property
    def file(self):
        """Returns the file node of this node.

        :rtype: Node
        """
        node = self
        while node.parent is not None:
            node = node.parent
        if isinstance(node, File):
            return node
        else:
            return None

    @property
    def attrs(self):
        """Returns HDF5 attributes of this node.

        :rtype: dict
        """
        if self._is_editable():
            return self.__attrs
        else:
            return _MappingProxyType(self.__attrs)

    @property
    def name(self):
        """Returns the HDF5 name of this node.
        """
        parent = self.parent
        if parent is None:
            return "/"
        if parent.name == "/":
            return "/" + self.basename
        return parent.name + "/" + self.basename

    @property
    def basename(self):
        """Returns the HDF5 basename of this node.
        """
        return self.__basename

    def _is_editable(self):
        """Returns true if the file is editable or if the node is not linked
        to a tree.

        :rtype: bool
        """
        f = self.file
        return f is None or f.mode == "w"


class Dataset(Node):
    """This class handles a numpy data object, as a mimicry of a
    *h5py.Dataset*.
    """

    def __init__(self, name, data, parent=None, attrs=None):
        Node.__init__(self, name, parent, attrs=attrs)
        if data is not None:
            self._check_data(data)
        self.__data = data

    def _check_data(self, data):
        """Check that the data provided by the dataset is valid.

        It is valid when it can be stored in a HDF5 using h5py.

        :param numpy.ndarray data: Data associated to the dataset
        :raises TypeError: In the case the data is not valid.
        """
        if isinstance(data, (six.text_type, six.binary_type)):
            return

        chartype = data.dtype.char
        if chartype == "U":
            pass
        elif chartype == "O":
            d = h5py.special_dtype(vlen=data.dtype)
            if d is not None:
                return
            d = h5py.special_dtype(ref=data.dtype)
            if d is not None:
                return
        else:
            return

        msg = "Type of the dataset '%s' is not supported. Found '%s'."
        raise TypeError(msg % (self.name, data.dtype))

    def _set_data(self, data):
        """Set the data exposed by the dataset.

        It have to be called only one time before the data is used. It should
        not be edited after use.

        :param numpy.ndarray data: Data associated to the dataset
        """
        self._check_data(data)
        self.__data = data

    def _get_data(self):
        """Returns the exposed data

        :rtype: numpy.ndarray
        """
        return self.__data

    @property
    def h5_class(self):
        """Returns the HDF5 class which is mimicked by this class.

        :rtype: H5Type
        """
        return utils.H5Type.DATASET

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

    def __iter__(self):
        """Iterate over the first axis. TypeError if scalar."""
        if len(self.shape) == 0:
            raise TypeError("Can't iterate over a scalar dataset")
        return self._get_data().__iter__()

    # make comparisons and operations on the data
    def __eq__(self, other):
        """When comparing datasets, compare the actual data."""
        if utils.is_dataset(other):
            return self[()] == other[()]
        return self[()] == other

    def __add__(self, other):
        return self[()] + other

    def __radd__(self, other):
        return other + self[()]

    def __sub__(self, other):
        return self[()] - other

    def __rsub__(self, other):
        return other - self[()]

    def __mul__(self, other):
        return self[()] * other

    def __rmul__(self, other):
        return other * self[()]

    def __truediv__(self, other):
        return self[()] / other

    def __rtruediv__(self, other):
        return other / self[()]

    def __floordiv__(self, other):
        return self[()] // other

    def __rfloordiv__(self, other):
        return other // self[()]

    def __neg__(self):
        return -self[()]

    def __abs__(self):
        return abs(self[()])

    def __float__(self):
        return float(self[()])

    def __int__(self):
        return int(self[()])

    def __bool__(self):
        if self[()]:
            return True
        return False

    def __nonzero__(self):
        # python 2
        return self.__bool__()

    def __ne__(self, other):
        if utils.is_dataset(other):
            return self[()] != other[()]
        else:
            return self[()] != other

    def __lt__(self, other):
        if utils.is_dataset(other):
            return self[()] < other[()]
        else:
            return self[()] < other

    def __le__(self, other):
        if utils.is_dataset(other):
            return self[()] <= other[()]
        else:
            return self[()] <= other

    def __gt__(self, other):
        if utils.is_dataset(other):
            return self[()] > other[()]
        else:
            return self[()] > other

    def __ge__(self, other):
        if utils.is_dataset(other):
            return self[()] >= other[()]
        else:
            return self[()] >= other

    def __getattr__(self, item):
        """Proxy to underlying numpy array methods.
        """
        data = self._get_data()
        if hasattr(data, item):
            return getattr(data, item)

        raise AttributeError("Dataset has no attribute %s" % item)


class DatasetProxy(Dataset):
    """Virtual dataset providing content of another dataset"""

    def __init__(self, name, target, parent=None):
        Dataset.__init__(self, name, data=None, parent=parent)
        if not utils.is_dataset(target):
            raise TypeError("A Dataset is expected but %s found", target.__class__)
        self.__target = target

    @property
    def shape(self):
        return self.__target.shape

    @property
    def size(self):
        return self.__target.size

    @property
    def dtype(self):
        return self.__target.dtype

    def _get_data(self):
        return self.__target[...]

    @property
    def attrs(self):
        return self.__target.attrs


class _LinkToDataset(Dataset):
    """Virtual dataset providing link to another dataset"""

    def __init__(self, name, target, parent=None):
        Dataset.__init__(self, name, data=None, parent=parent)
        self.__target = target

    def _get_data(self):
        return self.__target._get_data()

    @property
    def attrs(self):
        return self.__target.attrs


class LazyLoadableDataset(Dataset):
    """Abstract dataset which provides a lazy loading of the data.

    The class has to be inherited and the :meth:`_create_data` method has to be
    implemented to return the numpy data exposed by the dataset. This factory
    method is only called once, when the data is needed.
    """

    def __init__(self, name, parent=None, attrs=None):
        super(LazyLoadableDataset, self).__init__(name, None, parent, attrs=attrs)
        self._is_initialized = False

    def _create_data(self):
        """
        Factory to create the data exposed by the dataset when it is needed.

        It has to be implemented for the class to work.

        :rtype: numpy.ndarray
        """
        raise NotImplementedError()

    def _get_data(self):
        """Returns the data exposed by the dataset.

        Overwrite Dataset method :meth:`_get_data` to implement the lazy
        loading feature.

        :rtype: numpy.ndarray
        """
        if not self._is_initialized:
            data = self._create_data()
            # is_initialized before set_data to avoid infinit initialization
            # is case of wrong check of the data
            self._is_initialized = True
            self._set_data(data)
        return super(LazyLoadableDataset, self)._get_data()


class SoftLink(Node):
    """This class is a tree node that mimics a *h5py.Softlink*.

    In this implementation, the path to the target must be absolute.
    """
    def __init__(self, name, path, parent=None):
        assert str(path).startswith("/")  # TODO: h5py also allows a relative path

        Node.__init__(self, name, parent)

        # attr target defined for spech5 backward compatibility
        self.target = str(path)

    @property
    def h5_class(self):
        """Returns the HDF5 class which is mimicked by this class.

        :rtype: H5Type
        """
        return utils.H5Type.SOFT_LINK

    @property
    def path(self):
        """Soft link value. Not guaranteed to be a valid path."""
        return self.target


class Group(Node):
    """This class mimics a `h5py.Group`."""

    def __init__(self, name, parent=None, attrs=None):
        Node.__init__(self, name, parent, attrs=attrs)
        self.__items = collections.OrderedDict()

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
    def h5_class(self):
        """Returns the HDF5 class which is mimicked by this class.

        :rtype: H5Type
        """
        return utils.H5Type.GROUP

    def _get(self, name, getlink):
        """If getlink is True and name points to an existing SoftLink, this
        SoftLink is returned. In all other situations, we try to return a
        Group or Dataset, or we raise a KeyError if we fail."""
        if "/" not in name:
            result = self._get_items()[name]
        elif name.startswith("/"):
            root = self.file
            if name == "/":
                return root
            result = root._get(name[1:], getlink)
        else:
            path = name.split("/")
            result = self
            for item_name in path:
                if isinstance(result, SoftLink):
                    # traverse links
                    l_name, l_target = result.name, result.path
                    result = result.file.get(l_target)
                    if result is None:
                        raise KeyError(
                            "Unable to open object (broken SoftLink %s -> %s)" %
                            (l_name, l_target))
                if not item_name:
                    # trailing "/" in name (legal for accessing Groups only)
                    if isinstance(result, Group):
                        continue
                if not isinstance(result, Group):
                    raise KeyError("Unable to open object (Component not found)")
                result = result._get_items()[item_name]

        if isinstance(result, SoftLink) and not getlink:
            link = result
            target = result.file.get(link.path)
            if result is None:
                msg = "Unable to open object (broken SoftLink %s -> %s)"
                raise KeyError(msg % (link.name, link.path))
            # Convert SoftLink into typed group/dataset
            if isinstance(target, Group):
                result = _LinkToGroup(name=link.basename, target=target, parent=link.parent)
            elif isinstance(target, Dataset):
                result = _LinkToDataset(name=link.basename, target=target, parent=link.parent)
            else:
                raise TypeError("Unexpected target type %s" % type(target))

        return result

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
        if name not in self:
            return default

        node = self._get(name, getlink=True)
        if isinstance(node, SoftLink) and not getlink:
            # get target
            try:
                node = self._get(name, getlink=False)
            except KeyError:
                return default
        elif not isinstance(node, SoftLink) and getlink:
            # ExternalLink objects don't exist in silx, so it must be a HardLink
            node = h5py.HardLink()

        if getclass:
            obj = utils.get_h5py_class(node)
            if obj is None:
                obj = node.__class__
        else:
            obj = node
        return obj

    def __setitem__(self, name, obj):
        """Add an object to the group.

        :param str name: Location on the group to store the object.
            This path name must not exists.
        :param object obj: Object to store on the file. According to the type,
            the behaviour will not be the same.

            - `commonh5.SoftLink`: Create the corresponding link.
            - `numpy.ndarray`: The array is converted to a dataset object.
            - `commonh5.Node`: A hard link should be created pointing to the
                given object. This implementation uses a soft link.
                If the node do not have parent it is connected to the tree
                without using a link (that's a hard link behaviour).
            - other object: Convert first the object with ndarray and then
                store it. ValueError if the resulting array dtype is not
                supported.
        """
        if name in self:
            # From the h5py API
            raise RuntimeError("Unable to create link (name already exists)")

        elements = name.rsplit("/", 1)
        if len(elements) == 1:
            parent = self
            basename = elements[0]
        else:
            group_path, basename = elements
            if group_path in self:
                parent = self[group_path]
            else:
                parent = self.create_group(group_path)

        if isinstance(obj, SoftLink):
            obj._set_basename(basename)
            node = obj
        elif isinstance(obj, Node):
            if obj.parent is None:
                obj._set_basename(basename)
                node = obj
            else:
                node = SoftLink(basename, obj.name)
        elif isinstance(obj, numpy.dtype):
            node = Dataset(basename, data=obj)
        elif isinstance(obj, numpy.ndarray):
            node = Dataset(basename, data=obj)
        else:
            data = numpy.array(obj)
            try:
                node = Dataset(basename, data=data)
            except TypeError as e:
                raise ValueError(e.args[0])

        parent.add_node(node)

    def __getitem__(self, name):
        """Return a child from his name.

        :param str name: name of a member or a path throug members using '/'
            separator. A '/' as a prefix access to the root item of the tree.
        :rtype: Node
        """
        if name is None or name == "":
            raise ValueError("No name")
        return self._get(name, getlink=False)

    def __contains__(self, name):
        """Returns true if name is an existing child of this group.

        :rtype: bool
        """
        if "/" not in name:
            return name in self._get_items()

        if name.startswith("/"):
            # h5py allows to access any valid full path from any group
            node = self.file
        else:
            node = self

        name = name.lstrip("/")
        basenames = name.split("/")
        for basename in basenames:
            if basename.strip() == "":
                # presence of a trailing "/" in name
                # (OK for groups, not for datasets)
                if isinstance(node, SoftLink):
                    # traverse links
                    node = node.file.get(node.path, getlink=False)
                    if node is None:
                        # broken link
                        return False
                if utils.is_dataset(node):
                    return False
                continue
            if basename not in node._get_items():
                return False
            node = node[basename]

        return True

    def __len__(self):
        """Returns the number of children contained in this group.

        :rtype: int
        """
        return len(self._get_items())

    def __iter__(self):
        """Iterate over member names"""
        for x in self._get_items().__iter__():
            yield x

    if six.PY2:
        def keys(self):
            """Returns a list of the children's names."""
            return self._get_items().keys()

        def values(self):
            """Returns a list of the children nodes (groups and datasets).

            .. versionadded:: 0.6
            """
            return self._get_items().values()

        def items(self):
            """Returns a list of tuples containing (name, node) pairs.
            """
            return self._get_items().items()

    else:
        def keys(self):
            """Returns an iterator over the children's names in a group."""
            return self._get_items().keys()

        def values(self):
            """Returns an iterator over the children nodes (groups and datasets)
            in a group.

            .. versionadded:: 0.6
            """
            return self._get_items().values()

        def items(self):
            """Returns items iterator containing name-node mapping.

            :rtype: iterator
            """
            return self._get_items().items()

    def visit(self, func, visit_links=False):
        """Recursively visit all names in this group and subgroups.
        See the documentation for `h5py.Group.visit` for more help.

        :param func: Callable (function, method or callable object)
        :type func: callable
        """
        origin_name = self.name
        return self._visit(func, origin_name, visit_links)

    def visititems(self, func, visit_links=False):
        """Recursively visit names and objects in this group.
        See the documentation for `h5py.Group.visititems` for more help.

        :param func: Callable (function, method or callable object)
        :type func: callable
        :param bool visit_links: If *False*, ignore links. If *True*,
            call `func(name)` for links and recurse into target groups.
        """
        origin_name = self.name
        return self._visit(func, origin_name, visit_links,
                           visititems=True)

    def _visit(self, func, origin_name,
               visit_links=False, visititems=False):
        """

        :param origin_name: name of first group that initiated the recursion
            This is used to compute the relative path from each item's
            absolute path.
        """
        for member in self.values():
            ret = None
            if not isinstance(member, SoftLink) or visit_links:
                relative_name = member.name[len(origin_name):]
                # remove leading slash and unnecessary trailing slash
                relative_name = relative_name.strip("/")
                if visititems:
                    ret = func(relative_name, member)
                else:
                    ret = func(relative_name)
            if ret is not None:
                return ret
            if isinstance(member, Group):
                member._visit(func, origin_name, visit_links, visititems)

    def create_group(self, name):
        """Create and return a new subgroup.

        Name may be absolute or relative.  Fails if the target name already
        exists.

        :param str name: Name of the new group
        """
        if not self._is_editable():
            raise RuntimeError("File is not editable")
        if name in self:
            raise ValueError("Unable to create group (name already exists)")

        if name.startswith("/"):
            name = name[1:]
            return self.file.create_group(name)

        elements = name.split('/')
        group = self
        for basename in elements:
            if basename in group:
                group = group[basename]
                if not isinstance(group, Group):
                    raise RuntimeError("Unable to create group (group parent is missing")
            else:
                node = Group(basename)
                group.add_node(node)
                group = node
        return group

    def create_dataset(self, name, shape=None, dtype=None, data=None, **kwds):
        """Create and return a sub dataset.

        :param str name: Name of the dataset.
        :param shape: Dataset shape. Use "()" for scalar datasets.
            Required if "data" isn't provided.
        :param dtype: Numpy dtype or string.
            If omitted, dtype('f') will be used.
            Required if "data" isn't provided; otherwise, overrides data
            array's dtype.
        :param numpy.ndarray data: Provide data to initialize the dataset.
            If used, you can omit shape and dtype arguments.
        :param kwds: Extra arguments. Nothing yet supported.
        """
        if not self._is_editable():
            raise RuntimeError("File is not editable")
        if len(kwds) > 0:
            raise TypeError("Extra args provided, but nothing supported")
        if "/" in name:
            raise TypeError("Path are not supported")
        if data is None:
            if dtype is None:
                dtype = numpy.float
            data = numpy.empty(shape=shape, dtype=dtype)
        elif dtype is not None:
            data = data.astype(dtype)
        dataset = Dataset(name, data)
        self.add_node(dataset)
        return dataset


class _LinkToGroup(Group):
    """Virtual group providing link to another group"""

    def __init__(self, name, target, parent=None):
        Group.__init__(self, name, parent=parent)
        self.__target = target

    def _get_items(self):
        return self.__target._get_items()

    @property
    def attrs(self):
        return self.__target.attrs


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

    def __init__(self, name=None, mode=None, attrs=None):
        """
        Constructor

        :param str name: File name if it exists
        :param str mode: Access mode
            - "r": Read-only. Methods :meth:`create_dataset` and
                :meth:`create_group` are locked.
            - "w": File is editable. Methods :meth:`create_dataset` and
                :meth:`create_group` are available.
        :param dict attrs: Default attributes
        """
        Group.__init__(self, name="", parent=None, attrs=attrs)
        self._file_name = name
        if mode is None:
            mode = "r"
        assert(mode in ["r", "w"])
        self._mode = mode

    @property
    def filename(self):
        return self._file_name

    @property
    def mode(self):
        return self._mode

    @property
    def h5_class(self):
        """Returns the :class:`h5py.File` class"""
        return utils.H5Type.FILE

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close the object, and free up associated resources.
        """
        # should be implemented in subclass
        pass
