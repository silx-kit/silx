# coding: utf-8
#/*##########################################################################
# Copyright (C) 2016 European Synchrotron Radiation Facility
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
#############################################################################*/
"""This module provides functions to read fabio images as an HDF5 file.

    >>> import silx.io.fabioh5
    >>> f = silx.io.fabioh5.File("foobar.edf")

.. note:: This module has a dependency on the `h5py <http://www.h5py.org/>`_
    and `fabio <https://github.com/silx-kit/fabio>`_ libraries,
    which are not a mandatory dependencies for `silx`. You might need
    to install it if you don't already have it.
"""

import os.path
import collections
import numpy
import logging

_logger = logging.getLogger(__name__)

try:
    import fabio
except ImportError as e:
    _logger.error("Module %s requires fabio", __name__)
    raise e

try:
    import h5py
except ImportError as e:
    _logger.error("Module %s requires h5py", __name__)
    raise e


class Node(object):
    """Main class for all fabioh5 classes. Help to manage a tree."""

    def __init__(self, name, parent=None):
        self.__parent = parent
        self.__name = name

    @property
    def h5py_class(self):
        """h5py classes which is mimicked by this class."""
        raise NotImplementedError()

    @property
    def parent(self):
        return self.__parent

    @property
    def file(self):
        node = self
        while node.__parent is not None:
            node = node.__parent
        if isinstance(node, File):
            return node
        else:
            return None

    def _set_parent(self, parent):
        self.__parent = parent

    @property
    def attrs(self):
        return {}

    @property
    def name(self):
        return self.__name

class Group(Node):
    """Class which mimick a sinple h5py group."""

    def __init__(self, name, parent):
        Node.__init__(self, name, parent)
        self.__items = collections.OrderedDict()

    def add_node(self, group):
        self.__items[group.name] = group
        group._set_parent(self)

    @property
    def h5py_class(self):
        return h5py.Group

    def items(self):
        return self.__items.items()

    def get(self, name, default=None, getclass=False, getlink=False):
        """ Retrieve an item or other information.

        If getlink only is true, the returned value is always HardLink
            cause this implementation do not use links. Like the original
            implementation.

        :param name str: name of the item
        :param default object: default value returned if the name is not found
        :param getclass bool: if true, the returned object is the class of the object found
        :param getlink bool: if true, links object are returned instead of the target
        :return: An object, else None
        :rtype: object
        """
        if name not in self.__items:
            return default

        if getlink:
            node = h5py.HardLink()
        else:
            node = self.__items[name]

        if getclass:
            object = node.__class__
        else:
            object = node
        return object

    def __len__(self):
        """Number of members attached to this group"""
        return len(self.__items)

    def __iter__(self):
        """Iterate over member names"""
        for x in self.__items.__iter__():
            yield x

    def __getitem__(self, name):
        """Return a member from is name"""
        return self.__items[name]

    def __contains__(self, name):
        """Test if a member name exists"""
        return name in self.__items


class MetadataGroup(Group):
    """Class which contains all metadata from a fabio image."""

    def __init__(self, fabio_file, parent):
        Group.__init__(self, "metadata", parent)
        self.__fabio_image = fabio_file
        header = self.__fabio_image.getheader()
        self._create_child(header)

    def _convert_scalar_value(self, value):
        try:
            value = int(value)
            return numpy.int_(value)
        except ValueError:
            try:
                value = float(value)
                return numpy.double(value)
            except ValueError:
                return numpy.string_(value)

    def _convert_list(self, value):
        try:
            numpy_values = []
            values = value.split(" ")
            types = set([])
            for string_value in values:
                v = self._convert_scalar_value(string_value)
                numpy_values.append(v)
                types.add(v.dtype.type)

            if numpy.string_ in types:
                return numpy.string_(value)
            if numpy.double in types:
                return numpy.array(numpy_values, dtype=numpy.double)
            else:
                return numpy.array(numpy_values, dtype=numpy.int_)
        except ValueError:
            return numpy.string_(value)

    def _convert_value(self, value):
        if " " in value:
            result = self._convert_list(value)
        else:
            result = self._convert_scalar_value(value)
        return result

    def _ignore_key(self, key):
        return False

    def _create_child(self, header):
        for key, value in header.items():
            if self._ignore_key(key):
                continue
            numpy_value = self._convert_value(value)
            dataset = Dataset(key, numpy_value, self)
            self.add_node(dataset)


class EdfMetadataGroup(MetadataGroup):
    """Class which contains all metadata from a fabio EDF image.

    It is mostly the same as MetadataGroup, but counter_mne and
    motor_mne have there own sub groups.
    """

    def _ignore_key(self, key):
        return key in self.__ignore_keys

    def _create_child(self, header):
        self.__ignore_keys = set([])

        if "motor_pos" in header and "motor_mne" in header:
            try:
                group = self._create_mnemonic_group(header, "motor")
                self.add_node(group)
                self.__ignore_keys.add("motor_pos")
                self.__ignore_keys.add("motor_mne")
            except ValueError:
                pass
        if "counter_pos" in header and "counter_mne" in header:
            try:
                group = self._create_mnemonic_group(header, "counter")
                self.add_node(group)
                self.__ignore_keys.add("counter_pos")
                self.__ignore_keys.add("counter_mne")
            except ValueError:
                pass

        MetadataGroup._create_child(self, header)

    def _create_mnemonic_group(self, header, base_key):
        mnemonic_values_key = base_key + "_mne"
        mnemonic_values =  header.get(mnemonic_values_key, "")
        mnemonic_values = mnemonic_values.split()
        pos_values_key = base_key + "_pos"
        pos_values =  header.get(pos_values_key, "")
        pos_values = pos_values.split()

        group = Group(base_key, self)
        nbitems = max(len(mnemonic_values), len(pos_values))
        for i in range(nbitems):
            if i < len(mnemonic_values):
                mnemonic = mnemonic_values[i]
            else:
                mnemonic = "_"

            if i < len(pos_values):
                pos = pos_values[i]
            else:
                pos = "_"
            numpy_pos = self._convert_value(pos)
            dataset = Dataset(mnemonic, numpy_pos, group)
            group.add_node(dataset)

        return group

class FrameGroup(Group):
    """Class which contains all frames from a fabio image.
    """

    def __init__(self, fabio_file, parent):
        Group.__init__(self, "frames", parent)
        self.__fabio_image = fabio_file
        self._create_child()

    def _create_child(self):

        for frame in range(self.__fabio_image.nframes):
            if self.__fabio_image.nframes == 1:
                data = self.__fabio_image.data
            else:
                data = self.__fabio_image.getframe(frame).data

            dataset = Dataset("frame_%i" % (frame + 1), data, self)
            self.add_node(dataset)


class File(Group):
    """Class which handle a fabio image as a mimick of a h5py.File.
    """

    def __init__(self, file_name=None, fabio_image=None):
        if file_name is not None and fabio_image is not None:
            raise TypeError("Parameters file_name and fabio_image are mutually exclusive.")
        if file_name is not None:
            self.__fabio_image = fabio.open(file_name)
        elif fabio_image is not None:
            self.__fabio_image = fabio_image
        Group.__init__(self, os.path.basename(self.__fabio_image.filename), None)
        self.add_node(self.create_metadata_group(self.__fabio_image))
        self.add_node(self.create_frame_group(self.__fabio_image))

    def create_metadata_group(self, fabio_file):
        if isinstance(fabio_file, fabio.edfimage.EdfImage):
            metadata = EdfMetadataGroup(fabio_file, self)
        else:
            metadata = MetadataGroup(fabio_file, self)
        return metadata

    def create_frame_group(self, fabio_file):
        return FrameGroup(fabio_file, self)

    @property
    def h5py_class(self):
        return h5py.File

    @property
    def filename(self):
        return self.__fabio_image.filename


class Dataset(Node):
    """Class which handle a numpy data as a mimick of a h5py.Dataset.
    """

    def __init__(self, name, data, parent):
        self.__data = data
        Node.__init__(self, name, parent)

    @property
    def h5py_class(self):
        return h5py.Dataset

    @property
    def dtype(self):
        return self.__data.dtype

    @property
    def shape(self):
        return self.__data.shape

    @property
    def value(self):
        return self.__data
