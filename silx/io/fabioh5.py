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


class Dataset(Node):
    """Class which handle a numpy data as a mimick of a h5py.Dataset.
    """

    def __init__(self, name, data, parent=None):
        self.__data = data
        Node.__init__(self, name, parent)

    def _set_data(self, data):
        self.__data = data

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
    def size(self):
        return self.__data.size

    def __len__(self):
        return len(self.__data)

    def __getitem__(self, item):
        if not isinstance(self.__data, numpy.ndarray):
            if item == Ellipsis:
                return numpy.array(self.__data)
            elif item == tuple():
                return self.__data
            else:
                raise ValueError("Scalar can only be reached with an ellipsis or an empty tuple")
        return self.__data.__getitem__(item)

    def __getslice__(self, i, j):
        # deprecated but still in use for python 2.7
        return self.__getitem__(slice(i, j, None))

    @property
    def value(self):
        return self.__data

    @property
    def compression(self):
        return None

    @property
    def compression_opts(self):
        return None


class LazyLoadableDataset(Dataset):

    def __init__(self, name, parent=None):
        super(LazyLoadableDataset, self).__init__(name, None, parent)
        self.__initialized = False

    def _create_data(self):
        raise NotImplementedError()

    def __init_data(self):
        if self.__initialized is False:
            data = self._create_data()
            self._set_data(data)
            self.__initialized = True

    def __len__(self):
        self.__init_data()
        return super(LazyLoadableDataset, self).__len__()

    def __getitem__(self, item):
        self.__init_data()
        return super(LazyLoadableDataset, self).__getitem__(item)

    def __getslice__(self, i, j):
        self.__init_data()
        return self.__getitem__(slice(i, j, None))

    @property
    def dtype(self):
        self.__init_data()
        return super(LazyLoadableDataset, self).dtype

    @property
    def shape(self):
        self.__init_data()
        return super(LazyLoadableDataset, self).shape

    @property
    def value(self):
        self.__init_data()
        return super(LazyLoadableDataset, self).value

    @property
    def size(self):
        self.__init_data()
        return super(LazyLoadableDataset, self).size


class FrameData(LazyLoadableDataset):

    def __init__(self, name, fabio_file, parent=None):
        LazyLoadableDataset.__init__(self, name, parent)
        self.__fabio_file = fabio_file

    def _create_data(self):
        """Initialize hold data by merging all frames into a single cube.

        Choose the cube size which fit the best the data. If some images are
        smaller than expected, the empty space is set to 0.

        The computation is cached into the class, and only done ones.
        """
        images = []
        for frame in range(self.__fabio_file.nframes):
            if self.__fabio_file.nframes == 1:
                image = self.__fabio_file.data
            else:
                image = self.__fabio_file.getframe(frame).data
            images.append(image)

        # get the max size
        max_shape = [0, 0]
        for image in images:
            if image.shape[0] > max_shape[0]:
                max_shape[0] = image.shape[0]
            if image.shape[1] > max_shape[1]:
                max_shape[1] = image.shape[1]
        max_shape = tuple(max_shape)

        # fix smallest images
        for index, image in enumerate(images):
            if image.shape == max_shape:
                continue
            right_image = numpy.zeros(max_shape)
            right_image[0:image.shape[0], 0:image.shape[1]] = image
            images[index] = right_image

        # create a cube
        return numpy.array(images)


class Group(Node):
    """Class which mimick a sinple h5py group."""

    def __init__(self, name, parent=None, attrs=None):
        Node.__init__(self, name, parent)
        self.__items = collections.OrderedDict()
        if attrs is None:
            attrs = {}
        self.__attrs = attrs

    def add_node(self, node):
        self.__items[node.name] = node
        node._set_parent(self)

    @property
    def h5py_attrs(self):
        return self.__attrs

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
            obj = node.h5py_class
        else:
            obj = node
        return obj

    def __len__(self):
        """Number of members attached to this group"""
        return len(self.__items)

    def __iter__(self):
        """Iterate over member names"""
        for x in self.__items.__iter__():
            yield x

    def __getitem__(self, name):
        """Return a member from is name

        :param name str: name of a member or a path throug members using '/'
            separator. A '/' as a prefix access to the root item of the tree.
        :rtype: Node
        """

        if name is None or name == "":
            raise ValueError("No name")

        if "/" not in name:
            return self.__items[name]

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
            result = result.__items[item_name]

        return result

    def __contains__(self, name):
        """Test if a member name exists"""
        return name in self.__items


class MetadataGroup(Group):
    """Abstract class for groups containing a reference to a fabio image.
    """

    def __init__(self, name, metadata_reader, kind, parent=None, attrs=None):
        Group.__init__(self, name, parent, attrs)
        self.__metadata_reader = metadata_reader
        self.__kind = kind
        self._create_child()

    def _create_child(self):
        keys = self.__metadata_reader.get_keys(self.__kind)
        for name in keys:
            data = self.__metadata_reader.get_value(self.__kind, name)
            dataset = Dataset(name, data)
            self.add_node(dataset)

    @property
    def _metadata_reader(self):
        return self.__metadata_reader


class MetadataReader(object):
    """Class which contains all metadata from a fabio image."""

    MEASUREMENT = 0
    COUNTER = 1
    POSITIONER = 2

    def __init__(self, fabio_file):
        self.__fabio_file = fabio_file
        self.__counters = {}
        self.__positioners = {}
        self.__measurements = {}
        self.__frame_count = self.__fabio_file.nframes
        self._read(self.__fabio_file)

    def _get_dict(self, kind):
        if kind == self.MEASUREMENT:
            return self.__measurements
        elif kind == self.COUNTER:
            return self.__counters
        elif kind == self.POSITIONER:
            return self.__positioners
        else:
            raise Exception("Unexpected kind %s", kind)

    def get_keys(self, kind):
        return self._get_dict(kind).keys()

    def get_value(self, kind, name):
        value = self._get_dict(kind)[name]
        if not isinstance(value, numpy.ndarray):
            value = self._convert_metadata_vector(value)
            self._get_dict(kind)[name] = value
        return value

    def _set_counter_value(self, frame_id, name, value):
        if name not in self.__counters:
            self.__counters[name] = [None] * self.__frame_count
        self.__counters[name][frame_id] = value

    def _set_positioner_value(self, frame_id, name, value):
        if name not in self.__positioners:
            self.__positioners[name] = [None] * self.__frame_count
        self.__positioners[name][frame_id] = value

    def _set_measurement_value(self, frame_id, name, value):
        if name not in self.__measurements:
            self.__measurements[name] = [None] * self.__frame_count
        self.__measurements[name][frame_id] = value

    def _read(self, fabio_file):
        for frame in range(fabio_file.nframes):
            if fabio_file.nframes == 1:
                header = fabio_file.header
            else:
                header = fabio_file.getframe(frame).header
            self._read_frame(frame, header)

    def _read_frame(self, frame_id, header):
        for key, value in header.items():
            self._read_key(frame_id, key, value)

    def _read_key(self, frame_id, name, value):
        self._set_measurement_value(frame_id, name, value)

    def _convert_metadata_vector(self, values):
        converted = []
        types = set([])
        for v in values:
            if v is None:
                converted.append(None)
                types.add("None")
            else:
                c = self._convert_value(v)
                converted.append(c)
                types.add(c.dtype.kind)

        if len(types - set(["b", "i", "u", "f", "None"])) > 0:
            # use the raw data to create the array
            result = values
            result_type = "S"
        else:
            result = converted
            for t in ["f", "i", "u", "b"]:
                if t in types:
                    result_type = t
                    break

        if "None" in types:
            # Fix missing data according to the array type
            if result_type == "S":
                none_value = ""
            elif result_type == "f":
                none_value = numpy.float("NaN")
            elif result_type == "i":
                none_value = numpy.int(0)
            elif result_type == "u":
                none_value = numpy.int(0)
            elif result_type == "b":
                none_value = numpy.bool(False)
            else:
                none_value = None

            for index, r in enumerate(result):
                if r is not None:
                    continue
                result[index] = none_value

            if "None" in types:
                result = ["" if r is None else r for r in result]

            if "None" in types:
                result = ["" if r is None else r for r in result]

        return numpy.array(result)

    def _convert_value(self, value):
        if " " in value:
            result = self._convert_list(value)
        else:
            result = self._convert_scalar_value(value)
        return result

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


class EdfMetadataReader(MetadataReader):
    """Class which contains all metadata from a fabio EDF image.

    It is mostly the same as MetadataGroup, but counter_mne and
    motor_mne are parsed with a special way.
    """

    def _read_frame(self, frame_id, header):
        self.__catch_keys = set([])
        if "motor_pos" in header and "motor_mne" in header:
            self.__catch_keys.add("motor_pos")
            self.__catch_keys.add("motor_mne")
            self._read_mnemonic_key(frame_id, "motor", header)
        if "counter_pos" in header and "counter_mne" in header:
            self.__catch_keys.add("counter_pos")
            self.__catch_keys.add("counter_mne")
            self._read_mnemonic_key(frame_id, "counter", header)
        MetadataReader._read_frame(self, frame_id, header)

    def _read_key(self, frame_id, name, value):
        if name in self.__catch_keys:
            return
        MetadataReader._read_key(self, frame_id, name, value)

    def _read_mnemonic_key(self, frame_id, base_key, header):
        mnemonic_values_key = base_key + "_mne"
        mnemonic_values = header.get(mnemonic_values_key, "")
        mnemonic_values = mnemonic_values.split()
        pos_values_key = base_key + "_pos"
        pos_values = header.get(pos_values_key, "")
        pos_values = pos_values.split()

        is_counter = base_key == "counter"
        is_positioner = base_key == "motor"

        nbitems = max(len(mnemonic_values), len(pos_values))
        for i in range(nbitems):
            if i < len(mnemonic_values):
                mnemonic = mnemonic_values[i]
            else:
                # skip the element
                continue

            if i < len(pos_values):
                pos = pos_values[i]
            else:
                pos = None

            if is_counter:
                self._set_counter_value(frame_id, mnemonic, pos)
            elif is_positioner:
                self._set_positioner_value(frame_id, mnemonic, pos)
            else:
                raise Exception("State unexpected (base_key: %s)" % base_key)


class File(Group):
    """Class which handle a fabio image as a mimick of a h5py.File.
    """

    def __init__(self, file_name=None, fabio_image=None):
        self.__must_be_closed = False
        if file_name is not None and fabio_image is not None:
            raise TypeError("Parameters file_name and fabio_image are mutually exclusive.")
        if file_name is not None:
            self.__fabio_image = fabio.open(file_name)
            self.__must_be_closed = True
        elif fabio_image is not None:
            self.__fabio_image = fabio_image
        Group.__init__(self, os.path.basename(self.__fabio_image.filename), None)
        self.__metadata_reader = self.create_metadata_reader(self.__fabio_image)
        scan = self.create_scan_group(self.__fabio_image, self.__metadata_reader)
        self.add_node(scan)

    def create_scan_group(self, fabio_image, metadata_reader):
        scan = Group("scan", attrs={"NX_class": "NXentry"})
        instrument = Group("instrument", attrs={"NX_class": "NXinstrument"})
        measurement = Group("measurement", attrs={"NX_class": "NXcollection"})
        positioners = MetadataGroup("positioners", self.__metadata_reader, MetadataReader.POSITIONER, attrs={"NX_class": "NXcollection"})
        others = MetadataGroup("others", self.__metadata_reader, MetadataReader.MEASUREMENT, attrs={"NX_class": "NXcollection"})
        counters = MetadataGroup("counters", self.__metadata_reader, MetadataReader.COUNTER, attrs={"NX_class": "NXcollection"})
        data = FrameData("data", fabio_image, self)

        scan.add_node(instrument)
        instrument.add_node(positioners)
        scan.add_node(measurement)
        measurement.add_node(others)
        measurement.add_node(counters)
        measurement.add_node(data)
        return scan

    def create_metadata_reader(self, fabio_file):
        if isinstance(fabio_file, fabio.edfimage.EdfImage):
            metadata = EdfMetadataReader(fabio_file)
        else:
            metadata = MetadataReader(fabio_file)
        return metadata

    @property
    def h5py_class(self):
        return h5py.File

    @property
    def filename(self):
        return self.__fabio_image.filename

    def __enter__(self):
        pass

    def __exit__(self, type, value, tb):  # pylint: disable=W0622
        """Called at the end of a `with` statement.

        It will close the internal FabioImage only if the FabioImage was
        created by the class itself. The reference to the FabioImage is anyway
        broken.
        """
        if self.__must_be_closed:
            self.close()
        else:
            self.__fabio_image = None

    def close(self):
        """Close the object, and free up associated resources.

        The associated FabioImage is closed anyway the object was created from
        a filename or from a FabioImage.

        After calling this method, attempts to use the object may fail.
        """
        self.__fabio_image.close()
        self.__fabio_image = None
