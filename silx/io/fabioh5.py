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
        return self.__name


class Dataset(Node):
    """Class which handle a numpy data as a mimic of a h5py.Dataset.
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


class LazyLoadableDataset(Dataset):
    """Abstract dataset which provide a lazy loading of the data.

    The class have to be inherited and the :meth:`_create_data` have to be
    implemented to return the numpy data exposed by the dataset. This factory
    is only called ones, when the data is needed.
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


class Group(Node):
    """Class which mimic a `h5py.Group`."""

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
        self._get_items()[node.name] = node
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
        """ Retrieve an item or other information.

        If getlink only is true, the returned value is always HardLink
            cause this implementation do not use links. Like the original
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
        else:
            node = self._get_items()[name]

        if getclass:
            obj = node.h5py_class
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
        """Returns true is a name is an existing child of this group.

        :rtype: bool
        """
        return name in self._get_items()


class LazyLoadableGroup(Group):
    """Abstract group which provide a lazy loading of the child.

    The class have to be inherited and the :meth:`_create_child` have to be
    implemented to add (:meth:`_add_node`) all child. This factory
    is only called ones, when child are needed.
    """

    def __init__(self, name, parent=None, attrs=None):
        Group.__init__(self, name, parent, attrs)
        self.__is_initialized = False

    def _get_items(self):
        """Returns internal structure which contains child.

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

        It have to be implemented to work.
        """
        raise NotImplementedError()


class FrameData(LazyLoadableDataset):
    """Expose a cube of image from a Fabio file using `FabioReader` as
    cache."""

    def __init__(self, name, fabio_reader, parent=None):
        attrs = {"interpretation": "image"}
        LazyLoadableDataset.__init__(self, name, parent, attrs=attrs)
        self.__fabio_reader = fabio_reader

    def _create_data(self):
        return self.__fabio_reader.get_data()


class RawHeaderData(LazyLoadableDataset):
    """Lazy loadable raw header"""

    def __init__(self, name, fabio_file, parent=None):
        LazyLoadableDataset.__init__(self, name, parent)
        self.__fabio_file = fabio_file

    def _create_data(self):
        """Initialize hold data by merging all headers of each frames.
        """
        headers = []
        for frame in range(self.__fabio_file.nframes):
            if self.__fabio_file.nframes == 1:
                header = self.__fabio_file.header
            else:
                header = self.__fabio_file.getframe(frame).header

            data = []
            for key, value in header.items():
                data.append("%s: %s" % (str(key), str(value)))

            headers.append(u"\n".join(data))

        # create the header list
        return numpy.array(headers)


class MetadataGroup(LazyLoadableGroup):
    """Abstract class for groups containing a reference to a fabio image.
    """

    def __init__(self, name, metadata_reader, kind, parent=None, attrs=None):
        LazyLoadableGroup.__init__(self, name, parent, attrs)
        self.__metadata_reader = metadata_reader
        self.__kind = kind

    def _create_child(self):
        keys = self.__metadata_reader.get_keys(self.__kind)
        for name in keys:
            data = self.__metadata_reader.get_value(self.__kind, name)
            dataset = Dataset(name, data)
            self.add_node(dataset)

    @property
    def _metadata_reader(self):
        return self.__metadata_reader


class DetectorGroup(LazyLoadableGroup):
    """Define the detector group (sub group of instrument) using Fabio data.
    """

    def __init__(self, name, fabio_reader, parent=None, attrs=None):
        if attrs is None:
            attrs = {"NX_class": "NXdetector"}
        LazyLoadableGroup.__init__(self, name, parent, attrs)
        self.__fabio_reader = fabio_reader

    def _create_child(self):
        data = FrameData("data", self.__fabio_reader)
        self.add_node(data)

        # TODO we should add here Nexus informations we can extract from the
        # metadata

        others = MetadataGroup("others", self.__fabio_reader, kind=FabioReader.DEFAULT)
        self.add_node(others)


class ImageGroup(LazyLoadableGroup):
    """Define the image group (sub group of measurement) using Fabio data.
    """

    def __init__(self, name, fabio_reader, parent=None, attrs=None):
        LazyLoadableGroup.__init__(self, name, parent, attrs)
        self.__fabio_reader = fabio_reader

    def _create_child(self):
        data = FrameData("data", self.__fabio_reader)
        self.add_node(data)

        # TODO detector should be a real soft-link
        detector = DetectorGroup("info", self.__fabio_reader)
        self.add_node(detector)


class MeasurementGroup(LazyLoadableGroup):
    """Define the measurement group for fabio file.
    """

    def __init__(self, name, fabio_reader, parent=None, attrs=None):
        LazyLoadableGroup.__init__(self, name, parent, attrs)
        self.__fabio_reader = fabio_reader

    def _create_child(self):
        keys = self.__fabio_reader.get_keys(FabioReader.COUNTER)

        # create image measurement but take care that no other metadata use
        # this name
        for i in range(1000):
            name = "image_%i" % i
            if name not in keys:
                data = ImageGroup(name, self.__fabio_reader)
                self.add_node(data)
                break
        else:
            raise Exception("image_i for 0..1000 already used")

        # add all counters
        for name in keys:
            data = self.__fabio_reader.get_value(FabioReader.COUNTER, name)
            dataset = Dataset(name, data)
            self.add_node(dataset)


class FabioReader(object):
    """Class which read and cache data and metadata from a fabio image."""

    DEFAULT = 0
    COUNTER = 1
    POSITIONER = 2

    def __init__(self, fabio_file):
        self.__fabio_file = fabio_file
        self.__counters = {}
        self.__positioners = {}
        self.__measurements = {}
        self.__data = None
        self.__frame_count = self.__fabio_file.nframes
        self._read(self.__fabio_file)

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

    def __get_dict(self, kind):
        """Returns a dictionary from according to an expected kind"""
        if kind == self.DEFAULT:
            return self.__measurements
        elif kind == self.COUNTER:
            return self.__counters
        elif kind == self.POSITIONER:
            return self.__positioners
        else:
            raise Exception("Unexpected kind %s", kind)

    def get_data(self):
        """Returns a cube from all available data from frames

        :rtype: numpy.ndarray
        """
        if self.__data is None:
            self.__data = self._create_data()
        return self.__data

    def get_keys(self, kind):
        """Get all available keys according to a kind of metadata.

        :rtype: list
        """
        return self.__get_dict(kind).keys()

    def get_value(self, kind, name):
        """Get a metadata value according to the kind and the name.

        :rtype: numpy.ndarray
        """
        value = self.__get_dict(kind)[name]
        if not isinstance(value, numpy.ndarray):
            value = self._convert_metadata_vector(value)
            self.__get_dict(kind)[name] = value
        return value

    def _set_counter_value(self, frame_id, name, value):
        """Set a counter metadata according to the frame id"""
        if name not in self.__counters:
            self.__counters[name] = [None] * self.__frame_count
        self.__counters[name][frame_id] = value

    def _set_positioner_value(self, frame_id, name, value):
        """Set a positioner metadata according to the frame id"""
        if name not in self.__positioners:
            self.__positioners[name] = [None] * self.__frame_count
        self.__positioners[name][frame_id] = value

    def _set_measurement_value(self, frame_id, name, value):
        """Set a measurement metadata according to the frame id"""
        if name not in self.__measurements:
            self.__measurements[name] = [None] * self.__frame_count
        self.__measurements[name][frame_id] = value

    def _read(self, fabio_file):
        """Read all metadata from the fabio file and store it into this
        object."""
        for frame in range(fabio_file.nframes):
            if fabio_file.nframes == 1:
                header = fabio_file.header
            else:
                header = fabio_file.getframe(frame).header
            self._read_frame(frame, header)

    def _read_frame(self, frame_id, header):
        """Read all metadata from a frame and store it into this
        object."""
        for key, value in header.items():
            self._read_key(frame_id, key, value)

    def _read_key(self, frame_id, name, value):
        """Read a key from the metadata and cache it into this object."""
        self._set_measurement_value(frame_id, name, value)

    def _convert_metadata_vector(self, values):
        """Convert a list of numpy data into a numpy array with the better
        fitting type."""
        converted = []
        types = set([])
        has_none = False
        for v in values:
            if v is None:
                converted.append(None)
                has_none = True
            else:
                c = self._convert_value(v)
                converted.append(c)
                types.add(c.dtype)

        result_type = numpy.result_type(*types)

        if issubclass(result_type.type, numpy.string_):
            # use the raw data to create the array
            result = values
        elif issubclass(result_type.type, numpy.unicode_):
            # use the raw data to create the array
            result = values
        else:
            result = converted

        if has_none:
            # Fix missing data according to the array type
            if result_type.kind in ["S", "U"]:
                none_value = ""
            elif result_type.kind == "f":
                none_value = numpy.float("NaN")
            elif result_type.kind == "i":
                none_value = numpy.int(0)
            elif result_type.kind == "u":
                none_value = numpy.int(0)
            elif result_type.kind == "b":
                none_value = numpy.bool(False)
            else:
                none_value = None

            for index, r in enumerate(result):
                if r is not None:
                    continue
                result[index] = none_value

        return numpy.array(result, dtype=result_type)

    def _convert_value(self, value):
        """Convert a string into a numpy object (scalar or array).
        """
        if " " in value:
            result = self._convert_list(value)
        else:
            result = self._convert_scalar_value(value)
        return result

    def _convert_scalar_value(self, value):
        """Convert a string into a numpy int or float.

        If it is not possible it returns a numpy string.
        """
        try:
            value = int(value)
            dtype = numpy.min_scalar_type(value)
            return dtype.type(value)
        except ValueError:
            try:
                value = float(value)
                dtype = numpy.min_scalar_type(value)
                return dtype.type(value)
            except ValueError:
                return numpy.string_(value)

    def _convert_list(self, value):
        """Convert a string into a typed numpy array.

        If it is not possible it returns a numpy string.
        """
        try:
            numpy_values = []
            values = value.split(" ")
            types = set([])
            for string_value in values:
                v = self._convert_scalar_value(string_value)
                numpy_values.append(v)
                types.add(v.dtype.type)

            result_type = numpy.result_type(*types)

            if issubclass(result_type.type, numpy.string_):
                # use the raw data to create the result
                return numpy.string_(value)
            elif issubclass(result_type.type, numpy.unicode_):
                # use the raw data to create the result
                return numpy.unicode_(value)
            else:
                return numpy.array(numpy_values, dtype=result_type)
        except ValueError:
            return numpy.string_(value)


class EdfFabioReader(FabioReader):
    """Class which read and cache data and metadata from a fabio image.

    It is mostly the same as FabioReader, but counter_mne and
    motor_mne are parsed using a special way.
    """

    def _read_frame(self, frame_id, header):
        """Overwrite the method to check and parse special keys: counter and
        motors keys."""
        self.__catch_keys = set([])
        if "motor_pos" in header and "motor_mne" in header:
            self.__catch_keys.add("motor_pos")
            self.__catch_keys.add("motor_mne")
            self._read_mnemonic_key(frame_id, "motor", header)
        if "counter_pos" in header and "counter_mne" in header:
            self.__catch_keys.add("counter_pos")
            self.__catch_keys.add("counter_mne")
            self._read_mnemonic_key(frame_id, "counter", header)
        FabioReader._read_frame(self, frame_id, header)

    def _read_key(self, frame_id, name, value):
        """Overwrite the method to filter counter or motor keys."""
        if name in self.__catch_keys:
            return
        FabioReader._read_key(self, frame_id, name, value)

    def _read_mnemonic_key(self, frame_id, base_key, header):
        """Parse a mnemonic key"""
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
        Group.__init__(self, os.path.basename(self.__fabio_image.filename), attrs={"NX_class": "NXroot"})
        self.__fabio_reader = self.create_fabio_reader(self.__fabio_image)
        scan = self.create_scan_group(self.__fabio_image, self.__fabio_reader)
        self.add_node(scan)

    def create_scan_group(self, fabio_image, fabio_reader):
        """Factory to create the scan group.

        :param FabioImage fabio_image: A Fabio image
        :param FabioReader fabio_reader: A reader for the Fabio image
        :rtype: Group
        """

        scan = Group("scan_0", attrs={"NX_class": "NXentry"})
        instrument = Group("instrument", attrs={"NX_class": "NXinstrument"})
        measurement = MeasurementGroup("measurement", self.__fabio_reader, attrs={"NX_class": "NXcollection"})
        file_ = Group("file", attrs={"NX_class": "NXcollection"})
        positioners = MetadataGroup("positioners", self.__fabio_reader, FabioReader.POSITIONER, attrs={"NX_class": "NXpositioner"})
        raw_header = RawHeaderData("scan_header", fabio_image, self)
        detector = DetectorGroup("detector_0", self.__fabio_reader)

        scan.add_node(instrument)
        instrument.add_node(positioners)
        instrument.add_node(file_)
        instrument.add_node(detector)
        file_.add_node(raw_header)
        scan.add_node(measurement)
        return scan

    def create_fabio_reader(self, fabio_file):
        """Factory to create fabio reader.

        :rtype: FabioReader"""
        if isinstance(fabio_file, fabio.edfimage.EdfImage):
            metadata = EdfFabioReader(fabio_file)
        else:
            metadata = FabioReader(fabio_file)
        return metadata

    @property
    def h5py_class(self):
        return h5py.File

    @property
    def filename(self):
        return self.__fabio_image.filename

    def __enter__(self):
        return self

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
        # It looks like there is no close on FabioImage
        # self.__fabio_image.close()
        self.__fabio_image = None
