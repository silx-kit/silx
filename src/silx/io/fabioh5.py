# /*##########################################################################
# Copyright (C) 2016-2021 European Synchrotron Radiation Facility
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
    which are not mandatory dependencies for `silx`.

"""

import collections
import datetime
import logging
import numbers
import os

import fabio.file_series
import numpy

from . import commonh5
from silx import version as silx_version
import silx.utils.number
import h5py


_logger = logging.getLogger(__name__)


_fabio_extensions = set([])


def supported_extensions():
    """Returns all extensions supported by fabio.

    :returns: A set containing extensions like "*.edf".
    :rtype: Set[str]
    """
    global _fabio_extensions
    if len(_fabio_extensions) > 0:
        return _fabio_extensions

    formats = fabio.fabioformats.get_classes(reader=True)
    all_extensions = set([])

    for reader in formats:
        if not hasattr(reader, "DEFAULT_EXTENSIONS"):
            continue

        ext = reader.DEFAULT_EXTENSIONS
        ext = ["*.%s" % e for e in ext]
        all_extensions.update(ext)

    _fabio_extensions = set(all_extensions)
    return _fabio_extensions


class _FileSeries(fabio.file_series.file_series):
    """
    .. note:: Overwrite a function to fix an issue in fabio.
    """
    def jump(self, num):
        """
        Goto a position in sequence
        """
        assert num < len(self) and num >= 0, "num out of range"
        self._current = num
        return self[self._current]


class FrameData(commonh5.LazyLoadableDataset):
    """Expose a cube of image from a Fabio file using `FabioReader` as
    cache."""

    def __init__(self, name, fabio_reader, parent=None):
        if fabio_reader.is_spectrum():
            attrs = {"interpretation": "spectrum"}
        else:
            attrs = {"interpretation": "image"}
        commonh5.LazyLoadableDataset.__init__(self, name, parent, attrs=attrs)
        self.__fabio_reader = fabio_reader
        self._shape = None
        self._dtype = None

    def _create_data(self):
        return self.__fabio_reader.get_data()

    def _update_cache(self):
        if isinstance(self.__fabio_reader.fabio_file(),
                      fabio.file_series.file_series):
            # Reading all the files is taking too much time
            # Reach the information from the only first frame
            first_image = self.__fabio_reader.fabio_file().first_image()
            self._dtype = first_image.data.dtype
            shape0 = self.__fabio_reader.frame_count()
            shape1, shape2 = first_image.data.shape
            self._shape = shape0, shape1, shape2
        else:
            self._dtype = super(commonh5.LazyLoadableDataset, self).dtype
            self._shape = super(commonh5.LazyLoadableDataset, self).shape

    @property
    def dtype(self):
        if self._dtype is None:
            self._update_cache()
        return self._dtype

    @property
    def shape(self):
        if self._shape is None:
            self._update_cache()
        return self._shape

    def __iter__(self):
        for frame in self.__fabio_reader.iter_frames():
            yield frame.data

    def __getitem__(self, item):
        # optimization for fetching a single frame if data not already loaded
        if not self._is_initialized:
            if isinstance(item, int) and \
                    isinstance(self.__fabio_reader.fabio_file(),
                               fabio.file_series.file_series):
                if item < 0:
                    # negative indexing
                    item += len(self)
                return self.__fabio_reader.fabio_file().jump_image(item).data
        return super(FrameData, self).__getitem__(item)


class RawHeaderData(commonh5.LazyLoadableDataset):
    """Lazy loadable raw header"""

    def __init__(self, name, fabio_reader, parent=None):
        commonh5.LazyLoadableDataset.__init__(self, name, parent)
        self.__fabio_reader = fabio_reader

    def _create_data(self):
        """Initialize hold data by merging all headers of each frames.
        """
        headers = []
        types = set([])
        for fabio_frame in self.__fabio_reader.iter_frames():
            header = fabio_frame.header

            data = []
            for key, value in header.items():
                data.append("%s: %s" % (str(key), str(value)))

            data = "\n".join(data)
            try:
                line = data.encode("ascii")
                types.add(numpy.string_)
            except UnicodeEncodeError:
                try:
                    line = data.encode("utf-8")
                    types.add(numpy.unicode_)
                except UnicodeEncodeError:
                    # Fallback in void
                    line = numpy.void(data)
                    types.add(numpy.void)

            headers.append(line)

        if numpy.void in types:
            dtype = numpy.void
        elif numpy.unicode_ in types:
            dtype = numpy.unicode_
        else:
            dtype = numpy.string_

        if dtype == numpy.unicode_:
            # h5py only support vlen unicode
            dtype = h5py.special_dtype(vlen=str)

        return numpy.array(headers, dtype=dtype)


class MetadataGroup(commonh5.LazyLoadableGroup):
    """Abstract class for groups containing a reference to a fabio image.
    """

    def __init__(self, name, metadata_reader, kind, parent=None, attrs=None):
        commonh5.LazyLoadableGroup.__init__(self, name, parent, attrs)
        self.__metadata_reader = metadata_reader
        self.__kind = kind

    def _create_child(self):
        keys = self.__metadata_reader.get_keys(self.__kind)
        for name in keys:
            data = self.__metadata_reader.get_value(self.__kind, name)
            dataset = commonh5.Dataset(name, data)
            self.add_node(dataset)

    @property
    def _metadata_reader(self):
        return self.__metadata_reader


class DetectorGroup(commonh5.LazyLoadableGroup):
    """Define the detector group (sub group of instrument) using Fabio data.
    """

    def __init__(self, name, fabio_reader, parent=None, attrs=None):
        if attrs is None:
            attrs = {"NX_class": "NXdetector"}
        commonh5.LazyLoadableGroup.__init__(self, name, parent, attrs)
        self.__fabio_reader = fabio_reader

    def _create_child(self):
        data = FrameData("data", self.__fabio_reader)
        self.add_node(data)

        # TODO we should add here Nexus informations we can extract from the
        # metadata

        others = MetadataGroup("others", self.__fabio_reader, kind=FabioReader.DEFAULT)
        self.add_node(others)


class ImageGroup(commonh5.LazyLoadableGroup):
    """Define the image group (sub group of measurement) using Fabio data.
    """

    def __init__(self, name, fabio_reader, parent=None, attrs=None):
        commonh5.LazyLoadableGroup.__init__(self, name, parent, attrs)
        self.__fabio_reader = fabio_reader

    def _create_child(self):
        basepath = self.parent.parent.name
        data = commonh5.SoftLink("data", path=basepath + "/instrument/detector_0/data")
        self.add_node(data)
        detector = commonh5.SoftLink("info", path=basepath + "/instrument/detector_0")
        self.add_node(detector)


class NxDataPreviewGroup(commonh5.LazyLoadableGroup):
    """Define the NxData group which is used as the default NXdata to show the
    content of the file.
    """

    def __init__(self, name, fabio_reader, parent=None):
        if fabio_reader.is_spectrum():
            interpretation = "spectrum"
        else:
            interpretation = "image"
        attrs = {
            "NX_class": "NXdata",
            "interpretation": interpretation,
            "signal": "data",
        }
        commonh5.LazyLoadableGroup.__init__(self, name, parent, attrs)
        self.__fabio_reader = fabio_reader

    def _create_child(self):
        basepath = self.parent.name
        data = commonh5.SoftLink("data", path=basepath + "/instrument/detector_0/data")
        self.add_node(data)


class SampleGroup(commonh5.LazyLoadableGroup):
    """Define the image group (sub group of measurement) using Fabio data.
    """

    def __init__(self, name, fabio_reader, parent=None):
        attrs = {"NXclass": "NXsample"}
        commonh5.LazyLoadableGroup.__init__(self, name, parent, attrs)
        self.__fabio_reader = fabio_reader

    def _create_child(self):
        if self.__fabio_reader.has_ub_matrix():
            scalar = {"interpretation": "scalar"}
            data = self.__fabio_reader.get_unit_cell_abc()
            data = commonh5.Dataset("unit_cell_abc", data, attrs=scalar)
            self.add_node(data)
            unit_cell_data = numpy.zeros((1, 6), numpy.float32)
            unit_cell_data[0, :3] = data
            data = self.__fabio_reader.get_unit_cell_alphabetagamma()
            data = commonh5.Dataset("unit_cell_alphabetagamma", data, attrs=scalar)
            self.add_node(data)
            unit_cell_data[0, 3:] = data
            data = commonh5.Dataset("unit_cell", unit_cell_data, attrs=scalar)
            self.add_node(data)
            data = self.__fabio_reader.get_ub_matrix()
            data = commonh5.Dataset("ub_matrix", data, attrs=scalar)
            self.add_node(data)


class MeasurementGroup(commonh5.LazyLoadableGroup):
    """Define the measurement group for fabio file.
    """

    def __init__(self, name, fabio_reader, parent=None, attrs=None):
        commonh5.LazyLoadableGroup.__init__(self, name, parent, attrs)
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
            dataset = commonh5.Dataset(name, data)
            self.add_node(dataset)


class FabioReader(object):
    """Class which read and cache data and metadata from a fabio image."""

    DEFAULT = 0
    COUNTER = 1
    POSITIONER = 2

    def __init__(self, file_name=None, fabio_image=None, file_series=None):
        """
        Constructor

        :param str file_name: File name of the image file to read
        :param fabio.fabioimage.FabioImage fabio_image: An already openned
            :class:`fabio.fabioimage.FabioImage` instance.
        :param Union[list[str],fabio.file_series.file_series] file_series: An
            list of file name or a :class:`fabio.file_series.file_series`
            instance
        """
        self.__at_least_32bits = False
        self.__signed_type = False

        self.__load(file_name, fabio_image, file_series)
        self.__counters = {}
        self.__positioners = {}
        self.__measurements = {}
        self.__key_filters = set([])
        self.__data = None
        self.__frame_count = self.frame_count()
        self._read()

    def __load(self, file_name=None, fabio_image=None, file_series=None):
        if file_name is not None and fabio_image:
            raise TypeError("Parameters file_name and fabio_image are mutually exclusive.")
        if file_name is not None and fabio_image:
            raise TypeError("Parameters fabio_image and file_series are mutually exclusive.")

        self.__must_be_closed = False

        if file_name is not None:
            self.__fabio_file = fabio.open(file_name)
            self.__must_be_closed = True
        elif fabio_image is not None:
            if isinstance(fabio_image, fabio.fabioimage.FabioImage):
                self.__fabio_file = fabio_image
            else:
                raise TypeError("FabioImage expected but %s found.", fabio_image.__class__)
        elif file_series is not None:
            if isinstance(file_series, list):
                self.__fabio_file = _FileSeries(file_series)
            elif isinstance(file_series, fabio.file_series.file_series):
                self.__fabio_file = file_series
            else:
                raise TypeError("file_series or list expected but %s found.", file_series.__class__)

    def close(self):
        """Close the object, and free up associated resources.

        The associated FabioImage is closed only if the object was created from
        a filename by this class itself.

        After calling this method, attempts to use the object (and children)
        may fail.
        """
        if self.__must_be_closed:
            # Make sure the API of fabio provide it a 'close' method
            # TODO the test can be removed if fabio version >= 0.8
            if hasattr(self.__fabio_file, "close"):
                self.__fabio_file.close()
        self.__fabio_file = None

    def fabio_file(self):
        return self.__fabio_file

    def frame_count(self):
        """Returns the number of frames available."""
        if isinstance(self.__fabio_file, fabio.file_series.file_series):
            return len(self.__fabio_file)
        elif isinstance(self.__fabio_file, fabio.fabioimage.FabioImage):
            return self.__fabio_file.nframes
        else:
            raise TypeError("Unsupported type %s", self.__fabio_file.__class__)

    def iter_frames(self):
        """Iter all the available frames.

        A frame provides at least `data` and `header` attributes.
        """
        if isinstance(self.__fabio_file, fabio.file_series.file_series):
            for file_number in range(len(self.__fabio_file)):
                with self.__fabio_file.jump_image(file_number) as fabio_image:
                    # return the first frame only
                    assert(fabio_image.nframes == 1)
                    yield fabio_image
        elif isinstance(self.__fabio_file, fabio.fabioimage.FabioImage):
            for frame_count in range(self.__fabio_file.nframes):
                if self.__fabio_file.nframes == 1:
                    yield self.__fabio_file
                else:
                    yield self.__fabio_file.getframe(frame_count)
        else:
            raise TypeError("Unsupported type %s", self.__fabio_file.__class__)

    def _create_data(self):
        """Initialize hold data by merging all frames into a single cube.

        Choose the cube size which fit the best the data. If some images are
        smaller than expected, the empty space is set to 0.

        The computation is cached into the class, and only done ones.
        """
        images = []
        for fabio_frame in self.iter_frames():
            images.append(fabio_frame.data)

        # returns the data without extra dim in case of single frame
        if len(images) == 1:
            return images[0]

        # get the max size
        max_dim = max([i.ndim for i in images])
        max_shape = [0] * max_dim
        for image in images:
            for dim in range(image.ndim):
                if image.shape[dim] > max_shape[dim]:
                    max_shape[dim] = image.shape[dim]
        max_shape = tuple(max_shape)

        # fix smallest images
        for index, image in enumerate(images):
            if image.shape == max_shape:
                continue
            location = [slice(0, i) for i in image.shape]
            while len(location) < max_dim:
                location.append(0)
            normalized_image = numpy.zeros(max_shape, dtype=image.dtype)
            normalized_image[tuple(location)] = image
            images[index] = normalized_image

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
            if kind in [self.COUNTER, self.POSITIONER]:
                # Force normalization for counters and positioners
                old = self._set_vector_normalization(at_least_32bits=True, signed_type=True)
            else:
                old = None
            value = self._convert_metadata_vector(value)
            self.__get_dict(kind)[name] = value
            if old is not None:
                self._set_vector_normalization(*old)
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

    def _enable_key_filters(self, fabio_file):
        self.__key_filters.clear()
        if hasattr(fabio_file, "RESERVED_HEADER_KEYS"):
            # Provided in fabio 0.5
            for key in fabio_file.RESERVED_HEADER_KEYS:
                self.__key_filters.add(key.lower())

    def _read(self):
        """Read all metadata from the fabio file and store it into this
        object."""

        file_series = isinstance(self.__fabio_file, fabio.file_series.file_series)
        if not file_series:
            self._enable_key_filters(self.__fabio_file)

        for frame_id, fabio_frame in enumerate(self.iter_frames()):
            if file_series:
                self._enable_key_filters(fabio_frame)
            self._read_frame(frame_id, fabio_frame.header)

    def _is_filtered_key(self, key):
        """
        If this function returns True, the :meth:`_read_key` while not be
        called with this `key`while reading the metatdata frame.

        :param str key: A key of the metadata
        :rtype: bool
        """
        return key.lower() in self.__key_filters

    def _read_frame(self, frame_id, header):
        """Read all metadata from a frame and store it into this
        object."""
        for key, value in header.items():
            if self._is_filtered_key(key):
                continue
            self._read_key(frame_id, key, value)

    def _read_key(self, frame_id, name, value):
        """Read a key from the metadata and cache it into this object."""
        self._set_measurement_value(frame_id, name, value)

    def _set_vector_normalization(self, at_least_32bits, signed_type):
        previous = self.__at_least_32bits, self.__signed_type
        self.__at_least_32bits = at_least_32bits
        self.__signed_type = signed_type
        return previous

    def _normalize_vector_type(self, dtype):
        """Normalize the """
        if self.__at_least_32bits:
            if numpy.issubdtype(dtype, numpy.signedinteger):
                dtype = numpy.result_type(dtype, numpy.uint32)
            if numpy.issubdtype(dtype, numpy.unsignedinteger):
                dtype = numpy.result_type(dtype, numpy.uint32)
            elif numpy.issubdtype(dtype, numpy.floating):
                dtype = numpy.result_type(dtype, numpy.float32)
            elif numpy.issubdtype(dtype, numpy.complexfloating):
                dtype = numpy.result_type(dtype, numpy.complex64)
        if self.__signed_type:
            if numpy.issubdtype(dtype, numpy.unsignedinteger):
                signed = numpy.dtype("%s%i" % ('i', dtype.itemsize))
                dtype = numpy.result_type(dtype, signed)
        return dtype

    def _convert_metadata_vector(self, values):
        """Convert a list of numpy data into a numpy array with the better
        fitting type."""
        converted = []
        types = set([])
        has_none = False
        is_array = False
        array = []

        for v in values:
            if v is None:
                converted.append(None)
                has_none = True
                array.append(None)
            else:
                c = self._convert_value(v)
                if c.shape != tuple():
                    array.append(v.split(" "))
                    is_array = True
                else:
                    array.append(v)
                converted.append(c)
                types.add(c.dtype)

        if has_none and len(types) == 0:
            # That's a list of none values
            return numpy.array([0] * len(values), numpy.int8)

        result_type = numpy.result_type(*types)

        if issubclass(result_type.type, numpy.string_):
            # use the raw data to create the array
            result = values
        elif issubclass(result_type.type, numpy.unicode_):
            # use the raw data to create the array
            result = values
        else:
            result = converted

        result_type = self._normalize_vector_type(result_type)

        if has_none:
            # Fix missing data according to the array type
            if result_type.kind == "S":
                none_value = b""
            elif result_type.kind == "U":
                none_value = u""
            elif result_type.kind == "f":
                none_value = numpy.float64("NaN")
            elif result_type.kind == "i":
                none_value = numpy.int64(0)
            elif result_type.kind == "u":
                none_value = numpy.int64(0)
            elif result_type.kind == "b":
                none_value = numpy.bool_(False)
            else:
                none_value = None

            for index, r in enumerate(result):
                if r is not None:
                    continue
                result[index] = none_value
                values[index] = none_value
                array[index] = none_value

        if result_type.kind in "uifd" and len(types) > 1 and len(values) > 1:
            # Catch numerical precision
            if is_array and len(array) > 1:
                return numpy.array(array, dtype=result_type)
            else:
                return numpy.array(values, dtype=result_type)
        return numpy.array(result, dtype=result_type)

    def _convert_value(self, value):
        """Convert a string into a numpy object (scalar or array).

        The value is most of the time a string, but it can be python object
        in case if TIFF decoder for example.
        """
        if isinstance(value, list):
            # convert to a numpy array
            return numpy.array(value)
        if isinstance(value, dict):
            # convert to a numpy associative array
            key_dtype = numpy.min_scalar_type(list(value.keys()))
            value_dtype = numpy.min_scalar_type(list(value.values()))
            associative_type = [('key', key_dtype), ('value', value_dtype)]
            assert key_dtype.kind != "O" and value_dtype.kind != "O"
            return numpy.array(list(value.items()), dtype=associative_type)
        if isinstance(value, numbers.Number):
            dtype = numpy.min_scalar_type(value)
            assert dtype.kind != "O"
            return dtype.type(value)

        if isinstance(value, bytes):
            try:
                value = value.decode('utf-8')
            except UnicodeDecodeError:
                return numpy.void(value)

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
            numpy_type = silx.utils.number.min_numerical_convertible_type(value)
            converted = numpy_type(value)
        except ValueError:
            converted = numpy.string_(value)
        return converted

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

            if issubclass(result_type.type, (numpy.string_, bytes)):
                # use the raw data to create the result
                return numpy.string_(value)
            elif issubclass(result_type.type, (numpy.unicode_, str)):
                # use the raw data to create the result
                return numpy.unicode_(value)
            else:
                if len(types) == 1:
                    return numpy.array(numpy_values, dtype=result_type)
                else:
                    return numpy.array(values, dtype=result_type)
        except ValueError:
            return numpy.string_(value)

    def has_sample_information(self):
        """Returns true if there is information about the sample in the
        file

        :rtype: bool
        """
        return self.has_ub_matrix()

    def has_ub_matrix(self):
        """Returns true if a UB matrix is available.

        :rtype: bool
        """
        return False

    def is_spectrum(self):
        """Returns true if the data should be interpreted as
        MCA data.

        :rtype: bool
        """
        return False


class EdfFabioReader(FabioReader):
    """Class which read and cache data and metadata from a fabio image.

    It is mostly the same as FabioReader, but counter_mne and
    motor_mne are parsed using a special way.
    """

    def __init__(self, file_name=None, fabio_image=None, file_series=None):
        FabioReader.__init__(self, file_name, fabio_image, file_series)
        self.__unit_cell_abc = None
        self.__unit_cell_alphabetagamma = None
        self.__ub_matrix = None

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

    def _is_filtered_key(self, key):
        if key in self.__catch_keys:
            return True
        return FabioReader._is_filtered_key(self, key)

    def _get_mnemonic_key(self, base_key, header):
        mnemonic_values_key = base_key + "_mne"
        mnemonic_values = header.get(mnemonic_values_key, "")
        mnemonic_values = mnemonic_values.split()
        pos_values_key = base_key + "_pos"
        pos_values = header.get(pos_values_key, "")
        pos_values = pos_values.split()

        result = collections.OrderedDict()
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

            result[mnemonic] = pos
        return result

    def _read_mnemonic_key(self, frame_id, base_key, header):
        """Parse a mnemonic key"""
        is_counter = base_key == "counter"
        is_positioner = base_key == "motor"
        data = self._get_mnemonic_key(base_key, header)

        for mnemonic, pos in data.items():
            if is_counter:
                self._set_counter_value(frame_id, mnemonic, pos)
            elif is_positioner:
                self._set_positioner_value(frame_id, mnemonic, pos)
            else:
                raise Exception("State unexpected (base_key: %s)" % base_key)

    def _get_first_header(self):
        """
        ..note:: This function can be cached
        """
        fabio_file = self.fabio_file()
        if isinstance(fabio_file, fabio.file_series.file_series):
            return fabio_file.jump_image(0).header
        return fabio_file.header

    def has_ub_matrix(self):
        """Returns true if a UB matrix is available.

        :rtype: bool
        """
        header = self._get_first_header()
        expected_keys = set(["UB_mne", "UB_pos", "sample_mne", "sample_pos"])
        return expected_keys.issubset(header)

    def parse_ub_matrix(self):
        header = self._get_first_header()
        ub_data = self._get_mnemonic_key("UB", header)
        s_data = self._get_mnemonic_key("sample", header)
        if len(ub_data) > 9:
            _logger.warning("UB_mne and UB_pos contains more than expected keys.")
        if len(s_data) > 6:
            _logger.warning("sample_mne and sample_pos contains more than expected keys.")

        data = numpy.array([s_data["U0"], s_data["U1"], s_data["U2"]], dtype=float)
        unit_cell_abc = data

        data = numpy.array([s_data["U3"], s_data["U4"], s_data["U5"]], dtype=float)
        unit_cell_alphabetagamma = data

        ub_matrix = numpy.array([[
            [ub_data["UB0"], ub_data["UB1"], ub_data["UB2"]],
            [ub_data["UB3"], ub_data["UB4"], ub_data["UB5"]],
            [ub_data["UB6"], ub_data["UB7"], ub_data["UB8"]]]], dtype=float)

        self.__unit_cell_abc = unit_cell_abc
        self.__unit_cell_alphabetagamma = unit_cell_alphabetagamma
        self.__ub_matrix = ub_matrix

    def get_unit_cell_abc(self):
        """Get a numpy array data as defined for the dataset unit_cell_abc
        from the NXsample dataset.

        :rtype: numpy.ndarray
        """
        if self.__unit_cell_abc is None:
            self.parse_ub_matrix()
        return self.__unit_cell_abc

    def get_unit_cell_alphabetagamma(self):
        """Get a numpy array data as defined for the dataset
        unit_cell_alphabetagamma from the NXsample dataset.

        :rtype: numpy.ndarray
        """
        if self.__unit_cell_alphabetagamma is None:
            self.parse_ub_matrix()
        return self.__unit_cell_alphabetagamma

    def get_ub_matrix(self):
        """Get a numpy array data as defined for the dataset ub_matrix
        from the NXsample dataset.

        :rtype: numpy.ndarray
        """
        if self.__ub_matrix is None:
            self.parse_ub_matrix()
        return self.__ub_matrix

    def is_spectrum(self):
        """Returns true if the data should be interpreted as
        MCA data.
        EDF files or file series, with two or more header names starting with
        "MCA", should be interpreted as MCA data.

        :rtype: bool
        """
        count = 0
        for key in self._get_first_header():
            if key.lower().startswith("mca"):
                count += 1
            if count >= 2:
                return True
        return False


class File(commonh5.File):
    """Class which handle a fabio image as a mimick of a h5py.File.
    """

    def __init__(self, file_name=None, fabio_image=None, file_series=None):
        """
        Constructor

        :param str file_name: File name of the image file to read
        :param fabio.fabioimage.FabioImage fabio_image: An already openned
            :class:`fabio.fabioimage.FabioImage` instance.
        :param Union[list[str],fabio.file_series.file_series] file_series: An
            list of file name or a :class:`fabio.file_series.file_series`
            instance
        """
        self.__fabio_reader = self.create_fabio_reader(file_name, fabio_image, file_series)
        if fabio_image is not None:
            file_name = fabio_image.filename
        scan = self.create_scan_group(self.__fabio_reader)

        attrs = {"NX_class": "NXroot",
                 "file_time": datetime.datetime.now().isoformat(),
                 "creator": "silx %s" % silx_version,
                 "default": scan.basename}
        if file_name is not None:
            attrs["file_name"] = file_name
        commonh5.File.__init__(self, name=file_name, attrs=attrs)
        self.add_node(scan)

    def create_scan_group(self, fabio_reader):
        """Factory to create the scan group.

        :param FabioImage fabio_image: A Fabio image
        :param FabioReader fabio_reader: A reader for the Fabio image
        :rtype: commonh5.Group
        """
        nxdata = NxDataPreviewGroup("image", fabio_reader)
        scan_attrs = {
            "NX_class": "NXentry",
            "default": nxdata.basename,
        }
        scan = commonh5.Group("scan_0", attrs=scan_attrs)
        instrument = commonh5.Group("instrument", attrs={"NX_class": "NXinstrument"})
        measurement = MeasurementGroup("measurement", fabio_reader, attrs={"NX_class": "NXcollection"})
        file_ = commonh5.Group("file", attrs={"NX_class": "NXcollection"})
        positioners = MetadataGroup("positioners", fabio_reader, FabioReader.POSITIONER, attrs={"NX_class": "NXpositioner"})
        raw_header = RawHeaderData("scan_header", fabio_reader, self)
        detector = DetectorGroup("detector_0", fabio_reader)

        scan.add_node(instrument)
        instrument.add_node(positioners)
        instrument.add_node(file_)
        instrument.add_node(detector)
        file_.add_node(raw_header)
        scan.add_node(measurement)
        scan.add_node(nxdata)

        if fabio_reader.has_sample_information():
            sample = SampleGroup("sample", fabio_reader)
            scan.add_node(sample)

        return scan

    def create_fabio_reader(self, file_name, fabio_image, file_series):
        """Factory to create fabio reader.

        :rtype: FabioReader"""
        use_edf_reader = False
        first_file_name = None
        first_image = None

        if isinstance(file_series, list):
            first_file_name = file_series[0]
        elif isinstance(file_series, fabio.file_series.file_series):
            first_image = file_series.first_image()
        elif fabio_image is not None:
            first_image = fabio_image
        else:
            first_file_name = file_name

        if first_file_name is not None:
            _, ext = os.path.splitext(first_file_name)
            ext = ext[1:]
            edfimage = fabio.edfimage.EdfImage
            if hasattr(edfimage, "DEFAULT_EXTENTIONS"):
                # Typo on fabio 0.5
                edf_extensions = edfimage.DEFAULT_EXTENTIONS
            else:
                edf_extensions = edfimage.DEFAULT_EXTENSIONS
            use_edf_reader = ext in edf_extensions
        elif first_image is not None:
            use_edf_reader = isinstance(first_image, fabio.edfimage.EdfImage)
        else:
            assert(False)

        if use_edf_reader:
            reader = EdfFabioReader(file_name, fabio_image, file_series)
        else:
            reader = FabioReader(file_name, fabio_image, file_series)
        return reader

    def close(self):
        """Close the object, and free up associated resources.

        After calling this method, attempts to use the object (and children)
        may fail.
        """
        self.__fabio_reader.close()
        self.__fabio_reader = None
