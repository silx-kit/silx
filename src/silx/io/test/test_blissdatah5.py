import numpy
import pytest

pytest.importorskip("blissdata")

from blissdata.h5api import abstract, redis_hdf5  # noqa: E402

from .. import blissdatah5  # noqa: E402
from ..blissdatah5 import BlissDataDataset, BlissDataLima1Dataset  # noqa: E402


class _FakeStream:
    """Mimics a blissdata array stream.

    It fails for accesses which would make blissdata wait for new data or
    for the end of the scan (negative indices, slices without explicit
    positive stop, indices out of the available data, iteration) and for
    indices not supported by blissdata streams (negative steps, numpy integers).
    """

    kind = "array"

    def __init__(self, array: numpy.ndarray, plugin: str | None = None):
        self._array = array
        self.plugin = plugin
        self.sealed = False
        self.isSealedCallCount = 0
        self.getitem_calls = []

    @property
    def shape(self):
        return self._array.shape[1:]

    @property
    def dtype(self):
        return self._array.dtype

    def is_sealed(self):
        self.isSealedCallCount += 1
        return self.sealed

    def wait_seal(self, timeout: float = 0.0):
        raise AssertionError("Blocking until the end of the stream")

    def cursor(self):
        raise AssertionError("Blocking iteration until the end of the stream")

    def __len__(self):
        return len(self._array)

    def __getitem__(self, idx):
        self.getitem_calls.append(idx)
        if type(idx) is int:
            assert 0 <= idx < len(self), f"Blocking index {idx}"
        elif isinstance(idx, slice):
            assert idx.start is None or idx.start >= 0
            assert idx.stop is not None and 0 <= idx.stop <= len(self)
            assert idx.step is None or idx.step > 0
        else:
            raise AssertionError(f"Unsupported index: {idx!r}")
        return self._array[idx]


@pytest.mark.parametrize(
    "index",
    [
        (),
        Ellipsis,
        0,
        4,
        -1,
        -5,
        numpy.int64(2),
        numpy.int32(-2),
        slice(None),
        slice(1, None),
        slice(None, -1),
        slice(-2, None),
        slice(None, None, 2),
        slice(None, None, -1),
        slice(4, 0, -2),
        slice(10, 20),
        slice(3, 1),
        (0, 1, 2),
        (-1, slice(None), 0),
        (numpy.int64(1), 2),
        (slice(None, None, 2), 1),
        (Ellipsis, 0),
        (Ellipsis, 1, 2),
        (Ellipsis, 1, 2, 3),
        (0, Ellipsis),
        (1, Ellipsis, 2),
        (slice(1, 3), Ellipsis, slice(None, None, -1)),
        (None, 0),
        ([0, 2],),
        ([0, 2], 1),
        numpy.array([True, False, True, False, True]),
    ],
)
def testGetItem(index):
    """Indexing works as for numpy without making blissdata wait"""
    array = numpy.arange(5 * 3 * 4).reshape(5, 3, 4)
    stream = _FakeStream(array)
    dataset = BlissDataDataset(
        "data", redis_hdf5.Dataset(None, "/data", {}, redis_hdf5.StreamAsArray(stream))
    )

    result = dataset[index]

    numpy.testing.assert_array_equal(result, array[index])
    assert numpy.asarray(result).shape == array[index].shape


@pytest.mark.parametrize("index", [5, -6, (5, 0)])
def testGetItemOutOfRange(index):
    array = numpy.arange(5 * 3 * 4).reshape(5, 3, 4)
    stream = _FakeStream(array)
    dataset = BlissDataDataset(
        "data", redis_hdf5.Dataset(None, "/data", {}, redis_hdf5.StreamAsArray(stream))
    )
    with pytest.raises(IndexError):
        dataset[index]


def testGetItemReadsOnlyNeededRange():
    array = numpy.arange(5 * 3 * 4).reshape(5, 3, 4)
    stream = _FakeStream(array)
    dataset = BlissDataDataset(
        "data", redis_hdf5.Dataset(None, "/data", {}, redis_hdf5.StreamAsArray(stream))
    )
    dataset[0, 0, 0]
    dataset[1:3, 0]
    dataset[::2]
    dataset[4:0:-2]
    assert stream.getitem_calls == [0, slice(1, 3, 1), slice(0, 5, 2), slice(2, 5, 2)]


@pytest.mark.parametrize("index", [(), Ellipsis])
def testGetItemScalar(index):
    dataset = BlissDataDataset(
        "data", redis_hdf5.Dataset(None, "/data", {}, numpy.array(42))
    )
    assert dataset[index] == 42


def testValueAndArray():
    array = numpy.arange(5 * 3 * 4).reshape(5, 3, 4)
    stream = _FakeStream(array)
    dataset = BlissDataDataset(
        "data", redis_hdf5.Dataset(None, "/data", {}, redis_hdf5.StreamAsArray(stream))
    )
    numpy.testing.assert_array_equal(dataset.value, array)
    numpy.testing.assert_array_equal(numpy.asarray(dataset), array)


def testIter(monkeypatch):
    """Iteration reads chunks of available data and does not wait"""
    array = numpy.arange(5 * 3 * 4).reshape(5, 3, 4)
    monkeypatch.setattr(
        blissdatah5.BlissDataDataset, "_ITER_CHUNK_NBYTES", 2 * array[0].nbytes
    )
    stream = _FakeStream(array)
    dataset = BlissDataDataset(
        "data", redis_hdf5.Dataset(None, "/data", {}, redis_hdf5.StreamAsArray(stream))
    )

    items = list(dataset)  # iterate

    assert len(items) == len(array)
    for item, expected in zip(items, array):
        numpy.testing.assert_array_equal(item, expected)
    assert stream.getitem_calls == [slice(0, 2, 1), slice(2, 4, 1), slice(4, 5, 1)]


def testIter1D():
    array = numpy.arange(10)
    stream = _FakeStream(array)
    dataset = BlissDataDataset(
        "data", redis_hdf5.Dataset(None, "/data", {}, redis_hdf5.StreamAsArray(stream))
    )
    assert list(dataset) == list(array)


def testIterEmpty():
    stream = _FakeStream(numpy.empty((0, 2)))
    dataset = BlissDataDataset(
        "data", redis_hdf5.Dataset(None, "/data", {}, redis_hdf5.StreamAsArray(stream))
    )
    assert list(dataset) == []


def testIterScalar():
    dataset = BlissDataDataset(
        "data", redis_hdf5.Dataset(None, "/data", {}, numpy.array(42))
    )
    with pytest.raises(TypeError):
        list(dataset)


class _BlissdataGroup(abstract.Group):
    """Minimal blissdata group holding the given children"""

    name = "/"
    attrs = {}
    parent = None
    file = None

    def __init__(self, **children: abstract.Node):
        self._children = children

    def __getitem__(self, key: str) -> abstract.Node:
        return self._children[key]

    def __iter__(self):
        return iter(self._children)

    def __len__(self) -> int:
        return len(self._children)


def testLima1Unsealed():
    """Dataset of an unsealed lima1 stream is empty"""
    array = numpy.arange(5 * 3 * 4).reshape(5, 3, 4)
    stream = _FakeStream(array, plugin="lima")
    data = redis_hdf5.Dataset(None, "/data", {}, redis_hdf5.StreamAsArray(stream))
    (dataset,) = blissdatah5._children(_BlissdataGroup(data=data))
    assert isinstance(dataset, BlissDataLima1Dataset)

    assert len(dataset) == 0
    assert dataset.shape == (0, 3, 4)
    assert dataset.size == 0
    assert dataset[()].shape == (0, 3, 4)
    assert dataset[1:].shape == (0, 3, 4)
    assert list(dataset) == []
    with pytest.raises(IndexError):
        dataset[0]


def testLima1Sealed():
    """Dataset of a lima1 stream exposes data once sealed"""
    array = numpy.arange(5 * 3 * 4).reshape(5, 3, 4)
    stream = _FakeStream(array, plugin="lima")
    data = redis_hdf5.Dataset(None, "/data", {}, redis_hdf5.StreamAsArray(stream))
    (dataset,) = blissdatah5._children(_BlissdataGroup(data=data))
    assert len(dataset) == 0

    stream.sealed = True
    assert len(dataset) == len(array)
    assert dataset.shape == array.shape
    assert dataset.size == array.size
    numpy.testing.assert_array_equal(dataset[()], array)
    numpy.testing.assert_array_equal(dataset[-1], array[-1])

    # Sealed state is cached
    count = stream.isSealedCallCount
    len(dataset)
    assert stream.isSealedCallCount == count
