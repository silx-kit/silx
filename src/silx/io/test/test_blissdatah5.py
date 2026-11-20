import numpy
import pytest

pytest.importorskip("blissdata")

from .. import blissdatah5  # noqa: E402
from ..blissdatah5 import BlissDataDataset  # noqa: E402


class _StreamLikeDataset:
    """Mimics a blissdata redis_hdf5 Dataset wrapping a stream.

    It fails for indices which would make blissdata wait for new data or
    for the end of the scan (negative indices, slices without explicit
    positive stop, indices out of the available data) and for indices not
    supported by blissdata streams (negative steps, tuples, numpy integers).
    """

    def __init__(self, array: numpy.ndarray):
        self._array = array
        self.calls = []

    @property
    def shape(self):
        return self._array.shape

    @property
    def ndim(self):
        return self._array.ndim

    @property
    def size(self):
        return self._array.size

    @property
    def dtype(self):
        return self._array.dtype

    def __len__(self):
        return len(self._array)

    def __iter__(self):
        raise AssertionError("Blocking iteration until the end of the stream")

    def __getitem__(self, idx):
        self.calls.append(idx)
        if self.ndim == 0:
            assert idx == ()
            return self._array[()]
        if type(idx) is int:
            assert 0 <= idx < len(self), f"Blocking index {idx}"
        elif isinstance(idx, slice):
            assert idx.start is not None and idx.start >= 0
            assert idx.stop is not None and 0 <= idx.stop <= len(self)
            assert idx.step is not None and idx.step > 0
        else:
            raise AssertionError(f"Unsupported index: {idx!r}")
        return self._array[idx]


@pytest.fixture
def array():
    return numpy.arange(5 * 3 * 4).reshape(5, 3, 4)


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
def testGetItem(array, index):
    """Indexing works as for numpy without making blissdata wait"""
    data = _StreamLikeDataset(array)
    dataset = BlissDataDataset("data", data)

    result = dataset[index]

    numpy.testing.assert_array_equal(result, array[index])
    assert numpy.asarray(result).shape == array[index].shape


@pytest.mark.parametrize("index", [5, -6, (5, 0)])
def testGetItemOutOfRange(array, index):
    dataset = BlissDataDataset("data", _StreamLikeDataset(array))
    with pytest.raises(IndexError):
        dataset[index]


def testGetItemEmptySliceDoesNotRead(array):
    data = _StreamLikeDataset(array)
    dataset = BlissDataDataset("data", data)
    result = dataset[3:1]
    assert result.shape == (0, 3, 4)
    assert result.dtype == array.dtype
    assert data.calls == []


def testGetItemReadsOnlyNeededRange(array):
    data = _StreamLikeDataset(array)
    dataset = BlissDataDataset("data", data)
    dataset[0, 0, 0]
    dataset[1:3, 0]
    assert data.calls == [0, slice(1, 3, 1)]


def testGetItemStepReadsOnlySelected(array):
    data = _StreamLikeDataset(array)
    dataset = BlissDataDataset("data", data)
    dataset[::2]
    dataset[4:0:-2]
    assert data.calls == [slice(0, 5, 2), slice(2, 5, 2)]


@pytest.mark.parametrize("index", [(), Ellipsis])
def testGetItemScalar(index):
    dataset = BlissDataDataset("data", _StreamLikeDataset(numpy.array(42)))
    assert dataset[index] == 42


def testValueAndArray(array):
    dataset = BlissDataDataset("data", _StreamLikeDataset(array))
    numpy.testing.assert_array_equal(dataset.value, array)
    numpy.testing.assert_array_equal(numpy.asarray(dataset), array)


def testIter(monkeypatch, array):
    """Iteration reads chunks of available data and does not wait"""
    monkeypatch.setattr(blissdatah5, "_ITER_CHUNK_NBYTES", 2 * array[0].nbytes)
    data = _StreamLikeDataset(array)
    dataset = BlissDataDataset("data", data)

    items = list(dataset)

    assert len(items) == len(array)
    for item, expected in zip(items, array):
        numpy.testing.assert_array_equal(item, expected)
    assert data.calls == [slice(0, 2, 1), slice(2, 4, 1), slice(4, 5, 1)]


def testIter1D():
    array = numpy.arange(10)
    assert list(BlissDataDataset("data", _StreamLikeDataset(array))) == list(array)


def testIterEmpty():
    dataset = BlissDataDataset("data", _StreamLikeDataset(numpy.empty((0, 2))))
    assert list(dataset) == []


def testIterScalar():
    dataset = BlissDataDataset("data", _StreamLikeDataset(numpy.array(42)))
    with pytest.raises(TypeError):
        list(dataset)
