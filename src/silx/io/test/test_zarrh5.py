"""Tests for zarrh5 wrapper"""

import h5py
import numpy
import pytest

zarr = pytest.importorskip("zarr")

from .. import zarrh5  # noqa: E402


@pytest.fixture
def zarr_dir(tmp_path):
    root = zarr.open_group(str(tmp_path), mode="w")
    root.attrs["file_attr"] = "hello"

    data = root.create_array("data", shape=(3, 4), dtype="f8")
    data[...] = numpy.arange(12).reshape(3, 4)
    data.attrs["units"] = "m"

    root.create_array("no_compression", shape=(5,), dtype="i4", compressors=None)[
        ...
    ] = numpy.arange(5)

    group = root.create_group("group")
    group.attrs["group_attr"] = 42
    group.create_array("sub_data", shape=(2, 3), dtype="i4")[...] = numpy.arange(
        6
    ).reshape(2, 3)

    return str(tmp_path)


def test_extra_attrs_are_merged_with_zarr_attrs(zarr_dir):
    h5 = zarrh5.ZarrH5(zarr_dir, attrs={"extra": "value"})
    assert h5.attrs["file_attr"] == "hello"
    assert h5.attrs["extra"] == "value"


def test_h5py_class(zarr_dir):
    h5 = zarrh5.ZarrH5(zarr_dir)
    assert h5.h5py_class == h5py.File
    assert h5["group"].h5py_class == h5py.Group
    assert h5["data"].h5py_class == h5py.Dataset


def test_group(zarr_dir):
    h5 = zarrh5.ZarrH5(zarr_dir)
    assert set(h5.keys()) == {"data", "no_compression", "group"}

    assert set(h5["group"].keys()) == {"sub_data"}

    assert "data" in h5
    assert "group/sub_data" in h5
    assert "not_there" not in h5


def test_dataset(zarr_dir):
    h5 = zarrh5.ZarrH5(zarr_dir)
    assert h5["data"].shape == (3, 4)
    assert h5["data"].size == 12
    assert h5["data"].dtype == numpy.dtype("f8")
    assert len(h5["data"]) == 3
    assert h5["data"].chunks == (3, 4)
    assert h5["data"].compression == "zstd"

    numpy.testing.assert_array_equal(h5["data"][...], numpy.arange(12).reshape(3, 4))
    numpy.testing.assert_array_equal(h5["data"].value, numpy.arange(12).reshape(3, 4))
    numpy.testing.assert_array_equal(h5["data"][0], numpy.arange(4))


def test_dataset_no_compression(zarr_dir):
    h5 = zarrh5.ZarrH5(zarr_dir)
    assert h5["no_compression"].compression is None


def test_attrs(zarr_dir):
    h5 = zarrh5.ZarrH5(zarr_dir)
    assert dict(h5.attrs) == {"file_attr": "hello"}
    assert dict(h5["group"].attrs) == {"group_attr": 42}
    assert dict(h5["data"].attrs) == {"units": "m"}
