import h5py
import numpy
import pytest
from fabio.edfimage import EdfImage
from fabio.tifimage import TifImage

try:
    import imageio
except ImportError:
    imageio = None

from ..dictdumplinks import link_from_hdf5
from ..dictdumplinks import link_from_serialized
from ..dictdumplinks._link_types import InternalLink
from ..dictdumplinks._link_types import ExternalLink
from ..dictdumplinks._link_types import VDSLink
from ..dictdumplinks._link_types import ExternalBinaryLink


def test_soft_link_from_str(tmp_path):
    data_file = str(tmp_path / "data.h5")
    link1 = link_from_serialized(f"{data_file}::/group2/link", "../group1/dataset")
    assert isinstance(link1, InternalLink)
    assert link1.data_path == "/group1/dataset"

    with h5py.File(data_file, mode="w") as fh:
        fh["/group1/dataset"] = 42
        link1.create(fh, "/group2/link")

    with h5py.File(data_file, mode="r") as fh:
        assert fh["/group2/link"][()] == 42
        link2 = link_from_hdf5(fh, "/group2/link")

    assert link1 == link2


def test_external_link_from_str(tmp_path):
    data_file = str(tmp_path / "data.h5")
    with h5py.File(data_file, mode="w") as fh:
        fh["/group/dataset"] = 42

    master_file = str(tmp_path / "master.h5")
    link1 = link_from_serialized(
        f"{master_file}::/group/link", "data.h5::/group/dataset"
    )
    assert isinstance(link1, ExternalLink)
    assert link1.file_path == "data.h5"
    assert link1.data_path == "/group/dataset"

    with h5py.File(master_file, mode="w") as fh:
        link1.create(fh, "/group/link")

    with h5py.File(master_file, mode="r") as fh:
        assert fh["/group/link"][()] == 42
        link2 = link_from_hdf5(fh, "/group/link")

    assert link1 == link2


def test_vds_from_str(tmp_path):
    data_file = str(tmp_path / "data.h5")
    data = numpy.random.uniform(size=(10, 10))
    with h5py.File(data_file, mode="w") as fh:
        fh["/group/dataset"] = data

    sliced_data = data[1:5, 2:3]
    master_file = str(tmp_path / "master.h5")
    link1 = link_from_serialized(
        f"{master_file}::/group/link",
        "data.h5?path=/group/dataset&slice=1:5,2:3",
    )
    assert isinstance(link1, VDSLink)
    assert link1.shape == sliced_data.shape
    assert link1.dtype == sliced_data.dtype

    with h5py.File(master_file, mode="w") as fh:
        link1.create(fh, "/group/link")

    with h5py.File(master_file, mode="r") as fh:
        data = fh["/group/link"][()]
        numpy.testing.assert_almost_equal(data, sliced_data)
        link2 = link_from_hdf5(fh, "/group/link")

    assert link2 is None


def test_external_edf_from_str(tmp_path):
    data_file = str(tmp_path / "data.edf")
    data = numpy.arange(40, dtype=numpy.uint16).reshape(10, 4)
    with EdfImage(data=data) as edfimage:
        edfimage.write(data_file)

    master_file = str(tmp_path / "master.h5")
    link1 = link_from_serialized(f"{master_file}::/group/link", data_file)
    assert isinstance(link1, ExternalBinaryLink)

    with h5py.File(tmp_path / "master.h5", mode="w") as fh:
        group = fh.create_group("group")
        link1.create(group, "link")

    with h5py.File(tmp_path / "master.h5", mode="r") as fh:
        ext_data = fh["/group/link"][()]
        numpy.testing.assert_almost_equal(ext_data, data)
        link2 = link_from_hdf5(fh, "/group/link")

    assert link2 is None


def test_external_multipage_edf_from_str(tmp_path):
    data_file = str(tmp_path / "data.edf")
    data1 = numpy.arange(40, dtype=numpy.uint16).reshape(10, 4)
    data2 = numpy.arange(40, dtype=numpy.uint16).reshape(10, 4)
    with EdfImage(data=data1) as edfimage:
        edfimage.append_frame(data=data2)
        edfimage.write(data_file)

    master_file = str(tmp_path / "master.h5")
    link1 = link_from_serialized(f"{master_file}::/group/link", data_file)
    assert isinstance(link1, ExternalBinaryLink)

    with h5py.File(tmp_path / "master.h5", mode="w") as fh:
        group = fh.create_group("group")
        link1.create(group, "link")

    with h5py.File(tmp_path / "master.h5", mode="r") as fh:
        ext_data1 = fh["/group/link"][0]
        ext_data2 = fh["/group/link"][1]
        numpy.testing.assert_almost_equal(ext_data1, data1)
        numpy.testing.assert_almost_equal(ext_data2, data2)


def test_external_tiff_from_str(tmp_path):
    data_file = str(tmp_path / "data.tiff")

    data = numpy.arange(40, dtype=numpy.uint8).reshape(10, 4)

    with TifImage(data=data) as tifimage:
        tifimage.write(data_file)

    master_file = str(tmp_path / "master.h5")
    link1 = link_from_serialized(f"{master_file}::/group/link", data_file)
    assert isinstance(link1, ExternalBinaryLink)

    with h5py.File(tmp_path / "master.h5", mode="w") as fh:
        group = fh.create_group("group")
        link1.create(group, "link")

    with h5py.File(tmp_path / "master.h5", mode="r") as fh:
        ext_data = fh["/group/link"][()]
        numpy.testing.assert_almost_equal(ext_data, data)


@pytest.mark.skipif(imageio is None, reason="requires imageio")
def test_external_multipage_tiff_from_str(tmp_path):
    data_file = str(tmp_path / "data.tiff")

    data1 = numpy.arange(40, dtype=numpy.uint8).reshape(10, 4)
    data2 = numpy.arange(40, dtype=numpy.uint8).reshape(10, 4)

    imageio.mimwrite(data_file, [data1, data2])

    master_file = str(tmp_path / "master.h5")
    link1 = link_from_serialized(f"{master_file}::/group/link", data_file)
    assert isinstance(link1, ExternalBinaryLink)

    with h5py.File(tmp_path / "master.h5", mode="w") as fh:
        group = fh.create_group("group")
        link1.create(group, "link")

    with h5py.File(tmp_path / "master.h5", mode="r") as fh:
        ext_data1 = fh["/group/link"][0]
        ext_data2 = fh["/group/link"][1]
        numpy.testing.assert_almost_equal(ext_data1, data1)
        numpy.testing.assert_almost_equal(ext_data2, data2)
