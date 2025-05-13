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
from ..dictdumplinks import VDSLink
from ..dictdumplinks import ExternalBinaryLink
from ..dictdumplinks import InternalLink
from ..dictdumplinks import ExternalLink


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


@pytest.mark.parametrize(
    "internal", [True, False], ids=lambda val: "internal" if val else "external"
)
@pytest.mark.parametrize(
    "ndatasets",
    [1, 3],
    ids=lambda val: f"{val}_dataset" if val == 1 else f"{val}_datasets",
)
@pytest.mark.parametrize(
    "nimages_per_dataset",
    [1, 4],
    ids=lambda val: f"{val}_image" if val == 1 else f"{val}_images",
)
def test_vds_from_str(tmp_path, ndatasets, nimages_per_dataset, internal):
    file_content = list()
    for i in range(ndatasets):
        if internal:
            data_file = str(tmp_path / "master.h5")
        else:
            data_file = str(tmp_path / f"data{i}.h5")
        offset = 40 * nimages_per_dataset * i
        if nimages_per_dataset == 1:
            shape = (10, 4)
        else:
            shape = (nimages_per_dataset, 10, 4)
        data = (
            numpy.arange(40 * nimages_per_dataset, dtype=numpy.uint16).reshape(shape)
            + offset
        )
        with h5py.File(data_file, mode="a") as fh:
            if internal:
                fh[f"/group/dataset{i}"] = data
            else:
                fh["/group/dataset"] = data
        file_content.append(data)

    if nimages_per_dataset == 1:
        if ndatasets == 1:
            concat_data = file_content[0][1:5, 2:3]
            if internal:
                target = "master.h5?path=/group/dataset0&slice=1:5,2:3"
            else:
                target = "data0.h5?path=/group/dataset&slice=1:5,2:3"
        else:
            concat_data = numpy.array([data[1:5, 2:3] for data in file_content])
            if internal:
                target = [
                    f"master.h5?path=/group/dataset{i}&slice=1:5,2:3"
                    for i in range(ndatasets)
                ]
            else:
                target = [
                    f"data{i}.h5?path=/group/dataset&slice=1:5,2:3"
                    for i in range(ndatasets)
                ]
    else:
        if ndatasets == 1:
            concat_data = file_content[0][:, 1:5, 2:3]
            if internal:
                target = "master.h5?path=dataset0&slice=:,1:5,2:3"
            else:
                target = "data0.h5?path=/group/dataset&slice=:,1:5,2:3"
        else:
            concat_data = numpy.vstack([data[:, 1:5, 2:3] for data in file_content])
            if internal:
                target = [
                    f"master.h5?path=/group/dataset{i}&slice=:,1:5,2:3"
                    for i in range(ndatasets)
                ]
            else:
                target = [
                    f"data{i}.h5?path=/group/dataset&slice=:,1:5,2:3"
                    for i in range(ndatasets)
                ]

    master_file = str(tmp_path / "master.h5")
    link1 = link_from_serialized(f"{master_file}::/group/link", target)
    assert isinstance(link1, VDSLink)
    assert link1.shape == concat_data.shape
    assert link1.dtype == concat_data.dtype

    with h5py.File(master_file, mode="a") as fh:
        link1.create(fh, "/group/link")

    with h5py.File(master_file, mode="r") as fh:
        vds_data = fh["/group/link"][()]
        numpy.testing.assert_almost_equal(vds_data, concat_data)
        link2 = link_from_hdf5(fh, "/group/link")

    assert link2 is None


@pytest.mark.parametrize(
    "nfiles", [1, 3], ids=lambda val: f"{val}_file" if val == 1 else f"{val}_files"
)
@pytest.mark.parametrize(
    "nimages_per_file",
    [1, 4],
    ids=lambda val: f"{val}_image" if val == 1 else f"{val}_images",
)
def test_external_edf_from_str(tmp_path, nfiles, nimages_per_file):
    file_content = list()
    for i in range(nfiles):
        data_file = str(tmp_path / f"data{i}.edf")
        offset = 40 * nimages_per_file * i
        if nimages_per_file == 1:
            shape = (10, 4)
        else:
            shape = (nimages_per_file, 10, 4)
        data = (
            numpy.arange(40 * nimages_per_file, dtype=numpy.uint16).reshape(shape)
            + offset
        )
        if nimages_per_file == 1:
            with EdfImage(data=data) as edfimage:
                edfimage.write(data_file)
        else:
            with EdfImage(data=data[0]) as edfimage:
                for d in data[1:]:
                    edfimage.append_frame(data=d)
                edfimage.write(data_file)
        file_content.append(data)

    if nimages_per_file == 1:
        if nfiles == 1:
            concat_data = file_content[0]
            target = str(tmp_path / "data0.edf")
        else:
            concat_data = numpy.array(file_content)
            target = [str(tmp_path / f"data{i}.edf") for i in range(nfiles)]
    else:
        if nfiles == 1:
            concat_data = file_content[0]
            target = str(tmp_path / "data0.edf")
        else:
            concat_data = numpy.vstack(file_content)
            target = [str(tmp_path / f"data{i}.edf") for i in range(nfiles)]

    master_file = str(tmp_path / "master.h5")
    link1 = link_from_serialized(f"{master_file}::/group/link", target)
    assert isinstance(link1, ExternalBinaryLink)
    assert link1.shape == concat_data.shape
    assert link1.dtype == concat_data.dtype

    with h5py.File(tmp_path / "master.h5", mode="w") as fh:
        group = fh.create_group("group")
        link1.create(group, "link")

    with h5py.File(tmp_path / "master.h5", mode="r") as fh:
        ext_data = fh["/group/link"][()]
        numpy.testing.assert_almost_equal(ext_data, concat_data)
        link2 = link_from_hdf5(fh, "/group/link")

    assert link2 is None


@pytest.mark.parametrize(
    "nfiles", [1, 3], ids=lambda val: f"{val}_file" if val == 1 else f"{val}_files"
)
@pytest.mark.parametrize(
    "nimages_per_file",
    [1, 4],
    ids=lambda val: f"{val}_image" if val == 1 else f"{val}_images",
)
def test_external_tiff_from_str(tmp_path, nfiles, nimages_per_file):
    file_content = list()
    for i in range(nfiles):
        data_file = str(tmp_path / f"data{i}.tiff")
        offset = 40 * nimages_per_file * i
        if nimages_per_file == 1:
            shape = (10, 4)
        else:
            shape = (nimages_per_file, 10, 4)
        data = (
            numpy.arange(40 * nimages_per_file, dtype=numpy.uint16).reshape(shape)
            + offset
        )
        if nimages_per_file == 1:
            with TifImage(data=data) as tifimage:
                tifimage.write(data_file)
        else:
            if imageio is None:
                pytest.skip(reason="requires imageio")
            imageio.mimwrite(data_file, list(data))
        file_content.append(data)

    if nimages_per_file == 1:
        if nfiles == 1:
            concat_data = file_content[0]
            target = str(tmp_path / "data0.tiff")
        else:
            concat_data = numpy.array(file_content)
            target = [str(tmp_path / f"data{i}.tiff") for i in range(nfiles)]
    else:
        if nfiles == 1:
            concat_data = file_content[0]
            target = str(tmp_path / "data0.tiff")
        else:
            concat_data = numpy.vstack(file_content)
            target = [str(tmp_path / f"data{i}.tiff") for i in range(nfiles)]

    master_file = str(tmp_path / "master.h5")
    link1 = link_from_serialized(f"{master_file}::/group/link", target)
    assert isinstance(link1, ExternalBinaryLink)
    assert link1.shape == concat_data.shape
    assert link1.dtype == concat_data.dtype

    with h5py.File(tmp_path / "master.h5", mode="w") as fh:
        group = fh.create_group("group")
        link1.create(group, "link")

    with h5py.File(tmp_path / "master.h5", mode="r") as fh:
        ext_data = fh["/group/link"][()]
        numpy.testing.assert_almost_equal(ext_data, concat_data)
        link2 = link_from_hdf5(fh, "/group/link")

    assert link2 is None
