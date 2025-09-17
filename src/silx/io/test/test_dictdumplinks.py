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
from ..dictdumplinks import ExternalBinaryLink


def test_soft_link_from_str(tmp_path):
    data_file = str(tmp_path / "data.h5")
    link1 = link_from_serialized(f"{data_file}::/group2/link", "../group1/dataset")
    assert isinstance(link1, h5py.SoftLink)
    assert link1.path == "/group1/dataset"

    with h5py.File(data_file, mode="w") as fh:
        fh["/group1/dataset"] = 42
        fh["/group2/link"] = link1

    with h5py.File(data_file, mode="r") as fh:
        assert fh["/group2/link"][()] == 42
        link2 = link_from_hdf5(fh, "/group2/link")

    assert link1.path == link2.path


def test_external_link_from_str(tmp_path):
    data_file = str(tmp_path / "data.h5")
    with h5py.File(data_file, mode="w") as fh:
        fh["/group/dataset"] = 42

    master_file = str(tmp_path / "master.h5")
    link1 = link_from_serialized(
        f"{master_file}::/group/link", "data.h5::/group/dataset"
    )
    assert isinstance(link1, h5py.ExternalLink)
    assert link1.filename == "data.h5"
    assert link1.path == "/group/dataset"

    with h5py.File(master_file, mode="w") as fh:
        fh["/group/link"] = link1

    with h5py.File(master_file, mode="r") as fh:
        assert fh["/group/link"][()] == 42
        link2 = link_from_hdf5(fh, "/group/link")

    assert link1.filename == link2.filename
    assert link1.path == link2.path


@pytest.mark.parametrize(
    "internal", [True, False], ids=lambda val: "internal" if val else "external"
)
@pytest.mark.parametrize("len_urls", [0, 1, 3], ids=lambda val: f"len_{val}")
@pytest.mark.parametrize(
    "nimages_per_dataset",
    [1, 4],
    ids=lambda val: f"{val}_image" if val == 1 else f"{val}_images",
)
@pytest.mark.parametrize(
    "read_sources", [True, False], ids=lambda val: "read" if val else "no-read"
)
def test_vds_from_str(tmp_path, len_urls, nimages_per_dataset, internal, read_sources):
    file_content = list()
    ndatasets = max(len_urls, 1)
    dtype = numpy.uint16
    if nimages_per_dataset == 1:
        shape = (10, 4)
    else:
        shape = (nimages_per_dataset, 10, 4)

    for i in range(ndatasets):
        offset = 40 * nimages_per_dataset * i
        data = (
            numpy.arange(40 * nimages_per_dataset, dtype=dtype).reshape(shape) + offset
        )
        file_content.append(data)

    def save_sources():
        for i, data in enumerate(file_content):
            if internal:
                data_file = str(tmp_path / "master.h5")
            else:
                data_file = str(tmp_path / f"data{i}.h5")
            with h5py.File(data_file, mode="a") as fh:
                if internal:
                    fh[f"/group/dataset{i}"] = data
                else:
                    fh["/group/dataset"] = data

    scalar_target = len_urls == 0
    if nimages_per_dataset == 1:
        if scalar_target:
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
        if scalar_target:
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

    if read_sources:
        # shape and dtype will be read from the sources
        save_sources()
    else:
        # shape and dtype are known and the same for all sources
        target = {
            "dictdump_schema": "vds_urls_v1",
            "source_shape": shape,
            "source_dtype": "uint16",
            "sources": target,
        }

    master_file = str(tmp_path / "master.h5")
    link1 = link_from_serialized(f"{master_file}::/group/link", target)
    assert isinstance(link1, h5py.VirtualLayout)
    assert link1.shape == concat_data.shape
    assert link1.dtype == concat_data.dtype

    with h5py.File(master_file, mode="a") as fh:
        fh.create_virtual_dataset("/group/link", link1)

    if not read_sources:
        save_sources()

    with h5py.File(master_file, mode="r") as fh:
        vds_data = fh["/group/link"][()]
        numpy.testing.assert_almost_equal(vds_data, concat_data)
        link2 = link_from_hdf5(fh, "/group/link")

    assert link2 is None


@pytest.mark.parametrize("len_urls", [0, 1, 3], ids=lambda val: f"len_{val}")
@pytest.mark.parametrize(
    "nimages_per_file",
    [1, 4],
    ids=lambda val: f"{val}_image" if val == 1 else f"{val}_images",
)
def test_external_edf_from_str(tmp_path, len_urls, nimages_per_file):
    file_content = list()
    nfiles = max(len_urls, 1)
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

    scalar_target = len_urls == 0
    if nimages_per_file == 1:
        if scalar_target:
            concat_data = file_content[0]
            target = str(tmp_path / "data0.edf")
        else:
            concat_data = numpy.array(file_content)
            target = [str(tmp_path / f"data{i}.edf") for i in range(nfiles)]
    else:
        if scalar_target:
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


@pytest.mark.parametrize("len_urls", [0, 1, 3], ids=lambda val: f"len_{val}")
@pytest.mark.parametrize(
    "nimages_per_file",
    [1, 4],
    ids=lambda val: f"{val}_image" if val == 1 else f"{val}_images",
)
def test_external_tiff_from_str(tmp_path, len_urls, nimages_per_file):
    file_content = list()
    nfiles = max(len_urls, 1)
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
                pytest.skip(reason="requires imageio[tifffile]")
            imageio.mimwrite(data_file, list(data))
        file_content.append(data)

    scalar_target = len_urls == 0
    if nimages_per_file == 1:
        if scalar_target:
            concat_data = file_content[0]
            target = str(tmp_path / "data0.tiff")
        else:
            concat_data = numpy.array(file_content)
            target = [str(tmp_path / f"data{i}.tiff") for i in range(nfiles)]
    else:
        if scalar_target:
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


_SINGLE_IDX = [
    None,
    tuple(),
    slice(None),
    (slice(None),),
    (None,),
    (slice(None), slice(None)),
    (None, slice(None)),
    (slice(None), None),
    (None, None),
]


@pytest.mark.parametrize("target_type", ["vds_v1", "vds_v1_extra"])
@pytest.mark.parametrize("source_index", _SINGLE_IDX)
@pytest.mark.parametrize("target_index", _SINGLE_IDX)
def test_single_hdf5_source_full_schema(
    tmp_path, target_type, source_index, target_index
):
    _assert_single_hdf5_source(tmp_path, target_type, source_index, target_index)


@pytest.mark.parametrize("target_type", ["vds_urls_v1", "string", "string_slice"])
def test_single_hdf5_source(tmp_path, target_type):
    _assert_single_hdf5_source(tmp_path, target_type, NotImplemented, NotImplemented)


def _assert_single_hdf5_source(tmp_path, target_type, source_index, target_index):
    data = numpy.ones((3, 2), dtype=numpy.uint16)
    with h5py.File(tmp_path / "ext.h5", "w") as h5f:
        h5f["data"] = data

    if target_type == "vds_v1_extra":
        vds = True
        extra_dim = True
        target = {
            "dictdump_schema": "vds_v1",
            "shape": (1, 3, 2),
            "dtype": "uint16",
            "sources": [
                {
                    "data_path": "/data",
                    "dtype": "uint16",
                    "file_path": "ext.h5",
                    "shape": (3, 2),
                    "source_index": source_index,
                    "target_index": target_index,
                }
            ],
        }
    elif target_type == "vds_v1":
        vds = True
        extra_dim = False
        target = {
            "dictdump_schema": "vds_v1",
            "shape": (3, 2),
            "dtype": "uint16",
            "sources": [
                {
                    "data_path": "/data",
                    "dtype": "uint16",
                    "file_path": "ext.h5",
                    "shape": (3, 2),
                    "source_index": source_index,
                    "target_index": target_index,
                }
            ],
        }
    elif target_type == "vds_urls_v1":
        vds = True
        extra_dim = True
        target = {
            "dictdump_schema": "vds_urls_v1",
            "source_shape": (3, 2),
            "source_dtype": "uint16",
            "sources": ["ext.h5?path=/data"],
        }
    elif target_type == "string":
        vds = False
        extra_dim = False
        target = "ext.h5?path=/data"
    elif target_type == "string_slice":
        vds = True
        extra_dim = False
        target = "ext.h5?path=/data&slice=:"
    else:
        raise ValueError(target_type)

    master_file = tmp_path / "master.h5"
    link = link_from_serialized(f"{master_file}::/data", target)
    with h5py.File(master_file, "w") as h5f:
        if vds:
            h5f.create_virtual_dataset("data", link)
        else:
            h5f["data"] = link

    with h5py.File(master_file, "r") as h5f:
        link = h5f.get("data", getlink=True)
        dset = h5f["data"]
        is_virtual = dset.is_virtual
        data_final = dset[()]
        if extra_dim:
            data = data[numpy.newaxis, ...]
        numpy.testing.assert_equal(data, data_final)
        assert vds is is_virtual
        if vds:
            assert isinstance(link, h5py.HardLink)
        else:
            assert isinstance(link, h5py.ExternalLink)
