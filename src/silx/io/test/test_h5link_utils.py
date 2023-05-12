import os
import pytest
import h5py
import numpy
from silx.io import open
from silx.io import h5link_utils


@pytest.fixture(scope="module")
def hdf5_with_external_data(tmpdir_factory):
    tmpdir = tmpdir_factory.mktemp("hdf5_with_external_data")
    master = str(tmpdir / "master.h5")
    external_h5 = str(tmpdir / "external.h5")
    external_raw = str(tmpdir / "external.raw")

    data = numpy.array([100, 1000, 10000], numpy.uint16)
    tshape = (1,) + data.shape

    with h5py.File(master, "w") as fmaster:
        dset = fmaster.create_dataset("data", data=data)

        fmaster["int"] = h5py.SoftLink("data")

        layout = h5py.VirtualLayout(shape=tshape, dtype=data.dtype)
        layout[0] = h5py.VirtualSource(".", "data", shape=data.shape)
        fmaster.create_virtual_dataset("vds0", layout)

        with h5py.File(external_h5, "w") as f:
            dset = f.create_dataset("data", data=data)
            layout = h5py.VirtualLayout(shape=tshape, dtype=data.dtype)
            layout[0] = h5py.VirtualSource(dset)
            fmaster.create_virtual_dataset("vds1", layout)

            layout = h5py.VirtualLayout(shape=tshape, dtype=data.dtype)
            layout[0] = h5py.VirtualSource(
                external_h5,
                "data",
                shape=data.shape,
            )
            fmaster.create_virtual_dataset("vds2", layout)
            fmaster["ext1"] = h5py.ExternalLink(external_h5, "data")

            layout = h5py.VirtualLayout(shape=tshape, dtype=data.dtype)
            layout[0] = h5py.VirtualSource(
                "external.h5",
                "data",
                shape=data.shape,
            )
            fmaster.create_virtual_dataset("vds3", layout)
            fmaster["ext2"] = h5py.ExternalLink("external.h5", "data")

            layout = h5py.VirtualLayout(shape=tshape, dtype=data.dtype)
            layout[0] = h5py.VirtualSource(
                "./external.h5",
                "data",
                shape=data.shape,
            )
            fmaster.create_virtual_dataset("vds4", layout)
            fmaster["ext3"] = h5py.ExternalLink("./external.h5", "data")

        data.tofile(external_raw)

        external = [(external_raw, 0, 16 * 3)]
        fmaster.create_dataset(
            "raw1", external=external, shape=tshape, dtype=data.dtype
        )

        external = [("external.raw", 0, 16 * 3)]
        fmaster.create_dataset(
            "raw2", external=external, shape=tshape, dtype=data.dtype
        )

        external = [("./external.raw", 0, 16 * 3)]
        fmaster.create_dataset(
            "raw3", external=external, shape=tshape, dtype=data.dtype
        )

    # Validate links
    expected = data.tolist()
    cwd = os.getcwd()
    with h5py.File(master, "r") as master:
        for name in master:
            if name in ("raw2", "raw3"):
                os.chdir(str(tmpdir))
            try:
                data = master[name][()].flatten().tolist()
            except Exception:
                assert False, name
            finally:
                if name in ("raw2", "raw3"):
                    os.chdir(cwd)
            assert data == expected, name

    return tmpdir


@pytest.mark.skipif("VirtualLayout" not in dir(h5py), reason="h5py is too old")
def test_external_dataset_info(hdf5_with_external_data):
    tmpdir = hdf5_with_external_data
    master = str(tmpdir / "master.h5")
    external_h5 = str(tmpdir / "external.h5")
    external_raw = str(tmpdir / "external.raw")
    with open(master) as f:
        for name in f:
            hdf5obj = f[name]
            info = h5link_utils.external_dataset_info(hdf5obj)
            if name in ("data", "int", "ext1", "ext2", "ext3"):
                assert info is None, name
            elif name == "vds0":
                assert info.first_source_url == f"{master}::/data"
            elif name in ("vds1", "vds2", "vds3", "vds4"):
                assert info.first_source_url == f"{external_h5}::/data"
            elif name in ("raw1", "raw2", "raw3"):
                assert info.first_source_url == external_raw
            else:
                assert False, name
