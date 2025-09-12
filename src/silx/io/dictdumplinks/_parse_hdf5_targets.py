from collections.abc import Sequence
from typing import Any

import h5py
import numpy

from ..url import DataUrl
from ._utils import absolute_file_path
from ._utils import is_same_file
from ._utils import normalize_vds_source_url
from ._vds import VdsModelV1


def hdf5_url_to_vds(source: DataUrl, target: DataUrl) -> h5py.VirtualLayout:
    """Single HDF5 dataset: keep original shape (no new axis)."""
    datasets = [_read_target_info(source, target, allow_new_axis=False)]
    target_desc = _build_vds_schema(datasets)
    return VdsModelV1(**target_desc).tolink(source)


def hdf5_urls_to_vds(source: DataUrl, targets: Sequence[DataUrl]) -> h5py.VirtualLayout:
    """Multiple HDF5 datasets: stack when ndim<3, concatenate when ndim>=3.

    Examples for Nt targets

    - target `shape=()`               : VDS shape `(Nt,)`
    - target `shape=(N0,)`            : VDS shape `(Nt,N0)`
    - target `shape=(N0,N1)`          : VDS shape `(Nt,N0,N1)`
    - target `shape=(N0,N1,N2)`       : VDS shape `(Nt*N0,N1,N2)`
    - target `shape=(N0,N1,N2,N3)`    : VDS shape `(Nt*N0,N1,N2,N3)`
    - target `shape=(N0,N1,N2,N3,N4)` : VDS shape `(Nt*N0,N1,N2,N3,N4)`
    - ...
    """
    datasets = [_read_target_info(source, t) for t in targets]
    target_desc = _build_vds_schema(datasets)
    return VdsModelV1(**target_desc).tolink(source)


def _read_target_info(
    source: DataUrl, target: DataUrl, allow_new_axis: bool = True
) -> dict:
    """Collect dataset information needed to build the VDS schema."""
    assert source.data_path()
    assert target.data_path()

    file_path = "." if is_same_file(source, target) else target.file_path()
    data_path = target.data_path()
    data_slice = target.data_slice()
    file_path, data_path = normalize_vds_source_url(file_path, data_path, source)

    abs_file_path = absolute_file_path(target.file_path(), source.file_path())
    source_dtype, source_shape, data_shape = _read_hdf5_dataset_info(
        abs_file_path, data_path, data_slice
    )

    ndim = len(data_shape)
    if allow_new_axis and ndim < 3:
        size_dim0 = 1
        vds_shape = (1,) + data_shape
    else:
        size_dim0 = data_shape[0]
        vds_shape = data_shape

    return dict(
        file_path=file_path,
        data_path=data_path,
        data_slice=data_slice,
        source_shape=source_shape,
        source_dtype=source_dtype,
        vds_shape=vds_shape,
        ndim=ndim,
        size_dim0=size_dim0,
    )


def _read_hdf5_dataset_info(
    file_path: str, data_path: str, data_slice
) -> tuple[str, tuple[int, ...], tuple[int, ...]]:
    """Open HDF5 file and read dataset dtype, source and target shape"""
    with h5py.File(file_path, locking=False, mode="r") as f:
        dset = f[data_path]
        if data_slice:
            dummy = numpy.empty(dset.shape, dtype=bool)
            data_shape = dummy[data_slice].shape
        else:
            data_shape = tuple()
        return dset.dtype, dset.shape, data_shape


def _build_vds_schema(datasets: list[dict]) -> dict:
    """Build target_desc from collected dataset metadata."""
    target_desc: dict[str, Any] = {
        "dictdump_schema": "virtual_dataset_v1",
        "sources": [],
        "shape": None,
        "dtype": None,
    }

    i0 = 0
    for idx, info in enumerate(datasets):
        vsource = {
            "file_path": info["file_path"],
            "data_path": info["data_path"],
            "shape": info["source_shape"],
            "dtype": info["source_dtype"],
            "source_index": info["data_slice"],
            "target_index": slice(i0, i0 + info["size_dim0"]),
        }

        if idx == 0:
            target_desc["shape"] = info["vds_shape"]
            target_desc["dtype"] = info["source_dtype"]
        else:
            if info["ndim"] < 3:
                # stack
                target_desc["shape"] = (target_desc["shape"][0] + 1,) + target_desc[
                    "shape"
                ][1:]
            else:
                # concatenate
                target_desc["shape"] = (
                    target_desc["shape"][0] + info["size_dim0"],
                ) + target_desc["shape"][1:]

        target_desc["sources"].append(vsource)
        i0 += info["size_dim0"]

    return target_desc
