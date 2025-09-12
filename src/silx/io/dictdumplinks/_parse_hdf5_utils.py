from collections.abc import Sequence
from typing import Any

import h5py
import numpy
from numpy.typing import DTypeLike

from ..url import DataUrl
from ._utils import absolute_file_path
from ._utils import is_same_file
from ._utils import normalize_vds_source_url


def hdf5_url_to_vds_schema(
    source: DataUrl,
    target: DataUrl,
    target_shape: tuple[int, ...] | None = None,
    target_dtype: DTypeLike | None = None,
) -> dict:
    """Single HDF5 dataset: keep original shape (no new axis).

    When ``target_shape`` and ``target_dtype`` are not ``None``
    the target file will not be opened.
    """
    datasets = [
        _get_target_info(
            source,
            target,
            target_shape=target_shape,
            target_dtype=target_dtype,
            allow_new_axis=False,
        )
    ]
    return _build_vds_schema(datasets)


def hdf5_urls_to_vds_schema(
    source: DataUrl,
    targets: Sequence[DataUrl],
    target_shape: tuple[int, ...] | None = None,
    target_dtype: DTypeLike | None = None,
) -> dict:
    """Multiple HDF5 datasets: stack when ndim<3, concatenate when ndim>=3.

    Examples for Nt targets

    - target `shape=()`               : VDS shape `(Nt,)`
    - target `shape=(N0,)`            : VDS shape `(Nt,N0)`
    - target `shape=(N0,N1)`          : VDS shape `(Nt,N0,N1)`
    - target `shape=(N0,N1,N2)`       : VDS shape `(Nt*N0,N1,N2)`
    - target `shape=(N0,N1,N2,N3)`    : VDS shape `(Nt*N0,N1,N2,N3)`
    - target `shape=(N0,N1,N2,N3,N4)` : VDS shape `(Nt*N0,N1,N2,N3,N4)`
    - ...

    When ``target_shape`` and ``target_dtype`` are not ``None``
    the target files will not be opened and all targets are assumed
    to have the same shape and dtype.
    """
    datasets = [
        _get_target_info(
            source, target, target_shape=target_shape, target_dtype=target_dtype
        )
        for target in targets
    ]
    return _build_vds_schema(datasets)


def _get_target_info(
    source: DataUrl,
    target: DataUrl,
    target_shape: tuple[int, ...] | None,
    target_dtype: DTypeLike | None,
    allow_new_axis: bool = True,
) -> dict:
    """Collect dataset information needed to build the VDS schema."""
    assert source.data_path()
    assert target.data_path()

    file_path = "." if is_same_file(source, target) else target.file_path()
    data_path = target.data_path()
    data_slice = target.data_slice()
    file_path, data_path = normalize_vds_source_url(file_path, data_path, source)

    if target_shape is None or target_dtype is None:
        abs_file_path = absolute_file_path(target.file_path(), source.file_path())
        source_dtype, source_shape, data_shape = _read_hdf5_dataset_info(
            abs_file_path, data_path, data_slice
        )
    else:
        source_shape = target_shape
        source_dtype = target_dtype
        data_shape = _get_slice_shape(target_shape, data_slice)

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
        data_shape = _get_slice_shape(dset.shape, data_slice)
        return dset.dtype, dset.shape, data_shape


def _get_slice_shape(shape: tuple[int, ...], data_slice) -> tuple[int, ...]:
    if data_slice in (None, tuple()):
        return tuple()
    else:
        dummy = numpy.empty(shape, dtype=bool)
        return dummy[data_slice].shape


def _build_vds_schema(datasets: list[dict]) -> dict:
    """Build target_desc from collected dataset metadata."""
    target_desc: dict[str, Any] = {
        "dictdump_schema": "vds_v1",
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
