from collections.abc import Sequence

import h5py
from numpy.typing import DTypeLike

from ..url import DataUrl
from ._parse_hdf5_utils import hdf5_url_to_vds_schema
from ._parse_hdf5_utils import hdf5_urls_to_vds_schema
from ._vds import VdsModelV1


def hdf5_url_to_vds(
    source: DataUrl,
    target: DataUrl,
    target_shape: tuple[int, ...] | None = None,
    target_dtype: DTypeLike | None = None,
) -> h5py.VirtualLayout:
    """Single HDF5 dataset: keep original shape (no new axis).

    When ``target_shape`` and ``target_dtype`` are not ``None``
    the target file will not be opened.
    """
    target_desc = hdf5_url_to_vds_schema(
        source, target, target_shape=target_shape, target_dtype=target_dtype
    )
    return VdsModelV1(**target_desc).tolink(source)


def hdf5_urls_to_vds(
    source: DataUrl,
    targets: Sequence[DataUrl],
    target_shape: tuple[int, ...] | None = None,
    target_dtype: DTypeLike | None = None,
) -> h5py.VirtualLayout:
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
    target_desc = hdf5_urls_to_vds_schema(
        source, targets, target_shape=target_shape, target_dtype=target_dtype
    )
    return VdsModelV1(**target_desc).tolink(source)
