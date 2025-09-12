import os
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any
from typing import cast

import h5py

from ..url import DataUrl
from ._link_types import Hdf5Link
from ._link_types import Hdf5LinkType
from ._link_types import SerializedHdf5LinkType
from ._parse_fabio_targets import fabio_url_to_external_data
from ._parse_fabio_targets import fabio_urls_to_external_data
from ._parse_hdf5_targets import hdf5_url_to_vds
from ._parse_hdf5_targets import hdf5_urls_to_vds
from ._parse_tiff_targets import tiff_url_to_external_data
from ._parse_tiff_targets import tiff_urls_to_external_data
from ._schemas import deserialize_mapping
from ._utils import absolute_data_path
from ._utils import absolute_file_path
from ._utils import is_same_file


def link_from_serialized(
    source: str | DataUrl,
    target: Hdf5LinkType | SerializedHdf5LinkType | Any,
) -> Hdf5LinkType | None:
    """Convert the target to a link instance when it describes a link.
    Otherwise return `None`.

    The target can be a single URL or a list of URL's. These are URL's to
    HDF5 datasets, EDF files or TIFF files.

    For example the source is this

    .. code-block:: python

        source = "/path/to/file.h5?path=/group/link"

    An `h5py.SoftLink` is returned for these targets

    .. code-block:: python

        target = "/path/to/file.h5?path=/group/dataset"
        target = "/path/to/file.h5?path=dataset"
        target = "/path/to/file.h5?path=../group/dataset"

    An `h5py.ExternalLink` is returned for this target

    .. code-block:: python

        target = "/path/to/ext_file.h5?path=/group/dataset"

    An `h5py.VirtualLayout` is returned for these targets

    .. code-block:: python

        target = "/path/to/file.h5?path=/group/dataset&slice=1:5,2:3"
        target = ["/path/to/file1.h5?path=/group/dataset",
                  "/path/to/file2.h5?path=/group/dataset"]
        target = ["/path/to/file1.h5?path=/group/dataset&slice=1:5,2:3",
                  "/path/to/file2.h5?path=/group/dataset&slice=1:5,2:3"]

    An `ExternalBinaryLink` is returned for these targets

    .. code-block:: python

        target = "/path/to/ext_file.edf"
        target = ["/path/to/ext_file1.edf", "/path/to/ext_file1.edf"]
        target = "/path/to/ext_file.tiff"
        target = ["/path/to/ext_file1.tiff", "/path/to/ext_file1.tiff"]

    The target can also be a dictionary of type `VdsSchemaV1` or `ExtSchemaV1`
    for full control of how the links are created. This may be useful when the
    link has several targets that need to be merged, for example concatenated
    along the first dimension or a new dimension.

    :param source: URL of the link.
    :param target: URL or schema of the target.
    :returns: Link instance or `None`.
    """
    if isinstance(
        target,
        (h5py.SoftLink, h5py.ExternalLink, h5py.VirtualLayout, Hdf5Link),
    ):
        # Already a link instance.
        return target

    if not isinstance(source, DataUrl):
        source = DataUrl(source)

    if isinstance(target, Mapping):
        # A mapping could be a link schema or just any mapping.
        return deserialize_mapping(source, target)

    if isinstance(target, (str, DataUrl)):
        # Possibly a URL to a link target.
        return _url_to_hdf5_link(source, target)

    if isinstance(target, Sequence) and all(
        isinstance(v, (str, DataUrl)) for v in target
    ):
        # Possibly URL's to concatenate as a link target.
        return _urls_to_hdf5_link(source, target)

    return None


def _url_to_hdf5_link(source: DataUrl, target: str | DataUrl) -> Hdf5LinkType | None:
    if not isinstance(target, DataUrl):
        if "::" in target or "?" in target:
            # target refers to a data item in a file
            target = DataUrl(target)
        elif _get_file_type(target):
            # target refers to a file
            target = DataUrl(target)
        else:
            # target refers to a dataset
            target = DataUrl(f"{os.path.abspath(source.file_path())}::{target}")

    file_type = _get_target_file_type(source, target)
    if file_type == "hdf5":
        if is_same_file(source, target) and not target.data_slice():
            return _url_to_soft_link(source, target)
        elif target.data_slice():
            return hdf5_url_to_vds(source, target)
        else:
            return _url_to_external_link(target)
    elif file_type == "tiff":
        return tiff_url_to_external_data(source, target)
    elif file_type == "edf":
        return fabio_url_to_external_data(source, target)
    else:
        return None


def _urls_to_hdf5_link(
    source: DataUrl, targets: Sequence[str | DataUrl]
) -> Hdf5LinkType | None:
    targets = cast(
        list[DataUrl],
        [
            target if isinstance(target, DataUrl) else DataUrl(target)
            for target in targets
        ],
    )

    file_types = {_get_target_file_type(source, target) for target in targets}
    if len(file_types) != 1:
        return None
    file_type = list(file_types)[0]

    if file_type == "hdf5":
        return hdf5_urls_to_vds(source, targets)
    elif file_type == "tiff":
        return tiff_urls_to_external_data(source, targets)
    elif file_type == "edf":
        return fabio_urls_to_external_data(source, targets)
    else:
        return None


def _get_target_file_type(source: DataUrl, target: DataUrl) -> str | None:
    if is_same_file(source, target):
        return "hdf5"
    abs_file_path = absolute_file_path(target.file_path(), source.file_path())
    return _get_file_type(abs_file_path)


def _get_file_type(abs_file_path: str) -> str | None:
    if os.path.exists(abs_file_path) and h5py.is_hdf5(abs_file_path):
        return "hdf5"
    ext = os.path.splitext(abs_file_path)[-1].lower()
    if ext in {".nx", ".nxs", ".h5", ".hdf", ".hdf5"}:
        return "hdf5"
    if ext in {".tif", ".tiff"}:
        return "tiff"
    if ext in {".edf"}:
        return "edf"
    return None


def _url_to_soft_link(source: DataUrl, target: DataUrl) -> h5py.SoftLink:
    data_path = target.data_path() or "/"
    if ".." in data_path.split("/"):
        # Up links are not supported in soft links
        data_path = absolute_data_path(data_path, source.data_path() or "/")
    return h5py.SoftLink(data_path)


def _url_to_external_link(target: DataUrl) -> h5py.ExternalLink:
    return h5py.ExternalLink(target.file_path(), target.data_path() or "/")
