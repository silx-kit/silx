import os
from typing import TypeAlias, Any, cast, Dict
from collections.abc import Mapping, Sequence

import h5py
import fabio
from fabio.TiffIO import TiffIO
from fabio.edfimage import EdfFrame
from fabio.fabioimage import FabioFrame
from numpy.typing import DTypeLike

from ..url import DataUrl

from ._base_types import LinkInterface
from ._base_types import NativeHdf5LinkType
from ._internal_link_types import InternalLink
from ._external_link_types import ExternalLink
from ._vds_types import VDSLink
from ._vds_types import VdsSchemaV1
from ._external_binary_types import ExternalBinaryLink
from ._external_binary_types import ExtSchemaV1
from ._schemas import deserialize_schema


SerializedHdf5LinkType: TypeAlias = (
    str | DataUrl | Sequence[str | DataUrl] | VdsSchemaV1 | ExtSchemaV1
)


def link_from_serialized(
    source: str | DataUrl,
    target: LinkInterface | NativeHdf5LinkType | SerializedHdf5LinkType | Any,
) -> LinkInterface | None:
    """Convert the target to a link instance when it describes a link.
    Otherwise return `None`.

    The target can be a single URL or a list of URL's. These are URL's to
    HDF5 datasets, EDF files or TIFF files.

    For example the source is this

    .. code-block:: python

        source = "/path/to/file.h5?path=/group/link"

    An `InternalLink` is returned for these targets

    .. code-block:: python

        target = "/path/to/file.h5?path=/group/dataset"
        target = "/path/to/file.h5?path=dataset"
        target = "/path/to/file.h5?path=../group/dataset"

    An `ExternalLink` is returned for this target

    .. code-block:: python

        target = "/path/to/ext_file.h5?path=/group/dataset"

    A `VDSLink` is returned for these targets

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
        (InternalLink, ExternalLink, VDSLink, ExternalBinaryLink),
    ):
        # Already a link instance.
        return target

    if isinstance(target, h5py.SoftLink):
        # A native h5py link instance.
        return InternalLink(target.path)
    if isinstance(target, h5py.ExternalLink):
        # A native h5py link instance.
        return ExternalLink(target.filename, target.path)
    if isinstance(target, Mapping):
        # A mapping could be a link schema or just any mapping.
        return deserialize_schema(target)
    if isinstance(target, (str, DataUrl)):
        # Possibly a URL to a link target.
        return _url_to_hdf5_link(source, target)
    if isinstance(target, Sequence) and all(
        isinstance(v, (str, DataUrl)) for v in target
    ):
        # Possibly URL's to concatenate as a link target.
        return _urls_to_hdf5_link(source, target)

    return None


def _url_to_hdf5_link(
    source: str | DataUrl, target: str | DataUrl
) -> LinkInterface | None:
    if not isinstance(source, DataUrl):
        source = DataUrl(source)

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
        if _is_same_file(source, target) and not target.data_slice():
            return _url_to_soft_link(source, target)
        elif target.data_slice():
            return _url_to_vds(source, target)
        else:
            return _url_to_external_link(source, target)
    elif file_type == "tiff":
        return _tiff_url_to_external_data(source, target)
    elif file_type == "edf":
        return _fabio_url_to_external_data(source, target)
    else:
        return None


def _urls_to_hdf5_link(
    source: str | DataUrl, targets: Sequence[str | DataUrl]
) -> LinkInterface | None:
    if not isinstance(source, DataUrl):
        source = DataUrl(source)
    targets: list[DataUrl] = [
        target if isinstance(target, DataUrl) else DataUrl(target) for target in targets
    ]

    file_types = {_get_target_file_type(source, target) for target in targets}
    if len(file_types) != 1:
        return None
    file_type = list(file_types)[0]

    if file_type == "hdf5":
        return _urls_to_vds(source, targets)
    elif file_type == "tiff":
        return _tiff_urls_to_external_data(source, targets)
    elif file_type == "edf":
        return _fabio_urls_to_external_data(source, targets)
    else:
        return None


def _is_same_file(source: DataUrl, target: DataUrl) -> bool:
    source_file_path = source.file_path()
    target_file_path = target.file_path()

    if target_file_path == "." or source_file_path == target_file_path:
        return True

    if not os.path.isabs(target_file_path):
        target_file_path = os.path.join(
            os.path.dirname(source_file_path), target_file_path
        )

    return os.path.realpath(source_file_path) == os.path.realpath(target_file_path)


def _get_target_file_type(source: DataUrl, target: DataUrl) -> str | None:
    if _is_same_file(source, target):
        return "hdf5"
    abs_file_path = _absolute_file_path(target.file_path(), source.file_path())
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


def _url_to_soft_link(source: DataUrl, target: DataUrl) -> InternalLink:
    data_path = target.data_path() or "/"
    if ".." in data_path.split("/"):
        # Up links are not supported in soft links
        data_path = _absolute_data_path(data_path, source.data_path() or "/")
    return InternalLink(data_path)


def _url_to_external_link(source: DataUrl, target: DataUrl) -> ExternalLink:
    return ExternalLink(target.file_path(), target.data_path() or "/")


def _url_to_vds(source: DataUrl, target: DataUrl) -> VDSLink:
    target_desc: Dict[str, Any] = dict()
    _ = _add_url_to_vds_schema(source, target, target_desc)
    return cast(VDSLink, deserialize_schema(target_desc))


def _urls_to_vds(source: DataUrl, targets: Sequence[DataUrl]) -> VDSLink:
    target_desc: Dict[str, Any] = dict()
    nimages = [
        _add_url_to_vds_schema(source, target, target_desc) for target in targets
    ]
    i0 = 0
    for source, n in zip(target_desc["sources"], nimages):
        source["target_index"] = slice(i0, i0 + n)
        i0 += n
    return cast(VDSLink, deserialize_schema(target_desc))


def _add_url_to_vds_schema(source: DataUrl, target: DataUrl, target_desc: dict) -> int:
    assert source.data_path()
    assert target.data_path()

    if _is_same_file(source, target):
        file_path = "."
    else:
        file_path = target.file_path()

    data_path = target.data_path()
    data_slice = target.data_slice()

    if file_path == "." and not data_path.startswith("/"):
        data_path = _absolute_data_path(data_path, source.data_path())

    if ".." in data_path.split("/"):
        if file_path == ".":
            # Up links are not supported in internal virtual datasets
            data_path = _absolute_data_path(data_path, source.data_path())
        else:
            raise ValueError(
                f"VDS target data path in a different file cannot be relative ({data_path})"
            )

    abs_file_path = _absolute_file_path(target.file_path(), source.file_path())
    source_dtype, source_shape, data_shape = _get_hdf5_dataset_info(
        abs_file_path, data_path, data_slice
    )
    vsource = {
        "file_path": file_path,
        "data_path": data_path,
        "shape": source_shape,
        "dtype": source_dtype,
        "source_index": data_slice,
        "target_index": tuple(),
    }

    if len(data_shape) < 3:
        nextra = 3 - len(data_shape) - 1
        nimages = 1
        image_shape = (1,) * nextra + data_shape
    else:
        nimages = data_shape[0]
        image_shape = data_shape[1:]

    if target_desc:
        target_desc["sources"].append(vsource)
        target_desc["shape"] = _concatenate_items(
            target_desc["shape"],
            target_desc["dtype"],
            nimages,
            image_shape,
            source_dtype,
        )
    else:
        target_desc["dictdump_schema"] = "virtual_dataset_v1"
        target_desc["shape"] = data_shape
        target_desc["dtype"] = source_dtype
        target_desc["sources"] = [vsource]

    return nimages


def _fabio_url_to_external_data(source: DataUrl, target: DataUrl) -> ExternalBinaryLink:
    target_desc: Dict[str, Any] = dict()
    _add_fabio_url_to_schema(source, target, target_desc)
    return cast(ExternalBinaryLink, deserialize_schema(target_desc))


def _fabio_urls_to_external_data(
    source: DataUrl, targets: Sequence[DataUrl]
) -> ExternalBinaryLink:
    target_desc: Dict[str, Any] = dict()
    for target in targets:
        _add_fabio_url_to_schema(source, target, target_desc)
    return cast(ExternalBinaryLink, deserialize_schema(target_desc))


def _add_fabio_url_to_schema(
    source: DataUrl, target: DataUrl, target_desc: dict
) -> None:
    """
    Updates target_desc with external binary offsets from a Fabio file.

    :param source: The original data URL providing base path context.
    :param target: The target Fabio file to process.
    :param target_desc: A mutable schema dictionary to update in-place.
    """
    if target.data_slice():
        raise ValueError(f"Cannot handle fabio image slicing: {target.path()}")

    file_path = target.file_path()
    abs_file_path = _absolute_file_path(file_path, source.file_path())

    with fabio.open(abs_file_path) as fabioimage:
        if target_desc:
            target_desc["shape"] = _concatenate_items(
                target_desc["shape"],
                target_desc["dtype"],
                fabioimage.nframes,
                fabioimage.shape,
                fabioimage.dtype,
            )
            sources: list[tuple[str, int, int]] = target_desc["sources"]
        else:
            sources: list[tuple[str, int, int]] = list()
            if fabioimage.nframes > 1:
                shape = (fabioimage.nframes,) + fabioimage.shape
            else:
                shape = fabioimage.shape
            target_desc["dictdump_schema"] = "external_binary_link_v1"
            target_desc["shape"] = shape
            target_desc["dtype"] = fabioimage.dtype
            target_desc["sources"] = sources

        for frame in fabioimage.frames():
            offset, bytecount = _fabio_frame_info(file_path, frame)
            sources.append((file_path, offset, bytecount))


def _tiff_url_to_external_data(source: DataUrl, target: DataUrl) -> ExternalBinaryLink:
    target_desc: Dict[str, Any] = dict()
    _add_tiff_url_to_schema(source, target, target_desc)
    return cast(ExternalBinaryLink, deserialize_schema(target_desc))


def _tiff_urls_to_external_data(
    source: DataUrl, targets: Sequence[DataUrl]
) -> ExternalBinaryLink:
    target_desc = {}
    for target in targets:
        _add_tiff_url_to_schema(source, target, target_desc)
    return cast(ExternalBinaryLink, deserialize_schema(target_desc))


def _add_tiff_url_to_schema(
    source: DataUrl, target: DataUrl, target_desc: dict
) -> None:
    """
    Updates target_desc with external binary offsets from a TIFF file.

    :param source: The original data URL providing base path context.
    :param target: The target TIFF file to process.
    :param target_desc: A mutable schema dictionary to update in-place.
    """
    if target.data_slice():
        raise ValueError(f"Cannot handle fabio image slicing: {target.path()}")

    file_path = target.file_path()
    abs_file_path = _absolute_file_path(file_path, source.file_path())

    with TiffIO(abs_file_path) as tiffimage:
        nimages = tiffimage.getNumberOfImages()
        image = tiffimage.getImage(0)

        if target_desc:
            target_desc["shape"] = _concatenate_items(
                target_desc["shape"],
                target_desc["dtype"],
                nimages,
                image.shape,
                image.dtype,
            )
            sources: list[tuple[str, int, int]] = target_desc["sources"]
        else:
            sources: list[tuple[str, int, int]] = list()
            if nimages > 1:
                shape = (nimages,) + image.shape
            else:
                shape = image.shape
            target_desc["dictdump_schema"] = "external_binary_link_v1"
            target_desc["shape"] = shape
            target_desc["dtype"] = image.dtype
            target_desc["sources"] = sources

        for i in range(nimages):
            info = tiffimage.getInfo(i)
            if info["compression"]:
                raise RuntimeError(
                    f"{file_path!r} (frame {i}): external datasets do not support compression"
                )

            # shape = info["nRows"], info["nColumns"]
            for offset, bytecount in zip(info["stripOffsets"], info["stripByteCounts"]):
                sources.append((file_path, offset, bytecount))


def _concatenate_items(
    concat_shape: tuple[int, ...],
    dtype: DTypeLike | str,
    nitems: int,
    item_shape: tuple[int, ...],
    item_dtype: DTypeLike | str,
) -> tuple[int, ...]:
    if len(concat_shape) == len(item_shape):
        concat_shape = (1, *concat_shape)
    current_nitems, *current_item_shape = concat_shape
    current_item_shape = tuple(current_item_shape)
    if item_shape != current_item_shape:
        raise ValueError("Cannot concatenate data with different shapes")
    if item_dtype != dtype:
        raise ValueError("Cannot concatenate data with different dtype")
    return (current_nitems + nitems, *current_item_shape)


def _fabio_frame_info(file_path: str, frame: FabioFrame) -> tuple[int, int]:
    if isinstance(frame, EdfFrame):
        return _edf_frame_info(file_path, frame)
    raise NotImplementedError(f"Fabio {type(frame).__name__} is not supported")


def _edf_frame_info(file_path: str, frame: EdfFrame) -> tuple[int, int]:
    if frame.swap_needed():
        raise RuntimeError(
            "{} (frame {}): external datasets do not support byte-swap".format(
                repr(file_path), frame.iFrame
            )
        )
    compression = frame._data_compression
    if compression:
        compression = compression.lower()
    if compression == "none":
        compression = None
    if compression is not None:
        raise RuntimeError(
            "{} (frame {}): external datasets with compression not supported".format(
                repr(file_path), frame.iFrame
            )
        )
    offset = frame.start
    bytecount = frame.size  # uncompressed size
    return offset, bytecount


def _absolute_file_path(file_path: str, start_file_path: str) -> str:
    if os.path.isabs(file_path):
        return file_path
    root = os.path.dirname(start_file_path)
    return os.path.abspath(os.path.join(root, file_path))


def _absolute_data_path(data_path: str, start_data_path: str) -> str:
    inparts = [s for s in start_data_path.split("/") if s][:-1]
    inparts += data_path.split("/")
    outparts: list[str] = []
    for part in inparts:
        if part == "." or not part:
            pass
        elif part == "..":
            outparts = outparts[:-1]
        else:
            outparts.append(part)
    return "/".join([""] + outparts)


def _get_hdf5_dataset_info(
    file_path: str, data_path: str, data_slice
) -> tuple[str, tuple[int, ...], tuple[int, ...]]:
    with h5py.File(file_path, locking=False, mode="r") as f:
        dset = f[data_path]
        source_shape = dset.shape
        if data_slice:
            data_shape = dset[data_slice].shape
        else:
            data_shape = tuple()
        return dset.dtype, source_shape, data_shape
