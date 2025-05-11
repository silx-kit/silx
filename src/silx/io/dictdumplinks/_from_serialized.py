import os
from typing import TypeAlias, Any, cast
from collections.abc import Mapping, Sequence

import h5py
import fabio
from fabio.TiffIO import TiffIO
from fabio.edfimage import EdfFrame
from fabio.fabioimage import FabioFrame

from ..url import DataUrl

from ._vds_types import VdsSchemaV1
from ._ext_types import ExtSchemaV1
from ._link_types import Hdf5LinkType
from ._link_types import InternalLink
from ._link_types import ExternalLink
from ._link_types import VDSLink
from ._link_types import ExternalBinaryLink
from ._schemas import parse_schema


SerializedHdf5LinkType: TypeAlias = (
    str | DataUrl | Sequence[str | DataUrl] | VdsSchemaV1 | ExtSchemaV1
)

NativeHdf5LinkType: TypeAlias = h5py.SoftLink | h5py.ExternalLink


def link_from_serialized(
    source: str | DataUrl,
    target: Hdf5LinkType | NativeHdf5LinkType | SerializedHdf5LinkType | Any,
) -> Hdf5LinkType | None:
    """Convert the target to a link instance when it describes a link.
    Otherwise return `None`.

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
        return parse_schema(target)
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
) -> Hdf5LinkType | None:
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
    source: str | DataUrl, target: Sequence[str | DataUrl]
) -> Hdf5LinkType | None:
    # TODO: stack of targets
    raise NotImplementedError


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
    if _is_same_file(source, target):
        file_path = "."
    else:
        file_path = target.file_path()

    data_path = target.data_path() or "/"
    data_slice = target.data_slice()

    if ".." in data_path.split("/"):
        if file_path == ".":
            # Up links are not supported in internal virtual datasets
            data_path = _absolute_data_path(data_path, source.data_path() or "/")
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
    target_desc = {
        "dictdump_schema": "virtual_dataset_v1",
        "shape": data_shape,
        "dtype": source_dtype,
        "sources": [vsource],
    }
    return cast(VDSLink, parse_schema(target_desc))


def _fabio_url_to_external_data(source: DataUrl, target: DataUrl) -> ExternalBinaryLink:
    if target.data_slice():
        raise ValueError(f"Cannot handle fabio image slicing: {target.path()}")

    file_path = target.file_path()
    abs_file_path = _absolute_file_path(file_path, source.file_path())

    with fabio.open(abs_file_path) as fabioimage:
        sources: list[tuple[str, int, int]] = list()

        if fabioimage.nframes > 1:
            shape = (fabioimage.nframes,) + fabioimage.shape
        else:
            shape = fabioimage.shape
        target_desc = {
            "dictdump_schema": "external_binary_link_v1",
            "shape": shape,
            "dtype": fabioimage.dtype,
            "sources": sources,
        }
        for frame in fabioimage.frames():
            offset, bytecount = _fabio_frame_info(file_path, frame)
            sources.append((file_path, offset, bytecount))

    return cast(ExternalBinaryLink, parse_schema(target_desc))


def _tiff_url_to_external_data(source: DataUrl, target: DataUrl) -> ExternalBinaryLink:
    if target.data_slice():
        raise ValueError(f"Cannot handle fabio image slicing: {target.path()}")

    file_path = target.file_path()
    abs_file_path = _absolute_file_path(file_path, source.file_path())

    with TiffIO(abs_file_path) as tiffimage:
        sources: list[tuple[str, int, int]] = list()
        nframes = tiffimage.getNumberOfImages()
        img = tiffimage.getImage(0)
        if nframes > 1:
            shape = (nframes,) + img.shape
        else:
            shape = img.shape

        target_desc = {
            "dictdump_schema": "external_binary_link_v1",
            "shape": shape,
            "dtype": img.dtype,
            "sources": sources,
        }
        for i in range(nframes):
            info = tiffimage.getInfo(i)
            if info["compression"]:
                raise RuntimeError(
                    f"{file_path!r} (frame {i}): external datasets do not support compression"
                )

            # shape = info["nRows"], info["nColumns"]
            for offset, bytecount in zip(info["stripOffsets"], info["stripByteCounts"]):
                sources.append((file_path, offset, bytecount))

    return cast(ExternalBinaryLink, parse_schema(target_desc))


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
