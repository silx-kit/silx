from collections.abc import Sequence
from typing import Any

import fabio
from fabio.edfimage import EdfFrame
from fabio.fabioimage import FabioFrame

from ..url import DataUrl
from ._external_binary import ExternalLinkModelV1
from ._link_types import ExternalBinaryLink
from ._utils import normalize_ext_source_path


def fabio_url_to_external_data(source: DataUrl, target: DataUrl) -> ExternalBinaryLink:
    """Single Fabio file: keep original shape (no new axis)."""
    datasets = [_read_fabio_info(source, target, allow_new_axis=False)]
    target_desc = _build_fabio_schema(datasets)
    return ExternalLinkModelV1(**target_desc).tolink(source)


def fabio_urls_to_external_data(
    source: DataUrl, targets: Sequence[DataUrl]
) -> ExternalBinaryLink:
    """Multiple Fabio files: stack when ndim<3, concatenate when ndim>=3.

    Examples for Nt targets

    - target `shape=()`               : VDS shape `(Nt,)`
    - target `shape=(N0,)`            : VDS shape `(Nt,N0)`
    - target `shape=(N0,N1)`          : VDS shape `(Nt,N0,N1)`
    - target `shape=(N0,N1,N2)`       : VDS shape `(Nt*N0,N1,N2)`
    - target `shape=(N0,N1,N2,N3)`    : VDS shape `(Nt*N0,N1,N2,N3)`
    - target `shape=(N0,N1,N2,N3,N4)` : VDS shape `(Nt*N0,N1,N2,N3,N4)`
    - ...
    """
    datasets = [_read_fabio_info(source, t) for t in targets]
    target_desc = _build_fabio_schema(datasets)
    return ExternalLinkModelV1(**target_desc).tolink(source)


def _read_fabio_info(
    source: DataUrl, target: DataUrl, allow_new_axis: bool = True
) -> dict:
    """Extract dtype, shape, frame count, and external offsets from a Fabio file."""
    if target.data_slice():
        raise ValueError(f"Cannot handle fabio image slicing: {target.path()}")

    file_path = normalize_ext_source_path(target.file_path(), source)

    with fabio.open(file_path) as fabioimage:
        dtype = fabioimage.dtype
        shape = fabioimage.shape
        nframes = fabioimage.nframes
        ndim = len(shape) + (1 if nframes > 1 else 0)

        if allow_new_axis and ndim < 3:
            nimages = 1
            vds_shape = (1,) + ((nframes,) + shape if nframes > 1 else shape)
        else:
            nimages = nframes
            vds_shape = (nframes,) + shape if nframes > 1 else shape

        # Gather all frame offsets
        frame_infos = [
            _fabio_frame_info(file_path, frame) for frame in fabioimage.frames()
        ]

    return dict(
        file_path=file_path,
        dtype=dtype,
        vds_shape=vds_shape,
        ndim=ndim,
        nimages=nimages,
        frame_infos=frame_infos,
    )


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


def _build_fabio_schema(datasets: list[dict]) -> dict:
    """Build the external binary schema from Fabio dataset metadata."""
    target_desc: dict[str, Any] = {
        "dictdump_schema": "external_binary_link_v1",
        "sources": [],
        "shape": None,
        "dtype": None,
    }

    for idx, info in enumerate(datasets):
        if idx == 0:
            target_desc["shape"] = info["vds_shape"]
            target_desc["dtype"] = info["dtype"]
        else:
            if info["ndim"] < 3:
                # stack
                target_desc["shape"] = (target_desc["shape"][0] + 1,) + target_desc[
                    "shape"
                ][1:]
            else:
                # concatenate
                target_desc["shape"] = (
                    target_desc["shape"][0] + info["nimages"],
                ) + target_desc["shape"][1:]

        # Append frame offsets
        target_desc["sources"].extend(
            [
                (info["file_path"], offset, bytecount)
                for offset, bytecount in info["frame_infos"]
            ]
        )

    return target_desc
