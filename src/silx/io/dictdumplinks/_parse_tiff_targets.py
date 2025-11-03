from collections.abc import Sequence
from typing import Any

from fabio.TiffIO import TiffIO

from ..url import DataUrl
from ._external_binary import ExternalBinaryLink
from ._external_binary import ExternalLinkModelV1
from ._utils import normalize_ext_source_path


def tiff_url_to_external_data(source: DataUrl, target: DataUrl) -> ExternalBinaryLink:
    """Single TIFF file: keep original shape (no new axis)."""
    datasets = [_read_tiff_info(source, target, allow_new_axis=False)]
    target_desc = _build_tiff_schema(datasets)
    return ExternalLinkModelV1(**target_desc).tolink(source)


def tiff_urls_to_external_data(
    source: DataUrl, targets: Sequence[DataUrl]
) -> ExternalBinaryLink:
    """Multiple TIFF files: stack when ndim<3, concatenate when ndim>=3.

    Examples for Nt targets

    - target `shape=()`               : VDS shape `(Nt,)`
    - target `shape=(N0,)`            : VDS shape `(Nt,N0)`
    - target `shape=(N0,N1)`          : VDS shape `(Nt,N0,N1)`
    - target `shape=(N0,N1,N2)`       : VDS shape `(Nt*N0,N1,N2)`
    - target `shape=(N0,N1,N2,N3)`    : VDS shape `(Nt*N0,N1,N2,N3)`
    - target `shape=(N0,N1,N2,N3,N4)` : VDS shape `(Nt*N0,N1,N2,N3,N4)`
    - ...
    """
    datasets = [_read_tiff_info(source, t) for t in targets]
    target_desc = _build_tiff_schema(datasets)
    return ExternalLinkModelV1(**target_desc).tolink(source)


def _read_tiff_info(
    source: DataUrl, target: DataUrl, allow_new_axis: bool = True
) -> dict:
    """Collect TIFF metadata and offsets."""
    if target.data_slice():
        raise ValueError(f"Cannot handle tiff image slicing: {target.path()}")

    file_path = normalize_ext_source_path(target.file_path(), source)

    with TiffIO(file_path) as tiffimage:
        nimages = tiffimage.getNumberOfImages()
        image = tiffimage.getImage(0)  # representative frame
        dtype = image.dtype
        shape = image.shape

        ndim = len(shape) + (1 if nimages > 1 else 0)

        if allow_new_axis and ndim < 3:
            nframes = 1
            vds_shape = (1,) + ((nimages,) + shape if nimages > 1 else shape)
        else:
            nframes = nimages
            vds_shape = (nimages,) + shape if nimages > 1 else shape

        # Collect all strips from all frames
        frame_infos: list[tuple[int, int]] = []
        for i in range(nimages):
            info = tiffimage.getInfo(i)
            if info["compression"]:
                raise RuntimeError(
                    f"{file_path!r} (frame {i}): external datasets do not support compression"
                )
            for offset, bytecount in zip(info["stripOffsets"], info["stripByteCounts"]):
                frame_infos.append((offset, bytecount))

    return dict(
        file_path=file_path,
        dtype=dtype,
        vds_shape=vds_shape,
        ndim=ndim,
        nimages=nframes,
        frame_infos=frame_infos,
    )


def _build_tiff_schema(datasets: list[dict]) -> dict:
    """Build the external binary schema for TIFF data."""
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

        # Add all offsets for this TIFF
        target_desc["sources"].extend(
            [
                {"file_path": info["file_path"], "offset": offset, "size": size}
                for offset, size in info["frame_infos"]
            ]
        )

    return target_desc
