import os

from ..url import DataUrl


def absolute_file_path(file_path: str, start_file_path: str) -> str:
    if os.path.isabs(file_path):
        return file_path
    root = os.path.dirname(start_file_path)
    return os.path.abspath(os.path.join(root, file_path))


def absolute_data_path(data_path: str, start_data_path: str) -> str:
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


def normalize_vds_source_url(
    file_path: str, data_path: str, source: DataUrl
) -> tuple[str, str]:
    """Normalize URL for saving in HDF5."""
    if file_path == "." and not data_path.startswith("/"):
        data_path = absolute_data_path(data_path, source.data_path())

    if ".." in data_path.split("/"):
        if file_path == ".":
            # Up links are not supported in internal virtual datasets
            data_path = absolute_data_path(data_path, source.data_path())
        else:
            raise ValueError(
                f"VDS source data path in a different file cannot be relative ({data_path})"
            )

    return file_path, data_path


def normalize_ext_source_path(file_path: str, source: DataUrl) -> str:
    """Normalize URL for saving in HDF5."""
    return absolute_file_path(file_path, source.file_path())
