import os
from typing import NamedTuple, Optional
from .utils import is_dataset


class ExternalDatasetInfo(NamedTuple):
    type: str
    nfiles: int
    first_file_path: str
    first_data_path: Optional[str] = None

    @property
    def first_source_url(self):
        if self.first_data_path:
            if self.first_data_path.startswith("/"):
                return self.first_file_path + "::" + self.first_data_path
            else:
                return self.first_file_path + "::/" + self.first_data_path
        return self.first_file_path


def external_dataset_info(hdf5obj) -> Optional[ExternalDatasetInfo]:
    """When the object is a virtual dataset or an external dataset,
    return information on the external files. Return `None` otherwise.

    Note that this has nothing to do with external HDF5 links."""
    if not is_dataset(hdf5obj):
        return
    if hasattr(hdf5obj, "is_virtual") and hdf5obj.is_virtual:
        sources = hdf5obj.virtual_sources()
        if not sources:
            return ExternalDatasetInfo(
                type="Virtual",
                nfiles=0,
                first_file_path="",
            )

        first_source = sources[0]
        first_file_path = first_source.file_name
        if first_file_path == ".":
            first_file_path = hdf5obj.file.filename
        elif not os.path.isabs(first_file_path):
            dirname = os.path.dirname(hdf5obj.file.filename)
            first_file_path = os.path.normpath(
                os.path.join(
                    dirname,
                    first_file_path,
                )
            )

        return ExternalDatasetInfo(
            type="Virtual",
            nfiles=len(sources),
            first_file_path=first_file_path,
            first_data_path=first_source.dset_name,
        )
    if hasattr(hdf5obj, "external"):
        sources = hdf5obj.external
        if not sources:
            return

        first_source = sources[0]
        first_file_path = first_source[0]
        if not os.path.isabs(first_file_path):
            dirname = os.path.dirname(hdf5obj.file.filename)
            first_file_path = os.path.normpath(
                os.path.join(
                    dirname,
                    first_file_path,
                )
            )

        return ExternalDatasetInfo(
            type="Raw",
            nfiles=len(sources),
            first_file_path=first_file_path,
        )
