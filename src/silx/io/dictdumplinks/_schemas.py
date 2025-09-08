from typing import cast
from collections.abc import Mapping

import h5py

from ._vds_types import VdsSchemaV1
from ._vds_types import deserialize_vds_schema_v1

from ._external_binary_types import ExtSchemaV1
from ._external_binary_types import ExternalBinaryLink
from ._external_binary_types import deserialize_ext_schema_v1


def deserialize_schema(
    target: Mapping,
) -> h5py.VirtualLayout | ExternalBinaryLink | None:
    """Convert a mapping into a link when it describes a link.
    Otherwise return `None`.
    """
    dictdump_schema = target.get("dictdump_schema")
    if dictdump_schema == "external_binary_link_v1":
        ext_model = deserialize_ext_schema_v1(cast(ExtSchemaV1, target))
        return ExternalBinaryLink(ext_model.shape, ext_model.dtype, ext_model.sources)
    if dictdump_schema == "virtual_dataset_v1":
        vds_layout = deserialize_vds_schema_v1(cast(VdsSchemaV1, target))
        return vds_layout
    return None
