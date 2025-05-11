from typing import cast
from collections.abc import Mapping

from ._vds_types import parse_vds_schema_v1
from ._ext_types import parse_ext_schema_v1
from ._vds_types import VdsSchemaV1
from ._ext_types import ExtSchemaV1
from ._link_types import VDSLink
from ._link_types import ExternalBinaryLink


def parse_schema(target: Mapping) -> VDSLink | ExternalBinaryLink | None:
    """Convert a mapping into a link when it describes a link.
    Otherwise return `None`.
    """
    dictdump_schema = target.get("dictdump_schema")
    if dictdump_schema == "external_binary_link_v1":
        ext_model = parse_ext_schema_v1(cast(ExtSchemaV1, target))
        return ExternalBinaryLink(ext_model.shape, ext_model.dtype, ext_model.sources)
    if dictdump_schema == "virtual_dataset_v1":
        vds_model = parse_vds_schema_v1(cast(VdsSchemaV1, target))
        return VDSLink(vds_model.shape, vds_model.dtype, vds_model.sources)
    return None
