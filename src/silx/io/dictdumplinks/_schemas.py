from collections.abc import Mapping

from ._vds_types import parse_vds_schema_v1
from ._ext_types import parse_ext_schema_v1
from ._link_types import ExternalVirtualLink
from ._link_types import ExternalBinaryLink


def parse_schema(target: Mapping) -> ExternalVirtualLink | ExternalBinaryLink | None:
    dictdump_schema = target.get("dictdump_schema")
    if dictdump_schema == "external_binary_link_v1":
        model = parse_ext_schema_v1(target)
        return ExternalBinaryLink(model.shape, model.dtype, model.sources)
    if dictdump_schema == "external_virtual_link_v1":
        model = parse_vds_schema_v1(target)
        return ExternalVirtualLink(model.shape, model.dtype, model.sources)
