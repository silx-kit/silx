from collections.abc import Mapping

import h5py
from pydantic import ValidationError

from ._external_binary import ExternalBinaryLink
from ._external_binary import ExternalLinkModelV1
from ._vds import VdsModelV1
from ._vds import deserialize_vds


def deserialize_mapping(
    target: Mapping,
) -> h5py.VirtualLayout | ExternalBinaryLink | None:
    """Convert a mapping into a recognized link object.
    Returns `None` if the mapping does not match a known schema.
    """
    parsers = [
        (ExternalLinkModelV1, ExternalBinaryLink),
        (VdsModelV1, deserialize_vds),
    ]

    for model_cls, factory in parsers:
        try:
            model = model_cls(**target)
        except ValidationError:
            continue
        else:
            return factory(model)

    return None
