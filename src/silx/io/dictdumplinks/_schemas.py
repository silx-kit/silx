from collections.abc import Mapping

import h5py
from pydantic import ValidationError

from ..url import DataUrl
from ._external_binary import ExternalBinaryLink
from ._external_binary import ExternalLinkModelV1
from ._vds import VdsModelV1
from ._vds import VdsUrlsModelV1


def deserialize_mapping(
    source: DataUrl, target: Mapping
) -> h5py.VirtualLayout | ExternalBinaryLink | None:
    """Convert a mapping into a recognized link object.
    Returns `None` if the mapping does not match a known schema.
    """
    for model_cls in [ExternalLinkModelV1, VdsModelV1, VdsUrlsModelV1]:
        try:
            model = model_cls(**target)
        except ValidationError:
            continue
        else:
            return model.tolink(source)

    return None
