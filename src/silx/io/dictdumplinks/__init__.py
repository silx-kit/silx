"""
Supports representations of these HDF5 link types:

* Soft links: `InternalLink`
* External links: `ExternalLink`
* Virtual datasets: `VDSLink`
* External raw binary data: `ExternalBinaryLink`

An instance of these link classes can be created from a serialized
link with `link_from_serialized` or from an HDF5 item with `link_from_hdf5`.

Serialized instances of HDF5 links have the following type

* Soft link: string or `DataUrl`
* External link: string or `DataUrl`
* Virtual dataset: mapping matching the VDS schema
* External raw binary data: mapping matching EXT schema
"""

from ._base_types import LinkInterface  # noqa F401
from ._internal_link_types import InternalLink  # noqa F401
from ._external_link_types import ExternalLink  # noqa F401
from ._vds_types import VdsSource  # noqa F401
from ._vds_types import VDSLink  # noqa F401
from ._vds_types import VdsSchemaV1 as VdsSchema  # noqa F401
from ._external_binary_types import ExternalBinaryLink  # noqa F401
from ._external_binary_types import ExtSchemaV1 as ExtSchema  # noqa F401
from ._from_hdf5 import link_from_hdf5  # noqa F401
from ._from_serialized import link_from_serialized  # noqa F401
