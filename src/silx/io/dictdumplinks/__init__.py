"""
Supports representations of these HDF5 link types:

* Soft links: `InternalLink`
* External links: `ExternalLink`
* Virtual datasets: `VDSLink`
* External raw binary data: `ExternalBinaryLink`

An instance of these link classes can be created from a serialized
link with `link_from_serialized` or from an HDF5 item with `link_from_hdf5`.

Serialized instances of HDF5 links have the following type

* Soft links: string or `DataUrl`
* External links: string or `DataUrl`
* Virtual datasets: mapping matching the VDS schemas
* External raw binary data: mapping matching EXT schemas
"""

from ._from_hdf5 import link_from_hdf5  # noqa F401
from ._from_serialized import link_from_serialized  # noqa F401
