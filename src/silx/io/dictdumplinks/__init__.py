"""
Supports representations of these HDF5 link types:

* Internal links: `h5py.SoftLink`
* External links: `h5py.ExternalLink`
* Virtual datasets: `h5py.VirtualLayout`
* External raw binary data: `ExternalBinaryLink`

An instance of these link classes can be created from a serialized
link with `link_from_serialized` or from an HDF5 item with `link_from_hdf5`.

Serialized instances of HDF5 links have the following type

* Internal link: string or `DataUrl`
* External link: string or `DataUrl`
* Virtual dataset: mapping matching the VDS schema
* External raw binary data: mapping matching EXT schema
"""

from ._external_binary import ExternalBinaryLink  # noqa F401
from ._from_hdf5 import link_from_hdf5  # noqa F401
from ._from_serialized import link_from_serialized  # noqa F401
