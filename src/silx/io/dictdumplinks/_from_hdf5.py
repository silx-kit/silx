import h5py

from ._link_types import InternalLink
from ._link_types import ExternalLink


def link_from_hdf5(parent: h5py.Group, name: str) -> InternalLink | ExternalLink | None:
    """
    External binary datasets and virtual datasets are not supported
    as h5py does not provide an API for it.

    :param parent: HDF5 parent group.
    :param name: HDF5 name of the child.
    :return: Link instance or `None` when not a link.
    """
    link = parent.get(name, getlink=True)
    if isinstance(link, h5py.SoftLink):
        return InternalLink(link.path)
    if isinstance(link, h5py.ExternalLink):
        return ExternalLink(link.filename, link.path)
    return None
