import os

import h5py

from ._link_types import Hdf5LinkType
from ._link_types import InternalLink
from ._link_types import ExternalLink
from ._link_types import ExternalVirtualLink
from ._schemas import parse_schema


def link_from_hdf5(parent: h5py.Group, name: str) -> Hdf5LinkType | None:
    """
    :param parent: HDF5 parent group.
    :param name: HDF5 name of the child.
    :return: Link instance or `None` when `parent[name]` is raises a `KeyError`
             or is not a link.
    """
    link = parent.get(name, getlink=True)
    if isinstance(link, h5py.SoftLink):
        return InternalLink(link.path)
    if isinstance(link, h5py.ExternalLink):
        return ExternalLink(link.filename, link.path)

    try:
        item = parent[name]
    except KeyError:
        return

    if not isinstance(item, h5py.Dataset):
        return

    if item.is_virtual:
        return _vdsmaps_to_vds(item)

    # TODO: no h5py API to get external sources


def _vdsmaps_to_vds(dataset: h5py.Dataset) -> ExternalVirtualLink:
    sources = list()
    rootdir = os.path.dirname(dataset.file.filename)
    for vdsmap in dataset.virtual_sources():
        file_path = vdsmap.file_name
        if not os.path.isabs(file_path):
            file_path = os.path.abspath(os.path.join(rootdir, file_path))

        source_dtype = dataset.dtype  # TODO

        # TODO:
        source_index = _get_index(vdsmap.src_space)
        target_index = _get_index(vdsmap.vspace)

        vsource = {
            "file_path": vdsmap.file_name,
            "data_path": vdsmap.dset_name,
            "shape": vdsmap.src_space.shape,  # TODO: not correct
            "dtype": source_dtype,
            "source_index": source_index,
            "target_index": target_index,  # shape is vdsmap.vspace.shape
        }
        sources.append(vsource)
    target_desc = {
        "dictdump_schema": "external_virtual_link_v1",
        "shape": dataset.shape,
        "dtype": dataset.dtype,
        "sources": sources,
    }
    return parse_schema(target_desc)


def _get_index(space: h5py.h5s.SpaceID) -> tuple[slice]:
    return tuple()
