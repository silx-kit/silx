"""Provides a wrapper to expose `Tiled <https://blueskyproject.io/tiled/>`_"""
from __future__ import annotations


from functools import lru_cache
import logging
import numpy
from . import commonh5
import h5py
import tiled.client


_logger = logging.getLogger(__name__)


def _get_children(parent, container):
    children = {}
    for key, client in container.items():
        if isinstance(client, tiled.client.container.Container):
            children[key] = TiledGroup(client, name=key, parent=parent)
        elif isinstance(client, tiled.client.array.ArrayClient):
            children[key] = TiledDataset(client, name=key, parent=parent)
        else:
            _logger.warning(f"Unsupported child type: {key}: {client}")
            children[key] = commonh5.Dataset(
                key,
                numpy.array("Unsupported", h5py.special_dtype(vlen=str)),
                parent=parent,
            )
    return children


class TiledH5(commonh5.File):
    def __init__(self, name=None, mode=None, attrs=None):
        assert mode in ("r", None)
        super().__init__(name, mode, attrs)
        self.__container = tiled.client.from_uri(
            name[6:] if name.startswith("tiled:") else name
        )
        assert isinstance(self.__container, tiled.client.container.Container)

    def close(self):
        super().close()
        self.__container = None

    @lru_cache
    def _get_items(self):
        return _get_children(self, self.__container)


class TiledGroup(commonh5.Group):
    """tiled Container wrapper"""

    def __init__(self, container, name, parent=None, attrs=None):
        super().__init__(name, parent, attrs)
        self.__container = container

    @lru_cache
    def _get_items(self):
        return _get_children(self, self.__container)


class TiledDataset(commonh5.LazyLoadableDataset):
    """tiled ArrayClient wrapper"""

    def __init__(self, client, name, parent=None, attrs=None):
        super().__init__(name, parent, attrs)
        self.__client = client

    def _create_data(self) -> numpy.ndarray:
        return self.__client[()]

    @property
    def dtype(self):
        return self.__client.dtype

    @property
    def shape(self):
        return self.__client.shape

    @property
    def size(self):
        return self.__client.size

    def __len__(self):
        return len(self.__client)

    def __getitem__(self, item):
        return self.__client[item]

    @property
    def value(self):
        return self.__client[()]

    def __iter__(self):
        return self.__client.__iter__()

    def __getattr__(self, item):
        return getattr(self.__client, item)
