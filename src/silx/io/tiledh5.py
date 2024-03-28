"""Provides a wrapper to expose `Tiled <https://blueskyproject.io/tiled/>`_"""
from __future__ import annotations


from functools import lru_cache
import numpy
from . import commonh5
import tiled.client


def _getChildren(parent, container):
    children = {}
    for key, client in container.items():
        if isinstance(client, tiled.client.container.Container):
            children[key] = TiledGroup(client, name=key, parent=parent)
        elif isinstance(client, tiled.client.array.ArrayClient):
            children[key] = TiledDataset(client, name=key, parent=parent)
    return children


class TiledH5(commonh5.File):
    def __init__(self, name=None, mode=None, attrs=None):
        assert mode in ("r", None)
        super().__init__(name, mode, attrs)
        self.__container = tiled.client.from_uri(name)
        assert isinstance(self.__container, tiled.client.container.Container)

    def close(self):
        super().close()
        self.__container = None

    @lru_cache
    def _get_items(self):
        return _getChildren(self, self.__container)


class TiledGroup(commonh5.Group):
    """tiled Container wrapper"""

    def __init__(self, container, name, parent=None, attrs=None):
        super().__init__(name, parent, attrs)
        self.__container = container

    @lru_cache
    def _get_items(self):
        return _getChildren(self, self.__container)


class TiledDataset(commonh5.LazyLoadableDataset):
    """tiled ArrayClient wrapper"""

    def __init__(self, client, name, parent=None, attrs=None):
        super().__init__(name, parent, attrs)
        self.__client = client

    def _create_data(self) -> numpy.ndarray:
        return self.__client[()]
