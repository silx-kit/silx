# /*##########################################################################
# Copyright (C) 2024 European Synchrotron Radiation Facility
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ############################################################################*/
"""
Provides a wrapper to expose `Tiled <https://blueskyproject.io/tiled/>`_

This is a preview feature.
"""
from __future__ import annotations


from functools import lru_cache
import logging
import numpy
from . import commonh5
import h5py
import tiled.client


_logger = logging.getLogger(__name__)


def _get_children(
    parent: TiledH5 | TiledGroup,
    container: tiled.client.container.Container,
    max_children: int | None = None,
):
    """Return first max_children entries of given container as a dict of commonh5 wrappers.

    :param parent: The commonh5 wrapper for which to retrieve children.
    :param container: The tiled container from which to retrieve the entries.
    :param max_children: The maximum number of children to retrieve.
    """
    items = container.items()

    if max_children is not None and len(items) > max_children:
        items = items.head(max_children)
        _logger.warning(
            f"{container.uri} contains too many entries: Only loading first {max_children}."
        )

    children = {}
    for key, client in items:
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
    """tiled client wrapper"""

    MAX_CHILDREN_PER_GROUP: int | None = None
    """Maximum number of children to instantiate for each group.

    Set to None for allowing an unbound number of children per group.
    """

    def __init__(
        self,
        name: str,
        mode: str | None = None,
        attrs: dict | None = None,
    ):
        assert mode in ("r", None)
        super().__init__(name, mode, attrs)
        self.__container = tiled.client.from_uri(name)
        assert isinstance(self.__container, tiled.client.container.Container)

    def close(self):
        super().close()
        self.__container = None

    @lru_cache
    def _get_items(self):
        return _get_children(self, self.__container, self.MAX_CHILDREN_PER_GROUP)


class TiledGroup(commonh5.Group):
    """tiled Container wrapper"""

    def __init__(
        self,
        container: tiled.client.container.Container,
        name: str,
        parent: TiledH5 | TiledGroup | None = None,
        attrs: dict | None = None,
    ):
        super().__init__(name, parent, attrs)
        self.__container = container

    @lru_cache
    def _get_items(self):
        return _get_children(self, self.__container, self.file.MAX_CHILDREN_PER_GROUP)


class TiledDataset(commonh5.Dataset):
    """tiled ArrayClient wrapper"""

    def __init__(
        self,
        client: tiled.client.array.ArrayClient,
        name: str,
        parent: TiledH5 | TiledGroup | None = None,
        attrs: dict | None = None,
    ):
        super().__init__(name, client, parent, attrs)

    @property
    def shape(self):
        return self._get_data().shape

    @property
    def size(self):
        return self._get_data().size

    def __len__(self):
        return len(self.__client)

    def __getitem__(self, item):
        return self._get_data()[item]

    @property
    def value(self):
        return self._get_data()[()]
