from __future__ import annotations

import logging
from typing import Generator

import numpy

from . import commonh5

from blissdata.h5api import abstract as abc
from blissdata.h5api.redis_hdf5 import File


_logger = logging.getLogger(__name__)
logging.getLogger("blissdata").setLevel(logging.DEBUG)


class BlissDataH5(commonh5.File):
    def __init__(
        self,
        name: str,
        mode: str | None = None,
        attrs: dict | None = None,
    ) -> None:
        assert mode in ("r", None)

        if attrs is None:
            attrs = {}

        self.__file = File(name)

        super().__init__(name, mode, attrs={**self.__file.attrs, **attrs})

        for child in _children(self.__file):
            self.add_node(child)

        _logger.warning(
            "blissdata support is a preview feature: This may change or be removed without notice."
        )

    def close(self) -> None:
        super().close()
        self.__file.close()
        self.__file = None


class BlissDataGroup(commonh5.LazyLoadableGroup):
    def __init__(
        self,
        name: str,
        group: abc.Group,
        parent: BlissDataH5 | BlissDataGroup | None = None,
        attrs: dict | None = None,
    ) -> None:
        super().__init__(name, parent, attrs)
        self.__group = group

    def _create_child(self) -> None:
        for child in _children(self.__group):
            self.add_node(child)


class BlissDataDataset(commonh5.Dataset):

    @property
    def shape(self) -> tuple[int, ...]:
        return self._get_data().shape

    @property
    def size(self) -> int:
        return self._get_data().size

    def __len__(self) -> int:
        return len(self._get_data())

    def __getitem__(self, item):
        print("getitem", item)
        if isinstance(item, tuple) and len(item):
            print("special case", item)
            return self._get_data()[()][item]
        return self._get_data()[item]

    @property
    def value(self) -> numpy.ndarray:
        return self._get_data()[()]


def _children(group: abc.Group) -> Generator[BlissDataDataset | BlissDataGroup]:
    for name in group.keys():
        item = group[name]
        print("child", name, item)
        if isinstance(item, abc.Group):
            yield BlissDataGroup(name, item)
        elif isinstance(item, abc.Dataset):
            yield BlissDataDataset(name, item)
        else:
            _logger.warning(f"Cannot map child {name}: Ignored")
