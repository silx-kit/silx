# /*##########################################################################
# Copyright (C) 2025 European Synchrotron Radiation Facility
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
Provides a wrapper to expose `Zarr <https://zarr.readthedocs.io/>`_
This is a preview feature.
"""
from __future__ import annotations

import logging
import urllib.parse
from typing import Generator
import numpy
from . import commonh5
import zarr


_logger = logging.getLogger(__name__)


def _children(group: zarr.Group) -> Generator[ZarrDataset | ZarrGroup]:
    for name, item in group.items():
        if isinstance(item, zarr.Group):
            yield ZarrGroup(name, item)
        elif isinstance(item, zarr.Array):
            yield ZarrDataset(name, item)
        else:
            _logger.warning(f"Cannot map child {name}: Ignored")


class ZarrH5(commonh5.File):
    """Zarr client wrapper"""

    def __init__(
        self,
        name: str,
        mode: str | None = None,
        attrs: dict | None = None,
    ) -> None:
        assert mode in ("r", None)
        if name.startswith("zarr+"):
            name = name[5:]
        full_url = urllib.parse.urlparse(name)
        if full_url.fragment:
            raise ValueError("URL fragment is not supported")

        base_url = urllib.parse.urlunparse(
            (full_url.scheme, full_url.netloc, full_url.path, "", "", "")
        )

        # quick&dirty storage_options parsing: it would need pydantic model
        storage_options = {}
        for key, values in urllib.parse.parse_qs(full_url.query).items():
            value = values[-1]
            if key == "use_ssl":
                value = True if value.lower() == "true" else False
            storage_options[key] = value
        self.__group = zarr.open_group(base_url, storage_options=storage_options)

        if attrs is None:
            attrs = {}
        super().__init__(
            base_url.rstrip("/"), mode, attrs={**self.__group.attrs, **attrs}
        )

        for child in _children(self.__group):
            self.add_node(child)

        _logger.warning(
            "Zarr support is a preview feature: This may change or be removed without notice."
        )

    def close(self) -> None:
        super().close()
        self.__group = None


class ZarrGroup(commonh5.LazyLoadableGroup):
    """Zarr Group wrapper"""

    def __init__(
        self,
        name: str,
        group: zarr.Group,
        parent: ZarrH5 | ZarrGroup | None = None,
        attrs: dict | None = None,
    ) -> None:
        super().__init__(name, parent, attrs)
        self.__group = group

    def _create_child(self) -> None:
        for child in _children(self.__group):
            self.add_node(child)


class ZarrDataset(commonh5.Dataset):
    """Zarr Array wrapper"""

    def __init__(
        self,
        name: str,
        array: zarr.Array,
        parent: ZarrH5 | ZarrGroup | None = None,
        attrs: dict | None = None,
    ) -> None:
        super().__init__(name, array, parent, attrs)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._get_data().shape

    @property
    def size(self) -> int:
        return self._get_data().size

    def __len__(self) -> int:
        return len(self._get_data())

    def __getitem__(self, item):
        return self._get_data()[item]

    @property
    def value(self) -> numpy.ndarray:
        return self._get_data()[()]

    @property
    def compression(self):
        return self._get_data().compressor.codec_id

    @property
    def chunks(self):
        return self._get_data().chunks
