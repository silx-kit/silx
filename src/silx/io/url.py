# /*##########################################################################
#
# Copyright (c) 2016-2023 European Synchrotron Radiation Facility
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
# ###########################################################################*/
"""URL module"""

from __future__ import annotations

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "29/01/2018"

import logging
from collections.abc import Iterable
from typing import Union
import urllib.parse
from pathlib import Path


_logger = logging.getLogger(__name__)

SliceLike = Union[slice, int, type(Ellipsis)]


def _slice_to_string(s: SliceLike) -> str:
    """Convert a Python slice into a string"""
    if s == Ellipsis:
        return "..."
    elif isinstance(s, slice):
        result = ""
        if s.start is None:
            result += ":"
        else:
            result += f"{s.start}:"
        if s.stop is not None:
            result += f"{s.stop}"
        if s.step is not None:
            result += f":{s.step}"
        return result
    elif isinstance(s, int):
        return str(s)
    else:
        raise TypeError("Unexpected slicing type. Found %s" % type(s))


def slice_sequence_to_string(data_slice: Iterable[SliceLike] | SliceLike) -> str:
    """Convert a Python slice sequence or a slice into a string"""
    if isinstance(data_slice, Iterable):
        return ",".join([_slice_to_string(s) for s in data_slice])
    else:
        return _slice_to_string(data_slice)


class DataUrl(object):
    """Non-mutable object to parse a string representing a resource data
    locator.

    It supports:

    - path to file and path inside file to the data
    - data slicing
    - fabio or silx access to the data
    - absolute and relative file access

    >>> # fabio access using absolute path
    >>> DataUrl("fabio:///data/image.edf?slice=2")
    >>> DataUrl("fabio:///C:/data/image.edf?slice=2")

    >>> # silx access using absolute path
    >>> DataUrl("silx:///data/image.h5?path=/data/dataset&slice=1,5")
    >>> DataUrl("silx:///data/image.edf?path=/scan_0/detector/data")
    >>> DataUrl("silx:///C:/data/image.edf?path=/scan_0/detector/data")

    >>> # `path=` can be omitted if there are no other query keys
    >>> DataUrl("silx:///data/image.h5?/data/dataset")
    >>> # is the same as
    >>> DataUrl("silx:///data/image.h5?path=/data/dataset")

    >>> # `::` can be used instead of `?` which can be useful with shell in
    >>> # command lines
    >>> DataUrl("silx:///data/image.h5::/data/dataset")
    >>> # is the same as
    >>> DataUrl("silx:///data/image.h5?/data/dataset")

    >>> # Relative path access
    >>> DataUrl("silx:./image.h5")
    >>> DataUrl("fabio:./image.edf")
    >>> DataUrl("silx:image.h5")
    >>> DataUrl("fabio:image.edf")

    >>> # It also supports parsing of file access for convenience
    >>> DataUrl("./foo/bar/image.edf")
    >>> DataUrl("C:/data/")

    :param path: Path representing a link to a data. If specified, other
        arguments must not be provided.
    :param file_path: Link to the file containing the the data.
        None if there is no data selection.
    :param data_path: Data selection applied to the data file selected.
        None if there is no data selection.
    :param data_slice: Slicing applied of the selected
        data. None if no slicing applied.
    :param scheme: Scheme of the URL. "silx", "fabio"
        is supported. Other strings can be provided, but :meth:`is_valid` will
        be false.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        file_path: str | Path | None = None,
        data_path: str | None = None,
        data_slice: tuple[SliceLike, ...] | None = None,
        scheme: str | None = None,
    ):
        self.__is_valid = False
        if path is not None:
            assert file_path is None
            assert data_path is None
            assert data_slice is None
            assert scheme is None
            self.__parse_from_path(str(path))
        else:
            if file_path is not None:
                file_path = str(file_path)
            self.__file_path = file_path
            self.__data_path = data_path
            self.__data_slice = data_slice
            self.__scheme = scheme
            self.__path = None
            self.__check_validity()

    def __eq__(self, other):
        if not isinstance(other, DataUrl):
            return False
        if self.is_valid() != other.is_valid():
            return False
        if self.is_valid():
            if self.__scheme != other.scheme():
                return False
            if self.__file_path != other.file_path():
                return False
            if self.__data_path != other.data_path():
                return False
            if self.__data_slice != other.data_slice():
                return False
            return True
        else:
            return self.__path == other.path()

    def __ne__(self, other):
        return not (self == other)

    def __repr__(self):
        return str(self)

    def __str__(self):
        if self.is_valid() or self.__path is None:

            def quote_string(string):
                if isinstance(string, str):
                    return "'%s'" % string
                else:
                    return string

            template = "DataUrl(valid=%s, scheme=%s, file_path=%s, data_path=%s, data_slice=%s)"
            return template % (
                self.__is_valid,
                quote_string(self.__scheme),
                quote_string(self.__file_path),
                quote_string(self.__data_path),
                self.__data_slice,
            )
        else:
            template = "DataUrl(valid=%s, string=%s)"
            return template % (self.__is_valid, self.__path)

    def __check_validity(self):
        """Check the validity of the attributes."""
        if self.__file_path in [None, ""]:
            self.__is_valid = False
            return

        if self.__scheme is None:
            self.__is_valid = True
        elif self.__scheme == "fabio":
            self.__is_valid = self.__data_path is None
        elif self.__scheme == "silx":
            # If there is a slice you must have a data path
            # But you can have a data path without slice
            slice_implies_data = (
                self.__data_path is None and self.__data_slice is None
            ) or self.__data_path is not None
            self.__is_valid = slice_implies_data
        else:
            self.__is_valid = False

    @staticmethod
    def _parse_slice(slice_string: str) -> tuple[SliceLike, ...]:
        """Parse a slicing sequence and return an associated tuple.

        It supports a sequence of `...`, `:`, and integers separated by a coma.
        """

        def string_to_slice(string: str) -> SliceLike:
            """Convert a string to a Python slice"""
            if string == "...":
                return Ellipsis
            elif ":" in string:
                if string == ":":
                    return slice(None)
                else:

                    def get_value(my_str):
                        if my_str in ("", None):
                            return None
                        else:
                            return int(my_str)

                    sss = string.split(":")
                    start = get_value(sss[0])
                    stop = get_value(sss[1] if len(sss) > 1 else None)
                    step = get_value(sss[2] if len(sss) > 2 else None)
                    return slice(start, stop, step)
            else:
                return int(string)

        if slice_string == "":
            raise ValueError("An empty slice is not valid")

        tokens = slice_string.split(",")
        data_slice: list[SliceLike] = []
        for t in tokens:
            try:
                data_slice.append(string_to_slice(t))
            except ValueError:
                raise ValueError("'%s' is not a valid slicing" % t)
        return tuple(data_slice)

    def __parse_from_path(self, path: str):
        """Parse the path and initialize attributes.

        :param path: Path representing the URL.
        """
        self.__path = path
        # only replace if ? not here already. Otherwise can mess sith
        # data_slice if == ::2 for example
        if "?" not in path:
            path = path.replace("::", "?", 1)
        url = urllib.parse.urlparse(path)

        is_valid = True

        if len(url.scheme) <= 2:
            # Windows driver
            scheme = None
            pos = self.__path.index(url.path)
            file_path = self.__path[0:pos] + url.path
        else:
            scheme = url.scheme if url.scheme != "" else None
            file_path = url.path

            # Check absolute windows path
            if len(file_path) > 2 and file_path[0] == "/":
                if file_path[1] == ":" or file_path[2] == ":":
                    file_path = file_path[1:]

        self.__scheme = scheme
        self.__file_path = file_path

        query = urllib.parse.parse_qsl(url.query, keep_blank_values=True)
        if len(query) == 1 and query[0][1] == "":
            # there is no query keys
            data_path = query[0][0]
            data_slice = None
        else:
            merged_query = {}
            for name, value in query:
                if name in query:
                    merged_query[name].append(value)
                else:
                    merged_query[name] = [value]

            def pop_single_value(merged_query, name):
                if name in merged_query:
                    values = merged_query.pop(name)
                    if len(values) > 1:
                        _logger.warning(
                            "More than one query key named '%s'. The last one is used.",
                            name,
                        )
                    value = values[-1]
                else:
                    value = None
                return value

            data_path = pop_single_value(merged_query, "path")
            data_slice = pop_single_value(merged_query, "slice")
            if data_slice is not None:
                try:
                    data_slice = self._parse_slice(data_slice)
                except ValueError:
                    is_valid = False
                    data_slice = None

            for key in merged_query.keys():
                _logger.warning("Query key %s unsupported. Key skipped.", key)

        self.__data_path = data_path
        self.__data_slice = data_slice

        if is_valid:
            self.__check_validity()
        else:
            self.__is_valid = False

    def is_valid(self) -> bool:
        """Returns true if the URL is valid. Else attributes can be None."""
        return self.__is_valid

    def path(self) -> str:
        """Returns the string representing the URL."""
        if self.__path is not None:
            return self.__path

        if self.__data_path is not None and self.__data_slice is None:
            query = self.__data_path
        else:
            queries = []
            if self.__data_path is not None:
                queries.append("path=" + self.__data_path)
            if self.__data_slice is not None:
                data_slice = slice_sequence_to_string(self.__data_slice)
                queries.append("slice=" + data_slice)
            query = "&".join(queries)

        path = ""
        if self.__file_path is not None:
            path += self.__file_path

        if query != "":
            path = path + "?" + query

        if self.__scheme is not None:
            if self.is_absolute():
                if path.startswith("/"):
                    path = self.__scheme + "://" + path
                else:
                    path = self.__scheme + ":///" + path
            else:
                path = self.__scheme + ":" + path

        return path

    def is_absolute(self) -> bool:
        """Returns true if the file path is an absolute path."""
        file_path = self.file_path()
        if file_path is None:
            return False
        if len(file_path) > 0:
            if file_path[0] == "/":
                return True
        if len(file_path) > 2:
            # Windows
            if file_path[1] == ":" or file_path[2] == ":":
                return True
        elif len(file_path) > 1:
            # Windows
            if file_path[1] == ":":
                return True
        return False

    def file_path(self) -> str:
        """Returns the path to the file containing the data."""
        return self.__file_path

    def data_path(self) -> str | None:
        """Returns the path inside the file to the data."""
        return self.__data_path

    def data_slice(self) -> tuple[SliceLike, ...] | None:
        """Returns the slicing applied to the data.

        It is a tuple containing numbers, slice or ellipses.
        """
        return self.__data_slice

    def scheme(self) -> str | None:
        """Returns the scheme. It can be None if no scheme is specified."""
        return self.__scheme
