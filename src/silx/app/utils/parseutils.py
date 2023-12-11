# /*##########################################################################
# Copyright (C) 2018-2023 European Synchrotron Radiation Facility
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
"""Utils related to parsing"""

from __future__ import annotations

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "28/05/2018"

from collections.abc import Sequence
import glob
import logging
from typing import Generator, Iterable, Union, Any, Optional
from pathlib import Path


_logger = logging.getLogger(__name__)
"""Module logger"""


_trueStrings = {"yes", "true", "1"}
_falseStrings = {"no", "false", "0"}


def _string_to_bool(string: str) -> bool:
    """Returns a boolean from a string.

    :raise ValueError: If the string do not contains a boolean information.
    """
    lower = string.lower()
    if lower in _trueStrings:
        return True
    if lower in _falseStrings:
        return False
    raise ValueError("'%s' is not a valid boolean" % string)


def to_bool(thing: Any, default: Optional[bool] = None) -> bool:
    """Returns a boolean from an object.

    :raise ValueError: If the thing can't be interpreted as a boolean and
                       no default is set
    """
    if isinstance(thing, bool):
        return thing
    try:
        return _string_to_bool(thing)
    except ValueError:
        if default is not None:
            return default
        raise


def filenames_to_dataurls(
    filenames: Iterable[Union[str, Path]],
    slices: Sequence[int] = tuple(),
) -> Generator[object, None, None]:
    """Expand filenames and HDF5 data path in files input argument"""
    # Imports here so they are performed after setting HDF5_USE_FILE_LOCKING and logging level
    import silx.io
    from silx.io.utils import match
    from silx.io.url import DataUrl
    import silx.utils.files

    extra_slices = tuple(slices)

    for filename in filenames:
        url = DataUrl(filename)

        for file_path in sorted(silx.utils.files.expand_filenames([url.file_path()])):
            if url.data_path() is not None and glob.has_magic(url.data_path()):
                try:
                    with silx.io.open(file_path) as f:
                        data_paths = list(match(f, url.data_path()))
                except BaseException as e:
                    _logger.error(
                        f"Error searching HDF5 path pattern '{url.data_path()}' in file '{file_path}': Ignored"
                    )
                    _logger.error(e.args[0])
                    _logger.debug("Backtrace", exc_info=True)
                    continue
            else:
                data_paths = [url.data_path()]

            if not extra_slices:
                data_slices = (url.data_slice(),)
            elif not url.data_slice():
                data_slices = extra_slices
            else:
                data_slices = [tuple(url.data_slice()) + (s,) for s in extra_slices]

            for data_path in data_paths:
                for data_slice in data_slices:
                    yield DataUrl(
                        file_path=file_path,
                        data_path=data_path,
                        data_slice=data_slice,
                        scheme=url.scheme(),
                    )


def to_enum(thing: Any, enum_type, default: Optional[object] = None):
    """Parse this string as this enum_type."""
    try:
        v = getattr(enum_type, str(thing))
        if isinstance(v, enum_type):
            return v
        raise ValueError(f"{thing} is not a {enum_type.__name__}")
    except (AttributeError, ValueError) as e:
        if default is not None:
            return default
        raise
