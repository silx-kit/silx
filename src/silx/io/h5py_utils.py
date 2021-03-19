# coding: utf-8
# /*##########################################################################
# Copyright (C) 2016-2021 European Synchrotron Radiation Facility
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
This module provides utility methods on top of h5py, mainly to handle
parallel writing and reading.
"""

__authors__ = ["W. de Nolf"]
__license__ = "MIT"
__date__ = "27/01/2020"


import os
import traceback
import h5py

from .._version import calc_hexversion
from ..utils import retry as retry_mod

H5PY_HEX_VERSION = calc_hexversion(*h5py.version.version_tuple[:3])
HDF5_HEX_VERSION = calc_hexversion(*h5py.version.hdf5_version_tuple[:3])

HDF5_SWMR_VERSION = calc_hexversion(*h5py.get_config().swmr_min_hdf5_version[:3])
HDF5_TRACK_ORDER_VERSION = calc_hexversion(2, 9, 0)

HAS_SWMR = HDF5_HEX_VERSION >= HDF5_SWMR_VERSION
HAS_TRACK_ORDER = H5PY_HEX_VERSION >= HDF5_TRACK_ORDER_VERSION


def _is_h5py_exception(e):
    for frame in traceback.walk_tb(e.__traceback__):
        if frame[0].f_locals.get("__package__", None) == "h5py":
            return True
    return False


def _retry_h5py_error(e):
    """
    :param BaseException e:
    :returns bool:
    """
    if _is_h5py_exception(e):
        if isinstance(e, (OSError, RuntimeError)):
            return True
        elif isinstance(e, KeyError):
            # For example this needs to be retried:
            # KeyError: 'Unable to open object (bad object header version number)'
            return "Unable to open object" in str(e)
    elif isinstance(e, retry_mod.RetryError):
        return True
    return False


def retry(**kw):
    """Decorator for a method that needs to be executed until it not longer
    fails on HDF5 IO. Mainly used for reading an HDF5 file that is being
    written.

    :param \**kw: see `silx.utils.retry`
    """
    kw.setdefault("retry_on_error", _retry_h5py_error)
    return retry_mod.retry(**kw)


def retry_contextmanager(**kw):
    """Decorator to make a context manager from a method that needs to be
    entered until it not longer fails on HDF5 IO. Mainly used for reading
    an HDF5 file that is being written.

    :param \**kw: see `silx.utils.retry_contextmanager`
    """
    kw.setdefault("retry_on_error", _retry_h5py_error)
    return retry_mod.retry_contextmanager(**kw)


def retry_in_subprocess(**kw):
    """Same as `retry` but it also retries segmentation faults.

    On Window you cannot use this decorator with the "@" syntax:

    .. code-block:: python

        def _method(*args, **kw):
            ...

        method = retry_in_subprocess()(_method)

    :param \**kw: see `silx.utils.retry_in_subprocess`
    """
    kw.setdefault("retry_on_error", _retry_h5py_error)
    return retry_mod.retry_in_subprocess(**kw)


def group_has_end_time(h5item):
    """Returns True when the HDF5 item is a Group with an "end_time"
    dataset. A reader can use this as an indication that the Group
    has been fully written (at least if the writer supports this).

    :param Union[h5py.Group,h5py.Dataset] h5item:
    :returns bool:
    """
    if isinstance(h5item, h5py.Group):
        return "end_time" in h5item
    else:
        return False


@retry_contextmanager()
def open_item(filename, name, retry_invalid=False, validate=None):
    """Yield an HDF5 dataset or group (retry until it can be instantiated).

    :param str filename:
    :param bool retry_invalid: retry when item is missing or not valid
    :param callable or None validate:
    :yields Dataset, Group or None:
    """
    with File(filename) as h5file:
        try:
            item = h5file[name]
        except KeyError as e:
            if "doesn't exist" in str(e):
                if retry_invalid:
                    raise retry_mod.RetryError
                else:
                    item = None
            else:
                raise
        if callable(validate) and item is not None:
            if not validate(item):
                if retry_invalid:
                    raise retry_mod.RetryError
                else:
                    item = None
        yield item


def _top_level_names(filename, include_only=group_has_end_time):
    """Return all valid top-level HDF5 names.

    :param str filename:
    :param callable or None include_only:
    :returns list(str):
    """
    with File(filename) as h5file:
        try:
            if callable(include_only):
                return [name for name in h5file["/"] if include_only(h5file[name])]
            else:
                return list(h5file["/"])
        except KeyError:
            raise retry_mod.RetryError


top_level_names = retry()(_top_level_names)
safe_top_level_names = retry_in_subprocess()(_top_level_names)


class File(h5py.File):
    """Takes care of HDF5 file locking and SWMR mode without the need
    to handle those explicitely.

    When using this class, you cannot open different files simultatiously
    with different modes because the locking flag is an environment variable.
    """

    _HDF5_FILE_LOCKING = None
    _NOPEN = 0
    _SWMR_LIBVER = "latest"

    def __init__(
        self,
        filename,
        mode=None,
        enable_file_locking=None,
        swmr=None,
        libver=None,
        **kwargs
    ):
        """The arguments `enable_file_locking` and `swmr` should not be
        specified explicitly for normal use cases.

        :param str filename:
        :param str or None mode: read-only by default
        :param bool or None enable_file_locking: by default it is disabled for `mode='r'`
                                                 and `swmr=False` and enabled for all
                                                 other modes.
        :param bool or None swmr: try both modes when `mode='r'` and `swmr=None`
        :param **kwargs: see `h5py.File.__init__`
        """
        if mode is None:
            mode = "r"
        elif mode not in ("r", "w", "w-", "x", "a", "r+"):
            raise ValueError("invalid mode {}".format(mode))
        if not HAS_SWMR:
            swmr = False

        if enable_file_locking is None:
            enable_file_locking = bool(mode != "r" or swmr)
        if self._NOPEN:
            self._check_locking_env(enable_file_locking)
        else:
            self._set_locking_env(enable_file_locking)

        if swmr and libver is None:
            libver = self._SWMR_LIBVER

        if HAS_TRACK_ORDER:
            kwargs.setdefault("track_order", True)
        try:
            super().__init__(filename, mode=mode, swmr=swmr, libver=libver, **kwargs)
        except OSError as e:
            #   wlock   wSWMR   rlock   rSWMR   OSError: Unable to open file (...)
            # 1 TRUE    FALSE   FALSE   FALSE   -
            # 2 TRUE    FALSE   FALSE   TRUE    -
            # 3 TRUE    FALSE   TRUE    FALSE   unable to lock file, errno = 11, error message = 'Resource temporarily unavailable'
            # 4 TRUE    FALSE   TRUE    TRUE    unable to lock file, errno = 11, error message = 'Resource temporarily unavailable'
            # 5 TRUE    TRUE    FALSE   FALSE   file is already open for write (may use <h5clear file> to clear file consistency flags)
            # 6 TRUE    TRUE    FALSE   TRUE    -
            # 7 TRUE    TRUE    TRUE    FALSE   file is already open for write (may use <h5clear file> to clear file consistency flags)
            # 8 TRUE    TRUE    TRUE    TRUE    -
            if (
                mode == "r"
                and swmr is None
                and "file is already open for write" in str(e)
            ):
                # Try reading in SWMR mode (situation 5 and 7)
                swmr = True
                if libver is None:
                    libver = self._SWMR_LIBVER
                super().__init__(
                    filename, mode=mode, swmr=swmr, libver=libver, **kwargs
                )
            else:
                raise
        else:
            self._add_nopen(1)
            try:
                if mode != "r" and swmr:
                    # Try setting writer in SWMR mode
                    self.swmr_mode = True
            except Exception:
                self.close()
                raise

    @classmethod
    def _add_nopen(cls, v):
        cls._NOPEN = max(cls._NOPEN + v, 0)

    def close(self):
        super().close()
        self._add_nopen(-1)
        if not self._NOPEN:
            self._restore_locking_env()

    def _set_locking_env(self, enable):
        self._backup_locking_env()
        if enable:
            os.environ["HDF5_USE_FILE_LOCKING"] = "TRUE"
        elif enable is None:
            try:
                del os.environ["HDF5_USE_FILE_LOCKING"]
            except KeyError:
                pass
        else:
            os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    def _get_locking_env(self):
        v = os.environ.get("HDF5_USE_FILE_LOCKING")
        if v == "TRUE":
            return True
        elif v is None:
            return None
        else:
            return False

    def _check_locking_env(self, enable):
        if enable != self._get_locking_env():
            if enable:
                raise RuntimeError(
                    "Close all HDF5 files before enabling HDF5 file locking"
                )
            else:
                raise RuntimeError(
                    "Close all HDF5 files before disabling HDF5 file locking"
                )

    def _backup_locking_env(self):
        v = os.environ.get("HDF5_USE_FILE_LOCKING")
        if v is None:
            self._HDF5_FILE_LOCKING = None
        else:
            self._HDF5_FILE_LOCKING = v == "TRUE"

    def _restore_locking_env(self):
        self._set_locking_env(self._HDF5_FILE_LOCKING)
        self._HDF5_FILE_LOCKING = None
