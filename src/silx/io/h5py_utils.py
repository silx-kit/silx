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
import sys
import traceback
import logging
import h5py

from .._version import calc_hexversion
from ..utils import retry as retry_mod
from silx.utils.deprecation import deprecated_warning

_logger = logging.getLogger(__name__)

IS_WINDOWS = sys.platform == "win32"

H5PY_HEX_VERSION = calc_hexversion(*h5py.version.version_tuple[:3])
HDF5_HEX_VERSION = calc_hexversion(*h5py.version.hdf5_version_tuple[:3])

HDF5_SWMR_VERSION = calc_hexversion(*h5py.get_config().swmr_min_hdf5_version[:3])
HAS_SWMR = HDF5_HEX_VERSION >= HDF5_SWMR_VERSION

HAS_TRACK_ORDER = H5PY_HEX_VERSION >= calc_hexversion(2, 9, 0)

if h5py.version.hdf5_version_tuple[:2] == (1, 10):
    HDF5_HAS_LOCKING_ARGUMENT = HDF5_HEX_VERSION >= calc_hexversion(1, 10, 7)
else:
    HDF5_HAS_LOCKING_ARGUMENT = HDF5_HEX_VERSION >= calc_hexversion(1, 12, 1)
H5PY_HAS_LOCKING_ARGUMENT = H5PY_HEX_VERSION >= calc_hexversion(3, 5, 0)
HAS_LOCKING_ARGUMENT = HDF5_HAS_LOCKING_ARGUMENT & H5PY_HAS_LOCKING_ARGUMENT

LATEST_LIBVER_IS_V108 = HDF5_HEX_VERSION < calc_hexversion(1, 10, 0)


def _libver_low_bound_is_v108(libver) -> bool:
    if libver is None:
        return True
    if LATEST_LIBVER_IS_V108:
        return True
    if isinstance(libver, str):
        low = libver
    else:
        low = libver[0]
    if low == "latest":
        return False
    return low == "v108"


def _hdf5_file_locking(mode="r", locking=None, swmr=None, libver=None, **_):
    """Concurrent access by disabling file locking is not supported
    in these cases:

        * mode != "r": causes file corruption
        * SWMR: does not work
        * libver > v108 and file already locked: does not work
        * windows and HDF5_HAS_LOCKING_ARGUMENT and file already locked: does not work

    :param str or None mode: read-only by default
    :param bool or None locking: by default it is disabled for `mode='r'`
                                 and `swmr=False` and enabled for all
                                 other modes.
    :param bool or None swmr: try both modes when `mode='r'` and `swmr=None`
    :param None or str or tuple libver:
    :returns bool:
    """
    if locking is None:
        locking = bool(mode != "r" or swmr)
    if not locking:
        if mode != "r":
            raise ValueError("Locking is mandatory for HDF5 writing")
        if swmr:
            raise ValueError("Locking is mandatory for HDF5 SWMR mode")
        if IS_WINDOWS and HDF5_HAS_LOCKING_ARGUMENT:
            _logger.debug(
                "Non-locking readers will fail when a writer has already locked the HDF5 file (this restriction applies to libhdf5 >= 1.12.1 or libhdf5 >= 1.10.7 on Windows)"
            )
        if not _libver_low_bound_is_v108(libver):
            _logger.debug(
                "Non-locking readers will fail when a writer has already locked the HDF5 file (this restriction applies to libver >= v110)"
            )
    return locking


def _is_h5py_exception(e):
    """
    :param BaseException e:
    :returns bool:
    """
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
    r"""Decorator for a method that needs to be executed until it not longer
    fails on HDF5 IO. Mainly used for reading an HDF5 file that is being
    written.

    :param \**kw: see `silx.utils.retry`
    """
    kw.setdefault("retry_on_error", _retry_h5py_error)
    return retry_mod.retry(**kw)


def retry_contextmanager(**kw):
    r"""Decorator to make a context manager from a method that needs to be
    entered until it not longer fails on HDF5 IO. Mainly used for reading
    an HDF5 file that is being written.

    :param \**kw: see `silx.utils.retry_contextmanager`
    """
    kw.setdefault("retry_on_error", _retry_h5py_error)
    return retry_mod.retry_contextmanager(**kw)


def retry_in_subprocess(**kw):
    r"""Same as `retry` but it also retries segmentation faults.

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
def open_item(filename, name, retry_invalid=False, validate=None, **open_options):
    r"""Yield an HDF5 dataset or group (retry until it can be instantiated).

    :param str filename:
    :param bool retry_invalid: retry when item is missing or not valid
    :param callable or None validate:
    :param \**open_options: see `File.__init__`
    :yields Dataset, Group or None:
    """
    with File(filename, **open_options) as h5file:
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


def _top_level_names(filename, include_only=group_has_end_time, **open_options):
    r"""Return all valid top-level HDF5 names.

    :param str filename:
    :param callable or None include_only:
    :param \**open_options: see `File.__init__`
    :returns list(str):
    """
    with File(filename, **open_options) as h5file:
        try:
            if callable(include_only):
                return [name for name in h5file["/"] if include_only(h5file[name])]
            else:
                return list(h5file["/"])
        except KeyError:
            raise retry_mod.RetryError


top_level_names = retry()(_top_level_names)
safe_top_level_names = retry_in_subprocess()(_top_level_names)


class Hdf5FileLockingManager:
    """Manage HDF5 file locking in the current process through the HDF5_USE_FILE_LOCKING
    environment variable.
    """

    def __init__(self) -> None:
        self._hdf5_file_locking = None
        self._nfiles_open = 0

    def opened(self):
        self._add_nopen(1)

    def closed(self):
        self._add_nopen(-1)
        if not self._nfiles_open:
            self._restore_locking_env()

    def set_locking(self, locking):
        if self._nfiles_open:
            self._check_locking_env(locking)
        else:
            self._set_locking_env(locking)

    def _add_nopen(self, v):
        self._nfiles_open = max(self._nfiles_open + v, 0)

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
            self._hdf5_file_locking = None
        else:
            self._hdf5_file_locking = v == "TRUE"

    def _restore_locking_env(self):
        self._set_locking_env(self._hdf5_file_locking)
        self._hdf5_file_locking = None


class File(h5py.File):
    """Takes care of HDF5 file locking and SWMR mode without the need
    to handle those explicitely.

    When file locking is managed through the HDF5_USE_FILE_LOCKING environment
    variable, you cannot open different files simultaneously with different modes.
    """

    _SWMR_LIBVER = "latest"

    if HAS_LOCKING_ARGUMENT:
        _LOCKING_MGR = None
    else:
        _LOCKING_MGR = Hdf5FileLockingManager()

    def __init__(
        self,
        filename,
        mode=None,
        locking=None,
        enable_file_locking=None,
        swmr=None,
        libver=None,
        **kwargs,
    ):
        r"""The arguments `locking` and `swmr` should not be
        specified explicitly for normal use cases.

        :param str filename:
        :param str or None mode: read-only by default
        :param bool or None locking: by default it is disabled for `mode='r'`
                                        and `swmr=False` and enabled for all
                                        other modes.
        :param bool or None enable_file_locking: deprecated
        :param bool or None swmr: try both modes when `mode='r'` and `swmr=None`
        :param None or str or tuple libver:
        :param \**kwargs: see `h5py.File.__init__`
        """
        # File locking behavior has changed in recent versions of libhdf5
        if HDF5_HAS_LOCKING_ARGUMENT != H5PY_HAS_LOCKING_ARGUMENT:
            _logger.critical(
                "The version of libhdf5 ({}) used by h5py ({}) is not supported: "
                "Do not expect file locking to work.".format(
                    h5py.version.hdf5_version, h5py.version.version
                )
            )

        if mode is None:
            mode = "r"
        elif mode not in ("r", "w", "w-", "x", "a", "r+"):
            raise ValueError("invalid mode {}".format(mode))
        if not HAS_SWMR:
            swmr = False
        if swmr and libver is None:
            libver = self._SWMR_LIBVER

        if enable_file_locking is not None:
            deprecated_warning(
                type_="argument",
                name="enable_file_locking",
                replacement="locking",
                since_version="1.0",
            )
            if locking is None:
                locking = enable_file_locking
        locking = _hdf5_file_locking(
            mode=mode, locking=locking, swmr=swmr, libver=libver
        )
        if self._LOCKING_MGR is None:
            kwargs.setdefault("locking", locking)
        else:
            self._LOCKING_MGR.set_locking(locking)

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
            self._file_open_callback()
            try:
                if mode != "r" and swmr:
                    # Try setting writer in SWMR mode
                    self.swmr_mode = True
            except Exception:
                self.close()
                raise

    def close(self):
        super().close()
        self._file_close_callback()

    def _file_open_callback(self):
        if self._LOCKING_MGR is not None:
            self._LOCKING_MGR.opened()

    def _file_close_callback(self):
        if self._LOCKING_MGR is not None:
            self._LOCKING_MGR.closed()
