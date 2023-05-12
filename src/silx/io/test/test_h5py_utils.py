# /*##########################################################################
# Copyright (C) 2016-2017 European Synchrotron Radiation Facility
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
"""Tests for h5py utilities"""

__authors__ = ["W. de Nolf"]
__license__ = "MIT"
__date__ = "27/01/2020"


import unittest
import os
import sys
import time
import shutil
import logging
import tempfile
import multiprocessing
from contextlib import contextmanager

from .. import h5py_utils
from ...utils.retry import RetryError, RetryTimeoutError

IS_WINDOWS = sys.platform == "win32"
logger = logging.getLogger()


def _subprocess_context_main(queue, contextmgr, *args, **kw):
    try:
        with contextmgr(*args, **kw):
            queue.put(None)
            queue.get()
    except Exception:
        queue.put(None)
        raise


@contextmanager
def _subprocess_context(contextmgr, *args, **kw):
    print("\nSTART", os.getpid())
    timeout = kw.pop("timeout", 10)
    queue = multiprocessing.Queue(maxsize=1)
    p = multiprocessing.Process(
        target=_subprocess_context_main, args=(queue, contextmgr) + args, kwargs=kw
    )
    p.start()
    try:
        queue.get(timeout=timeout)
        yield
    finally:
        queue.put(None)
        p.join(timeout)
        print(" EXIT", os.getpid())


@contextmanager
def _open_context(filename, **kw):
    try:
        print(os.getpid(), "OPEN", filename, kw)
        with h5py_utils.File(filename, **kw) as f:
            if kw.get("mode") == "w":
                f["check"] = True
                f.flush()
            yield f
    except Exception:
        print(" ", os.getpid(), "FAILED", filename, kw)
        raise
    else:
        print(" ", os.getpid(), "CLOSED", filename, kw)


def _cause_segfault():
    import ctypes

    i = ctypes.c_char(b"a")
    j = ctypes.pointer(i)
    c = 0
    while True:
        j[c] = b"a"
        c += 1


def _top_level_names_test(txtfilename, *args, **kw):
    sys.stderr = open(os.devnull, "w")

    with open(txtfilename, mode="r") as f:
        failcounter = int(f.readline().strip())

    ncausefailure = kw.pop("ncausefailure")
    faildelay = kw.pop("faildelay")
    if failcounter < ncausefailure:
        time.sleep(faildelay)
        failcounter += 1
        with open(txtfilename, mode="w") as f:
            f.write(str(failcounter))
        if failcounter % 2:
            raise RetryError
        else:
            _cause_segfault()
    return h5py_utils._top_level_names(*args, **kw)


top_level_names_test = h5py_utils.retry_in_subprocess()(_top_level_names_test)


def subtests(test):
    def wrapper(self):
        for subtest_options in self._subtests():
            print("\n====SUB TEST===\n")
            print(f"sub test options: {subtest_options}")
            with self.subTest(str(subtest_options)):
                test(self)

    return wrapper


class TestH5pyUtils(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _subtests(self):
        self._subtest_options = {"mode": "w"}
        self.filename_generator = self._filenames()
        yield self._subtest_options
        self._subtest_options = {"mode": "w", "libver": "latest"}
        self.filename_generator = self._filenames()
        yield

    def _filenames(self):
        i = 1
        while True:
            filename = os.path.join(self.test_dir, "file{}.h5".format(i))
            with self._open_context(filename):
                pass
            yield filename
            i += 1

    def _new_filename(self):
        return next(self.filename_generator)

    @contextmanager
    def _open_context(self, filename, **kwargs):
        kw = dict(self._subtest_options)
        kw.update(kwargs)
        with _open_context(filename, **kw) as f:
            yield f

    @contextmanager
    def _open_context_subprocess(self, filename, **kwargs):
        kw = dict(self._subtest_options)
        kw.update(kwargs)
        with _subprocess_context(_open_context, filename, **kw):
            yield

    def _assert_hdf5_data(self, f):
        self.assertTrue(f["check"][()])

    def _validate_hdf5_data(self, filename, swmr=False):
        with self._open_context(filename, mode="r") as f:
            self.assertEqual(f.swmr_mode, swmr)
            self._assert_hdf5_data(f)

    @subtests
    def test_modes_single_process(self):
        """Test concurrent access to the different files from the same process"""
        # When using HDF5_USE_FILE_LOCKING, open files with and without
        # locking should raise an exception. HDF5_USE_FILE_LOCKING should
        # be reset when all files are closed.

        orig = os.environ.get("HDF5_USE_FILE_LOCKING")
        filename1 = self._new_filename()
        self.assertEqual(orig, os.environ.get("HDF5_USE_FILE_LOCKING"))
        filename2 = self._new_filename()
        self.assertEqual(orig, os.environ.get("HDF5_USE_FILE_LOCKING"))

        with self._open_context(filename1, mode="r"):
            locking1 = False
            for mode in ["r", "w", "a"]:
                locking2 = mode != "r"
                raise_condition = not h5py_utils.HAS_LOCKING_ARGUMENT
                raise_condition &= locking1 != locking2
                with self.assertRaisesIf(raise_condition, RuntimeError):
                    with self._open_context(filename2, mode=mode):
                        pass
        self._validate_hdf5_data(filename1)
        self._validate_hdf5_data(filename2)
        self.assertEqual(orig, os.environ.get("HDF5_USE_FILE_LOCKING"))

        with self._open_context(filename1, mode="a"):
            locking1 = True
            for mode in ["r", "w", "a"]:
                locking2 = mode != "r"
                raise_condition = not h5py_utils.HAS_LOCKING_ARGUMENT
                raise_condition &= locking1 != locking2
                with self.assertRaisesIf(raise_condition, RuntimeError):
                    with self._open_context(filename2, mode=mode):
                        pass
        self._validate_hdf5_data(filename1)
        self._validate_hdf5_data(filename2)
        self.assertEqual(orig, os.environ.get("HDF5_USE_FILE_LOCKING"))

    @property
    def _libver_low_bound_is_v108(self):
        libver = self._subtest_options.get("libver")
        return h5py_utils._libver_low_bound_is_v108(libver)

    @property
    def _nonlocking_reader_before_writer(self):
        """A non-locking reader must open the file before it is locked by a writer"""
        if IS_WINDOWS and h5py_utils.HDF5_HAS_LOCKING_ARGUMENT:
            return True
        if not self._libver_low_bound_is_v108:
            return True
        return False

    @contextmanager
    def assertRaisesIf(self, condition, *args, **kw):
        if condition:
            with self.assertRaises(*args, **kw):
                yield
        else:
            yield

    @unittest.skipIf(
        h5py_utils.HDF5_HAS_LOCKING_ARGUMENT != h5py_utils.H5PY_HAS_LOCKING_ARGUMENT,
        "Versions of libhdf5 and h5py use incompatible file locking behaviour",
    )
    @subtests
    def test_modes_multi_process(self):
        """Test concurrent access to the same file from different processes"""
        filename = self._new_filename()

        nonlocking_reader_before_writer = self._nonlocking_reader_before_writer
        writer_before_nonlocking_reader_exception = OSError
        old_hdf5_on_windows = IS_WINDOWS and not h5py_utils.HDF5_HAS_LOCKING_ARGUMENT
        locked_exception = OSError

        # File locked by a writer
        unexpected_access = old_hdf5_on_windows and self._libver_low_bound_is_v108
        for wmode in ["w", "a"]:
            with self._open_context_subprocess(filename, mode=wmode):
                # Access by a second non-locking reader
                with self.assertRaisesIf(
                    nonlocking_reader_before_writer,
                    writer_before_nonlocking_reader_exception,
                ):
                    with self._open_context(filename, mode="r") as f:
                        self._assert_hdf5_data(f)
                # No access by a second locking reader
                if unexpected_access:
                    logger.warning("unexpected concurrent access by a locking reader")
                with self.assertRaisesIf(not unexpected_access, locked_exception):
                    with self._open_context(filename, mode="r", locking=True) as f:
                        self._assert_hdf5_data(f)
                # No access by a second writer
                if unexpected_access:
                    logger.warning("unexpected concurrent access by a writer")
                with self.assertRaisesIf(not unexpected_access, locked_exception):
                    with self._open_context(filename, mode="a") as f:
                        self._assert_hdf5_data(f)
                # Check for file corruption
                if not nonlocking_reader_before_writer:
                    self._validate_hdf5_data(filename)
            self._validate_hdf5_data(filename)

        # File locked by a reader
        unexpected_access = old_hdf5_on_windows
        with _subprocess_context(_open_context, filename, mode="r", locking=True):
            # Access by a non-locking reader
            with self._open_context(filename, mode="r") as f:
                self._assert_hdf5_data(f)
            # Access by a locking reader
            with self._open_context(filename, mode="r", locking=True) as f:
                self._assert_hdf5_data(f)
            # No access by a second writer
            if unexpected_access:
                logger.warning("unexpected concurrent access by a writer")
            raise_condition = not unexpected_access
            with self.assertRaisesIf(raise_condition, locked_exception):
                with self._open_context(filename, mode="a") as f:
                    self._assert_hdf5_data(f)
            # Check for file corruption
            self._validate_hdf5_data(filename)
        self._validate_hdf5_data(filename)

        # File open by a non-locking reader
        with self._open_context_subprocess(filename, mode="r"):
            # Access by a second non-locking reader
            with self._open_context(filename, mode="r") as f:
                self._assert_hdf5_data(f)
            # Access by a second locking reader
            with self._open_context(filename, mode="r", locking=True) as f:
                self._assert_hdf5_data(f)
            # Access by a second writer
            with self._open_context(filename, mode="a") as f:
                self._assert_hdf5_data(f)
            # Check for file corruption
            self._validate_hdf5_data(filename)
        self._validate_hdf5_data(filename)

    @subtests
    @unittest.skipIf(not h5py_utils.HAS_SWMR, "SWMR not supported")
    def test_modes_multi_process_swmr(self):
        filename = self._new_filename()

        with self._open_context(filename, mode="w", libver="latest") as f:
            pass

        # File open by SWMR writer
        with self._open_context_subprocess(filename, mode="a", swmr=True):
            with self._open_context(filename, mode="r") as f:
                assert f.swmr_mode
                self._assert_hdf5_data(f)
            with self.assertRaises(OSError):
                with self._open_context(filename, mode="a") as f:
                    pass
            self._validate_hdf5_data(filename, swmr=True)

    @subtests
    def test_retry_defaults(self):
        filename = self._new_filename()

        names = h5py_utils.top_level_names(filename)
        self.assertEqual(names, [])

        names = h5py_utils.safe_top_level_names(filename)
        self.assertEqual(names, [])

        names = h5py_utils.top_level_names(filename, include_only=None)
        self.assertEqual(names, ["check"])

        names = h5py_utils.safe_top_level_names(filename, include_only=None)
        self.assertEqual(names, ["check"])

        with h5py_utils.open_item(filename, "/check", validate=lambda x: False) as item:
            self.assertEqual(item, None)

        with h5py_utils.open_item(filename, "/check", validate=None) as item:
            self.assertTrue(item[()])

        with self.assertRaises(RetryTimeoutError):
            with h5py_utils.open_item(
                filename,
                "/check",
                retry_timeout=0.1,
                retry_invalid=True,
                validate=lambda x: False,
            ) as item:
                pass

        ncall = 0

        def validate(item):
            nonlocal ncall
            if ncall >= 1:
                return True
            else:
                ncall += 1
                raise RetryError

        with h5py_utils.open_item(
            filename,
            "/check",
            validate=validate,
            retry_timeout=1,
            retry_invalid=True,
        ) as item:
            self.assertTrue(item[()])

    @subtests
    def test_retry_custom(self):
        filename = self._new_filename()
        ncausefailure = 3
        faildelay = 0.1
        sufficient_timeout = ncausefailure * (faildelay + 10)
        insufficient_timeout = ncausefailure * faildelay * 0.5

        @h5py_utils.retry_contextmanager()
        def open_item(filename, name):
            nonlocal failcounter
            if failcounter < ncausefailure:
                time.sleep(faildelay)
                failcounter += 1
                raise RetryError
            with h5py_utils.File(filename) as h5file:
                yield h5file[name]

        failcounter = 0
        kw = {"retry_timeout": sufficient_timeout}
        with open_item(filename, "/check", **kw) as item:
            self.assertTrue(item[()])

        failcounter = 0
        kw = {"retry_timeout": insufficient_timeout}
        with self.assertRaises(RetryTimeoutError):
            with open_item(filename, "/check", **kw) as item:
                pass

    @subtests
    def test_retry_in_subprocess(self):
        filename = self._new_filename()
        txtfilename = os.path.join(self.test_dir, "failcounter.txt")
        ncausefailure = 3
        faildelay = 0.1
        sufficient_timeout = ncausefailure * (faildelay + 10)
        insufficient_timeout = ncausefailure * faildelay * 0.5

        kw = {
            "retry_timeout": sufficient_timeout,
            "include_only": None,
            "ncausefailure": ncausefailure,
            "faildelay": faildelay,
        }
        with open(txtfilename, mode="w") as f:
            f.write("0")
        names = top_level_names_test(txtfilename, filename, **kw)
        self.assertEqual(names, ["check"])

        kw = {
            "retry_timeout": insufficient_timeout,
            "include_only": None,
            "ncausefailure": ncausefailure,
            "faildelay": faildelay,
        }
        with open(txtfilename, mode="w") as f:
            f.write("0")
        with self.assertRaises(RetryTimeoutError):
            top_level_names_test(txtfilename, filename, **kw)

    @subtests
    def test_retry_generator(self):
        filename = self._new_filename()
        ncausefailure = 3
        faildelay = 0.1
        sufficient_timeout = ncausefailure * (faildelay + 10)
        insufficient_timeout = ncausefailure * faildelay * 0.5

        @h5py_utils.retry()
        def iter_data(filename, name, start_index=0):
            nonlocal failcounter
            if start_index <= 0:
                with h5py_utils.File(filename) as h5file:
                    yield h5file[name][()]
            if failcounter < ncausefailure:
                time.sleep(faildelay)
                failcounter += 1
                raise RetryError
            if start_index <= 1:
                with h5py_utils.File(filename) as h5file:
                    yield not h5file[name][()]

        failcounter = 0
        kw = {"retry_timeout": sufficient_timeout}
        data = list(iter_data(filename, "/check", **kw))
        self.assertEqual(data, [True, False])

        failcounter = 0
        kw = {"retry_timeout": insufficient_timeout}
        with self.assertRaises(RetryTimeoutError):
            list(iter_data(filename, "/check", **kw))
