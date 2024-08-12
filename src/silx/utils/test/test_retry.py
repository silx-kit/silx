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
"""Tests for retry utilities"""

__authors__ = ["W. de Nolf"]
__license__ = "MIT"
__date__ = "05/02/2020"


import unittest
import os
import sys
import time
import tempfile
import pytest

from .. import retry


def _cause_segfault():
    import ctypes

    i = ctypes.c_char(b"a")
    j = ctypes.pointer(i)
    c = 0
    while True:
        j[c] = b"a"
        c += 1


def _submain(filename, kwcheck=None, ncausefailure=0, faildelay=0):
    assert filename
    assert kwcheck
    sys.stderr = open(os.devnull, "w")

    with open(filename, mode="r") as f:
        failcounter = int(f.readline().strip())

    if failcounter < ncausefailure:
        time.sleep(faildelay)
        failcounter += 1
        with open(filename, mode="w") as f:
            f.write(str(failcounter))
        if failcounter % 2:
            raise retry.RetryError
        else:
            _cause_segfault()
    return True


_wsubmain = retry.retry_in_subprocess()(_submain)


class TestRetry(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.ctr_file = os.path.join(self.test_dir, "failcounter.txt")

    def tearDown(self):
        if os.path.exists(self.ctr_file):
            os.unlink(self.ctr_file)
        os.rmdir(self.test_dir)

    def test_retry(self):
        ncausefailure = 3
        faildelay = 0.1
        sufficient_timeout = ncausefailure * (faildelay + 10)
        insufficient_timeout = ncausefailure * faildelay * 0.5

        @retry.retry()
        def method(check, kwcheck=None):
            assert check
            assert kwcheck
            nonlocal failcounter
            if failcounter < ncausefailure:
                time.sleep(faildelay)
                failcounter += 1
                raise retry.RetryError
            return True

        failcounter = 0
        kw = {
            "kwcheck": True,
            "retry_timeout": sufficient_timeout,
        }
        self.assertTrue(method(True, **kw))

        failcounter = 0
        kw = {
            "kwcheck": True,
            "retry_timeout": insufficient_timeout,
        }
        with self.assertRaises(retry.RetryTimeoutError):
            method(True, **kw)

    def test_retry_contextmanager(self):
        ncausefailure = 3
        faildelay = 0.1
        sufficient_timeout = ncausefailure * (faildelay + 10)
        insufficient_timeout = ncausefailure * faildelay * 0.5

        @retry.retry_contextmanager()
        def context(check, kwcheck=None):
            assert check
            assert kwcheck
            nonlocal failcounter
            if failcounter < ncausefailure:
                time.sleep(faildelay)
                failcounter += 1
                raise retry.RetryError
            yield True

        failcounter = 0
        kw = {"kwcheck": True, "retry_timeout": sufficient_timeout}
        with context(True, **kw) as result:
            self.assertTrue(result)

        failcounter = 0
        kw = {"kwcheck": True, "retry_timeout": insufficient_timeout}
        with self.assertRaises(retry.RetryTimeoutError):
            with context(True, **kw) as result:
                pass

    def test_retry_in_subprocess(self):
        ncausefailure = 3
        faildelay = 0.1
        sufficient_timeout = ncausefailure * (faildelay + 10)
        insufficient_timeout = ncausefailure * faildelay * 0.5

        kw = {
            "ncausefailure": ncausefailure,
            "faildelay": faildelay,
            "kwcheck": True,
            "retry_timeout": sufficient_timeout,
        }
        with open(self.ctr_file, mode="w") as f:
            f.write("0")
        self.assertTrue(_wsubmain(self.ctr_file, **kw))

        kw = {
            "ncausefailure": ncausefailure,
            "faildelay": faildelay,
            "kwcheck": True,
            "retry_timeout": insufficient_timeout,
        }
        with open(self.ctr_file, mode="w") as f:
            f.write("0")
        with self.assertRaises(retry.RetryTimeoutError):
            _wsubmain(self.ctr_file, **kw)

    def test_retry_generator(self):
        ncausefailure = 3
        faildelay = 0.1
        sufficient_timeout = ncausefailure * (faildelay + 10)
        insufficient_timeout = ncausefailure * faildelay * 0.5

        @retry.retry()
        def method(check, kwcheck=None, start_index=0):
            if start_index <= 0:
                yield 0
            assert check
            assert kwcheck
            nonlocal failcounter
            if failcounter < ncausefailure:
                time.sleep(faildelay)
                failcounter += 1
                if start_index <= 1:
                    yield 1
                raise retry.RetryError
            else:
                if start_index <= 1:
                    yield 1
            if start_index <= 2:
                yield 2

        failcounter = 0
        kw = {"kwcheck": True, "retry_timeout": sufficient_timeout}
        self.assertEqual(list(method(True, **kw)), [0, 1, 2])

        failcounter = 0
        kw = {
            "kwcheck": True,
            "retry_timeout": insufficient_timeout,
        }
        with self.assertRaises(retry.RetryTimeoutError):
            list(method(True, **kw))

    def test_retry_wrong_generator(self):
        with self.assertRaises(TypeError):

            @retry.retry()
            def method():
                yield from range(3)

    def test_retry_iter_reset(self):
        """Test would fail when the timer does not get reset after every iteration"""
        failure_t0 = None
        failure_count = 0
        retry_period = 0.1
        failure_duration = 0.7
        nretry_offset = 10

        nfailures = int(failure_duration / retry_period + 0.5)
        retry_timeout = failure_duration + retry_period
        xfail_timeout = failure_duration + retry_period * 0.5

        @retry.retry(retry_period=retry_period, retry_timeout=retry_timeout)
        def iter_with_failure(start_index=0):
            nonlocal failure_count, failure_t0

            # This takes `nretry_offset * retry_period` seconds
            if start_index == 0:
                for i in range(nretry_offset):
                    time.sleep(retry_period)
                    yield i
                failure_t0 = time.time()

            # This will fail for slightly longer than `failure_duration` seconds
            if failure_count <= nfailures:
                failure_count += 1
                if failure_count > 1:
                    iter_time = (time.time() - failure_t0) / (failure_count - 1)
                    if (iter_time * nfailures) >= xfail_timeout:
                        pytest.xfail("iteration takes much longer than retry_period")
                raise retry.RetryError()

            yield nretry_offset

        yielded = list(iter_with_failure())
        expected = list(range(nretry_offset + 1))
        assert yielded == expected
