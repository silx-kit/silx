# coding: utf-8
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
import tempfile

from .. import retry


def _cause_segfault():
    import ctypes

    i = ctypes.c_char(b"a")
    j = ctypes.pointer(i)
    c = 0
    while True:
        j[c] = b"a"
        c += 1


def _submain(filename, kwcheck=None, nsleep=0):
    assert filename
    assert kwcheck
    sys.stderr = open(os.devnull, "w")

    with open(filename, mode="r") as f:
        counter = int(f.readline().strip())

    if counter < nsleep:
        counter += 1
        with open(filename, mode="w") as f:
            f.write(str(counter))
        if counter % 2:
            raise retry.RetryError
        else:
            _cause_segfault()
    return True


_wsubmain = retry.retry_in_subprocess()(_submain)


class TestRetry(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.ctr_file = os.path.join(self.test_dir, "counter.txt")

    def tearDown(self):
        if os.path.exists(self.ctr_file):
            os.unlink(self.ctr_file)
        os.rmdir(self.test_dir)

    def test_retry(self):
        nsleep = 3
        retry_period = 0.01

        # just to make sure the test doesn't hang
        overhead = 10

        @retry.retry()
        def method(check, kwcheck=None):
            assert check
            assert kwcheck
            nonlocal counter
            if counter < nsleep:
                counter += 1
                raise retry.RetryError
            return True

        counter = 0
        kw = {
            "kwcheck": True,
            "retry_timeout": nsleep * (retry_period + overhead),
            "retry_period": retry_period,
        }
        self.assertTrue(method(True, **kw))

        counter = 0
        kw = {
            "kwcheck": True,
            "retry_timeout": nsleep * (retry_period * 0.9),
            "retry_period": retry_period,
        }
        with self.assertRaises(retry.RetryTimeoutError):
            method(True, **kw)

    def test_retry_contextmanager(self):
        nsleep = 3
        retry_period = 0.01

        # just to make sure the test doesn't hang
        overhead = 10

        @retry.retry_contextmanager()
        def context(check, kwcheck=None):
            assert check
            assert kwcheck
            nonlocal counter
            if counter < nsleep:
                counter += 1
                raise retry.RetryError
            yield True

        counter = 0
        kw = {
            "kwcheck": True,
            "retry_timeout": nsleep * (retry_period + overhead),
            "retry_period": retry_period,
        }
        with context(True, **kw) as result:
            self.assertTrue(result)

        counter = 0
        kw = {
            "kwcheck": True,
            "retry_timeout": nsleep * (retry_period * 0.9),
            "retry_period": retry_period,
        }
        with self.assertRaises(retry.RetryTimeoutError):
            with context(True, **kw) as result:
                pass

    def test_retry_in_subprocess(self):
        nsleep = 3
        retry_period = 0.01

        # just to make sure the test doesn't hang
        overhead = 10

        kw = {
            "nsleep": nsleep,
            "kwcheck": True,
            "retry_timeout": nsleep * (retry_period + overhead),
            "retry_period": retry_period,
        }
        with open(self.ctr_file, mode="w") as f:
            f.write("0")
        self.assertTrue(_wsubmain(self.ctr_file, **kw))

        kw = {
            "nsleep": nsleep,
            "kwcheck": True,
            "retry_timeout": nsleep * (retry_period - 0.001),
            "retry_period": retry_period,
        }
        with open(self.ctr_file, mode="w") as f:
            f.write("0")
        with self.assertRaises(retry.RetryTimeoutError):
            _wsubmain(self.ctr_file, **kw)


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestRetry))
    return test_suite


if __name__ == "__main__":
    unittest.main(defaultTest="suite")
