# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2018 European Synchrotron Radiation Facility
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
"""Test of async module."""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "09/03/2018"


import threading
import unittest


from silx.third_party.concurrent_futures import wait
from silx.gui import qt
from silx.gui.utils.testutils import TestCaseQt

from silx.gui.utils import concurrent


class TestSubmitToQtThread(TestCaseQt):
    """Test submission of tasks to Qt main thread"""

    def setUp(self):
        # Reset executor to test lazy-loading in different conditions
        concurrent._executor = None
        super(TestSubmitToQtThread, self).setUp()

    def _task(self, value1, value2):
        return value1, value2

    def _taskWithException(self, *args, **kwargs):
        raise RuntimeError('task exception')

    def testFromMainThread(self):
        """Call submitToQtMainThread from the main thread"""
        value1, value2 = 0, 1
        future = concurrent.submitToQtMainThread(self._task, value1, value2=value2)
        self.assertTrue(future.done())
        self.assertEqual(future.result(1), (value1, value2))
        self.assertIsNone(future.exception(1))

        future = concurrent.submitToQtMainThread(self._taskWithException)
        self.assertTrue(future.done())
        with self.assertRaises(RuntimeError):
            future.result(1)
        self.assertIsInstance(future.exception(1), RuntimeError)

    def _threadedTest(self):
        """Function run in a thread for the tests"""
        value1, value2 = 0, 1
        future = concurrent.submitToQtMainThread(self._task, value1, value2=value2)

        wait([future], 3)

        self.assertTrue(future.done())
        self.assertEqual(future.result(1), (value1, value2))
        self.assertIsNone(future.exception(1))

        future = concurrent.submitToQtMainThread(self._taskWithException)

        wait([future], 3)

        self.assertTrue(future.done())
        with self.assertRaises(RuntimeError):
            future.result(1)
        self.assertIsInstance(future.exception(1), RuntimeError)

    def testFromPythonThread(self):
        """Call submitToQtMainThread from a Python thread"""
        thread = threading.Thread(target=self._threadedTest)
        thread.start()
        for i in range(100):  # Loop over for 10 seconds
            self.qapp.processEvents()
            thread.join(0.1)
            if not thread.is_alive():
                break
        else:
            self.fail(('Thread task still running'))

    def testFromQtThread(self):
        """Call submitToQtMainThread from a Qt thread pool"""
        class Runner(qt.QRunnable):
            def __init__(self, fn):
                super(Runner, self).__init__()
                self._fn = fn

            def run(self):
                self._fn()

            def autoDelete(self):
                return True

        threadPool = qt.silxGlobalThreadPool()
        runner = Runner(self._threadedTest)
        threadPool.start(runner)
        for i in range(100):  # Loop over for 10 seconds
            self.qapp.processEvents()
            done = threadPool.waitForDone(100)
            if done:
                break
        else:
            self.fail('Thread pool task still running')


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(
        TestSubmitToQtThread))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
