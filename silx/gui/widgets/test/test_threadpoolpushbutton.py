# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016 European Synchrotron Radiation Facility
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
"""Test for silx.gui.hdf5 module"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "11/10/2016"


import unittest
import time
from silx.gui import qt
from silx.gui import testutils
from silx.gui.widgets.ThreadPoolPushButton import ThreadPoolPushButton


class TestThreadPoolPushButton(testutils.TestCaseQt):

    def setUp(self):
        super(TestThreadPoolPushButton, self).setUp()
        self._result = []

    def _trace(self, name, delay=0):
        self._result.append(name)
        if delay != 0:
            time.sleep(delay / 1000.0)

    def _compute(self):
        return "result"

    def _computeFail(self):
        raise Exception("exception")

    def testExecute(self):
        button = ThreadPoolPushButton()
        button.setCallable(self._trace, "a", 0)
        button.executeCallable()
        time.sleep(0.1)
        self.assertListEqual(self._result, ["a"])
        self.qapp.processEvents()

    def testMultiExecution(self):
        button = ThreadPoolPushButton()
        button.setCallable(self._trace, "a", 0)
        number = qt.QThreadPool.globalInstance().maxThreadCount() * 2
        for _ in range(number):
            button.executeCallable()
        time.sleep(number * 0.01 + 0.1)
        self.assertListEqual(self._result, ["a"] * number)
        self.qapp.processEvents()

    def testSaturateThreadPool(self):
        button = ThreadPoolPushButton()
        button.setCallable(self._trace, "a", 100)
        number = qt.QThreadPool.globalInstance().maxThreadCount() * 2
        for _ in range(number):
            button.executeCallable()
        time.sleep(number * 0.1 + 0.1)
        self.assertListEqual(self._result, ["a"] * number)
        self.qapp.processEvents()

    def testSuccess(self):
        button = ThreadPoolPushButton()
        button.setCallable(self._compute)
        button.beforeExecuting.connect(lambda: self._result.append("be"))
        button.started.connect(lambda: self._result.append("s"))
        button.succeeded.connect(lambda r: self._result.append(r))
        button.failed.connect(lambda e: self.fail("Unexpected exception"))
        button.finished.connect(lambda: self._result.append("f"))
        button.executeCallable()
        self.qapp.processEvents()
        time.sleep(0.1)
        self.qapp.processEvents()
        self.assertListEqual(self._result, ["be", "s", "result", "f"])

    def testFail(self):
        button = ThreadPoolPushButton()
        button.setCallable(self._computeFail)
        button.beforeExecuting.connect(lambda: self._result.append("be"))
        button.started.connect(lambda: self._result.append("s"))
        button.succeeded.connect(lambda r: self.fail("Unexpected success"))
        button.failed.connect(lambda e: self._result.append(str(e)))
        button.finished.connect(lambda: self._result.append("f"))
        button.executeCallable()
        self.qapp.processEvents()
        time.sleep(0.1)
        self.qapp.processEvents()
        self.assertListEqual(self._result, ["be", "s", "exception", "f"])


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestThreadPoolPushButton))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
