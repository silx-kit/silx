# /*##########################################################################
#
# Copyright (c) 2016-2021 European Synchrotron Radiation Facility
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
__date__ = "17/01/2018"


import unittest
import time
from silx.gui import qt
from silx.gui.utils.testutils import TestCaseQt
from silx.gui.utils.testutils import SignalListener
from silx.gui.widgets.ThreadPoolPushButton import ThreadPoolPushButton
from silx.utils.testutils import LoggingValidator


class TestThreadPoolPushButton(TestCaseQt):

    def setUp(self):
        super(TestThreadPoolPushButton, self).setUp()
        self._result = []

    def waitForPendingOperations(self, object):
        for i in range(50):
            if not object.hasPendingOperations():
                break
            self.qWait(10)
        else:
            raise RuntimeError("Still waiting for a pending operation")

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
        self.waitForPendingOperations(button)

    def testMultiExecution(self):
        button = ThreadPoolPushButton()
        button.setCallable(self._trace, "a", 0)
        number = qt.silxGlobalThreadPool().maxThreadCount()
        for _ in range(number):
            button.executeCallable()
        self.waitForPendingOperations(button)
        self.assertListEqual(self._result, ["a"] * number)

    def testSaturateThreadPool(self):
        button = ThreadPoolPushButton()
        button.setCallable(self._trace, "a", 100)
        number = qt.silxGlobalThreadPool().maxThreadCount() * 2
        for _ in range(number):
            button.executeCallable()
        self.waitForPendingOperations(button)
        self.assertListEqual(self._result, ["a"] * number)

    def testSuccess(self):
        listener = SignalListener()
        button = ThreadPoolPushButton()
        button.setCallable(self._compute)
        button.beforeExecuting.connect(listener.partial(test="be"))
        button.started.connect(listener.partial(test="s"))
        button.succeeded.connect(listener.partial(test="result"))
        button.failed.connect(listener.partial(test="Unexpected exception"))
        button.finished.connect(listener.partial(test="f"))
        button.executeCallable()
        self.qapp.processEvents()
        time.sleep(0.1)
        self.qapp.processEvents()
        result = listener.karguments(argumentName="test")
        self.assertListEqual(result, ["be", "s", "result", "f"])

    def testFail(self):
        listener = SignalListener()
        button = ThreadPoolPushButton()
        button.setCallable(self._computeFail)
        button.beforeExecuting.connect(listener.partial(test="be"))
        button.started.connect(listener.partial(test="s"))
        button.succeeded.connect(listener.partial(test="Unexpected success"))
        button.failed.connect(listener.partial(test="exception"))
        button.finished.connect(listener.partial(test="f"))
        with LoggingValidator('silx.gui.widgets.ThreadPoolPushButton', error=1):
            button.executeCallable()
            self.qapp.processEvents()
            time.sleep(0.1)
            self.qapp.processEvents()
        result = listener.karguments(argumentName="test")
        self.assertListEqual(result, ["be", "s", "exception", "f"])
        listener.clear()
