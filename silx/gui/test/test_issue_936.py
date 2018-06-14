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

import unittest
from .. import qt
from . import utils
import time


class Test(utils.TestCaseQt):

    def test_globalthreadpool(self):
        qt.QThreadPool.globalInstance()

    def test_threadpool(self):
        class Runnable(qt.QRunnable):
            def run(self):
                print("AAAAAAAAAAAAA")
            def autoDelete(self):
                return True
        tp = qt.QThreadPool.globalInstance()
        tp.start(Runnable())

    def test_saturate_threadpool(self):
        class Runnable(qt.QRunnable):
            def run(self):
                print("AAAAAAAAAAAAA")
            def autoDelete(self):
                return True
        tp = qt.QThreadPool.globalInstance()
        for i in range(tp.maxThreadCount() * 5):
            tp.start(Runnable())


def suite_1():
    test_suite = unittest.TestSuite()
    test_suite.addTest(Test("test_globalthreadpool"))
    return test_suite


def suite_2():
    test_suite = unittest.TestSuite()

    from silx.gui.widgets.test import test_threadpoolpushbutton
    from . import test_icons

    test_suite.addTest(test_threadpoolpushbutton.TestThreadPoolPushButton("testMultiExecution"))
    test_suite.addTest(test_icons.TestIcons("testPngIcon"))
    test_suite.addTest(test_icons.TestIcons("testCacheReleased"))

    return test_suite


def suite_3():
    loadTests = unittest.defaultTestLoader.loadTestsFromTestCase
    test_suite = unittest.TestSuite()

    from silx.gui.hdf5.test import test_hdf5
    test_suite.addTest(loadTests(test_hdf5.TestHdf5TreeModel))
    test_suite.addTest(loadTests(test_hdf5.TestHdf5TreeView))

    from silx.gui.widgets.test import test_threadpoolpushbutton
    test_threadpoolpushbutton.suite()

    return test_suite


def main():
    unittest.main(defaultTest='suite')

if __name__ == '__main__':
    main()
