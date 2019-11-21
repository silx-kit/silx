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
"""Tests for testutils module"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "18/11/2019"


import unittest
import logging
from .. import testutils


class TestTestLogging(unittest.TestCase):
    """Tests for TestLogging."""

    def testRight(self):
        logger = logging.getLogger(__name__ + "testRight")
        listener = testutils.TestLogging(logger, error=1)
        with listener:
            logger.error("expected")
            logger.info("ignored")

    def testCustomLevel(self):
        logger = logging.getLogger(__name__ + "testCustomLevel")
        listener = testutils.TestLogging(logger, error=1)
        with listener:
            logger.error("expected")
            logger.log(666, "custom level have to be ignored")

    def testWrong(self):
        logger = logging.getLogger(__name__ + "testWrong")
        listener = testutils.TestLogging(logger, error=1)
        with self.assertRaises(RuntimeError):
            with listener:
                logger.error("expected")
                logger.error("not expected")

    def testManyErrors(self):
        logger = logging.getLogger(__name__ + "testManyErrors")
        listener = testutils.TestLogging(logger, error=1, warning=2)
        with self.assertRaises(RuntimeError):
            with listener:
                pass

    def testCanBeChecked(self):
        logger = logging.getLogger(__name__ + "testCanBreak")
        listener = testutils.TestLogging(logger, error=1, warning=2)
        with self.assertRaises(RuntimeError):
            with listener:
                logger.error("aaa")
                logger.warning("aaa")
                self.assertFalse(listener.can_be_checked())
                logger.error("aaa")
                # Here we know that it's already wrong without a big cost
                self.assertTrue(listener.can_be_checked())

    def testWithAs(self):
        logger = logging.getLogger(__name__ + "testCanBreak")
        with testutils.TestLogging(logger) as listener:
            logger.error("aaa")
            self.assertIsNotNone(listener)


def suite():
    loadTests = unittest.defaultTestLoader.loadTestsFromTestCase
    test_suite = unittest.TestSuite()
    test_suite.addTest(loadTests(TestTestLogging))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
