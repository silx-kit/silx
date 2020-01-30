# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2020 European Synchrotron Radiation Facility
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
"""Tests for the silx.gui.utils.glutils module."""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "15/01/2020"


import logging
import unittest
from silx.gui.utils.glutils import isOpenGLAvailable


_logger = logging.getLogger(__name__)


class TestIsOpenGLAvailable(unittest.TestCase):
    """Test isOpenGLAvailable"""

    def test(self):
        for version in ((2, 1), (2, 1), (1000, 1)):
            with self.subTest(version=version):
                result = isOpenGLAvailable(version=version)
                _logger.info("isOpenGLAvailable returned: %s", str(result))
                if version[0] == 1000:
                    self.assertFalse(result)
                if not result:
                    self.assertFalse(result.status)
                    self.assertTrue(len(result.error) > 0)
                else:
                    self.assertTrue(result.status)
                    self.assertTrue(len(result.error) == 0)


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(
        TestIsOpenGLAvailable))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
