# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2019 European Synchrotron Radiation Facility
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
"""Tests of Enum class with extra class methods"""

from __future__ import absolute_import

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "29/04/2019"


import sys
import unittest

import enum
from silx.utils.enum import Enum


class TestEnum(unittest.TestCase):
    """Tests for enum module."""

    def test(self):
        """Test with Enum"""
        class Success(Enum):
            A = 1
            B = 'B'
        self._check_enum_content(Success)

    @unittest.skipIf(sys.version_info.major <= 2, 'Python3 only')
    def test(self):
        """Test Enum with member redefinition"""
        with self.assertRaises(TypeError):
            class Failure(Enum):
                A = 1
                A = 'B'

    def test_unique(self):
        """Test with enum.unique"""
        with self.assertRaises(ValueError):
            @enum.unique
            class Failure(Enum):
                A = 1
                B = 1

        @enum.unique
        class Success(Enum):
            A = 1
            B = 'B'
        self._check_enum_content(Success)

    def _check_enum_content(self, enum_):
        """Check that the content of an enum is: <A: 1, B: 2>.

        :param Enum enum_:
        """
        self.assertEqual(enum_.members(), (enum_.A, enum_.B))
        self.assertEqual(enum_.names(), ('A', 'B'))
        self.assertEqual(enum_.values(), (1, 'B'))

        self.assertEqual(enum_.from_value(1), enum_.A)
        self.assertEqual(enum_.from_value('B'), enum_.B)
        with self.assertRaises(ValueError):
            enum_.from_value(3)


def suite():
    test_suite = unittest.TestSuite()
    loadTests = unittest.defaultTestLoader.loadTestsFromTestCase
    test_suite.addTest(loadTests(TestEnum))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
