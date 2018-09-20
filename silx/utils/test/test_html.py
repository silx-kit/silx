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
"""Tests for html module"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "19/09/2016"


import unittest
from .. import html


class TestHtml(unittest.TestCase):
    """Tests for html module."""

    def testLtGt(self):
        result = html.escape("<html>'\"")
        self.assertEqual("&lt;html&gt;&apos;&quot;", result)

    def testLtAmpGt(self):
        # '&' have to be escaped first
        result = html.escape("<&>")
        self.assertEqual("&lt;&amp;&gt;", result)

    def testNoQuotes(self):
        result = html.escape("\"m&m's\"", quote=False)
        self.assertEqual("\"m&amp;m's\"", result)


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestHtml))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
