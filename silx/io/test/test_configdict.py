# coding: utf-8
#/*##########################################################################
# Copyright (C) 2016 European Synchrotron Radiation Facility
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
#############################################################################*/
"""Tests for configdict module"""

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "18/04/2016"

import numpy
import os
import tempfile
import unittest

from ..configdict import ConfigDict


class TestConfigDict(unittest.TestCase):
    def setUp(self):
        self.dir_path = tempfile.mkdtemp()
        self.ini_fname = os.path.join(self.dir_path, "test.ini")

    def tearDown(self):
        os.unlink(self.ini_fname)
        os.rmdir(self.dir_path)

    def testConfigDictIO(self):
        testdict = {
            'simple_types': {
                'float': 1.0,
                'int': 1,
                'string': 'Hello World',
            },
            'containers': {
                'list': [-1, 'string', 3.0, False],
                'array': numpy.array([1.0, 2.0, 3.0]),
                'dict': {
                    'key1': 'Hello World',
                    'key2': 2.0,
                }
            }
        }

        writeinstance = ConfigDict(initdict=testdict)
        writeinstance.write(self.ini_fname)

        #read the data back
        readinstance = ConfigDict()
        readinstance.read(self.ini_fname)

        testdictkeys = list(testdict.keys())
        readkeys = list(readinstance.keys())

        self.assertTrue(len(readkeys) == len(testdictkeys),
                        "Number of read keys not equal")

        for key in testdict["simple_types"]:
            original = testdict['simple_types'][key]
            read = readinstance['simple_types'][key]
            self.assertEqual(read, original,
                             "Read <%s> instead of <%s>" % (read, original))

        for key in testdict["containers"]:
            original = testdict["containers"][key]
            read = readinstance["containers"][key]
            if key == 'array':
                self.assertEqual(read.all(), original.all(),
                            "Read <%s> instead of <%s>" % (read, original))
            else:
                self.assertEqual(read, original, 
                                 "Read <%s> instead of <%s>" % (read, original))


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestConfigDict))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest="suite")