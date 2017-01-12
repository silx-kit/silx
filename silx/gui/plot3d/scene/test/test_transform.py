# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2015-2017 European Synchrotron Radiation Facility
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

from __future__ import absolute_import, division, unicode_literals

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "05/01/2017"


import numpy
import unittest

from silx.gui.plot3d.scene import transform


class TestTransformList(unittest.TestCase):

    def assertSameArrays(self, a, b):
        return self.assertTrue(numpy.allclose(a, b, atol=1e-06))

    def testTransformList(self):
        """Minimalistic test of TransformList"""
        transforms = transform.TransformList()
        refmatrix = numpy.identity(4, dtype=numpy.float32)
        self.assertSameArrays(refmatrix, transforms.matrix)

        # Append translate
        transforms.append(transform.Translate(1., 1., 1.))
        refmatrix = numpy.array(((1., 0., 0., 1.),
                                 (0., 1., 0., 1.),
                                 (0., 0., 1., 1.),
                                 (0., 0., 0., 1.)), dtype=numpy.float32)
        self.assertSameArrays(refmatrix, transforms.matrix)

        # Extend scale
        transforms.extend([transform.Scale(0.1, 2., 1.)])
        refmatrix = numpy.dot(refmatrix,
                              numpy.array(((0.1, 0., 0., 0.),
                                           (0., 2., 0., 0.),
                                           (0., 0., 1., 0.),
                                           (0., 0., 0., 1.)),
                                          dtype=numpy.float32))
        self.assertSameArrays(refmatrix, transforms.matrix)

        # Insert rotate
        transforms.insert(0, transform.Rotate(360.))
        self.assertSameArrays(refmatrix, transforms.matrix)

        # Update translate and check for listener called
        self._callCount = 0

        def listener(source):
            self._callCount += 1
        transforms.addListener(listener)

        transforms[1].tx += 1
        self.assertEqual(self._callCount, 1)


def suite():
    testsuite = unittest.TestSuite()
    testsuite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestTransformList))
    return testsuite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
