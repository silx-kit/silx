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
# ###########################################################################*/
"""Test SceneWidget"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "06/03/2019"


import unittest

import numpy

from silx.utils.testutils import ParametricTestCase
from silx.gui.utils.testutils import TestCaseQt
from silx.gui import qt

from silx.gui.plot3d.SceneWidget import SceneWidget


class TestSceneWidget(TestCaseQt, ParametricTestCase):
    """Tests SceneWidget picking feature"""

    def setUp(self):
        super(TestSceneWidget, self).setUp()
        self.widget = SceneWidget()
        self.widget.show()
        self.qWaitForWindowExposed(self.widget)

    def tearDown(self):
        self.qapp.processEvents()
        self.widget.setAttribute(qt.Qt.WA_DeleteOnClose)
        self.widget.close()
        del self.widget
        super(TestSceneWidget, self).tearDown()

    def testFogEffect(self):
        """Test fog effect on scene primitive"""
        image = self.widget.addImage(numpy.arange(100).reshape(10, 10))
        scatter = self.widget.add3DScatter(*numpy.random.random(4000).reshape(4, -1))
        scatter.setTranslation(10, 10)
        scatter.setScale(10, 10, 10)

        self.widget.resetZoom('front')
        self.qapp.processEvents()

        self.widget.setFogMode(self.widget.FogMode.LINEAR)
        self.qapp.processEvents()

        self.widget.setFogMode(self.widget.FogMode.NONE)
        self.qapp.processEvents()


def suite():
    testsuite = unittest.TestSuite()
    testsuite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(
            TestSceneWidget))
    return testsuite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
