# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2018 European Synchrotron Radiation Facility
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
"""Test PositionInfoWidget"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "03/10/2018"


import unittest

import numpy

from silx.gui.test.utils import TestCaseQt
from silx.gui import qt

from silx.gui.plot3d.SceneWidget import SceneWidget
from silx.gui.plot3d.tools.PositionInfoWidget import PositionInfoWidget


class TestPositionInfoWidget(TestCaseQt):
    """Tests PositionInfoWidget"""

    def setUp(self):
        super(TestPositionInfoWidget, self).setUp()
        self.sceneWidget = SceneWidget()
        self.sceneWidget.resize(300, 300)
        self.sceneWidget.show()

        self.positionInfoWidget = PositionInfoWidget()
        self.positionInfoWidget.setSceneWidget(self.sceneWidget)
        self.positionInfoWidget.show()
        self.qWaitForWindowExposed(self.positionInfoWidget)

        # self.qWaitForWindowExposed(self.widget)

    def tearDown(self):
        self.qapp.processEvents()

        self.sceneWidget.setAttribute(qt.Qt.WA_DeleteOnClose)
        self.sceneWidget.close()
        del self.sceneWidget

        self.positionInfoWidget.setAttribute(qt.Qt.WA_DeleteOnClose)
        self.positionInfoWidget.close()
        del self.positionInfoWidget
        super(TestPositionInfoWidget, self).tearDown()

    def test(self):
        """Test PositionInfoWidget"""
        self.assertIs(self.positionInfoWidget.getSceneWidget(),
                      self.sceneWidget)

        data = numpy.arange(100)
        self.sceneWidget.add2DScatter(x=data, y=data, value=data)
        self.sceneWidget.resetZoom('front')

        # Double click at the center
        self.mouseDClick(self.sceneWidget, button=qt.Qt.LeftButton)

        # Clear displayed value
        self.positionInfoWidget.clear()

        # Update info from API
        self.positionInfoWidget.pick(x=10, y=10)

        # Remove SceneWidget
        self.positionInfoWidget.setSceneWidget(None)


def suite():
    testsuite = unittest.TestSuite()
    testsuite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(
            TestPositionInfoWidget))
    return testsuite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
