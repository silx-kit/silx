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
"""Test SceneWindow"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "22/03/2019"


import unittest

import numpy

from silx.utils.testutils import ParametricTestCase
from silx.gui.utils.testutils import TestCaseQt
from silx.gui import qt

from silx.gui.plot3d.SceneWindow import SceneWindow


class TestSceneWindow(TestCaseQt, ParametricTestCase):
    """Tests SceneWidget picking feature"""

    def setUp(self):
        super(TestSceneWindow, self).setUp()
        self.window = SceneWindow()
        self.window.show()
        self.qWaitForWindowExposed(self.window)

    def tearDown(self):
        self.qapp.processEvents()
        self.window.setAttribute(qt.Qt.WA_DeleteOnClose)
        self.window.close()
        del self.window
        super(TestSceneWindow, self).tearDown()

    def testAdd(self):
        """Test add basic scene primitive"""
        sceneWidget = self.window.getSceneWidget()
        items = []

        # RGB image
        image = sceneWidget.addImage(numpy.random.random(
            10*10*3).astype(numpy.float32).reshape(10, 10, 3))
        image.setLabel('RGB image')
        items.append(image)
        self.assertEqual(sceneWidget.getItems(), tuple(items))

        # Data image
        image = sceneWidget.addImage(
            numpy.arange(100, dtype=numpy.float32).reshape(10, 10))
        image.setTranslation(10.)
        items.append(image)
        self.assertEqual(sceneWidget.getItems(), tuple(items))

        # 2D scatter
        scatter = sceneWidget.add2DScatter(
            *numpy.random.random(3000).astype(numpy.float32).reshape(3, -1),
            index=0)
        scatter.setTranslation(0, 10)
        scatter.setScale(10, 10, 10)
        items.insert(0, scatter)
        self.assertEqual(sceneWidget.getItems(), tuple(items))

        # 3D scatter
        scatter = sceneWidget.add3DScatter(
            *numpy.random.random(4000).astype(numpy.float32).reshape(4, -1))
        scatter.setTranslation(10, 10)
        scatter.setScale(10, 10, 10)
        items.append(scatter)
        self.assertEqual(sceneWidget.getItems(), tuple(items))

        # 3D array of float
        volume = sceneWidget.addVolume(
            numpy.arange(10**3, dtype=numpy.float32).reshape(10, 10, 10))
        volume.setTranslation(0, 0, 10)
        volume.setRotation(45, (0, 0, 1))
        volume.addIsosurface(500, 'red')
        volume.getCutPlanes()[0].getColormap().setName('viridis')
        items.append(volume)
        self.assertEqual(sceneWidget.getItems(), tuple(items))

        # 3D array of complex
        volume = sceneWidget.addVolume(
            numpy.arange(10**3).reshape(10, 10, 10).astype(numpy.complex64))
        volume.setTranslation(10, 0, 10)
        volume.setRotation(45, (0, 0, 1))
        volume.setComplexMode(volume.ComplexMode.REAL)
        volume.addIsosurface(500, (1., 0., 0., .5))
        items.append(volume)
        self.assertEqual(sceneWidget.getItems(), tuple(items))

        sceneWidget.resetZoom('front')
        self.qapp.processEvents()

    def testChangeContent(self):
        """Test add/remove/clear items"""
        sceneWidget = self.window.getSceneWidget()
        items = []

        # Add 2 images
        image = numpy.arange(100, dtype=numpy.float32).reshape(10, 10)
        items.append(sceneWidget.addImage(image))
        items.append(sceneWidget.addImage(image))
        self.qapp.processEvents()
        self.assertEqual(sceneWidget.getItems(), tuple(items))

        # Clear
        sceneWidget.clearItems()
        self.qapp.processEvents()
        self.assertEqual(sceneWidget.getItems(), ())

        # Add 2 images and remove first one
        image = numpy.arange(100, dtype=numpy.float32).reshape(10, 10)
        sceneWidget.addImage(image)
        items = (sceneWidget.addImage(image),)
        self.qapp.processEvents()

        sceneWidget.removeItem(sceneWidget.getItems()[0])
        self.qapp.processEvents()
        self.assertEqual(sceneWidget.getItems(), items)

    def testColors(self):
        """Test setting scene colors"""
        sceneWidget = self.window.getSceneWidget()

        color = qt.QColor(128, 128, 128)
        sceneWidget.setBackgroundColor(color)
        self.assertEqual(sceneWidget.getBackgroundColor(), color)

        color = qt.QColor(0, 0, 0)
        sceneWidget.setForegroundColor(color)
        self.assertEqual(sceneWidget.getForegroundColor(), color)

        color = qt.QColor(255, 0, 0)
        sceneWidget.setTextColor(color)
        self.assertEqual(sceneWidget.getTextColor(), color)

        color = qt.QColor(0, 255, 0)
        sceneWidget.setHighlightColor(color)
        self.assertEqual(sceneWidget.getHighlightColor(), color)

        self.qapp.processEvents()

    def testInteractiveMode(self):
        """Test changing interactive mode"""
        sceneWidget = self.window.getSceneWidget()
        center = numpy.array((sceneWidget.width() //2, sceneWidget.height() // 2))

        self.mouseMove(sceneWidget, pos=center)
        self.mouseClick(sceneWidget, qt.Qt.LeftButton, pos=center)

        volume = sceneWidget.addVolume(
            numpy.arange(10**3).astype(numpy.float32).reshape(10, 10, 10))
        sceneWidget.selection().setCurrentItem( volume.getCutPlanes()[0])
        sceneWidget.resetZoom('side')

        for mode in (None, 'rotate', 'pan', 'panSelectedPlane'):
            with self.subTest(mode=mode):
                sceneWidget.setInteractiveMode(mode)
                self.qapp.processEvents()
                self.assertEqual(sceneWidget.getInteractiveMode(), mode)

                self.mouseMove(sceneWidget, pos=center)
                self.mousePress(sceneWidget, qt.Qt.LeftButton, pos=center)
                self.mouseMove(sceneWidget, pos=center-10)
                self.mouseMove(sceneWidget, pos=center-20)
                self.mouseRelease(sceneWidget, qt.Qt.LeftButton, pos=center-20)

                self.keyPress(sceneWidget, qt.Qt.Key_Control)
                self.mouseMove(sceneWidget, pos=center)
                self.mousePress(sceneWidget, qt.Qt.LeftButton, pos=center)
                self.mouseMove(sceneWidget, pos=center-10)
                self.mouseMove(sceneWidget, pos=center-20)
                self.mouseRelease(sceneWidget, qt.Qt.LeftButton, pos=center-20)
                self.keyRelease(sceneWidget, qt.Qt.Key_Control)


def suite():
    testsuite = unittest.TestSuite()
    testsuite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(
            TestSceneWindow))
    return testsuite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
