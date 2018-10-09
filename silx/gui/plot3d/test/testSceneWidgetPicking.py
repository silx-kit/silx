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
"""Test SceneWidget picking feature"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "03/10/2018"


import unittest

import numpy

from silx.utils.testutils import ParametricTestCase
from silx.gui.test.utils import TestCaseQt
from silx.gui import qt

from silx.gui.plot3d.SceneWidget import SceneWidget, items


class TestSceneWidgetPicking(TestCaseQt, ParametricTestCase):
    """Tests SceneWidget picking feature"""

    def setUp(self):
        super(TestSceneWidgetPicking, self).setUp()
        self.widget = SceneWidget()
        self.widget.resize(300, 300)
        self.widget.show()
        # self.qWaitForWindowExposed(self.widget)

    def tearDown(self):
        self.qapp.processEvents()
        self.widget.setAttribute(qt.Qt.WA_DeleteOnClose)
        self.widget.close()
        del self.widget
        super(TestSceneWidgetPicking, self).tearDown()

    def _widgetCenter(self):
        """Returns widget center"""
        size = self.widget.size()
        return size.width() // 2, size.height() // 2

    def testPickImage(self):
        """Test picking of ImageData and ImageRgba items"""
        imageData = items.ImageData()
        imageData.setData(numpy.arange(100).reshape(10, 10))

        imageRgba = items.ImageRgba()
        imageRgba.setData(
            numpy.arange(300, dtype=numpy.uint8).reshape(10, 10, 3))

        for item in (imageData, imageRgba):
            with self.subTest(item=item.__class__.__name__):
                # Add item
                self.widget.clearItems()
                self.widget.addItem(item)
                self.widget.resetZoom('front')
                self.qapp.processEvents()

                # Picking on data (at widget center)
                picking = list(self.widget.pickItems(*self._widgetCenter()))

                self.assertEqual(len(picking), 1)
                self.assertIs(picking[0].getItem(), item)
                self.assertEqual(picking[0].getPositions('ndc').shape, (1, 3))
                data = picking[0].getData()
                self.assertEqual(len(data), 1)
                self.assertTrue(numpy.array_equal(
                    data,
                    item.getData()[picking[0].getIndices()]))

                # Picking outside data
                picking = list(self.widget.pickItems(1, 1))
                self.assertEqual(len(picking), 0)

    def testPickScatter(self):
        """Test picking of Scatter2D and Scatter3D items"""
        data = numpy.arange(100)

        scatter2d = items.Scatter2D()
        scatter2d.setData(x=data, y=data, value=data)

        scatter3d = items.Scatter3D()
        scatter3d.setData(x=data, y=data, z=data, value=data)

        for item in (scatter2d, scatter3d):
            with self.subTest(item=item.__class__.__name__):
                # Add item
                self.widget.clearItems()
                self.widget.addItem(item)
                self.widget.resetZoom('front')
                self.qapp.processEvents()

                # Picking on data (at widget center)
                picking = list(self.widget.pickItems(*self._widgetCenter()))

                self.assertEqual(len(picking), 1)
                self.assertIs(picking[0].getItem(), item)
                nbPos = len(picking[0].getPositions('ndc'))
                data = picking[0].getData()
                self.assertEqual(nbPos, len(data))
                self.assertTrue(numpy.array_equal(
                    data,
                    item.getValues()[picking[0].getIndices()]))

                # Picking outside data
                picking = list(self.widget.pickItems(1, 1))
                self.assertEqual(len(picking), 0)

    def testPickScalarField3D(self):
        """Test picking of volume CutPlane and Isosurface items"""
        volume = self.widget.add3DScalarField(
            numpy.arange(10**3, dtype=numpy.float32).reshape(10, 10, 10))
        self.widget.resetZoom('front')

        cutplane = volume.getCutPlanes()[0]
        cutplane.getColormap().setVRange(0, 100)
        cutplane.setNormal((0, 0, 1))

        # Picking on data without anything displayed
        cutplane.setVisible(False)
        picking = list(self.widget.pickItems(*self._widgetCenter()))
        self.assertEqual(len(picking), 0)

        # Picking on data with the cut plane
        cutplane.setVisible(True)
        picking = list(self.widget.pickItems(*self._widgetCenter()))

        self.assertEqual(len(picking), 1)
        self.assertIs(picking[0].getItem(), cutplane)
        data = picking[0].getData()
        self.assertEqual(len(data), 1)
        self.assertEqual(picking[0].getPositions().shape, (1, 3))
        self.assertTrue(numpy.array_equal(
            data,
            volume.getData(copy=False)[picking[0].getIndices()]))

        # Picking on data with an isosurface
        isosurface = volume.addIsosurface(level=500, color=(1., 0., 0., .5))
        picking = list(self.widget.pickItems(*self._widgetCenter()))
        self.assertEqual(len(picking), 2)
        self.assertIs(picking[0].getItem(), cutplane)
        self.assertIs(picking[1].getItem(), isosurface)
        self.assertEqual(picking[1].getPositions().shape, (1, 3))
        data = picking[1].getData()
        self.assertEqual(len(data), 1)
        self.assertTrue(numpy.array_equal(
            data,
            volume.getData(copy=False)[picking[1].getIndices()]))

        # Picking outside data
        picking = list(self.widget.pickItems(1, 1))
        self.assertEqual(len(picking), 0)

    def testPickMesh(self):
        """Test picking of Mesh items"""

        triangles = items.Mesh()
        triangles.setData(
            position=((0, 0, 0), (1, 0, 0), (1, 1, 0),
                      (0, 0, 0), (1, 1, 0), (0, 1, 0)),
            color=(1, 0, 0, 1),
            mode='triangles')
        triangleStrip = items.Mesh()
        triangleStrip.setData(
            position=(((1, 0, 0), (0, 0, 0), (1, 1, 0), (0, 1, 0))),
            color=(0, 1, 0, 1),
            mode='triangle_strip')
        triangleFan = items.Mesh()
        triangleFan.setData(
            position=((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)),
            color=(0, 0, 1, 1),
            mode='fan')

        for item in (triangles, triangleStrip, triangleFan):
            with self.subTest(mode=item.getDrawMode()):
                # Add item
                self.widget.clearItems()
                self.widget.addItem(item)
                self.widget.resetZoom('front')
                self.qapp.processEvents()

                # Picking on data (at widget center)
                picking = list(self.widget.pickItems(*self._widgetCenter()))

                self.assertEqual(len(picking), 1)
                self.assertIs(picking[0].getItem(), item)
                nbPos = len(picking[0].getPositions())
                data = picking[0].getData()
                self.assertEqual(nbPos, len(data))
                self.assertTrue(numpy.array_equal(
                    data,
                    item.getPositionData()[picking[0].getIndices()]))

                # Picking outside data
                picking = list(self.widget.pickItems(1, 1))
                self.assertEqual(len(picking), 0)

    def testPickCylindricalMesh(self):
        """Test picking of Box, Cylinder and Hexagon items"""

        positions = numpy.array(((0., 0., 0.), (1., 1., 0.), (2., 2., 0.)))
        box = items.Box()
        box.setData(position=positions)
        cylinder = items.Cylinder()
        cylinder.setData(position=positions)
        hexagon = items.Hexagon()
        hexagon.setData(position=positions)

        for item in (box, cylinder, hexagon):
            with self.subTest(item=item.__class__.__name__):
                # Add item
                self.widget.clearItems()
                self.widget.addItem(item)
                self.widget.resetZoom('front')
                self.qapp.processEvents()

                # Picking on data (at widget center)
                picking = list(self.widget.pickItems(*self._widgetCenter()))

                self.assertEqual(len(picking), 1)
                self.assertIs(picking[0].getItem(), item)
                nbPos = len(picking[0].getPositions())
                data = picking[0].getData()
                print(item.__class__.__name__, [positions[1]], data)
                self.assertTrue(numpy.all(numpy.equal(positions[1], data)))
                self.assertEqual(nbPos, len(data))
                self.assertTrue(numpy.array_equal(
                    data,
                    item.getPosition()[picking[0].getIndices()]))

                # Picking outside data
                picking = list(self.widget.pickItems(1, 1))
                self.assertEqual(len(picking), 0)


def suite():
    testsuite = unittest.TestSuite()
    testsuite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(
            TestSceneWidgetPicking))
    return testsuite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
