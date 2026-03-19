# /*##########################################################################
#
# Copyright (c) 2019-2021 European Synchrotron Radiation Facility
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
"""Test SceneWindow with OpenGL and pygfx backends"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "22/03/2019"


import pytest

import numpy

from silx.utils.testutils import ParametricTestCase
from silx.gui.utils.testutils import TestCaseQt
from silx.gui import qt

from silx.gui.plot3d.SceneWindow import SceneWindow
from silx.gui.plot3d.items import HeightMapData, HeightMapRGBA

# --- Parametrized fixture for both backends ---


@pytest.fixture(
    params=[
        pytest.param(None, id="opengl"),
        pytest.param("pygfx", id="pygfx"),
    ]
)
def scene_window(request, qapp, test_options):
    """SceneWindow fixture parametrized by backend."""
    backend = request.param
    if backend is None and not test_options.WITH_GL_TEST:
        pytest.skip(test_options.WITH_GL_TEST_REASON)
    if backend == "pygfx" and not test_options.WITH_PYGFX_TEST:
        pytest.skip(test_options.WITH_PYGFX_TEST_REASON)

    window = SceneWindow(backend=backend)
    window.show()
    qapp.processEvents()
    yield window
    window.setAttribute(qt.Qt.WA_DeleteOnClose)
    window.close()
    qapp.processEvents()


# --- Tests for both backends ---


class TestSceneWindow:
    """Tests SceneWindow features shared across backends"""

    def test_add(self, scene_window, qapp):
        """Test add basic scene primitives"""
        sceneWidget = scene_window.getSceneWidget()
        items = []

        # RGB image
        image = sceneWidget.addImage(
            numpy.random.random(10 * 10 * 3).astype(numpy.float32).reshape(10, 10, 3)
        )
        image.setLabel("RGB image")
        items.append(image)
        assert sceneWidget.getItems() == tuple(items)

        # Data image
        image = sceneWidget.addImage(
            numpy.arange(100, dtype=numpy.float32).reshape(10, 10)
        )
        image.setTranslation(10.0)
        items.append(image)
        assert sceneWidget.getItems() == tuple(items)

        # 2D scatter
        scatter = sceneWidget.add2DScatter(
            *numpy.random.random(3000).astype(numpy.float32).reshape(3, -1), index=0
        )
        scatter.setTranslation(0, 10)
        scatter.setScale(10, 10, 10)
        items.insert(0, scatter)
        assert sceneWidget.getItems() == tuple(items)

        # 3D scatter
        scatter = sceneWidget.add3DScatter(
            *numpy.random.random(4000).astype(numpy.float32).reshape(4, -1)
        )
        scatter.setTranslation(10, 10)
        scatter.setScale(10, 10, 10)
        items.append(scatter)
        assert sceneWidget.getItems() == tuple(items)

        # 3D array of float
        volume = sceneWidget.addVolume(
            numpy.arange(10**3, dtype=numpy.float32).reshape(10, 10, 10)
        )
        volume.setTranslation(0, 0, 10)
        volume.setRotation(45, (0, 0, 1))
        volume.addIsosurface(500, "red")
        items.append(volume)
        assert sceneWidget.getItems() == tuple(items)

        # 3D array of complex
        volume = sceneWidget.addVolume(
            numpy.arange(10**3).reshape(10, 10, 10).astype(numpy.complex64)
        )
        volume.setTranslation(10, 0, 10)
        volume.setRotation(45, (0, 0, 1))
        volume.setComplexMode(volume.ComplexMode.REAL)
        volume.addIsosurface(500, (1.0, 0.0, 0.0, 0.5))
        items.append(volume)
        assert sceneWidget.getItems() == tuple(items)

        sceneWidget.resetZoom("front")
        qapp.processEvents()

    def test_change_content(self, scene_window, qapp):
        """Test add/remove/clear items"""
        sceneWidget = scene_window.getSceneWidget()
        items = []

        # Add 2 images
        image = numpy.arange(100, dtype=numpy.float32).reshape(10, 10)
        items.append(sceneWidget.addImage(image))
        items.append(sceneWidget.addImage(image))
        qapp.processEvents()
        assert sceneWidget.getItems() == tuple(items)

        # Clear
        sceneWidget.clearItems()
        qapp.processEvents()
        assert sceneWidget.getItems() == ()

        # Add 2 images and remove first one
        image = numpy.arange(100, dtype=numpy.float32).reshape(10, 10)
        sceneWidget.addImage(image)
        items = (sceneWidget.addImage(image),)
        qapp.processEvents()

        sceneWidget.removeItem(sceneWidget.getItems()[0])
        qapp.processEvents()
        assert sceneWidget.getItems() == items

    def test_colors(self, scene_window, qapp):
        """Test setting scene colors"""
        sceneWidget = scene_window.getSceneWidget()

        color = qt.QColor(128, 128, 128)
        sceneWidget.setBackgroundColor(color)
        assert sceneWidget.getBackgroundColor() == color

        color = qt.QColor(0, 0, 0)
        sceneWidget.setForegroundColor(color)
        assert sceneWidget.getForegroundColor() == color

        color = qt.QColor(255, 0, 0)
        sceneWidget.setTextColor(color)
        assert sceneWidget.getTextColor() == color

        color = qt.QColor(0, 255, 0)
        sceneWidget.setHighlightColor(color)
        assert sceneWidget.getHighlightColor() == color

        qapp.processEvents()

    def test_interactive_mode(self, scene_window, qapp):
        """Test changing interactive mode"""
        sceneWidget = scene_window.getSceneWidget()

        for mode in ("rotate", "pan"):
            sceneWidget.setInteractiveMode(mode)
            qapp.processEvents()
            assert sceneWidget.getInteractiveMode() == mode

    def test_model(self, scene_window, qapp):
        """Test that model is properly set up"""
        sceneWidget = scene_window.getSceneWidget()
        model = sceneWidget.model()
        assert model is not None
        assert model.rowCount() == 2  # Settings + Data

        # Add item and check model updates
        scatter = sceneWidget.add3DScatter(
            *numpy.random.random(4000).astype(numpy.float32).reshape(4, -1)
        )
        scatter.setLabel("Test scatter")

        # Data group should have children now
        data_index = model.index(1, 0)
        assert model.rowCount(data_index) > 0


# --- OpenGL-only tests ---


@pytest.mark.usefixtures("use_opengl")
class TestSceneWindowOpenGL(TestCaseQt, ParametricTestCase):
    """Tests specific to OpenGL backend"""

    def setUp(self):
        super().setUp()
        self.window = SceneWindow()
        self.window.show()
        self.qWaitForWindowExposed(self.window)

    def tearDown(self):
        self.qapp.processEvents()
        self.window.setAttribute(qt.Qt.WA_DeleteOnClose)
        self.window.close()
        del self.window
        super().tearDown()

    def testHeightMap(self):
        """Test height map items"""
        sceneWidget = self.window.getSceneWidget()

        height = numpy.arange(10000).reshape(100, 100) / 100.0

        for shape in ((100, 100), (4, 5), (150, 20), (110, 110)):
            with self.subTest(shape=shape):
                items = []

                # Colormapped data height map
                data = (
                    numpy.arange(numpy.prod(shape)).astype(numpy.float32).reshape(shape)
                )

                heightmap = HeightMapData()
                heightmap.setData(height)
                heightmap.setColormappedData(data)
                heightmap.getColormap().setName("viridis")
                items.append(heightmap)
                sceneWidget.addItem(heightmap)

                # RGBA height map
                colors = numpy.zeros(shape + (3,), dtype=numpy.float32)
                colors[:, :, 1] = numpy.random.random(shape)

                heightmap = HeightMapRGBA()
                heightmap.setData(height)
                heightmap.setColorData(colors)
                heightmap.setTranslation(100.0, 0.0, 0.0)
                items.append(heightmap)
                sceneWidget.addItem(heightmap)

                self.assertEqual(sceneWidget.getItems(), tuple(items))
                sceneWidget.resetZoom("front")
                self.qapp.processEvents()
                sceneWidget.clearItems()

    def testInteractiveMode(self):
        """Test changing interactive mode with mouse events"""
        sceneWidget = self.window.getSceneWidget()
        center = numpy.array((sceneWidget.width() // 2, sceneWidget.height() // 2))

        self.mouseMove(sceneWidget, pos=center)
        self.mouseClick(sceneWidget, qt.Qt.LeftButton, pos=center)

        volume = sceneWidget.addVolume(
            numpy.arange(10**3).astype(numpy.float32).reshape(10, 10, 10)
        )
        sceneWidget.selection().setCurrentItem(volume.getCutPlanes()[0])
        sceneWidget.resetZoom("side")

        for mode in (None, "rotate", "pan", "panSelectedPlane"):
            with self.subTest(mode=mode):
                sceneWidget.setInteractiveMode(mode)
                self.qapp.processEvents()
                self.assertEqual(sceneWidget.getInteractiveMode(), mode)

                self.mouseMove(sceneWidget, pos=center)
                self.mousePress(sceneWidget, qt.Qt.LeftButton, pos=center)
                self.mouseMove(sceneWidget, pos=center - 10)
                self.mouseMove(sceneWidget, pos=center - 20)
                self.mouseRelease(sceneWidget, qt.Qt.LeftButton, pos=center - 20)

                self.keyPress(sceneWidget, qt.Qt.Key_Control)
                self.mouseMove(sceneWidget, pos=center)
                self.mousePress(sceneWidget, qt.Qt.LeftButton, pos=center)
                self.mouseMove(sceneWidget, pos=center - 10)
                self.mouseMove(sceneWidget, pos=center - 20)
                self.mouseRelease(sceneWidget, qt.Qt.LeftButton, pos=center - 20)
                self.keyRelease(sceneWidget, qt.Qt.Key_Control)
