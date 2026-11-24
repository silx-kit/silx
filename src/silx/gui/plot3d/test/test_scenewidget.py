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


import pytest

import numpy

from silx.gui import qt
from silx.gui.plot3d.SceneWidget import SceneWidget


@pytest.mark.usefixtures("use_opengl")
class TestPlot3DWidget:
    def testFogMode(self, qWidgetFactory, qapp):
        sceneWidget = qWidgetFactory(SceneWidget)
        _ = sceneWidget.addImage(numpy.arange(100).reshape(10, 10))
        scatter = sceneWidget.add3DScatter(*numpy.random.random(400).reshape(4, -1))
        scatter.setTranslation(10, 10)
        scatter.setScale(10, 10, 10)

        sceneWidget.resetZoom("front")
        qapp.processEvents()
        assert sceneWidget.getFogMode() == sceneWidget.FogMode.NONE

        sceneWidget.setFogMode(sceneWidget.FogMode.LINEAR)
        qapp.processEvents()
        assert sceneWidget.getFogMode() == sceneWidget.FogMode.LINEAR

        sceneWidget.setFogMode(sceneWidget.FogMode.NONE)
        qapp.processEvents()
        assert sceneWidget.getFogMode() == sceneWidget.FogMode.NONE

    def testFirstRenderUsesResetCamera(self, qapp, qapp_utils):
        # On the first display the camera is reset to show the whole scene
        # (Plot3DWidget.paintGL). The first rendered frame must already use
        # this camera, so the reset view does not need a further repaint.
        sceneWidget = SceneWidget()
        sceneWidget.resize(500, 400)
        sceneWidget.add3DScatter(*numpy.random.default_rng(0).random((4, 500)) * 10)

        camera = sceneWidget.viewport.camera.extrinsic
        rendered = []
        originalRender = sceneWidget._window.render

        def tracedRender(*args, **kwargs):
            rendered.append(numpy.array(camera.position))
            originalRender(*args, **kwargs)

        sceneWidget._window.render = tracedRender

        sceneWidget.show()
        qapp_utils.qWaitForWindowExposed(sceneWidget)
        qapp.processEvents()

        sceneWidget.close()
        qapp.processEvents()

        # The first frame is rendered with the reset camera.
        assert rendered
        assert numpy.allclose(rendered[0], camera.position)

    def testLightMode(self, qWidgetFactory, qapp):
        sceneWidget = qWidgetFactory(SceneWidget)
        _ = sceneWidget.addImage(numpy.arange(100).reshape(10, 10))
        scatter = sceneWidget.add3DScatter(*numpy.random.random(400).reshape(4, -1))
        scatter.setTranslation(10, 10)
        scatter.setScale(10, 10, 10)
        sceneWidget.resetZoom("front")
        qapp.processEvents()

        assert sceneWidget.getLightMode() == "directional"

        sceneWidget.setLightMode(None)
        qapp.processEvents()
        assert sceneWidget.getLightMode() is None

        sceneWidget.setLightMode("directional")
        qapp.processEvents()
        assert sceneWidget.getLightMode() == "directional"

    def testWheelZoomStepsShrinkWithoutRepaint(self, qWidgetFactory, qapp):
        # Scene with an object under the zoom position and a rendered frame
        sceneWidget = qWidgetFactory(SceneWidget)
        sceneWidget.resize(600, 600)
        sceneWidget.add3DScatter(*numpy.random.default_rng(0).random((4, 2000)) * 10)
        sceneWidget.resetZoom()
        for _ in range(30):
            qapp.processEvents()

        camera = sceneWidget.viewport.camera.extrinsic
        steps = []
        for _ in range(4):
            before = numpy.array(camera.position)
            pos = qt.QPoint(100, 100)
            # Send the wheel event at once, without a repaint after the previous one
            event = qt.QWheelEvent(
                qt.QPointF(pos),
                qt.QPointF(sceneWidget.mapToGlobal(pos)),
                qt.QPoint(0, 0),
                qt.QPoint(0, 120),
                qt.Qt.NoButton,
                qt.Qt.NoModifier,
                qt.Qt.NoScrollPhase,
                False,
            )
            qt.QApplication.sendEvent(sceneWidget, event)
            steps.append(numpy.linalg.norm(camera.position - before))

        # A zoom step moves the camera by a fraction of its distance to the object
        # under the cursor. This distance comes from the depth buffer, which must
        # show the scene as of the previous zoom step, even without a repaint in
        # between. So each step is smaller than the one before.
        assert steps[1] < steps[0]
        assert steps[2] < steps[1]
        assert steps[3] < steps[2]
