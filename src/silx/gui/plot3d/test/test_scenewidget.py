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
