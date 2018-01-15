# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017-2018 European Synchrotron Radiation Facility
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
"""This module provides Plot3DAction controlling the viewpoint.

It provides QAction to rotate or pan a Plot3DWidget.
"""

from __future__ import absolute_import, division

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "03/10/2017"


import time
import logging

from silx.gui import qt
from silx.gui.icons import getQIcon
from .Plot3DAction import Plot3DAction


_logger = logging.getLogger(__name__)


class _SetViewpointAction(Plot3DAction):
    """Base class for actions setting a Plot3DWidget viewpoint

    :param parent: See :class:`QAction`
    :param str face: The name of the predefined viewpoint
    :param ~silx.gui.plot3d.Plot3DWidget.Plot3DWidget plot3d:
        Plot3DWidget the action is associated with
    """
    def __init__(self, parent, face, plot3d=None):
        super(_SetViewpointAction, self).__init__(parent, plot3d)
        assert face in ('side', 'front', 'back', 'left', 'right', 'top', 'bottom')
        self._face = face

        self.setIconVisibleInMenu(True)
        self.setCheckable(False)
        self.triggered[bool].connect(self._triggered)

    def _triggered(self, checked=False):
        plot3d = self.getPlot3DWidget()
        if plot3d is None:
            _logger.error(
                'Cannot start/stop rotation, no associated Plot3DWidget')
        else:
            plot3d.viewport.camera.extrinsic.reset(face=self._face)
            plot3d.centerScene()


class FrontViewpointAction(_SetViewpointAction):
    """QAction to set Plot3DWidget viewpoint to look from the front

    :param parent: See :class:`QAction`
    :param ~silx.gui.plot3d.Plot3DWidget.Plot3DWidget plot3d:
        Plot3DWidget the action is associated with
    """
    def __init__(self, parent, plot3d=None):
        super(FrontViewpointAction, self).__init__(parent, 'front', plot3d)

        self.setIcon(getQIcon('cube-front'))
        self.setText('Front')
        self.setToolTip('View along the -Z axis')


class BackViewpointAction(_SetViewpointAction):
    """QAction to set Plot3DWidget viewpoint to look from the back

    :param parent: See :class:`QAction`
    :param ~silx.gui.plot3d.Plot3DWidget.Plot3DWidget plot3d:
        Plot3DWidget the action is associated with
    """
    def __init__(self, parent, plot3d=None):
        super(BackViewpointAction, self).__init__(parent, 'back', plot3d)

        self.setIcon(getQIcon('cube-back'))
        self.setText('Back')
        self.setToolTip('View along the +Z axis')


class LeftViewpointAction(_SetViewpointAction):
    """QAction to set Plot3DWidget viewpoint to look from the left

    :param parent: See :class:`QAction`
    :param ~silx.gui.plot3d.Plot3DWidget.Plot3DWidget plot3d:
        Plot3DWidget the action is associated with
    """
    def __init__(self, parent, plot3d=None):
        super(LeftViewpointAction, self).__init__(parent, 'left', plot3d)

        self.setIcon(getQIcon('cube-left'))
        self.setText('Left')
        self.setToolTip('View along the +X axis')


class RightViewpointAction(_SetViewpointAction):
    """QAction to set Plot3DWidget viewpoint to look from the right

    :param parent: See :class:`QAction`
    :param ~silx.gui.plot3d.Plot3DWidget.Plot3DWidget plot3d:
        Plot3DWidget the action is associated with
    """
    def __init__(self, parent, plot3d=None):
        super(RightViewpointAction, self).__init__(parent, 'right', plot3d)

        self.setIcon(getQIcon('cube-right'))
        self.setText('Right')
        self.setToolTip('View along the -X axis')


class TopViewpointAction(_SetViewpointAction):
    """QAction to set Plot3DWidget viewpoint to look from the top

    :param parent: See :class:`QAction`
    :param ~silx.gui.plot3d.Plot3DWidget.Plot3DWidget plot3d:
        Plot3DWidget the action is associated with
    """
    def __init__(self, parent, plot3d=None):
        super(TopViewpointAction, self).__init__(parent, 'top', plot3d)

        self.setIcon(getQIcon('cube-top'))
        self.setText('Top')
        self.setToolTip('View along the -Y axis')


class BottomViewpointAction(_SetViewpointAction):
    """QAction to set Plot3DWidget viewpoint to look from the bottom

    :param parent: See :class:`QAction`
    :param ~silx.gui.plot3d.Plot3DWidget.Plot3DWidget plot3d:
        Plot3DWidget the action is associated with
    """
    def __init__(self, parent, plot3d=None):
        super(BottomViewpointAction, self).__init__(parent, 'bottom', plot3d)

        self.setIcon(getQIcon('cube-bottom'))
        self.setText('Bottom')
        self.setToolTip('View along the +Y axis')


class SideViewpointAction(_SetViewpointAction):
    """QAction to set Plot3DWidget viewpoint to look from the side

    :param parent: See :class:`QAction`
    :param ~silx.gui.plot3d.Plot3DWidget.Plot3DWidget plot3d:
        Plot3DWidget the action is associated with
    """
    def __init__(self, parent, plot3d=None):
        super(SideViewpointAction, self).__init__(parent, 'side', plot3d)

        self.setIcon(getQIcon('cube'))
        self.setText('Side')
        self.setToolTip('Side view')


class RotateViewpoint(Plot3DAction):
    """QAction to rotate the scene of a Plot3DWidget

    :param parent: See :class:`QAction`
    :param ~silx.gui.plot3d.Plot3DWidget.Plot3DWidget plot3d:
        Plot3DWidget the action is associated with
    """

    _TIMEOUT_MS = 50
    """Time interval between to frames (in milliseconds)"""

    _DEGREE_PER_SECONDS = 360. / 5.
    """Rotation speed of the animation"""

    def __init__(self, parent, plot3d=None):
        super(RotateViewpoint, self).__init__(parent, plot3d)

        self._previousTime = None

        self._timer = qt.QTimer(self)
        self._timer.setInterval(self._TIMEOUT_MS)  # 20fps
        self._timer.timeout.connect(self._rotate)

        self.setIcon(getQIcon('cube-rotate'))
        self.setText('Rotate scene')
        self.setToolTip('Rotate the 3D scene around the vertical axis')
        self.setCheckable(True)
        self.triggered[bool].connect(self._triggered)


    def _triggered(self, checked=False):
        plot3d = self.getPlot3DWidget()
        if plot3d is None:
            _logger.error(
                'Cannot start/stop rotation, no associated Plot3DWidget')
        elif checked:
            self._previousTime = time.time()
            self._timer.start()
        else:
            self._timer.stop()
            self._previousTime = None

    def _rotate(self):
        """Perform a step of the rotation"""
        if self._previousTime is None:
            _logger.error('Previous time not set!')
            angleStep = 0.
        else:
            angleStep = self._DEGREE_PER_SECONDS * (time.time() - self._previousTime)

        self.getPlot3DWidget().viewport.orbitCamera('left', angleStep)
        self._previousTime = time.time()
