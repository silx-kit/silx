# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017 European Synchrotron Radiation Facility
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


class RotateViewport(Plot3DAction):
    """QAction to rotate the scene of a Plot3DWidget

    :param parent: See :class:`QAction`
    :param Plot3DWidget plot3d: Plot3DWidget the action is associated with
    """

    _TIMEOUT_MS = 50
    """Time interval between to frames (in milliseconds)"""

    _DEGREE_PER_SECONDS = 360. / 5.
    """Rotation speed of the animation"""

    def __init__(self, parent, plot3d=None):
        super(RotateViewport, self).__init__(parent, plot3d)

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
