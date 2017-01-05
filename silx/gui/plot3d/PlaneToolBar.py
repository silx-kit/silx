# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2017 European Synchrotron Radiation Facility
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
"""This module provides a toolbar to control a cutting plane."""

from __future__ import absolute_import

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "05/10/2016"


import logging

from silx.gui import qt
from .icons import getQIcon  # TODO merge with silx icons

_logger = logging.getLogger(__name__)


class PlaneToolBar(qt.QToolBar):
    """A toolbar providing icons to orient a cutting plane.

    :param parent: See :class:`QToolBar`
    :param str title: Title of the toolbar
    """

    # Action information: icon name, text, tooltip
    _PLANE_ACTIONS = (
        ('3d-plane-normal-x', 'YZ-Plane', 'Set plane perpendicular to X axis'),
        ('3d-plane-normal-y', 'XZ-Plane', 'Set plane perpendicular to Y axis'),
        ('3d-plane-normal-z', 'XY-Plane', 'Set plane perpendicular to Z axis'),
    )

    def __init__(self, parent=None, title='Plane control'):
        super(PlaneToolBar, self).__init__(title, parent)
        self.setEnabled(False)

        self._plane = None

        self._group = qt.QActionGroup(self)
        self._group.setExclusive(False)
        self._group.setEnabled(False)

        # Show/Hide icon
        self._showAction = self.addAction(
            getQIcon('3d-plane'), 'Cutting plane')
        self._showAction.setToolTip('Show/Hide cutting plane')
        self._showAction.setCheckable(True)
        self._showAction.setChecked(False)
        self._showAction.triggered[bool].connect(self.setPlaneVisible)

        for iconName, text, tooltip in self._PLANE_ACTIONS:
            action = qt.QAction(None)
            action.setIcon(getQIcon(iconName))
            action.setText(text)
            action.setCheckable(False)
            action.setToolTip(tooltip)
            self._group.addAction(action)

        self._group.triggered.connect(self._actionTriggered)
        self.addActions(self._group.actions())

    def setPlane(self, plane):
        """Set the plane to control

        :param plane: The :class:`Plane` to control
        """
        if self._plane is not None:
            self._plane.removeListener(self._planeChanged)

        self._plane = plane
        self.setEnabled(self._plane is not None)

        if self._plane is not None:
            self._plane.visible = self._showAction.isChecked()
            self._plane.addListener(self._planeChanged)

    def getPlane(self):
        """Returns the plane currently controlled by this toolbar
        """
        return self._plane

    def _planeChanged(self, *args, **kwargs):
        """Handle events from the plane to check if visibility has changed"""
        plane = self.getPlane()
        assert plane is not None
        # Sync show action with plane
        if plane.visible != self._showAction.isChecked():
            self._showAction.setChecked(plane.visible)

    def setPlaneVisible(self, visibility):
        """Change the visibility of the plane if any.

        :param bool visibility: True to show plane, False otherwise
        """
        plane = self.getPlane()
        if plane is not None:
            plane.visible = visibility
            plane.moveToCenter()
            self._group.setEnabled(visibility)

    def getPlaneVisible(self):
        """Returns the visibility of the plane"""
        plane = self.getPlane()
        return plane.visible if plane is not None else False

    def _actionTriggered(self, action):
        plane = self.getPlane()
        if plane is not None:
            name = action.text().lower()
            if name.startswith('yz'):
                plane.plane.normal = (1., 0., 0.)
            elif name.startswith('xz'):
                plane.plane.normal = (0., 1., 0.)
            elif name.startswith('xy'):
                plane.plane.normal = (0., 0., 1.)
            else:
                _logger.error('Unknown action triggered %s', name)

            plane.moveToCenter()
