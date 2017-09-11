# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2015-2017 European Synchrotron Radiation Facility
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
"""This module provides a toolbar to control Plot3DWidget viewpoint."""

from __future__ import absolute_import

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "08/09/2017"


from silx.gui import qt
from silx.gui.icons import getQIcon


class _ViewpointActionGroup(qt.QActionGroup):
    """ActionGroup of actions to reset the viewpoint.

    As for QActionGroup, add group's actions to the widget with:
    `widget.addActions(actionGroup.actions())`

    :param Plot3DWidget plot3D: The widget for which to control the viewpoint
    :param parent: See :class:`QActionGroup`
    """

    # Action information: icon name, text, tooltip
    _RESET_CAMERA_ACTIONS = (
        ('cube-front', 'Front', 'View along the -Z axis'),
        ('cube-back', 'Back', 'View along the +Z axis'),
        ('cube-top', 'Top', 'View along the -Y'),
        ('cube-bottom', 'Bottom', 'View along the +Y'),
        ('cube-right', 'Right', 'View along the -X'),
        ('cube-left', 'Left', 'View along the +X'),
        ('cube', 'Side', 'Side view')
    )

    def __init__(self, plot3D, parent=None):
        super(_ViewpointActionGroup, self).__init__(parent)
        self.setExclusive(False)

        self._plot3D = plot3D

        for actionInfo in self._RESET_CAMERA_ACTIONS:
            iconname, text, tooltip = actionInfo

            action = qt.QAction(getQIcon(iconname), text, None)
            action.setIconVisibleInMenu(True)
            action.setCheckable(False)
            action.setToolTip(tooltip)
            self.addAction(action)

        self.triggered[qt.QAction].connect(self._actionGroupTriggered)

    def _actionGroupTriggered(self, action):
        actionname = action.text().lower()

        self._plot3D.viewport.camera.extrinsic.reset(face=actionname)
        self._plot3D.centerScene()


class ViewpointToolBar(qt.QToolBar):
    """A toolbar providing icons to reset the viewpoint.

    :param parent: See :class:`QToolBar`
    :param Plot3DWidget plot3D: The widget to control
    :param str title: Title of the toolbar
    """

    def __init__(self, parent=None, plot3D=None, title='Viewpoint control'):
        super(ViewpointToolBar, self).__init__(title, parent)

        self._actionGroup = _ViewpointActionGroup(plot3D)
        assert plot3D is not None
        self._plot3D = plot3D
        self.addActions(self._actionGroup.actions())

        # Choosing projection disabled for now
        # Add projection combo box
        # comboBoxProjection = qt.QComboBox()
        # comboBoxProjection.addItem('Perspective')
        # comboBoxProjection.addItem('Parallel')
        # comboBoxProjection.setToolTip(
        #     'Choose the projection:'
        #     ' perspective or parallel (i.e., orthographic)')
        # comboBoxProjection.currentIndexChanged[(str)].connect(
        #     self._comboBoxProjectionCurrentIndexChanged)
        # self.addWidget(qt.QLabel('Projection:'))
        # self.addWidget(comboBoxProjection)

    # def _comboBoxProjectionCurrentIndexChanged(self, text):
    #     """Projection combo box listener"""
    #     self._plot3D.setProjection(
    #         'perspective' if text == 'Perspective' else 'orthographic')


class ViewpointToolButton(qt.QToolButton):
    """A toolbutton with a drop-down list of ways to reset the viewpoint.

    :param parent: See :class:`QToolButton`
    :param Plot3DWiddget plot3D: The widget to control
    """

    def __init__(self, parent=None, plot3D=None):
        super(ViewpointToolButton, self).__init__(parent)

        self._actionGroup = _ViewpointActionGroup(plot3D)

        menu = qt.QMenu(self)
        menu.addActions(self._actionGroup.actions())
        self.setMenu(menu)
        self.setPopupMode(qt.QToolButton.InstantPopup)
        self.setIcon(getQIcon('cube'))
        self.setToolTip('Reset the viewpoint to a defined position')
