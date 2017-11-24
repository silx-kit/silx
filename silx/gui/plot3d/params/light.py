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
"""
This module provides a parameter tree item to set light direction.
"""

from __future__ import absolute_import

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "24/11/2017"

import numpy

from ... import qt
from .SubjectItem import SubjectItem


class _LightDirectionAngleBaseItem(SubjectItem):
    """Base class for directional light angle item."""
    editable = True
    persistent = True

    def _init(self):
        pass

    def getSignals(self):
        """Override to provide signals to listen"""
        raise NotImplementedError("MUST be implemented in subclass")

    def _pullData(self):
        """Override in subclass to get current angle"""
        raise NotImplementedError("MUST be implemented in subclass")

    def _pushData(self, value, role=qt.Qt.UserRole):
        """Override in subclass to set the angle"""
        raise NotImplementedError("MUST be implemented in subclass")

    def getEditor(self, parent, option, index):
        editor = qt.QSlider(parent)
        editor.setOrientation(qt.Qt.Horizontal)
        editor.setMinimum(-90)
        editor.setMaximum(90)
        editor.setValue(self._pullData())

        # Wrapping call in lambda is a workaround for PySide with Python 3
        editor.valueChanged.connect(
            lambda value: self._pushData(value))

        return editor

    def setEditorData(self, editor):
        editor.setValue(self._pullData())
        return True

    def _setModelData(self, editor):
        value = editor.value()
        self._pushData(value)
        return True


class _LightAzimuthAngleItem(_LightDirectionAngleBaseItem):
    """Light direction azimuth angle item."""

    def getSignals(self):
        return self.getSubject().sigAzimuthAngleChanged

    def _pullData(self):
         return self.getSubject().getAzimuthAngle()

    def _pushData(self, value, role=qt.Qt.UserRole):
         self.getSubject().setAzimuthAngle(value)


class _LightAltitudeAngleItem(_LightDirectionAngleBaseItem):
    """Light direction altitude angle item."""

    def getSignals(self):
        return self.getSubject().sigAltitudeAngleChanged

    def _pullData(self):
         return self.getSubject().getAltitudeAngle()

    def _pushData(self, value, role=qt.Qt.UserRole):
         self.getSubject().setAltitudeAngle(value)


class _DirectionalLightProxy(qt.QObject):
    """Proxy to handle directional light with angles rather than vector.
    """

    sigAzimuthAngleChanged = qt.Signal()
    """Signal sent when the azimuth angle has changed."""

    sigAltitudeAngleChanged = qt.Signal()
    """Signal sent when altitude angle has changed."""

    def __init__(self, light):
        super(_DirectionalLightProxy, self).__init__()
        self._light = light
        light.addListener(self._directionUpdated)
        self._azimuth = 0.
        self._altitude = 0.

    def getAzimuthAngle(self):
        """Returns the signed angle in the horizontal plane.

         Unit: degrees.
        The 0 angle corresponds to the axis perpendicular to the screen.

        :rtype: float
        """
        return self._azimuth

    def getAltitudeAngle(self):
        """Returns the signed vertical angle from the horizontal plane.

        Unit: degrees.
        Range: [-90, +90]

        :rtype: float
        """
        return self._altitude

    def setAzimuthAngle(self, angle):
        """Set the horizontal angle.

        :param float angle: Angle from -z axis in zx plane in degrees.
        """
        if angle != self._azimuth:
            self._azimuth = angle
            self._updateLight()
            self.sigAzimuthAngleChanged.emit()

    def setAltitudeAngle(self, angle):
        """Set the horizontal angle.

        :param float angle: Angle from -z axis in zy plane in degrees.
        """
        if angle != self._altitude:
            self._altitude = angle
            self._updateLight()
            self.sigAltitudeAngleChanged.emit()

    def _directionUpdated(self, *args, **kwargs):
        """Handle light direction update in the scene"""
        # Invert direction to manipulate the 'source' pointing to
        # the center of the viewport
        x, y, z = - self._light.direction

        # Horizontal plane is plane xz
        azimuth = numpy.degrees(numpy.arctan2(x, z))
        altitude = numpy.degrees(numpy.pi/2. - numpy.arccos(y))

        if (abs(azimuth - self.getAzimuthAngle()) > 0.01 and
                abs(abs(altitude) - 90.) >= 0.001):  # Do not update when at zenith
            self.setAzimuthAngle(azimuth)

        if abs(altitude - self.getAltitudeAngle()) > 0.01:
            self.setAltitudeAngle(altitude)

    def _updateLight(self):
        """Update light direction in the scene"""
        azimuth = numpy.radians(self._azimuth)
        delta = numpy.pi/2. - numpy.radians(self._altitude)
        z = - numpy.sin(delta) * numpy.cos(azimuth)
        x = - numpy.sin(delta) * numpy.sin(azimuth)
        y = - numpy.cos(delta)
        self._light.direction = x, y, z


class DirectionalLightItem(SubjectItem):
    """Root Item for directional light configuration.

    :param Plot3DWidget plot3dWidget:
        The :class:`Plot3DWidget` to configure
    """

    def __init__(self, plot3dWidget, *args):
        self._light = _DirectionalLightProxy(
            plot3dWidget.viewport.light)

        super(DirectionalLightItem, self).__init__(plot3dWidget, *args)

    def _init(self):

        nameItem = qt.QStandardItem('Azimuth')
        nameItem.setEditable(False)
        valueItem = _LightAzimuthAngleItem(self._light)
        self.appendRow([nameItem, valueItem])

        nameItem = qt.QStandardItem('Altitude')
        nameItem.setEditable(False)
        valueItem = _LightAltitudeAngleItem(self._light)
        self.appendRow([nameItem, valueItem])
