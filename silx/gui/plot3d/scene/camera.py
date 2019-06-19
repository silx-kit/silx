# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2015-2018 European Synchrotron Radiation Facility
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
"""This module provides classes to handle a perspective projection in 3D."""

from __future__ import absolute_import, division, unicode_literals

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "25/07/2016"


import numpy

from . import transform


# CameraExtrinsic #############################################################

class CameraExtrinsic(transform.Transform):
    """Transform matrix to handle camera position and orientation.

    :param position: Coordinates of the point of view.
    :type position: numpy.ndarray-like of 3 float32.
    :param direction: Sight direction vector.
    :type direction: numpy.ndarray-like of 3 float32.
    :param up: Vector pointing upward in the image plane.
    :type up: numpy.ndarray-like of 3 float32.
    """

    def __init__(self, position=(0., 0., 0.),
                 direction=(0., 0., -1.),
                 up=(0., 1., 0.)):

        super(CameraExtrinsic, self).__init__()
        self._position = None
        self.position = position  # set _position
        self._side = 1., 0., 0.
        self._up = 0., 1., 0.
        self._direction = 0., 0., -1.
        self.setOrientation(direction=direction, up=up)  # set _direction, _up

    def _makeMatrix(self):
        return transform.mat4LookAtDir(self._position,
                                       self._direction, self._up)

    def copy(self):
        """Return an independent copy"""
        return CameraExtrinsic(self.position, self.direction, self.up)

    def setOrientation(self, direction=None, up=None):
        """Set the rotation of the point of view.

        :param direction: Sight direction vector or
                          None to keep the current one.
        :type direction: numpy.ndarray-like of 3 float32 or None.
        :param up: Vector pointing upward in the image plane or
                   None to keep the current one.
        :type up: numpy.ndarray-like of 3 float32 or None.
        :raises RuntimeError: if the direction and up are parallel.
        """
        if direction is None:  # Use current direction
            direction = self.direction
        else:
            assert len(direction) == 3
            direction = numpy.array(direction, copy=True, dtype=numpy.float32)
            direction /= numpy.linalg.norm(direction)

        if up is None:  # Use current up
            up = self.up
        else:
            assert len(up) == 3
            up = numpy.array(up, copy=True, dtype=numpy.float32)

        # Update side and up to make sure they are perpendicular and normalized
        side = numpy.cross(direction, up)
        sidenormal = numpy.linalg.norm(side)
        if sidenormal == 0.:
            raise RuntimeError('direction and up vectors are parallel.')
            # Alternative: when one of the input parameter is None, it is
            # possible to guess correct vectors using previous direction and up
        side /= sidenormal
        up = numpy.cross(side, direction)
        up /= numpy.linalg.norm(up)

        self._side = side
        self._up = up
        self._direction = direction
        self.notify()

    @property
    def position(self):
        """Coordinates of the point of view as a numpy.ndarray of 3 float32."""
        return self._position.copy()

    @position.setter
    def position(self, position):
        assert len(position) == 3
        self._position = numpy.array(position, copy=True, dtype=numpy.float32)
        self.notify()

    @property
    def direction(self):
        """Sight direction (ndarray of 3 float32)."""
        return self._direction.copy()

    @direction.setter
    def direction(self, direction):
        self.setOrientation(direction=direction)

    @property
    def up(self):
        """Vector pointing upward in the image plane (ndarray of 3 float32).
        """
        return self._up.copy()

    @up.setter
    def up(self, up):
        self.setOrientation(up=up)

    @property
    def side(self):
        """Vector pointing towards the side of the image plane.

        ndarray of 3 float32"""
        return self._side.copy()

    def move(self, direction, step=1.):
        """Move the camera relative to the image plane.

        :param str direction: Direction relative to image plane.
                              One of: 'up', 'down', 'left', 'right',
                              'forward', 'backward'.
        :param float step: The step of the pan to perform in the coordinate
                           in which the camera position is defined.
        """
        if direction in ('up', 'down'):
            vector = self.up * (1. if direction == 'up' else -1.)
        elif direction in ('left', 'right'):
            vector = self.side * (1. if direction == 'right' else -1.)
        elif direction in ('forward', 'backward'):
            vector = self.direction * (1. if direction == 'forward' else -1.)
        else:
            raise ValueError('Unsupported direction: %s' % direction)

        self.position += step * vector

    def rotate(self, direction, angle=1.):
        """First-person rotation of the camera towards the direction.

        :param str direction: Direction of movement relative to image plane.
                              In: 'up', 'down', 'left', 'right'.
        :param float angle: The angle in degrees of the rotation.
        """
        if direction in ('up', 'down'):
            axis = self.side * (1. if direction == 'up' else -1.)
        elif direction in ('left', 'right'):
            axis = self.up * (1. if direction == 'left' else -1.)
        else:
            raise ValueError('Unsupported direction: %s' % direction)

        matrix = transform.mat4RotateFromAngleAxis(numpy.radians(angle), *axis)
        newdir = numpy.dot(matrix[:3, :3], self.direction)

        if direction in ('up', 'down'):
            # Rotate up to avoid up and new direction to be (almost) co-linear
            newup = numpy.dot(matrix[:3, :3], self.up)
            self.setOrientation(newdir, newup)
        else:
            # No need to rotate up here as it is the rotation axis
            self.direction = newdir

    def orbit(self, direction, center=(0., 0., 0.), angle=1.):
        """Rotate the camera around a point.

        :param str direction: Direction of movement relative to image plane.
                              In: 'up', 'down', 'left', 'right'.
        :param center: Position around which to rotate the point of view.
        :type center: numpy.ndarray-like of 3 float32.
        :param float angle: he angle in degrees of the rotation.
        """
        if direction in ('up', 'down'):
            axis = self.side * (1. if direction == 'down' else -1.)
        elif direction in ('left', 'right'):
            axis = self.up * (1. if direction == 'right' else -1.)
        else:
            raise ValueError('Unsupported direction: %s' % direction)

        # Rotate viewing direction
        rotmatrix = transform.mat4RotateFromAngleAxis(
            numpy.radians(angle), *axis)
        self.direction = numpy.dot(rotmatrix[:3, :3], self.direction)

        # Rotate position around center
        center = numpy.array(center, copy=False, dtype=numpy.float32)
        matrix = numpy.dot(transform.mat4Translate(*center), rotmatrix)
        matrix = numpy.dot(matrix, transform.mat4Translate(*(-center)))
        position = numpy.append(self.position, 1.)
        self.position = numpy.dot(matrix, position)[:3]

    _RESET_CAMERA_ORIENTATIONS = {
        'side': ((-1., -1., -1.), (0., 1., 0.)),
        'front': ((0., 0., -1.), (0., 1., 0.)),
        'back': ((0., 0., 1.), (0., 1., 0.)),
        'top': ((0., -1., 0.), (0., 0., -1.)),
        'bottom': ((0., 1., 0.), (0., 0., 1.)),
        'right': ((-1., 0., 0.), (0., 1., 0.)),
        'left': ((1., 0., 0.), (0., 1., 0.))
    }

    def reset(self, face=None):
        """Reset the camera position to pre-defined orientations.

        :param str face: The direction of the camera in:
                         side, front, back, top, bottom, right, left.
        """
        if face not in self._RESET_CAMERA_ORIENTATIONS:
            raise ValueError('Unsupported face: %s' % face)

        distance = numpy.linalg.norm(self.position)
        direction, up = self._RESET_CAMERA_ORIENTATIONS[face]
        self.setOrientation(direction, up)
        self.position = - self.direction * distance


class Camera(transform.Transform):
    """Combination of camera projection and position.

    See :class:`Perspective` and :class:`CameraExtrinsic`.

    :param float fovy: Vertical field-of-view in degrees.
    :param float near: The near clipping plane Z coord (strictly positive).
    :param float far: The far clipping plane Z coord (> near).
    :param size:
        Viewport's size used to compute the aspect ratio (width, height).
    :type size: 2-tuple of float
    :param position: Coordinates of the point of view.
    :type position: numpy.ndarray-like of 3 float32.
    :param direction: Sight direction vector.
    :type direction: numpy.ndarray-like of 3 float32.
    :param up: Vector pointing upward in the image plane.
    :type up: numpy.ndarray-like of 3 float32.
    """

    def __init__(self, fovy=30., near=0.1, far=1., size=(1., 1.),
                 position=(0., 0., 0.),
                 direction=(0., 0., -1.), up=(0., 1., 0.)):
        super(Camera, self).__init__()
        self._intrinsic = transform.Perspective(fovy, near, far, size)
        self._intrinsic.addListener(self._transformChanged)
        self._extrinsic = CameraExtrinsic(position, direction, up)
        self._extrinsic.addListener(self._transformChanged)

    def _makeMatrix(self):
        return numpy.dot(self.intrinsic.matrix, self.extrinsic.matrix)

    def _transformChanged(self, source):
        """Listener of intrinsic and extrinsic camera parameters instances."""
        if source is not self:
            self.notify()

    def resetCamera(self, bounds):
        """Change camera to have the bounds in the viewing frustum.

        It updates the camera position and depth extent.
        Camera sight direction and up are not affected.

        :param bounds: The axes-aligned bounds to include.
        :type bounds: numpy.ndarray: ((xMin, yMin, zMin), (xMax, yMax, zMax))
        """

        center = 0.5 * (bounds[0] + bounds[1])
        radius = numpy.linalg.norm(0.5 * (bounds[1] - bounds[0]))
        if radius == 0.:  # bounds are all collapsed
            radius = 1.

        if isinstance(self.intrinsic, transform.Perspective):
            # Get the viewpoint distance from the bounds center
            minfov = numpy.radians(self.intrinsic.fovy)
            width, height = self.intrinsic.size
            if width < height:
                minfov *= width / height

            offset = radius / numpy.sin(0.5 * minfov)

            # Update camera
            self.extrinsic.position = \
                center - offset * self.extrinsic.direction
            self.intrinsic.setDepthExtent(offset - radius, offset + radius)

        elif isinstance(self.intrinsic, transform.Orthographic):
            # Y goes up
            self.intrinsic.setClipping(
                left=center[0] - radius,
                right=center[0] + radius,
                bottom=center[1] - radius,
                top=center[1] + radius)

            # Update camera
            self.extrinsic.position = 0, 0, 0
            self.intrinsic.setDepthExtent(center[2] - radius,
                                          center[2] + radius)
        else:
            raise RuntimeError('Unsupported camera: %s' % self.intrinsic)

    @property
    def intrinsic(self):
        """Intrinsic camera parameters, i.e., projection matrix."""
        return self._intrinsic

    @intrinsic.setter
    def intrinsic(self, intrinsic):
        self._intrinsic.removeListener(self._transformChanged)
        self._intrinsic = intrinsic
        self._intrinsic.addListener(self._transformChanged)

    @property
    def extrinsic(self):
        """Extrinsic camera parameters, i.e., position and orientation."""
        return self._extrinsic

    def move(self, *args, **kwargs):
        """See :meth:`CameraExtrinsic.move`."""
        self.extrinsic.move(*args, **kwargs)

    def rotate(self, *args, **kwargs):
        """See :meth:`CameraExtrinsic.rotate`."""
        self.extrinsic.rotate(*args, **kwargs)

    def orbit(self, *args, **kwargs):
        """See :meth:`CameraExtrinsic.orbit`."""
        self.extrinsic.orbit(*args, **kwargs)
