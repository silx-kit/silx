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
"""This module provides 4x4 matrix operation and classes to handle them."""

from __future__ import absolute_import, division, unicode_literals

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "25/07/2016"


import itertools
import numpy

from . import event


# Functions ###################################################################

# Projections

def mat4LookAtDir(position, direction, up):
    """Creates matrix to look in direction from position.

    :param position: Array-like 3 coordinates of the point of view position.
    :param direction: Array-like 3 coordinates of the sight direction vector.
    :param up: Array-like 3 coordinates of the upward direction
               in the image plane.
    :returns: Corresponding matrix.
    :rtype: numpy.ndarray of shape (4, 4)
    """
    assert len(position) == 3
    assert len(direction) == 3
    assert len(up) == 3

    direction = numpy.array(direction, copy=True, dtype=numpy.float32)
    dirnorm = numpy.linalg.norm(direction)
    assert dirnorm != 0.
    direction /= dirnorm

    side = numpy.cross(direction,
                       numpy.array(up, copy=False, dtype=numpy.float32))
    sidenorm = numpy.linalg.norm(side)
    assert sidenorm != 0.
    up = numpy.cross(side / sidenorm, direction)
    upnorm = numpy.linalg.norm(up)
    assert upnorm != 0.
    up /= upnorm

    matrix = numpy.identity(4, dtype=numpy.float32)
    matrix[0, :3] = side
    matrix[1, :3] = up
    matrix[2, :3] = -direction
    return numpy.dot(matrix,
                     mat4Translate(-position[0], -position[1], -position[2]))


def mat4LookAt(position, center, up):
    """Creates matrix to look at center from position.

    See gluLookAt.

    :param position: Array-like 3 coordinates of the point of view position.
    :param center: Array-like 3 coordinates of the center of the scene.
    :param up: Array-like 3 coordinates of the upward direction
               in the image plane.
    :returns: Corresponding matrix.
    :rtype: numpy.ndarray of shape (4, 4)
    """
    position = numpy.array(position, copy=False, dtype=numpy.float32)
    center = numpy.array(center, copy=False, dtype=numpy.float32)
    direction = center - position
    return mat4LookAtDir(position, direction, up)


def mat4Frustum(left, right, bottom, top, near, far):
    """Creates a frustum projection matrix.

    See glFrustum.
    """
    return numpy.array((
        (2.*near / (right-left), 0., (right+left) / (right-left), 0.),
        (0., 2.*near / (top-bottom), (top+bottom) / (top-bottom), 0.),
        (0., 0., -(far+near) / (far-near), -2.*far*near / (far-near)),
        (0., 0., -1., 0.)), dtype=numpy.float32)


def mat4Perspective(fovy, width, height, near, far):
    """Creates a perspective projection matrix.

    Similar to gluPerspective.

    :param float fovy: Field of view angle in degrees in the y direction.
    :param float width: Width of the viewport.
    :param float height: Height of the viewport.
    :param float near: Distance to the near plane (strictly positive).
    :param float far: Distance to the far plane (strictly positive).
    :return: Corresponding matrix.
    :rtype: numpy.ndarray of shape (4, 4)
    """
    assert fovy != 0
    assert height != 0
    assert width != 0
    assert near > 0.
    assert far > near
    aspectratio = width / height
    f = 1. / numpy.tan(numpy.radians(fovy) / 2.)
    return numpy.array((
        (f / aspectratio, 0., 0., 0.),
        (0., f, 0., 0.),
        (0., 0., (far + near) / (near - far), 2. * far * near / (near - far)),
        (0., 0., -1., 0.)), dtype=numpy.float32)


def mat4Orthographic(left, right, bottom, top, near, far):
    """Creates an orthographic (i.e., parallel) projection matrix.

    See glOrtho.
    """
    return numpy.array((
        (2. / (right - left), 0., 0., - (right + left) / (right - left)),
        (0., 2. / (top - bottom), 0., - (top + bottom) / (top - bottom)),
        (0., 0., -2. / (far - near), - (far + near) / (far - near)),
        (0., 0., 0., 1.)), dtype=numpy.float32)


# Affine

def mat4Translate(tx, ty, tz):
    """4x4 translation matrix."""
    return numpy.array((
        (1., 0., 0., tx),
        (0., 1., 0., ty),
        (0., 0., 1., tz),
        (0., 0., 0., 1.)), dtype=numpy.float32)


def mat4Scale(sx, sy, sz):
    """4x4 scale matrix."""
    return numpy.array((
        (sx, 0., 0., 0.),
        (0., sy, 0., 0.),
        (0., 0., sz, 0.),
        (0., 0., 0., 1.)), dtype=numpy.float32)


def mat4RotateFromAngleAxis(angle, x=0., y=0., z=1.):
    """4x4 rotation matrix from angle and axis.

    :param float angle: The rotation angle in radians.
    :param float x: The rotation vector x coordinate.
    :param float y: The rotation vector y coordinate.
    :param float z: The rotation vector z coordinate.
    """
    ca = numpy.cos(angle)
    sa = numpy.sin(angle)
    return numpy.array((
        ((1.-ca) * x*x + ca,   (1.-ca) * x*y - sa*z, (1.-ca) * x*z + sa*y, 0.),
        ((1.-ca) * x*y + sa*z, (1.-ca) * y*y + ca,   (1.-ca) * y*z - sa*x, 0.),
        ((1.-ca) * x*z - sa*y, (1.-ca) * y*z + sa*x, (1.-ca) * z*z + ca, 0.),
        (0., 0., 0., 1.)), dtype=numpy.float32)


def mat4RotateFromQuaternion(quaternion):
    """4x4 rotation matrix from quaternion.

    :param quaternion: Array-like unit quaternion stored as (x, y, z, w)
    """
    quaternion = numpy.array(quaternion, copy=True)
    quaternion /= numpy.linalg.norm(quaternion)

    qx, qy, qz, qw = quaternion
    return numpy.array((
        (1. - 2.*(qy**2 + qz**2), 2.*(qx*qy - qw*qz), 2.*(qx*qz + qw*qy), 0.),
        (2.*(qx*qy + qw*qz), 1. - 2.*(qx**2 + qz**2), 2.*(qy*qz - qw*qx), 0.),
        (2.*(qx*qz - qw*qy), 2.*(qy*qz + qw*qx), 1. - 2.*(qx**2 + qy**2), 0.),
        (0., 0., 0., 1.)), dtype=numpy.float32)


def mat4Shear(axis, sx=0., sy=0., sz=0.):
    """4x4 shear matrix: Skew two axes relative to a third fixed one.

    shearFactor = tan(shearAngle)

    :param str axis: The axis to keep constant and shear against.
                     In 'x', 'y', 'z'.
    :param float sx: The shear factor for the X axis relative to axis.
    :param float sy: The shear factor for the Y axis relative to axis.
    :param float sz: The shear factor for the Z axis relative to axis.
    """
    assert axis in ('x', 'y', 'z')

    matrix = numpy.identity(4, dtype=numpy.float32)

    # Make the shear column
    index = 'xyz'.find(axis)
    shearcolumn = numpy.array((sx, sy, sz, 0.), dtype=numpy.float32)
    shearcolumn[index] = 1.
    matrix[:, index] = shearcolumn
    return matrix


# Transforms ##################################################################

class Transform(event.Notifier):

    def __init__(self, static=False):
        """Base class for (row-major) 4x4 matrix transforms.

        :param bool static: False (default) to reset cache when changed,
                            True for static matrices.
        """
        super(Transform, self).__init__()
        self._matrix = None
        self._inverse = None
        if not static:
            self.addListener(self._changed)  # Listening self for changes

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__init__,
                           repr(self.getMatrix(copy=False)))

    def inverse(self):
        """Return the Transform of the inverse.

        The returned Transform is static, it is not updated when this
        Transform is modified.

        :return: A Transform which is the inverse of this Transform.
        """
        return Inverse(self)

    # Matrix

    def _makeMatrix(self):
        """Override to build matrix"""
        return numpy.identity(4, dtype=numpy.float32)

    def _makeInverse(self):
        """Override to build inverse matrix."""
        return numpy.linalg.inv(self.getMatrix(copy=False))

    def getMatrix(self, copy=True):
        """The 4x4 matrix of this transform.

        :param bool copy: True (the default) to get a copy of the matrix,
                          False to get the internal matrix, do not modify!
        :return: 4x4 matrix of this transform.
        """
        if self._matrix is None:
            self._matrix = self._makeMatrix()
        if copy:
            return self._matrix.copy()
        else:
            return self._matrix

    matrix = property(getMatrix, doc="The 4x4 matrix of this transform.")

    def getInverseMatrix(self, copy=False):
        """The 4x4 matrix of the inverse of this transform.

        :param bool copy: True (the default) to get a copy of the matrix,
                          False to get the internal matrix, do not modify!
        :return: 4x4 matrix of the inverse of this transform.
        """
        if self._inverse is None:
            self._inverse = self._makeInverse()
        if copy:
            return self._inverse.copy()
        else:
            return self._inverse

    inverseMatrix = property(
        getInverseMatrix,
        doc="The 4x4 matrix of the inverse of this transform.")

    # Listener

    def _changed(self, source):
        """Default self listener reseting matrix cache."""
        self._matrix = None
        self._inverse = None

    # Multiplication with vectors

    def transformPoints(self, points, direct=True, perspectiveDivide=False):
        """Apply the transform to an array of points.

        :param points: 2D array of N vectors of 3 or 4 coordinates
        :param bool direct: Whether to apply the direct (True, the default)
                            or inverse (False) transform.
        :param bool perspectiveDivide: Whether to apply the perspective divide
                                       (True) or not (False, the default).
        :return: The transformed points.
        :rtype: numpy.ndarray of same shape as points.
        """
        if direct:
            matrix = self.getMatrix(copy=False)
        else:
            matrix = self.getInverseMatrix(copy=False)

        points = numpy.array(points, copy=False)
        assert points.ndim == 2

        points = numpy.transpose(points)

        dimension = points.shape[0]
        assert dimension in (3, 4)

        if dimension == 3:  # Add 4th coordinate
            points = numpy.append(
                points,
                numpy.ones((1, points.shape[1]), dtype=points.dtype),
                axis=0)

        result = numpy.transpose(numpy.dot(matrix, points))

        if perspectiveDivide:
            mask = result[:, 3] != 0.
            result[mask] /= result[mask, 3][:, numpy.newaxis]

        return result[:, :3] if dimension == 3 else result

    @staticmethod
    def _prepareVector(vector, w):
        """Add 4th coordinate (w) to vector if missing."""
        assert len(vector) in (3, 4)
        vector = numpy.array(vector, copy=False, dtype=numpy.float32)
        if len(vector) == 3:
            vector = numpy.append(vector, w)
        return vector

    def transformPoint(self, point, direct=True, perspectiveDivide=False):
        """Apply the transform to a point.

        :param point: Array-like vector of 3 or 4 coordinates.
        :param bool direct: Whether to apply the direct (True, the default)
                            or inverse (False) transform.
        :param bool perspectiveDivide: Whether to apply the perspective divide
                                       (True) or not (False, the default).
        :return: The transformed point.
        :rtype: numpy.ndarray of same length as point.
        """
        if direct:
            matrix = self.getMatrix(copy=False)
        else:
            matrix = self.getInverseMatrix(copy=False)
        result = numpy.dot(matrix, self._prepareVector(point, 1.))

        if perspectiveDivide and result[3] != 0.:
            result /= result[3]

        if len(point) == 3:
            return result[:3]
        else:
            return result

    def transformDir(self, direction, direct=True):
        """Apply the transform to a direction.

        :param direction: Array-like vector of 3 coordinates.
        :param bool direct: Whether to apply the direct (True, the default)
                            or inverse (False) transform.
        :return: The transformed direction.
        :rtype: numpy.ndarray of length 3.
        """
        if direct:
            matrix = self.getMatrix(copy=False)
        else:
            matrix = self.getInverseMatrix(copy=False)
        return numpy.dot(matrix[:3, :3], direction[:3])

    def transformNormal(self, normal, direct=True):
        """Apply the transform to a normal: R = (M-1)t * V.

        :param normal: Array-like vector of 3 coordinates.
        :param bool direct: Whether to apply the direct (True, the default)
                            or inverse (False) transform.
        :return: The transformed normal.
        :rtype: numpy.ndarray of length 3.
        """
        if direct:
            matrix = self.getInverseMatrix(copy=False).T
        else:
            matrix = self.getMatrix(copy=False).T
        return numpy.dot(matrix[:3, :3], normal[:3])

    _CUBE_CORNERS = numpy.array(list(itertools.product((0., 1.), repeat=3)),
                                dtype=numpy.float32)
    """Unit cube corners used by :meth:`transformBounds`"""

    def transformBounds(self, bounds, direct=True):
        """Apply the transform to an axes-aligned rectangular box.

        :param bounds: Min and max coords of the box for each axes.
        :type bounds: 2x3 numpy.ndarray
        :param bool direct: Whether to apply the direct (True, the default)
                            or inverse (False) transform.
        :return: Axes-aligned rectangular box including the transformed box.
        :rtype: 2x3 numpy.ndarray of float32
        """
        corners = numpy.ones((8, 4), dtype=numpy.float32)
        corners[:, :3] = bounds[0] + \
            self._CUBE_CORNERS * (bounds[1] - bounds[0])

        if direct:
            matrix = self.getMatrix(copy=False)
        else:
            matrix = self.getInverseMatrix(copy=False)

        # Transform corners
        cornerstransposed = numpy.dot(matrix, corners.T)
        cornerstransposed = cornerstransposed / cornerstransposed[3]

        # Get min/max for each axis
        transformedbounds = numpy.empty((2, 3), dtype=numpy.float32)
        transformedbounds[0] = cornerstransposed.T[:, :3].min(axis=0)
        transformedbounds[1] = cornerstransposed.T[:, :3].max(axis=0)

        return transformedbounds


class Inverse(Transform):
    """Transform which is the inverse of another one.

    Static: It never gets updated.
    """

    def __init__(self, transform):
        """Initializer.

        :param Transform transform: The transform to invert.
        """

        super(Inverse, self).__init__(static=True)
        self._matrix = transform.getInverseMatrix(copy=True)
        self._inverse = transform.getMatrix(copy=True)


class TransformList(Transform, event.HookList):
    """List of transforms."""

    def __init__(self, iterable=()):
        Transform.__init__(self)
        event.HookList.__init__(self, iterable)

    def _listWillChangeHook(self, methodName, *args, **kwargs):
        for item in self:
            item.removeListener(self._transformChanged)

    def _listWasChangedHook(self, methodName, *args, **kwargs):
        for item in self:
            item.addListener(self._transformChanged)
        self.notify()

    def _transformChanged(self, source):
        """Listen to transform changes of the list and its items."""
        if source is not self:  # Avoid infinite recursion
            self.notify()

    def _makeMatrix(self):
        matrix = numpy.identity(4, dtype=numpy.float32)
        for transform in self:
            matrix = numpy.dot(matrix, transform.getMatrix(copy=False))
        return matrix


class StaticTransformList(Transform):
    """Transform that is a snapshot of a list of Transforms

    It does not keep reference to the list of Transforms.

    :param iterable: Iterable of Transform used for initialization
    """

    def __init__(self, iterable=()):
        super(StaticTransformList, self).__init__(static=True)
        matrix = numpy.identity(4, dtype=numpy.float32)
        for transform in iterable:
            matrix = numpy.dot(matrix, transform.getMatrix(copy=False))
        self._matrix = matrix  # Init matrix once


# Affine ######################################################################

class Matrix(Transform):

    def __init__(self, matrix=None):
        """4x4 Matrix.

        :param matrix: 4x4 array-like matrix or None for identity matrix.
        """
        super(Matrix, self).__init__(static=True)
        self.setMatrix(matrix)

    def setMatrix(self, matrix=None):
        """Update the 4x4 Matrix.

        :param matrix: 4x4 array-like matrix or None for identity matrix.
        """
        if matrix is None:
            self._matrix = numpy.identity(4, dtype=numpy.float32)
        else:
            matrix = numpy.array(matrix, copy=True, dtype=numpy.float32)
            assert matrix.shape == (4, 4)
            self._matrix = matrix
        # Reset cached inverse as Transform is declared static
        self._inverse = None
        self.notify()

    # Redefined here to add a setter
    matrix = property(Transform.getMatrix, setMatrix,
                      doc="The 4x4 matrix of this transform.")


class Translate(Transform):
    """4x4 translation matrix."""

    def __init__(self, tx=0., ty=0., tz=0.):
        super(Translate, self).__init__()
        self._tx, self._ty, self._tz = 0., 0., 0.
        self.setTranslate(tx, ty, tz)

    def _makeMatrix(self):
        return mat4Translate(self.tx, self.ty, self.tz)

    def _makeInverse(self):
        return mat4Translate(-self.tx, -self.ty, -self.tz)

    @property
    def tx(self):
        return self._tx

    @tx.setter
    def tx(self, tx):
        self.setTranslate(tx=tx)

    @property
    def ty(self):
        return self._ty

    @ty.setter
    def ty(self, ty):
        self.setTranslate(ty=ty)

    @property
    def tz(self):
        return self._tz

    @tz.setter
    def tz(self, tz):
        self.setTranslate(tz=tz)

    @property
    def translation(self):
        return numpy.array((self.tx, self.ty, self.tz), dtype=numpy.float32)

    @translation.setter
    def translation(self, translations):
        tx, ty, tz = translations
        self.setTranslate(tx, ty, tz)

    def setTranslate(self, tx=None, ty=None, tz=None):
        if tx is not None:
            self._tx = tx
        if ty is not None:
            self._ty = ty
        if tz is not None:
            self._tz = tz
        self.notify()


class Scale(Transform):
    """4x4 scale matrix."""

    def __init__(self, sx=1., sy=1., sz=1.):
        super(Scale, self).__init__()
        self._sx, self._sy, self._sz = 0., 0., 0.
        self.setScale(sx, sy, sz)

    def _makeMatrix(self):
        return mat4Scale(self.sx, self.sy, self.sz)

    def _makeInverse(self):
        return mat4Scale(1. / self.sx, 1. / self.sy, 1. / self.sz)

    @property
    def sx(self):
        return self._sx

    @sx.setter
    def sx(self, sx):
        self.setScale(sx=sx)

    @property
    def sy(self):
        return self._sy

    @sy.setter
    def sy(self, sy):
        self.setScale(sy=sy)

    @property
    def sz(self):
        return self._sz

    @sz.setter
    def sz(self, sz):
        self.setScale(sz=sz)

    @property
    def scale(self):
        return numpy.array((self._sx, self._sy, self._sz), dtype=numpy.float32)

    @scale.setter
    def scale(self, scales):
        sx, sy, sz = scales
        self.setScale(sx, sy, sz)

    def setScale(self, sx=None, sy=None, sz=None):
        if sx is not None:
            assert sx != 0.
            self._sx = sx
        if sy is not None:
            assert sy != 0.
            self._sy = sy
        if sz is not None:
            assert sz != 0.
            self._sz = sz
        self.notify()


class Rotate(Transform):

    def __init__(self, angle=0., ax=0., ay=0., az=1.):
        """4x4 rotation matrix.

        :param float angle: The rotation angle in degrees.
        :param float ax: The x coordinate of the rotation axis.
        :param float ay: The y coordinate of the rotation axis.
        :param float az: The z coordinate of the rotation axis.
        """
        super(Rotate, self).__init__()
        self._angle = 0.
        self._axis = None
        self.setAngleAxis(angle, (ax, ay, az))

    @property
    def angle(self):
        """The rotation angle in degrees."""
        return self._angle

    @angle.setter
    def angle(self, angle):
        self.setAngleAxis(angle=angle)

    @property
    def axis(self):
        """The normalized rotation axis as a numpy.ndarray."""
        return self._axis.copy()

    @axis.setter
    def axis(self, axis):
        self.setAngleAxis(axis=axis)

    def setAngleAxis(self, angle=None, axis=None):
        """Update the angle and/or axis of the rotation.

        :param float angle: The rotation angle in degrees.
        :param axis: Array-like axis vector (3 coordinates).
        """
        if angle is not None:
            self._angle = angle
        if axis is not None:
            assert len(axis) == 3
            axis = numpy.array(axis, copy=True, dtype=numpy.float32)
            assert axis.size == 3
            norm = numpy.linalg.norm(axis)
            if norm == 0.:  # No axis, set rotation angle to 0.
                self._angle = 0.
                self._axis = numpy.array((0., 0., 1.), dtype=numpy.float32)
            else:
                self._axis = axis / norm

        if angle is not None or axis is not None:
            self.notify()

    @property
    def quaternion(self):
        """Rotation unit quaternion as (x, y, z, w).

        Where: ||(x, y, z)|| = sin(angle/2),  w = cos(angle/2).
        """
        if numpy.linalg.norm(self._axis) == 0.:
            return numpy.array((0., 0., 0., 1.), dtype=numpy.float32)

        else:
            quaternion = numpy.empty((4,), dtype=numpy.float32)
            halfangle = 0.5 * numpy.radians(self.angle)
            quaternion[0:3] = numpy.sin(halfangle) * self._axis
            quaternion[3] = numpy.cos(halfangle)
            return quaternion

    @quaternion.setter
    def quaternion(self, quaternion):
        assert len(quaternion) == 4

        # Normalize quaternion
        quaternion = numpy.array(quaternion, copy=True)
        quaternion /= numpy.linalg.norm(quaternion)

        # Get angle
        sinhalfangle = numpy.linalg.norm(quaternion[0:3])
        coshalfangle = quaternion[3]
        angle = 2. * numpy.arctan2(sinhalfangle, coshalfangle)

        # Axis will be normalized in setAngleAxis
        self.setAngleAxis(numpy.degrees(angle), quaternion[0:3])

    def _makeMatrix(self):
        angle = numpy.radians(self.angle, dtype=numpy.float32)
        return mat4RotateFromAngleAxis(angle, *self.axis)

    def _makeInverse(self):
        return numpy.array(self.getMatrix(copy=False).transpose(),
                           copy=True, order='C',
                           dtype=numpy.float32)


class Shear(Transform):

    def __init__(self, axis, sx=0., sy=0., sz=0.):
        """4x4 shear/skew matrix of 2 axes relative to the third one.

        :param str axis: The axis to keep fixed, in 'x', 'y', 'z'
        :param float sx: The shear factor for the x axis.
        :param float sy: The shear factor for the y axis.
        :param float sz: The shear factor for the z axis.
        """
        assert axis in ('x', 'y', 'z')
        super(Shear, self).__init__()
        self._axis = axis
        self._factors = sx, sy, sz

    @property
    def axis(self):
        """The axis against which other axes are skewed."""
        return self._axis

    @property
    def factors(self):
        """The shear factors: shearFactor = tan(shearAngle)"""
        return self._factors

    def _makeMatrix(self):
        return mat4Shear(self.axis, *self.factors)

    def _makeInverse(self):
        sx, sy, sz = self.factors
        return mat4Shear(self.axis, -sx, -sy, -sz)


# Projection ##################################################################

class _Projection(Transform):
    """Base class for projection matrix.

    Handles near and far clipping plane values.
    Subclasses must implement :meth:`_makeMatrix`.

    :param float near: Distance to the near plane.
    :param float far: Distance to the far plane.
    :param bool checkDepthExtent: Toggle checks near > 0 and far > near.
    :param size:
        Viewport's size used to compute the aspect ratio (width, height).
    :type size: 2-tuple of float
    """

    def __init__(self, near, far, checkDepthExtent=False, size=(1., 1.)):
        super(_Projection, self).__init__()
        self._checkDepthExtent = checkDepthExtent
        self._depthExtent = 1, 10
        self.setDepthExtent(near, far)  # set _depthExtent
        self._size = 1., 1.
        self.size = size  # set _size

    def setDepthExtent(self, near=None, far=None):
        """Set the extent of the visible area along the viewing direction.

        :param float near: The near clipping plane Z coord.
        :param float far: The far clipping plane Z coord.
        """
        near = float(near) if near is not None else self._depthExtent[0]
        far = float(far) if far is not None else self._depthExtent[1]

        if self._checkDepthExtent:
            assert near > 0.
            assert far > near

        self._depthExtent = near, far
        self.notify()

    @property
    def near(self):
        """Distance to the near plane."""
        return self._depthExtent[0]

    @near.setter
    def near(self, near):
        if near != self.near:
            self.setDepthExtent(near=near)

    @property
    def far(self):
        """Distance to the far plane."""
        return self._depthExtent[1]

    @far.setter
    def far(self, far):
        if far != self.far:
            self.setDepthExtent(far=far)

    @property
    def size(self):
        """Viewport size as a 2-tuple of float (width, height)."""
        return self._size

    @size.setter
    def size(self, size):
        assert len(size) == 2
        self._size = tuple(size)
        self.notify()


class Orthographic(_Projection):
    """Orthographic (i.e., parallel) projection which keeps aspect ratio.

    Clipping planes are adjusted to match the aspect ratio of
    the :attr:`size` attribute.

    The left, right, bottom and top parameters defines the area which must
    always remain visible.
    Effective clipping planes are adjusted to keep the aspect ratio.

    :param float left: Coord of the left clipping plane.
    :param float right: Coord of the right clipping plane.
    :param float bottom: Coord of the bottom clipping plane.
    :param float top: Coord of the top clipping plane.
    :param float near: Distance to the near plane.
    :param float far: Distance to the far plane.
    :param size:
        Viewport's size used to compute the aspect ratio (width, height).
    :type size: 2-tuple of float
    """

    def __init__(self, left=0., right=1., bottom=1., top=0., near=-1., far=1.,
                 size=(1., 1.)):
        self._left, self._right = left, right
        self._bottom, self._top = bottom, top
        super(Orthographic, self).__init__(near, far, checkDepthExtent=False,
                                           size=size)
        # _update called when setting size

    def _makeMatrix(self):
        return mat4Orthographic(
            self.left, self.right, self.bottom, self.top, self.near, self.far)

    def _update(self, left, right, bottom, top):
        width, height = self.size
        aspect = width / height

        orthoaspect = abs(left - right) / abs(bottom - top)

        if orthoaspect >= aspect:  # Keep width, enlarge height
            newheight = \
                numpy.sign(top - bottom) * abs(left - right) / aspect
            bottom = 0.5 * (bottom + top) - 0.5 * newheight
            top = bottom + newheight

        else:  # Keep height, enlarge width
            newwidth = \
                numpy.sign(right - left) * abs(bottom - top) * aspect
            left = 0.5 * (left + right) - 0.5 * newwidth
            right = left + newwidth

        # Store values
        self._left, self._right = left, right
        self._bottom, self._top = bottom, top

    def setClipping(self, left=None, right=None, bottom=None, top=None):
        """Set the clipping planes of the projection.

        Parameters are adjusted to keep aspect ratio.
        If a clipping plane coord is not provided, it uses its current value

        :param float left: Coord of the left clipping plane.
        :param float right: Coord of the right clipping plane.
        :param float bottom: Coord of the bottom clipping plane.
        :param float top: Coord of the top clipping plane.
        """
        left = float(left) if left is not None else self.left
        right = float(right) if right is not None else self.right
        bottom = float(bottom) if bottom is not None else self.bottom
        top = float(top) if top is not None else self.top

        self._update(left, right, bottom, top)
        self.notify()

    left = property(lambda self: self._left,
                    doc="Coord of the left clipping plane.")

    right = property(lambda self: self._right,
                     doc="Coord of the right clipping plane.")

    bottom = property(lambda self: self._bottom,
                      doc="Coord of the bottom clipping plane.")

    top = property(lambda self: self._top,
                   doc="Coord of the top clipping plane.")

    @property
    def size(self):
        """Viewport size as a 2-tuple of float (width, height) or None."""
        return self._size

    @size.setter
    def size(self, size):
        assert len(size) == 2
        self._size = float(size[0]), float(size[1])
        self._update(self.left, self.right, self.bottom, self.top)
        self.notify()


class Ortho2DWidget(_Projection):
    """Orthographic projection with pixel as unit.

    Provides same coordinates as widgets:
    origin: top left, X axis goes left, Y axis goes down.

    :param float near: Z coordinate of the near clipping plane.
    :param float far: Z coordinante of the far clipping plane.
    :param size:
        Viewport's size used to compute the aspect ratio (width, height).
    :type size: 2-tuple of float
    """

    def __init__(self, near=-1., far=1., size=(1., 1.)):

        super(Ortho2DWidget, self).__init__(near, far, size)

    def _makeMatrix(self):
        width, height = self.size
        return mat4Orthographic(0., width, height, 0., self.near, self.far)


class Perspective(_Projection):
    """Perspective projection matrix defined by FOV and aspect ratio.

    :param float fovy: Vertical field-of-view in degrees.
    :param float near: The near clipping plane Z coord (stricly positive).
    :param float far: The far clipping plane Z coord (> near).
    :param size:
        Viewport's size used to compute the aspect ratio (width, height).
    :type size: 2-tuple of float
    """

    def __init__(self, fovy=90., near=0.1, far=1., size=(1., 1.)):

        super(Perspective, self).__init__(near, far, checkDepthExtent=True)
        self._fovy = 90.
        self.fovy = fovy  # Set _fovy
        self.size = size  # Set _ size

    def _makeMatrix(self):
        width, height = self.size
        return mat4Perspective(self.fovy, width, height, self.near, self.far)

    @property
    def fovy(self):
        """Vertical field-of-view in degrees."""
        return self._fovy

    @fovy.setter
    def fovy(self, fovy):
        self._fovy = float(fovy)
        self.notify()
