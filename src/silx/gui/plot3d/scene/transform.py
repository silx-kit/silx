# /*##########################################################################
#
# Copyright (c) 2015-2020 European Synchrotron Radiation Facility
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

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "25/07/2016"


from collections.abc import Iterable, Sequence
import itertools
from typing import Literal
import numpy
from numpy.typing import ArrayLike

from . import event
from .utils import Matrix4
from .utils import Vector3
from .utils import Vector4

AxisName = Literal["x", "y", "z"]

# Functions ###################################################################

# Projections


def mat4LookAtDir(position: Vector3, direction: Vector3, up: Vector3) -> Matrix4:
    """Creates matrix to look in direction from position.

    :param position: Array-like 3 coordinates of the point of view position.
    :param direction: Array-like 3 coordinates of the sight direction vector.
    :param up: Array-like 3 coordinates of the upward direction
               in the image plane.
    :returns: Corresponding 4x4 matrix.
    """
    assert len(position) == 3
    assert len(direction) == 3
    assert len(up) == 3

    direction = numpy.array(direction, copy=True, dtype=numpy.float32)
    dirnorm = numpy.linalg.norm(direction)
    assert dirnorm != 0.0
    direction /= dirnorm

    side = numpy.cross(direction, numpy.asarray(up, dtype=numpy.float32))
    sidenorm = numpy.linalg.norm(side)
    assert sidenorm != 0.0
    up = numpy.cross(side / sidenorm, direction)
    upnorm = numpy.linalg.norm(up)
    assert upnorm != 0.0
    up /= upnorm

    matrix = numpy.identity(4, dtype=numpy.float32)
    matrix[0, :3] = side
    matrix[1, :3] = up
    matrix[2, :3] = -direction
    return numpy.dot(matrix, mat4Translate(-position[0], -position[1], -position[2]))


def mat4LookAt(position: Vector3, center: Vector3, up: Vector3) -> Matrix4:
    """Creates matrix to look at center from position.

    See gluLookAt.

    :param position: Array-like 3 coordinates of the point of view position.
    :param center: Array-like 3 coordinates of the center of the scene.
    :param up: Array-like 3 coordinates of the upward direction
               in the image plane.
    :returns: Corresponding 4x4 matrix.
    """
    position = numpy.asarray(position, dtype=numpy.float32)
    center = numpy.asarray(center, dtype=numpy.float32)
    direction = center - position
    return mat4LookAtDir(position, direction, up)


def mat4Frustum(
    left: float, right: float, bottom: float, top: float, near: float, far: float
) -> Matrix4:
    """Creates a frustum projection matrix.

    See glFrustum.
    """
    return numpy.array(
        (
            (2.0 * near / (right - left), 0.0, (right + left) / (right - left), 0.0),
            (0.0, 2.0 * near / (top - bottom), (top + bottom) / (top - bottom), 0.0),
            (0.0, 0.0, -(far + near) / (far - near), -2.0 * far * near / (far - near)),
            (0.0, 0.0, -1.0, 0.0),
        ),
        dtype=numpy.float32,
    )


def mat4Perspective(
    fovy: float, width: float, height: float, near: float, far: float
) -> Matrix4:
    """Creates a perspective projection matrix.

    Similar to gluPerspective.

    :param fovy: Field of view angle in degrees in the y direction.
    :param width: Width of the viewport.
    :param height: Height of the viewport.
    :param near: Distance to the near plane (strictly positive).
    :param far: Distance to the far plane (strictly positive).
    :return: Corresponding 4x4 matrix.
    """
    assert fovy != 0
    assert height != 0
    assert width != 0
    assert near > 0.0
    assert far > near
    aspectratio = width / height
    f = 1.0 / numpy.tan(numpy.radians(fovy) / 2.0)
    return numpy.array(
        (
            (f / aspectratio, 0.0, 0.0, 0.0),
            (0.0, f, 0.0, 0.0),
            (0.0, 0.0, (far + near) / (near - far), 2.0 * far * near / (near - far)),
            (0.0, 0.0, -1.0, 0.0),
        ),
        dtype=numpy.float32,
    )


def mat4Orthographic(
    left: float, right: float, bottom: float, top: float, near: float, far: float
) -> Matrix4:
    """Creates an orthographic (i.e., parallel) projection matrix.

    See glOrtho.
    """
    return numpy.array(
        (
            (2.0 / (right - left), 0.0, 0.0, -(right + left) / (right - left)),
            (0.0, 2.0 / (top - bottom), 0.0, -(top + bottom) / (top - bottom)),
            (0.0, 0.0, -2.0 / (far - near), -(far + near) / (far - near)),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=numpy.float32,
    )


# Affine


def mat4Translate(tx: float, ty: float, tz: float) -> Matrix4:
    """4x4 translation matrix."""
    return numpy.array(
        (
            (1.0, 0.0, 0.0, tx),
            (0.0, 1.0, 0.0, ty),
            (0.0, 0.0, 1.0, tz),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=numpy.float32,
    )


def mat4Scale(sx: float, sy: float, sz: float) -> Matrix4:
    """4x4 scale matrix."""
    return numpy.array(
        (
            (sx, 0.0, 0.0, 0.0),
            (0.0, sy, 0.0, 0.0),
            (0.0, 0.0, sz, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=numpy.float32,
    )


def mat4RotateFromAngleAxis(
    angle: float, x: float = 0.0, y: float = 0.0, z: float = 1.0
) -> Matrix4:
    """4x4 rotation matrix from angle and axis.

    :param angle: The rotation angle in radians.
    :param x: The rotation vector x coordinate.
    :param y: The rotation vector y coordinate.
    :param z: The rotation vector z coordinate.
    """
    ca = numpy.cos(angle)
    sa = numpy.sin(angle)
    return numpy.array(
        (
            (
                (1.0 - ca) * x * x + ca,
                (1.0 - ca) * x * y - sa * z,
                (1.0 - ca) * x * z + sa * y,
                0.0,
            ),
            (
                (1.0 - ca) * x * y + sa * z,
                (1.0 - ca) * y * y + ca,
                (1.0 - ca) * y * z - sa * x,
                0.0,
            ),
            (
                (1.0 - ca) * x * z - sa * y,
                (1.0 - ca) * y * z + sa * x,
                (1.0 - ca) * z * z + ca,
                0.0,
            ),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=numpy.float32,
    )


def mat4RotateFromQuaternion(quaternion: Vector4) -> Matrix4:
    """4x4 rotation matrix from quaternion.

    :param quaternion: Array-like unit quaternion stored as (x, y, z, w)
    """
    quaternion = numpy.array(quaternion, copy=True)
    quaternion /= numpy.linalg.norm(quaternion)

    qx, qy, qz, qw = quaternion
    return numpy.array(
        (
            (
                1.0 - 2.0 * (qy**2 + qz**2),
                2.0 * (qx * qy - qw * qz),
                2.0 * (qx * qz + qw * qy),
                0.0,
            ),
            (
                2.0 * (qx * qy + qw * qz),
                1.0 - 2.0 * (qx**2 + qz**2),
                2.0 * (qy * qz - qw * qx),
                0.0,
            ),
            (
                2.0 * (qx * qz - qw * qy),
                2.0 * (qy * qz + qw * qx),
                1.0 - 2.0 * (qx**2 + qy**2),
                0.0,
            ),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=numpy.float32,
    )


def mat4Shear(
    axis: AxisName, sx: float = 0.0, sy: float = 0.0, sz: float = 0.0
) -> Matrix4:
    """4x4 shear matrix: Skew two axes relative to a third fixed one.

    shearFactor = tan(shearAngle)

    :param axis: The axis to keep constant and shear against.
    :param sx: The shear factor for the X axis relative to axis.
    :param sy: The shear factor for the Y axis relative to axis.
    :param sz: The shear factor for the Z axis relative to axis.
    """
    assert axis in ("x", "y", "z")

    matrix = numpy.identity(4, dtype=numpy.float32)

    # Make the shear column
    index = "xyz".find(axis)
    shearcolumn = numpy.array((sx, sy, sz, 0.0), dtype=numpy.float32)
    shearcolumn[index] = 1.0
    matrix[:, index] = shearcolumn
    return matrix


# Transforms ##################################################################


class Transform(event.Notifier):
    def __init__(self, static: bool = False):
        """Base class for (row-major) 4x4 matrix transforms.

        :param static: False (default) to reset cache when changed,
                       True for static matrices.
        """
        super().__init__()
        self._matrix = None
        self._inverse = None
        if not static:
            self.addListener(self._changed)  # Listening self for changes

    def __repr__(self) -> str:
        return f"{self.__class__.__init__}({repr(self.getMatrix(copy=False))})"

    def inverse(self) -> "Inverse":
        """Return the Transform of the inverse.

        The returned Transform is static, it is not updated when this
        Transform is modified.

        :return: A Transform which is the inverse of this Transform.
        """
        return Inverse(self)

    # Matrix

    def _makeMatrix(self) -> Matrix4:
        """Override to build matrix"""
        return numpy.identity(4, dtype=numpy.float32)

    def _makeInverse(self) -> Matrix4:
        """Override to build inverse matrix."""
        return numpy.linalg.inv(self.getMatrix(copy=False))

    def getMatrix(self, copy: bool = True) -> Matrix4:
        """The 4x4 matrix of this transform.

        :param copy: True (the default) to get a copy of the matrix,
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

    def getInverseMatrix(self, copy: bool = False) -> Matrix4:
        """The 4x4 matrix of the inverse of this transform.

        :param copy: True (the default) to get a copy of the matrix,
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
        getInverseMatrix, doc="The 4x4 matrix of the inverse of this transform."
    )

    # Listener

    def _changed(self, source):
        """Default self listener reseting matrix cache."""
        self._matrix = None
        self._inverse = None

    # Multiplication with vectors

    def transformPoints(
        self, points: ArrayLike, direct: bool = True, perspectiveDivide: bool = False
    ) -> numpy.ndarray:
        """Apply the transform to an array of points.

        :param points: 2D array of N vectors of 3 or 4 coordinates
        :param direct: Whether to apply the direct (True, the default)
                       or inverse (False) transform.
        :param perspectiveDivide: Whether to apply the perspective divide
                                  (True) or not (False, the default).
        :return: The transformed points as an array of the same shape as points.
        """
        if direct:
            matrix = self.getMatrix(copy=False)
        else:
            matrix = self.getInverseMatrix(copy=False)

        points = numpy.asarray(points)
        assert points.ndim == 2

        points = numpy.transpose(points)

        dimension = points.shape[0]
        assert dimension in (3, 4)

        if dimension == 3:  # Add 4th coordinate
            points = numpy.append(
                points, numpy.ones((1, points.shape[1]), dtype=points.dtype), axis=0
            )

        result = numpy.transpose(numpy.dot(matrix, points))

        if perspectiveDivide:
            mask = result[:, 3] != 0.0
            result[mask] /= result[mask, 3][:, numpy.newaxis]

        return result[:, :3] if dimension == 3 else result

    @staticmethod
    def _prepareVector(vector: ArrayLike, w: float) -> numpy.ndarray:
        """Add 4th coordinate (w) to vector if missing."""
        assert len(vector) in (3, 4)
        vector = numpy.asarray(vector, dtype=numpy.float32)
        if len(vector) == 3:
            vector = numpy.append(vector, w)
        return vector

    def transformPoint(
        self, point: ArrayLike, direct: bool = True, perspectiveDivide: bool = False
    ) -> numpy.ndarray:
        """Apply the transform to a point.

        :param point: Array-like vector of 3 or 4 coordinates.
        :param direct: Whether to apply the direct (True, the default)
                       or inverse (False) transform.
        :param perspectiveDivide: Whether to apply the perspective divide
                                  (True) or not (False, the default).
        :return: The transformed point as an array of same length as point.
        """
        if direct:
            matrix = self.getMatrix(copy=False)
        else:
            matrix = self.getInverseMatrix(copy=False)
        result = numpy.dot(matrix, self._prepareVector(point, 1.0))

        if perspectiveDivide and result[3] != 0.0:
            result /= result[3]

        if len(point) == 3:
            return result[:3]
        else:
            return result

    def transformDir(self, direction: Vector3, direct: bool = True) -> numpy.ndarray:
        """Apply the transform to a direction.

        :param direction: Array-like vector of 3 coordinates.
        :param direct: Whether to apply the direct (True, the default)
                       or inverse (False) transform.
        :return: The transformed direction as an array of length 3.
        """
        if direct:
            matrix = self.getMatrix(copy=False)
        else:
            matrix = self.getInverseMatrix(copy=False)
        return numpy.dot(matrix[:3, :3], direction[:3])

    def transformNormal(self, normal: Vector3, direct: bool = True) -> numpy.ndarray:
        """Apply the transform to a normal: R = (M-1)t * V.

        :param normal: Array-like vector of 3 coordinates.
        :param direct: Whether to apply the direct (True, the default)
                       or inverse (False) transform.
        :return: The transformed normal as an array of length 3.
        """
        if direct:
            matrix = self.getInverseMatrix(copy=False).T
        else:
            matrix = self.getMatrix(copy=False).T
        return numpy.dot(matrix[:3, :3], normal[:3])

    _CUBE_CORNERS = numpy.array(
        list(itertools.product((0.0, 1.0), repeat=3)), dtype=numpy.float32
    )
    """Unit cube corners used by :meth:`transformBounds`"""

    def transformBounds(self, bounds: ArrayLike, direct: bool = True) -> numpy.ndarray:
        """Apply the transform to an axes-aligned rectangular box.

        :param bounds: Min and max coords of the box for each axes as a 2x3 array
        :param direct: Whether to apply the direct (True, the default)
                       or inverse (False) transform.
        :return: Axes-aligned rectangular box including the transformed box.
                 as a 2x3 array of float32
        """
        corners = numpy.ones((8, 4), dtype=numpy.float32)
        corners[:, :3] = bounds[0] + self._CUBE_CORNERS * (bounds[1] - bounds[0])

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

    def __init__(self, transform: Transform):
        """Initializer.

        :param transform: The transform to invert.
        """

        super().__init__(static=True)
        self._matrix = transform.getInverseMatrix(copy=True)
        self._inverse = transform.getMatrix(copy=True)


class TransformList(Transform, event.HookList):
    """List of transforms."""

    def __init__(self, iterable: Iterable[Transform] = ()):
        Transform.__init__(self)
        event.HookList.__init__(self, iterable)

    def _listWillChangeHook(self, methodName: str, *args, **kwargs):
        for item in self:
            item.removeListener(self._transformChanged)

    def _listWasChangedHook(self, methodName: str, *args, **kwargs):
        for item in self:
            item.addListener(self._transformChanged)
        self.notify()

    def _transformChanged(self, source: Transform):
        """Listen to transform changes of the list and its items."""
        if source is not self:  # Avoid infinite recursion
            self.notify()

    def _makeMatrix(self) -> Matrix4:
        matrix = numpy.identity(4, dtype=numpy.float32)
        for transform in self:
            matrix = numpy.dot(matrix, transform.getMatrix(copy=False))
        return matrix


class StaticTransformList(Transform):
    """Transform that is a snapshot of a list of Transforms

    It does not keep reference to the list of Transforms.

    :param iterable: Iterable of Transform used for initialization
    """

    def __init__(self, iterable: Iterable[Transform] = ()):
        super().__init__(static=True)
        matrix = numpy.identity(4, dtype=numpy.float32)
        for transform in iterable:
            matrix = numpy.dot(matrix, transform.getMatrix(copy=False))
        self._matrix = matrix  # Init matrix once


# Affine ######################################################################


class Matrix(Transform):
    def __init__(self, matrix: ArrayLike | None = None):
        """4x4 Matrix.

        :param matrix: 4x4 array-like matrix or None for identity matrix.
        """
        super().__init__(static=True)
        self.setMatrix(matrix)

    def setMatrix(self, matrix: ArrayLike | None = None) -> None:
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
    matrix = property(
        Transform.getMatrix, setMatrix, doc="The 4x4 matrix of this transform."
    )


class Translate(Transform):
    """4x4 translation matrix."""

    def __init__(self, tx: float = 0.0, ty: float = 0.0, tz: float = 0.0):
        super().__init__()
        self._tx, self._ty, self._tz = 0.0, 0.0, 0.0
        self.setTranslate(tx, ty, tz)

    def _makeMatrix(self) -> Matrix4:
        return mat4Translate(self.tx, self.ty, self.tz)

    def _makeInverse(self) -> Matrix4:
        return mat4Translate(-self.tx, -self.ty, -self.tz)

    @property
    def tx(self) -> float:
        return self._tx

    @tx.setter
    def tx(self, tx: float) -> None:
        self.setTranslate(tx=tx)

    @property
    def ty(self) -> float:
        return self._ty

    @ty.setter
    def ty(self, ty: float) -> None:
        self.setTranslate(ty=ty)

    @property
    def tz(self) -> float:
        return self._tz

    @tz.setter
    def tz(self, tz: float) -> None:
        self.setTranslate(tz=tz)

    @property
    def translation(self) -> numpy.ndarray:
        return numpy.array((self.tx, self.ty, self.tz), dtype=numpy.float32)

    @translation.setter
    def translation(self, translations: Vector3) -> None:
        tx, ty, tz = translations
        self.setTranslate(tx, ty, tz)

    def setTranslate(
        self, tx: float | None = None, ty: float | None = None, tz: float | None = None
    ) -> None:
        if tx is not None:
            self._tx = tx
        if ty is not None:
            self._ty = ty
        if tz is not None:
            self._tz = tz
        self.notify()


class Scale(Transform):
    """4x4 scale matrix."""

    def __init__(self, sx: float = 1.0, sy: float = 1.0, sz: float = 1.0):
        super().__init__()
        self._sx, self._sy, self._sz = 0.0, 0.0, 0.0
        self.setScale(sx, sy, sz)

    def _makeMatrix(self) -> Matrix4:
        return mat4Scale(self.sx, self.sy, self.sz)

    def _makeInverse(self) -> Matrix4:
        return mat4Scale(1.0 / self.sx, 1.0 / self.sy, 1.0 / self.sz)

    @property
    def sx(self) -> float:
        return self._sx

    @sx.setter
    def sx(self, sx: float) -> None:
        self.setScale(sx=sx)

    @property
    def sy(self) -> float:
        return self._sy

    @sy.setter
    def sy(self, sy: float) -> None:
        self.setScale(sy=sy)

    @property
    def sz(self) -> float:
        return self._sz

    @sz.setter
    def sz(self, sz: float) -> None:
        self.setScale(sz=sz)

    @property
    def scale(self) -> numpy.ndarray:
        return numpy.array((self._sx, self._sy, self._sz), dtype=numpy.float32)

    @scale.setter
    def scale(self, scales: Vector3) -> None:
        sx, sy, sz = scales
        self.setScale(sx, sy, sz)

    def setScale(
        self, sx: float | None = None, sy: float | None = None, sz: float | None = None
    ) -> None:
        if sx is not None:
            assert sx != 0.0
            self._sx = sx
        if sy is not None:
            assert sy != 0.0
            self._sy = sy
        if sz is not None:
            assert sz != 0.0
            self._sz = sz
        self.notify()


class Rotate(Transform):
    def __init__(
        self, angle: float = 0.0, ax: float = 0.0, ay: float = 0.0, az: float = 1.0
    ):
        """4x4 rotation matrix.

        :param angle: The rotation angle in degrees.
        :param ax: The x coordinate of the rotation axis.
        :param ay: The y coordinate of the rotation axis.
        :param az: The z coordinate of the rotation axis.
        """
        super().__init__()
        self._angle = 0.0
        self._axis = None
        self.setAngleAxis(angle, (ax, ay, az))

    @property
    def angle(self) -> float:
        """The rotation angle in degrees."""
        return self._angle

    @angle.setter
    def angle(self, angle: float) -> None:
        self.setAngleAxis(angle=angle)

    @property
    def axis(self) -> numpy.ndarray:
        """The normalized rotation axis as a numpy.ndarray."""
        return self._axis.copy()

    @axis.setter
    def axis(self, axis: Vector3) -> None:
        self.setAngleAxis(axis=axis)

    def setAngleAxis(
        self, angle: float | None = None, axis: Vector3 | None = None
    ) -> None:
        """Update the angle and/or axis of the rotation.

        :param angle: The rotation angle in degrees.
        :param axis: Array-like axis vector.
        """
        if angle is not None:
            self._angle = angle
        if axis is not None:
            assert len(axis) == 3
            axis = numpy.array(axis, copy=True, dtype=numpy.float32)
            assert axis.size == 3
            norm = numpy.linalg.norm(axis)
            if norm == 0.0:  # No axis, set rotation angle to 0.
                self._angle = 0.0
                self._axis = numpy.array((0.0, 0.0, 1.0), dtype=numpy.float32)
            else:
                self._axis = axis / norm

        if angle is not None or axis is not None:
            self.notify()

    @property
    def quaternion(self) -> numpy.ndarray:
        """Rotation unit quaternion as (x, y, z, w).

        Where: ||(x, y, z)|| = sin(angle/2),  w = cos(angle/2).
        """
        if numpy.linalg.norm(self._axis) == 0.0:
            return numpy.array((0.0, 0.0, 0.0, 1.0), dtype=numpy.float32)

        else:
            quaternion = numpy.empty((4,), dtype=numpy.float32)
            halfangle = 0.5 * numpy.radians(self.angle)
            quaternion[0:3] = numpy.sin(halfangle) * self._axis
            quaternion[3] = numpy.cos(halfangle)
            return quaternion

    @quaternion.setter
    def quaternion(self, quaternion: Vector4):
        assert len(quaternion) == 4

        # Normalize quaternion
        quaternion = numpy.array(quaternion, copy=True)
        quaternion /= numpy.linalg.norm(quaternion)

        # Get angle
        sinhalfangle = numpy.linalg.norm(quaternion[0:3])
        coshalfangle = quaternion[3]
        angle = 2.0 * numpy.arctan2(sinhalfangle, coshalfangle)

        # Axis will be normalized in setAngleAxis
        self.setAngleAxis(numpy.degrees(angle), quaternion[0:3])

    def _makeMatrix(self) -> Matrix4:
        angle = numpy.radians(self.angle, dtype=numpy.float32)
        return mat4RotateFromAngleAxis(angle, *self.axis)

    def _makeInverse(self) -> Matrix4:
        return numpy.array(
            self.getMatrix(copy=False).transpose(),
            copy=True,
            order="C",
            dtype=numpy.float32,
        )


class Shear(Transform):
    def __init__(
        self, axis: AxisName, sx: float = 0.0, sy: float = 0.0, sz: float = 0.0
    ):
        """4x4 shear/skew matrix of 2 axes relative to the third one.

        :param axis: The axis to keep fixed, in 'x', 'y', 'z'
        :param sx: The shear factor for the x axis.
        :param sy: The shear factor for the y axis.
        :param sz: The shear factor for the z axis.
        """
        assert axis in ("x", "y", "z")
        super().__init__()
        self._axis = axis
        self._factors = sx, sy, sz

    @property
    def axis(self) -> AxisName:
        """The axis against which other axes are skewed."""
        return self._axis

    @property
    def factors(self) -> float:
        """The shear factors: shearFactor = tan(shearAngle)"""
        return self._factors

    def _makeMatrix(self) -> Matrix4:
        return mat4Shear(self.axis, *self.factors)

    def _makeInverse(self) -> Matrix4:
        sx, sy, sz = self.factors
        return mat4Shear(self.axis, -sx, -sy, -sz)


# Projection ##################################################################


class _Projection(Transform):
    """Base class for projection matrix.

    Handles near and far clipping plane values.
    Subclasses must implement :meth:`_makeMatrix`.

    :param near: Distance to the near plane.
    :param far: Distance to the far plane.
    :param checkDepthExtent: Toggle checks near > 0 and far > near.
    :param size:
        Viewport's size used to compute the aspect ratio (width, height).
    """

    def __init__(
        self,
        near: float,
        far: float,
        checkDepthExtent: bool = False,
        size: tuple[float, float] = (1.0, 1.0),
    ):
        super().__init__()
        self._checkDepthExtent = checkDepthExtent
        self._depthExtent = 1, 10
        self.setDepthExtent(near, far)  # set _depthExtent
        self._size = 1.0, 1.0
        self.size = size  # set _size

    def setDepthExtent(self, near: float | None = None, far: float | None = None):
        """Set the extent of the visible area along the viewing direction.

        :param near: The near clipping plane Z coord.
        :param far: The far clipping plane Z coord.
        """
        near = float(near) if near is not None else self._depthExtent[0]
        far = float(far) if far is not None else self._depthExtent[1]

        if self._checkDepthExtent:
            assert near > 0.0
            assert far > near

        self._depthExtent = near, far
        self.notify()

    @property
    def near(self) -> float:
        """Distance to the near plane."""
        return self._depthExtent[0]

    @near.setter
    def near(self, near: float) -> None:
        if near != self.near:
            self.setDepthExtent(near=near)

    @property
    def far(self) -> float:
        """Distance to the far plane."""
        return self._depthExtent[1]

    @far.setter
    def far(self, far: float) -> None:
        if far != self.far:
            self.setDepthExtent(far=far)

    @property
    def size(self) -> tuple[float, float]:
        """Viewport size (width, height)."""
        return self._size

    @size.setter
    def size(self, size: Sequence[float]) -> None:
        assert len(size) == 2
        self._size = tuple(size)
        self.notify()


class Orthographic(_Projection):
    """Orthographic (i.e., parallel) projection which can keep aspect ratio.

    Clipping planes are adjusted to match the aspect ratio of
    the :attr:`size` attribute if :attr:`keepaspect` is True.

    In this case, the left, right, bottom and top parameters defines the area
    which must always remain visible.
    Effective clipping planes are adjusted to keep the aspect ratio.

    :param left: Coord of the left clipping plane.
    :param right: Coord of the right clipping plane.
    :param bottom: Coord of the bottom clipping plane.
    :param top: Coord of the top clipping plane.
    :param near: Distance to the near plane.
    :param far: Distance to the far plane.
    :param size:
        Viewport's size used to compute the aspect ratio (width, height).
    :param keepaspect: True (default) to keep aspect ratio, False otherwise.
    """

    def __init__(
        self,
        left: float = 0.0,
        right: float = 1.0,
        bottom: float = 1.0,
        top: float = 0.0,
        near: float = -1.0,
        far: float = 1.0,
        size: tuple[float, float] = (1.0, 1.0),
        keepaspect: bool = True,
    ):
        self._left, self._right = left, right
        self._bottom, self._top = bottom, top
        self._keepaspect = bool(keepaspect)
        super().__init__(near, far, checkDepthExtent=False, size=size)
        # _update called when setting size

    def _makeMatrix(self) -> Matrix4:
        return mat4Orthographic(
            self.left, self.right, self.bottom, self.top, self.near, self.far
        )

    def _update(self, left: float, right: float, bottom: float, top: float) -> None:
        if self.keepaspect:
            width, height = self.size
            aspect = width / height

            orthoaspect = abs(left - right) / abs(bottom - top)

            if orthoaspect >= aspect:  # Keep width, enlarge height
                newheight = numpy.sign(top - bottom) * abs(left - right) / aspect
                bottom = 0.5 * (bottom + top) - 0.5 * newheight
                top = bottom + newheight

            else:  # Keep height, enlarge width
                newwidth = numpy.sign(right - left) * abs(bottom - top) * aspect
                left = 0.5 * (left + right) - 0.5 * newwidth
                right = left + newwidth

        # Store values
        self._left, self._right = left, right
        self._bottom, self._top = bottom, top

    def setClipping(
        self,
        left: float | None = None,
        right: float | None = None,
        bottom: float | None = None,
        top: float | None = None,
    ) -> None:
        """Set the clipping planes of the projection.

        Parameters are adjusted to keep aspect ratio.
        If a clipping plane coord is not provided, it uses its current value

        :param left: Coord of the left clipping plane.
        :param right: Coord of the right clipping plane.
        :param bottom: Coord of the bottom clipping plane.
        :param top: Coord of the top clipping plane.
        """
        left = float(left) if left is not None else self.left
        right = float(right) if right is not None else self.right
        bottom = float(bottom) if bottom is not None else self.bottom
        top = float(top) if top is not None else self.top

        self._update(left, right, bottom, top)
        self.notify()

    left = property(lambda self: self._left, doc="Coord of the left clipping plane.")

    right = property(lambda self: self._right, doc="Coord of the right clipping plane.")

    bottom = property(
        lambda self: self._bottom, doc="Coord of the bottom clipping plane."
    )

    top = property(lambda self: self._top, doc="Coord of the top clipping plane.")

    @property
    def size(self) -> tuple[float, float]:
        """Viewport size (width, height)"""
        return self._size

    @size.setter
    def size(self, size: Sequence[float]) -> None:
        assert len(size) == 2
        size = float(size[0]), float(size[1])
        if size != self._size:
            self._size = size
            self._update(self.left, self.right, self.bottom, self.top)
            self.notify()

    @property
    def keepaspect(self) -> bool:
        """True to keep aspect ratio, False otherwise."""
        return self._keepaspect

    @keepaspect.setter
    def keepaspect(self, aspect: bool) -> None:
        aspect = bool(aspect)
        if aspect != self._keepaspect:
            self._keepaspect = aspect
            self._update(self.left, self.right, self.bottom, self.top)
            self.notify()


class Ortho2DWidget(_Projection):
    """Orthographic projection with pixel as unit.

    Provides same coordinates as widgets:
    origin: top left, X axis goes left, Y axis goes down.

    :param near: Z coordinate of the near clipping plane.
    :param far: Z coordinante of the far clipping plane.
    :param size:
        Viewport's size used to compute the aspect ratio (width, height).
    """

    def __init__(
        self,
        near: float = -1.0,
        far: float = 1.0,
        size: tuple[float, float] = (1.0, 1.0),
    ):
        super().__init__(near, far, size)

    def _makeMatrix(self) -> Matrix4:
        width, height = self.size
        return mat4Orthographic(0.0, width, height, 0.0, self.near, self.far)


class Perspective(_Projection):
    """Perspective projection matrix defined by FOV and aspect ratio.

    :param fovy: Vertical field-of-view in degrees.
    :param near: The near clipping plane Z coord (stricly positive).
    :param far: The far clipping plane Z coord (> near).
    :param size:
        Viewport's size used to compute the aspect ratio (width, height).
    """

    def __init__(
        self,
        fovy: float = 90.0,
        near: float = 0.1,
        far: float = 1.0,
        size: tuple[float, float] = (1.0, 1.0),
    ):
        super().__init__(near, far, checkDepthExtent=True)
        self._fovy = 90.0
        self.fovy = fovy  # Set _fovy
        self.size = size  # Set _ size

    def _makeMatrix(self) -> Matrix4:
        width, height = self.size
        return mat4Perspective(self.fovy, width, height, self.near, self.far)

    @property
    def fovy(self) -> float:
        """Vertical field-of-view in degrees."""
        return self._fovy

    @fovy.setter
    def fovy(self, fovy: float) -> None:
        self._fovy = float(fovy)
        self.notify()
