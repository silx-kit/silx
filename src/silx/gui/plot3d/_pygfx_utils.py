"""Utility functions for pygfx 3D backend."""

import logging
import numpy

_logger = logging.getLogger(__name__)

# silx symbol -> pygfx marker shape mapping (same as 2D backend)
SYMBOL_MAP = {
    "o": "circle",
    ".": "circle",
    ",": "square",
    "+": "plus",
    "x": "cross",
    "d": "diamond",
    "s": "square",
    "^": "triangle_up",
    "v": "triangle_down",
    "<": "triangle_left",
    ">": "triangle_right",
    "*": "asterisk6",
}


def apply_colormap(colormap, data):
    """Apply a silx Colormap to data, returning (N, 4) float32 RGBA array.

    :param colormap: silx Colormap object
    :param data: 1D or 2D numpy array of scalar values
    :returns: RGBA array with shape (*data.shape, 4), float32 in [0, 1]
    """
    original_shape = data.shape
    flat = numpy.asarray(data, dtype=numpy.float32).ravel()

    # Get colormap range
    vmin, vmax = colormap.getColormapRange(flat)

    # Normalize data to [0, 1]
    if vmax == vmin:
        normalized = numpy.zeros_like(flat)
    else:
        normalized = numpy.clip((flat - vmin) / (vmax - vmin), 0, 1)

    # Get LUT (256 colors)
    lut = colormap.getNColors(nbColors=256)  # (256, 4) uint8
    lut_f = lut.astype(numpy.float32) / 255.0

    # Map normalized values to LUT indices
    indices = numpy.clip((normalized * 255).astype(int), 0, 255)
    colors = lut_f[indices]

    # Reshape to match original data shape
    return colors.reshape(*original_shape, 4)


def grid_to_triangles(H, W):
    """Convert (H, W) grid to triangle index array.

    Creates two triangles per grid cell for a total of (H-1)*(W-1)*2 triangles.

    :param int H: Number of rows
    :param int W: Number of columns
    :returns: (N, 3) uint32 index array
    """
    rows = numpy.arange(H - 1)
    cols = numpy.arange(W - 1)
    r, c = numpy.meshgrid(rows, cols, indexing="ij")
    r = r.ravel()
    c = c.ravel()

    # Vertex indices for each quad
    v00 = r * W + c
    v01 = r * W + (c + 1)
    v10 = (r + 1) * W + c
    v11 = (r + 1) * W + (c + 1)

    # Two triangles per quad
    tri1 = numpy.column_stack([v00, v10, v11])
    tri2 = numpy.column_stack([v00, v11, v01])
    indices = numpy.vstack([tri1, tri2]).astype(numpy.uint32)
    return indices


def compute_normals(positions, indices):
    """Compute per-vertex normals from positions and triangle indices.

    :param positions: (N, 3) float32 array of vertex positions
    :param indices: (M, 3) uint32 array of triangle indices
    :returns: (N, 3) float32 array of normalized per-vertex normals
    """
    positions = numpy.asarray(positions, dtype=numpy.float32)
    indices = numpy.asarray(indices, dtype=numpy.uint32)

    normals = numpy.zeros_like(positions)

    v0 = positions[indices[:, 0]]
    v1 = positions[indices[:, 1]]
    v2 = positions[indices[:, 2]]

    # Face normals
    face_normals = numpy.cross(v1 - v0, v2 - v0)

    # Accumulate face normals to vertices
    for i in range(3):
        numpy.add.at(normals, indices[:, i], face_normals)

    # Normalize
    lengths = numpy.linalg.norm(normals, axis=1, keepdims=True)
    lengths = numpy.maximum(lengths, 1e-10)
    normals /= lengths

    return normals


def apply_transform(item, world_object):
    """Apply Item3D's transforms to a pygfx WorldObject.

    Handles translation, scale, and rotation from DataItem3D.

    :param item: silx Item3D (DataItem3D) with transform methods
    :param world_object: pygfx WorldObject to apply transforms to
    """
    if not hasattr(item, "getTranslation"):
        return

    tx, ty, tz = item.getTranslation()
    world_object.local.position = (float(tx), float(ty), float(tz))

    sx, sy, sz = item.getScale()
    world_object.local.scale = (float(sx), float(sy), float(sz))

    angle, axis = item.getRotation()
    if angle != 0 and numpy.any(axis != 0):
        import pylinalg as la

        axis_f = numpy.asarray(axis, dtype=numpy.float64)
        norm = numpy.linalg.norm(axis_f)
        if norm > 0:
            axis_f /= norm
            angle_rad = numpy.radians(float(angle))
            quat = la.quat_from_axis_angle(axis_f, angle_rad)
            world_object.local.rotation = quat
