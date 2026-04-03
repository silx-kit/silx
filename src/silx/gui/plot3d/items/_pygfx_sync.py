"""Item3D -> pygfx WorldObject conversion functions."""

import logging

import numpy

import pygfx as gfx

from .._pygfx_utils import (
    SYMBOL_MAP,
    apply_colormap,
    apply_transform,
    compute_normals,
    grid_to_triangles,
)

_logger = logging.getLogger(__name__)


def sync_item(item, clip_planes=None):
    """Convert an Item3D to a pygfx WorldObject based on its type.

    :param item: silx Item3D instance
    :param clip_planes: list of (a, b, c, d) plane equations for clipping
    :returns: pygfx WorldObject or None
    """
    from . import (
        Scatter3D,
        Scatter2D,
        Mesh,
        ColormapMesh,
        Box,
        Cylinder,
        Hexagon,
        ImageData,
        ImageRgba,
        HeightMapData,
        HeightMapRGBA,
        ScalarField3D,
        GroupItem,
        ClipPlane,
    )

    obj = None

    if isinstance(item, ClipPlane):
        return None  # Handled by sync_group

    elif isinstance(item, GroupItem):
        obj = sync_group(item, clip_planes)

    elif isinstance(item, Scatter3D):
        obj = sync_scatter3d(item)

    elif isinstance(item, Scatter2D):
        obj = sync_scatter2d(item)

    elif isinstance(item, Box):
        obj = sync_box(item)

    elif isinstance(item, Cylinder):
        obj = sync_cylinder(item)

    elif isinstance(item, Hexagon):
        obj = sync_hexagon(item)

    elif isinstance(item, ColormapMesh):
        obj = sync_colormap_mesh(item)

    elif isinstance(item, Mesh):
        obj = sync_mesh(item)

    elif isinstance(item, HeightMapData):
        obj = sync_heightmap_data(item)

    elif isinstance(item, HeightMapRGBA):
        obj = sync_heightmap_rgba(item)

    elif isinstance(item, ImageData):
        obj = sync_image_data(item)

    elif isinstance(item, ImageRgba):
        obj = sync_image_rgba(item)

    elif isinstance(item, ScalarField3D):
        obj = sync_scalar_field_3d(item)

    else:
        _logger.warning("Unsupported item type for pygfx sync: %s", type(item).__name__)
        return None

    if obj is not None:
        obj.visible = item.isVisible()
        apply_transform(item, obj)

        # Apply clipping planes to materials
        if clip_planes:
            _apply_clip_planes(obj, clip_planes)

    return obj


def _apply_clip_planes(obj, clip_planes):
    """Recursively apply clipping planes to all materials in an object tree."""
    if hasattr(obj, "material") and obj.material is not None:
        if hasattr(obj.material, "clipping_planes"):
            obj.material.clipping_planes = [tuple(p) for p in clip_planes]
            obj.material.clipping_mode = "any"
    if hasattr(obj, "children"):
        for child in obj.children:
            _apply_clip_planes(child, clip_planes)


# --- Mesh items ---


def sync_mesh(item):
    """Convert a Mesh item to pygfx.Mesh.

    :param item: silx Mesh item
    :returns: pygfx.Mesh
    """
    positions = item.getPositionData(copy=False)
    if positions is None or len(positions) == 0:
        return None

    colors = item.getColorData(copy=False)
    normals = item.getNormalData(copy=False)
    indices = item.getIndices(copy=False)
    mode = item.getDrawMode()

    positions = numpy.ascontiguousarray(positions, dtype=numpy.float32)

    # Handle color: can be single color or per-vertex
    if colors is not None:
        colors = numpy.asarray(colors, dtype=numpy.float32)
        if colors.ndim == 1:
            # Single color for all vertices
            if len(colors) == 3:
                color = (*colors, 1.0)
            else:
                color = tuple(colors)
            geo = gfx.Geometry(positions=positions)
            if normals is not None:
                normals = numpy.ascontiguousarray(normals, dtype=numpy.float32)
                if normals.ndim == 1:
                    # Broadcast single normal
                    normals = numpy.tile(normals, (len(positions), 1))
                geo = gfx.Geometry(positions=positions, normals=normals)
            mat = gfx.MeshPhongMaterial(color=color)
        else:
            # Per-vertex colors
            if colors.shape[1] == 3:
                alpha = numpy.ones((len(colors), 1), dtype=numpy.float32)
                colors = numpy.hstack([colors, alpha])
            colors = numpy.ascontiguousarray(colors, dtype=numpy.float32)
            kwargs = {"positions": positions, "colors": colors}
            if normals is not None:
                normals = numpy.ascontiguousarray(normals, dtype=numpy.float32)
                if normals.ndim == 1:
                    normals = numpy.tile(normals, (len(positions), 1))
                kwargs["normals"] = normals
            geo = gfx.Geometry(**kwargs)
            mat = gfx.MeshPhongMaterial(color_mode="vertex")
    else:
        geo = gfx.Geometry(positions=positions)
        mat = gfx.MeshPhongMaterial(color=(0.8, 0.8, 0.8, 1.0))

    # Handle triangle strip/fan by expanding to triangles
    if mode == "triangle_strip" and indices is None:
        indices = _strip_to_triangles(len(positions))
    elif mode == "fan" and indices is None:
        indices = _fan_to_triangles(len(positions))
    elif mode == "triangles" and indices is None:
        # pygfx requires explicit indices
        indices = numpy.arange(len(positions), dtype=numpy.uint32).reshape(-1, 3)

    if indices is not None:
        indices = numpy.ascontiguousarray(indices, dtype=numpy.uint32)
        if indices.ndim == 1:
            indices = indices.reshape(-1, 3)
        geo.indices = gfx.Buffer(indices)

    return gfx.Mesh(geo, mat)


def sync_colormap_mesh(item):
    """Convert a ColormapMesh item to pygfx.Mesh with colormapped vertex colors."""
    positions = item.getPositionData(copy=False)
    if positions is None or len(positions) == 0:
        return None

    values = item.getValueData(copy=False)
    normals = item.getNormalData(copy=False)
    indices = item.getIndices(copy=False)

    positions = numpy.ascontiguousarray(positions, dtype=numpy.float32)

    # Apply colormap to get per-vertex colors
    colors = apply_colormap(item.getColormap(), values.ravel())
    colors = numpy.ascontiguousarray(colors, dtype=numpy.float32)

    kwargs = {"positions": positions, "colors": colors}
    if normals is not None:
        normals = numpy.ascontiguousarray(normals, dtype=numpy.float32)
        if normals.ndim == 1:
            normals = numpy.tile(normals, (len(positions), 1))
        kwargs["normals"] = normals

    geo = gfx.Geometry(**kwargs)

    if indices is not None:
        indices = numpy.ascontiguousarray(indices, dtype=numpy.uint32)
        if indices.ndim == 1:
            indices = indices.reshape(-1, 3)
        geo.indices = gfx.Buffer(indices)

    mat = gfx.MeshPhongMaterial(color_mode="vertex")
    return gfx.Mesh(geo, mat)


def _strip_to_triangles(n):
    """Convert triangle strip vertex count to triangle indices."""
    indices = []
    for i in range(n - 2):
        if i % 2 == 0:
            indices.append([i, i + 1, i + 2])
        else:
            indices.append([i, i + 2, i + 1])
    return numpy.array(indices, dtype=numpy.uint32)


def _fan_to_triangles(n):
    """Convert triangle fan vertex count to triangle indices."""
    indices = []
    for i in range(1, n - 1):
        indices.append([0, i, i + 1])
    return numpy.array(indices, dtype=numpy.uint32)


def sync_box(item):
    """Convert a Box item to pygfx.Mesh using box_geometry."""
    size = item.getSize()
    color = item.getColor(copy=False)
    positions = item.getPosition(copy=False)

    if len(color) == 3:
        color_rgba = (*color, 1.0)
    else:
        color_rgba = tuple(color[:4])

    group = gfx.Group()

    for pos in positions:
        geo = gfx.box_geometry(float(size[0]), float(size[1]), float(size[2]))
        mat = gfx.MeshPhongMaterial(color=color_rgba)
        if color_rgba[3] < 1.0:
            mat.opacity = color_rgba[3]
            mat.transparent = True
        mesh = gfx.Mesh(geo, mat)
        mesh.local.position = (float(pos[0]), float(pos[1]), float(pos[2]))
        group.add(mesh)

    if len(positions) == 1:
        # For single box, return mesh directly (simpler transform)
        mesh = group.children[0]
        group.remove(mesh)
        return mesh

    return group


def sync_cylinder(item):
    """Convert a Cylinder item to pygfx.Mesh using cylinder_geometry."""
    radius = item.getRadius()
    height = item.getHeight()
    color = item.getColor(copy=False)
    positions = item.getPosition(copy=False)

    if len(color) == 3:
        color_rgba = (*color, 1.0)
    else:
        color_rgba = tuple(color[:4])

    group = gfx.Group()

    for pos in positions:
        geo = gfx.cylinder_geometry(
            radius_bottom=float(radius),
            radius_top=float(radius),
            height=float(height),
            radial_segments=20,
        )
        mat = gfx.MeshPhongMaterial(color=color_rgba)
        if color_rgba[3] < 1.0:
            mat.opacity = color_rgba[3]
            mat.transparent = True
        mesh = gfx.Mesh(geo, mat)
        mesh.local.position = (float(pos[0]), float(pos[1]), float(pos[2]))
        group.add(mesh)

    if len(positions) == 1:
        mesh = group.children[0]
        group.remove(mesh)
        return mesh

    return group


def sync_hexagon(item):
    """Convert a Hexagon item to pygfx.Mesh using cylinder_geometry with 6 segments."""
    radius = item.getRadius()
    height = item.getHeight()
    color = item.getColor(copy=False)
    positions = item.getPosition(copy=False)

    if len(color) == 3:
        color_rgba = (*color, 1.0)
    else:
        color_rgba = tuple(color[:4])

    group = gfx.Group()

    for pos in positions:
        geo = gfx.cylinder_geometry(
            radius_bottom=float(radius),
            radius_top=float(radius),
            height=float(height),
            radial_segments=6,
        )
        mat = gfx.MeshPhongMaterial(color=color_rgba)
        if color_rgba[3] < 1.0:
            mat.opacity = color_rgba[3]
            mat.transparent = True
        mesh = gfx.Mesh(geo, mat)
        mesh.local.position = (float(pos[0]), float(pos[1]), float(pos[2]))
        group.add(mesh)

    if len(positions) == 1:
        mesh = group.children[0]
        group.remove(mesh)
        return mesh

    return group


# --- Scatter items ---


def sync_scatter3d(item):
    """Convert a Scatter3D item to pygfx.Points."""
    x, y, z, value = item.getData(copy=False)
    if x is None or len(x) == 0:
        return None

    positions = numpy.column_stack(
        [
            numpy.asarray(x, dtype=numpy.float32),
            numpy.asarray(y, dtype=numpy.float32),
            numpy.asarray(z, dtype=numpy.float32),
        ]
    )
    positions = numpy.ascontiguousarray(positions)

    # Apply colormap
    colors = apply_colormap(
        item.getColormap(), numpy.asarray(value, dtype=numpy.float32)
    )
    colors = numpy.ascontiguousarray(colors, dtype=numpy.float32)

    geo = gfx.Geometry(positions=positions, colors=colors)

    symbol = item.getSymbol()
    marker = SYMBOL_MAP.get(symbol, "circle")
    size = float(item.getSymbolSize())

    mat = gfx.PointsMarkerMaterial(
        marker=marker,
        size=size,
        color_mode="vertex",
        size_space="screen",
    )

    return gfx.Points(geo, mat)


def sync_scatter2d(item):
    """Convert a Scatter2D item to pygfx WorldObject.

    Supports solid, lines, and points visualization modes.
    """
    x = numpy.asarray(item.getXData(copy=False), dtype=numpy.float32)
    y = numpy.asarray(item.getYData(copy=False), dtype=numpy.float32)
    value = numpy.asarray(item.getValueData(copy=False), dtype=numpy.float32)

    if len(x) == 0:
        return None

    height_map = item.isHeightMap()
    z = value if height_map else numpy.zeros_like(x)

    positions = numpy.column_stack([x, y, z])
    positions = numpy.ascontiguousarray(positions, dtype=numpy.float32)

    colors = apply_colormap(item.getColormap(), value)
    colors = numpy.ascontiguousarray(colors, dtype=numpy.float32)

    vis = item.getVisualization()
    vis_name = vis.value if hasattr(vis, "value") else str(vis)

    if vis_name == "solid":
        return _scatter2d_solid(positions, colors)
    elif vis_name == "lines":
        return _scatter2d_lines(item, positions, colors, x, y)
    else:  # points
        return _scatter2d_points(item, positions, colors)


def _scatter2d_solid(positions, colors):
    """Create solid surface from 2D scatter data using Delaunay triangulation."""
    try:
        from scipy.spatial import Delaunay

        points_2d = positions[:, :2]
        tri = Delaunay(points_2d)
        indices = numpy.ascontiguousarray(tri.simplices.astype(numpy.uint32))
    except ImportError:
        _logger.warning("scipy not available, falling back to grid triangulation")
        # Try grid-based triangulation if data is on a grid
        n = int(numpy.sqrt(len(positions)))
        if n * n == len(positions):
            indices = grid_to_triangles(n, n)
        else:
            return None
    except Exception:
        _logger.warning("Delaunay triangulation failed")
        return None

    normals = compute_normals(positions, indices)

    geo = gfx.Geometry(
        positions=positions,
        normals=normals,
        colors=colors,
        indices=gfx.Buffer(indices),
    )
    mat = gfx.MeshPhongMaterial(color_mode="vertex")
    return gfx.Mesh(geo, mat)


def _scatter2d_lines(item, positions, colors, x, y):
    """Create wireframe from 2D scatter data."""
    # Try to detect grid structure
    unique_x = numpy.unique(x)
    unique_y = numpy.unique(y)
    nx, ny = len(unique_x), len(unique_y)

    if nx * ny == len(x):
        # Grid data - create line segments along rows and columns
        group = gfx.Group()

        # Reshape to grid
        pos_grid = positions.reshape(ny, nx, 3)
        col_grid = colors.reshape(ny, nx, 4)

        # Row lines
        for j in range(ny):
            row_pos = numpy.ascontiguousarray(pos_grid[j], dtype=numpy.float32)
            row_col = numpy.ascontiguousarray(col_grid[j], dtype=numpy.float32)
            geo = gfx.Geometry(positions=row_pos, colors=row_col)
            mat = gfx.LineMaterial(thickness=item.getLineWidth(), color_mode="vertex")
            group.add(gfx.Line(geo, mat))

        # Column lines
        for i in range(nx):
            col_pos = numpy.ascontiguousarray(pos_grid[:, i], dtype=numpy.float32)
            col_col = numpy.ascontiguousarray(col_grid[:, i], dtype=numpy.float32)
            geo = gfx.Geometry(positions=col_pos, colors=col_col)
            mat = gfx.LineMaterial(thickness=item.getLineWidth(), color_mode="vertex")
            group.add(gfx.Line(geo, mat))

        return group
    else:
        # Non-grid data - just connect points in order
        geo = gfx.Geometry(positions=positions, colors=colors)
        mat = gfx.LineMaterial(thickness=item.getLineWidth(), color_mode="vertex")
        return gfx.Line(geo, mat)


def _scatter2d_points(item, positions, colors):
    """Create point cloud from 2D scatter data."""
    geo = gfx.Geometry(positions=positions, colors=colors)

    symbol = item.getSymbol() if hasattr(item, "getSymbol") else "o"
    marker = SYMBOL_MAP.get(symbol, "circle")
    size = float(item.getSymbolSize()) if hasattr(item, "getSymbolSize") else 6.0

    mat = gfx.PointsMarkerMaterial(
        marker=marker,
        size=size,
        color_mode="vertex",
        size_space="screen",
    )
    return gfx.Points(geo, mat)


# --- Image items ---


def sync_image_data(item):
    """Convert ImageData item to pygfx.Image."""
    data = item.getData(copy=False)
    if data is None:
        return None

    colors = apply_colormap(item.getColormap(), data)
    colors = numpy.ascontiguousarray(colors, dtype=numpy.float32)

    tex = gfx.Texture(colors, dim=2)
    geo = gfx.Geometry(grid=tex)
    mat = gfx.ImageBasicMaterial(clim=(0, 1))
    return gfx.Image(geo, mat)


def sync_image_rgba(item):
    """Convert ImageRgba item to pygfx.Image."""
    data = item.getData(copy=False)
    if data is None:
        return None

    data = numpy.asarray(data)
    if data.dtype == numpy.uint8:
        data = data.astype(numpy.float32) / 255.0

    if data.ndim == 3 and data.shape[2] == 3:
        # Add alpha channel
        alpha = numpy.ones((*data.shape[:2], 1), dtype=numpy.float32)
        data = numpy.concatenate([data, alpha], axis=2)

    data = numpy.ascontiguousarray(data, dtype=numpy.float32)
    tex = gfx.Texture(data, dim=2)
    geo = gfx.Geometry(grid=tex)
    mat = gfx.ImageBasicMaterial(clim=(0, 1))
    return gfx.Image(geo, mat)


def sync_heightmap_data(item):
    """Convert HeightMapData item to pygfx.Mesh (height field as triangle mesh)."""
    height_data = item.getData(copy=False)
    if height_data is None:
        return None

    H, W = height_data.shape
    y_idx, x_idx = numpy.mgrid[0:H, 0:W]

    positions = numpy.column_stack(
        [
            x_idx.ravel().astype(numpy.float32),
            y_idx.ravel().astype(numpy.float32),
            height_data.ravel().astype(numpy.float32),
        ]
    )
    positions = numpy.ascontiguousarray(positions)

    indices = grid_to_triangles(H, W)

    # Use colormapped data if available, otherwise use height data
    colormap_data = item.getColormappedData(copy=False)
    if colormap_data is None or colormap_data.size == 0:
        colormap_data = height_data
    colors = apply_colormap(item.getColormap(), colormap_data.ravel())
    colors = numpy.ascontiguousarray(colors, dtype=numpy.float32)

    normals = compute_normals(positions, indices)

    geo = gfx.Geometry(
        positions=positions,
        normals=normals,
        colors=colors,
        indices=gfx.Buffer(indices),
    )
    mat = gfx.MeshPhongMaterial(color_mode="vertex")
    return gfx.Mesh(geo, mat)


def sync_heightmap_rgba(item):
    """Convert HeightMapRGBA item to pygfx.Mesh."""
    height_data = item.getData(copy=False)
    if height_data is None:
        return None

    H, W = height_data.shape
    y_idx, x_idx = numpy.mgrid[0:H, 0:W]

    positions = numpy.column_stack(
        [
            x_idx.ravel().astype(numpy.float32),
            y_idx.ravel().astype(numpy.float32),
            height_data.ravel().astype(numpy.float32),
        ]
    )
    positions = numpy.ascontiguousarray(positions)

    indices = grid_to_triangles(H, W)

    color_data = item.getColorData(copy=False)
    if color_data is not None:
        color_data = numpy.asarray(color_data, dtype=numpy.float32)
        if color_data.dtype == numpy.uint8:
            color_data = color_data.astype(numpy.float32) / 255.0
        if color_data.ndim == 3 and color_data.shape[2] == 3:
            alpha = numpy.ones((*color_data.shape[:2], 1), dtype=numpy.float32)
            color_data = numpy.concatenate([color_data, alpha], axis=2)
        colors = color_data.reshape(-1, 4)
    else:
        colors = numpy.ones((H * W, 4), dtype=numpy.float32)

    colors = numpy.ascontiguousarray(colors, dtype=numpy.float32)
    normals = compute_normals(positions, indices)

    geo = gfx.Geometry(
        positions=positions,
        normals=normals,
        colors=colors,
        indices=gfx.Buffer(indices),
    )
    mat = gfx.MeshPhongMaterial(color_mode="vertex")
    return gfx.Mesh(geo, mat)


# --- Volume items ---


def sync_scalar_field_3d(item):
    """Convert ScalarField3D item to pygfx.Group.

    Handles isosurfaces (marching cubes -> mesh) and cut planes (volume slice).
    """
    data = item.getData(copy=False)
    if data is None:
        return None

    group = gfx.Group()

    # Isosurfaces -> marching cubes -> gfx.Mesh
    for isosurface in item.getIsosurfaces():
        if not isosurface.isVisible():
            continue

        level = isosurface.getLevel()
        color = isosurface.getColor()

        try:
            from skimage.measure import marching_cubes

            verts, faces, _, _ = marching_cubes(data, level=level)
            # marching_cubes returns (z, y, x) order; swap to (x, y, z) + offset
            verts = verts[:, ::-1].copy() + 0.5  # z,y,x -> x,y,z and offset
            verts = numpy.ascontiguousarray(verts.astype(numpy.float32))
            faces = numpy.ascontiguousarray(faces.astype(numpy.uint32))

            normals = compute_normals(verts, faces)

            geo = gfx.Geometry(
                positions=verts,
                normals=normals,
                indices=gfx.Buffer(faces),
            )

            # Parse color
            r = color.redF() if hasattr(color, "redF") else color[0]
            g = color.greenF() if hasattr(color, "greenF") else color[1]
            b = color.blueF() if hasattr(color, "blueF") else color[2]
            a = (
                color.alphaF()
                if hasattr(color, "alphaF")
                else (color[3] if len(color) > 3 else 1.0)
            )

            mat = gfx.MeshPhongMaterial(color=(r, g, b, a))
            mat.opacity = a
            group.add(gfx.Mesh(geo, mat))

        except ImportError:
            _logger.warning("scikit-image not available for marching cubes")
        except Exception as e:
            _logger.warning("Marching cubes failed for level %s: %s", level, e)

    # Cut planes -> volume slice
    for cut_plane in item.getCutPlanes():
        if not cut_plane.isVisible():
            continue

        try:
            data_f32 = numpy.ascontiguousarray(data.astype(numpy.float32))
            tex = gfx.Texture(data_f32, dim=3)
            geo = gfx.Geometry(grid=tex)

            normal = numpy.asarray(cut_plane.getNormal(), dtype=numpy.float64)
            point = numpy.asarray(cut_plane.getPoint(), dtype=numpy.float64)
            d = -numpy.dot(normal, point)

            # Build colormap texture for the slice
            cmap = cut_plane.getColormap()
            lut = cmap.getNColors(nbColors=256)  # (256, 4) uint8
            lut_f = lut.astype(numpy.float32) / 255.0
            cmap_tex = gfx.Texture(lut_f, dim=1)

            vmin, vmax = cmap.getColormapRange(data_f32)

            mat = gfx.VolumeSliceMaterial(
                plane=(float(normal[0]), float(normal[1]), float(normal[2]), float(d)),
                map=cmap_tex,
                clim=(float(vmin), float(vmax)),
            )
            group.add(gfx.Volume(geo, mat))

        except Exception as e:
            _logger.warning("Cut plane rendering failed: %s", e)

    if len(group.children) == 0:
        return None

    return group


# --- Group and clipping ---


def sync_group(item, clip_planes=None):
    """Convert a GroupItem to pygfx.Group with recursive child sync.

    ClipPlane items in the group add clipping planes for subsequent siblings.
    """
    from . import ClipPlane

    group = gfx.Group()
    current_clips = list(clip_planes or [])

    for child in item.getItems():
        if isinstance(child, ClipPlane):
            if child.isVisible():
                normal = numpy.asarray(child.getNormal(), dtype=numpy.float64)
                point = numpy.asarray(child.getPoint(), dtype=numpy.float64)
                d = -numpy.dot(normal, point)
                current_clips.append(
                    (float(normal[0]), float(normal[1]), float(normal[2]), float(d))
                )
        else:
            obj = sync_item(child, clip_planes=current_clips)
            if obj is not None:
                group.add(obj)

    return group
