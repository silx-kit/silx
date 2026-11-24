# /*##########################################################################
#
# Copyright (c) 2026 European Synchrotron Radiation Facility
# Copyright (c) 2026 Pohang Accelerator Laboratory
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
# ############################################################################*/
"""pygfx (WGPU) Plot backend."""

from __future__ import annotations

__authors__ = ["S. Kim", "T. Vincent", "L. Huder"]
__license__ = "MIT"

from collections import OrderedDict
import functools
import logging
import math
import weakref

import numpy
from matplotlib.font_manager import FontProperties, findfont

from rendercanvas.qt import QRenderWidget
import pygfx as gfx
import wgpu
from pygfx.renderers.wgpu import register_wgpu_render_function
from pygfx.renderers.wgpu.engine.blender import Blender
from pygfx.renderers.wgpu.shaders.lineshader import LineShader
from pygfx.renderers.wgpu.shaders.meshshader import MeshShader
from wgpu.backends.wgpu_native.extras import set_instance_extras

from .. import items
from .._utils import axis_scale
from ..items.types import AxisScaleType
from . import BackendBase
from ... import colors
from ... import qt
from ...qt.inspect import isValid
from ..._geometry import segmentTrianglesIntersection
from ...utils.matplotlib import rasterMathText
from ._PlotFrameCore import PlotFrame2DCore
from ._PlotImageFile import saveImageToFile
from .utils import findDimToKeep, ensureAspectRatio
from silx.gui.colors import RGBAColorType

_logger = logging.getLogger(__name__)


@functools.cache
def _restrictWgpuToPrimaryBackends() -> None:
    """Restrict the wgpu instance to the Vulkan/Metal/DX12 backends.

    Probing wgpu's OpenGL (EGL) backend aborts the process if Qt already
    created an OpenGL context (e.g., a "gl" PlotWidget).

    Must be called before the first ``gfx.WgpuRenderer`` is created, since
    the wgpu instance is process-global; it has no effect if wgpu was already
    used in the process.
    """
    try:
        set_instance_extras(backends=["Primary"])
    except Exception:  # RuntimeError if the wgpu instance already exists
        _logger.debug("Could not restrict wgpu instance backends", exc_info=True)


@functools.cache
def _logDpiOnce(message: str) -> None:
    _logger.error(message)


def _closeCanvas(canvasRef: weakref.ref, *args) -> None:
    """Close the canvas of a deleted widget"""
    canvas = canvasRef()
    if canvas is not None:
        canvas.close()


_TEXT_RASTERS: OrderedDict = OrderedDict()
"""Least recently used text rasters, as cached by the OpenGL backend"""


def _rasterText(
    text: str, font: qt.QFont, dotsPerInch: float
) -> tuple[numpy.ndarray, float]:
    """rasterMathText of the text, the raster of the OpenGL backend"""
    key = text, font.key(), dotsPerInch
    raster = _TEXT_RASTERS.pop(key, None)
    if raster is None:
        raster = rasterMathText(text, font, dotsPerInch)
        if len(_TEXT_RASTERS) >= 128:
            _TEXT_RASTERS.popitem(last=False)
    _TEXT_RASTERS[key] = raster
    return raster


# silx linestyle -> (offset, dash pattern) in line widths, as in BackendOpenGL:
# a None pattern draws no line, () a solid line
_DASH_PATTERNS = {
    "": (0.0, None),
    " ": (0.0, None),
    "-": (0.0, ()),
    "--": (0.0, (3.7, 1.6, 3.7, 1.6)),
    "-.": (0.0, (6.4, 1.6, 1, 1.6)),
    ":": (0.0, (1, 1.65, 1, 1.65)),
    None: (0.0, None),
}


class _GLLineMaterial(gfx.LineMaterial):
    """Line material drawn as GLLines2D, which ends lines and dashes square"""


@register_wgpu_render_function(gfx.Line, _GLLineMaterial)
class _GLLineShader(LineShader):
    def get_code(self):
        # pygfx rounds dash ends, which lengthens dashes by the line width,
        # caps lines beyond their end points and broken joins, and takes the
        # square root of edge coverages, which blending in sRGB makes darker
        return (
            super()
            .get_code()
            .replace("alpha = sqrt(alpha);", "")
            .replace(
                "let dist_to_stroke_dash_p = length(vec_to_dash_p) - half_thickness_p;",
                "let dist_to_stroke_dash_p = select(-1e9, 1e9, dist_to_dash_p > 0.0);",
            )
            .replace(
                "var dist_to_stroke_p = length(segment_coord_p) - half_thickness_p;",
                "var dist_to_stroke_p = length(segment_coord_p) - half_thickness_p;\n"
                "if (!is_join && segment_coord_p.x != 0.0) { discard; }",
            )
        )

    def bake_function(self, wobject, camera, logical_size):
        # pygfx measures dashes over the canvas size instead of the render
        # rect the shader draws into, which shrinks them in the plot area
        super().bake_function(wobject, camera, camera._view_size)


class _GLLineSegmentMaterial(gfx.LineSegmentMaterial):
    """Line segment material drawn as GLLines2D in GL_LINES mode"""


@register_wgpu_render_function(gfx.Line, _GLLineSegmentMaterial)
class _GLLineSegmentShader(_GLLineShader):
    def __init__(self, wobject):
        super().__init__(wobject)
        self["line_type"] = "segment"


class _SRGBVertexMeshMaterial(gfx.MeshBasicMaterial):
    """Mesh material with vertex colors in sRGB, as those of lines and points"""


@register_wgpu_render_function(gfx.Mesh, _SRGBVertexMeshMaterial)
class _SRGBVertexMeshShader(MeshShader):
    def get_code(self):
        # pygfx takes mesh vertex colors as physical: convert them after their
        # interpolation, which OpenGL does between sRGB colors
        return (
            super()
            .get_code()
            .replace(
                "diffuse_color = varyings.color;",
                "diffuse_color = vec4<f32>(srgb2physical(varyings.color.rgb), varyings.color.a);",
            )
        )


class _SRGBBlender(Blender):
    """Blender blending sRGB colors, as OpenGL does, instead of linear ones.

    Fragments are written in sRGB to a color target that does not convert them,
    so that they are blended as stored, and the target is read back as sRGB.
    """

    def __init__(self):
        super().__init__()
        self.texture_info["color"]["format"] = wgpu.TextureFormat.rgba8unorm

    def get_texture_view(self, name, usage, *, create_if_not_exist=False):
        if name != "color":
            return super().get_texture_view(
                name, usage, create_if_not_exist=create_if_not_exist
            )
        texture = self._textures.get(name)
        if texture is None:
            if not create_if_not_exist:
                return None
            texture = self._textures[name] = self.device.create_texture(
                size=(*self.size, 1),
                usage=self.texture_info[name]["usage"],
                format=wgpu.TextureFormat.rgba8unorm,
                view_formats=[wgpu.TextureFormat.rgba8unorm_srgb],
            )
        if usage == wgpu.TextureUsage.TEXTURE_BINDING:
            return texture.create_view(
                format=wgpu.TextureFormat.rgba8unorm_srgb, usage=usage
            )
        return texture.create_view(usage=usage)

    def get_shader_kwargs(self, material_pick_write, alpha_config):
        kwargs = super().get_shader_kwargs(material_pick_write, alpha_config)
        if kwargs["alpha_method"] != "blended":
            raise RuntimeError(f"Unsupported alpha method: {kwargs['alpha_method']}")
        kwargs["fragment_output_code"] = (
            kwargs["fragment_output_code"].replace(
                "struct FragmentOutput {",
                "struct FragmentOutput {\n// virtualfield stub: bool = false;",
            )
            + """
            fn apply_virtual_fields_of_fragment_output(outp: ptr<function,FragmentOutput>, stub: bool) {
                let c = clamp((*outp).color.rgb, vec3f(0.0), vec3f(1.0));
                let srgb = select(1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055, 12.92 * c, c <= vec3f(0.0031308));
                (*outp).color = vec4f(srgb, (*outp).color.a);
            }
            """
        )
        return kwargs


def _lineStyleToDashOffsetPattern(linestyle):
    """Convert a silx linestyle to its dash (offset, pattern)"""
    if linestyle is None or isinstance(linestyle, str):
        return _DASH_PATTERNS[linestyle]
    offset, pattern = linestyle
    return float(offset), () if pattern is None else tuple(pattern)


# WGSL ports of the alphaSymbol functions of glutils.GLPlotCurve.Points2D.
# They set alpha from q, the pixel center offset to the symbol center with
# y downward as gl_PointCoord, and size, both in device pixels.
_SYMBOL_ALPHAS = {
    "d": "alpha = clamp(0.5 * size - abs(q.x) - abs(q.y), 0.0, 1.0);",
    "o": "alpha = clamp(0.5 * size - length(q), 0.0, 1.0);",
    "s": "alpha = 1.0;",
    "+": "alpha = 1.0 - smoothstep(0.5, 1.5, min(abs(q.x), abs(q.y)));",
    "x": """
        let pos = floor(q + 0.5 * size) + 0.5;
        let d = abs(vec2<f32>(pos.x - pos.y, pos.x + pos.y - size));
        alpha = 1.0 - smoothstep(0.5, 1.5, min(d.x, d.y));
    """,
    "*": """
        let pos = floor(q + 0.5 * size) + 0.5;
        let d = abs(vec2<f32>(pos.x - pos.y, pos.x + pos.y - size));
        if (min(abs(q.x), abs(q.y)) < 0.5) {
            alpha = 1.0;
        } else if (min(d.x, d.y) <= 0.5) {
            alpha = clamp(0.5 * size - length(q), 0.0, 1.0);
        }
    """,
    "_": "alpha = 1.0 - smoothstep(0.5, 1.5, abs(q.y));",
    "|": "alpha = 1.0 - smoothstep(0.5, 1.5, abs(q.x));",
    "♥": """
        var c = 1.5 * q / size;
        c.y += 0.25;
        let h = abs(atan2(c.x, -c.y) / 3.141593);
        let d = (13.0 * h - 22.0 * h * h + 10.0 * h * h * h) / (6.0 - 5.0 * h);
        alpha = 1.0 - smoothstep(0.001, 0.1, clamp(length(c) - d, 0.0, 1.0));
    """,
    "tickleft": "if (q.x <= 0.5) { alpha = 1.0 - smoothstep(0.5, 1.5, abs(q.y)); }",
    "tickright": "if (q.x >= -0.5) { alpha = 1.0 - smoothstep(0.5, 1.5, abs(q.y)); }",
    "tickup": "if (q.y <= 0.5) { alpha = 1.0 - smoothstep(0.5, 1.5, abs(q.x)); }",
    "tickdown": "if (q.y >= -0.5) { alpha = 1.0 - smoothstep(0.5, 1.5, abs(q.x)); }",
    "caretleft": """
        let d = abs(q.x) - abs(q.y);
        if (d >= -0.1 && q.x > 0.5) { alpha = smoothstep(-0.1, 0.1, d); }
    """,
    "caretright": """
        let d = abs(q.x) - abs(q.y);
        if (d >= -0.1 && q.x < 0.5) { alpha = smoothstep(-0.1, 0.1, d); }
    """,
    "caretup": """
        let d = abs(q.y) - abs(q.x);
        if (d >= -0.1 && q.y > 0.5) { alpha = smoothstep(-0.1, 0.1, d); }
    """,
    "caretdown": """
        let d = abs(q.y) - abs(q.x);
        if (d >= -0.1 && q.y < 0.5) { alpha = smoothstep(-0.1, 0.1, d); }
    """,
}
_SYMBOL_ALPHAS["."] = _SYMBOL_ALPHAS["o"]
_SYMBOL_ALPHAS[","] = _SYMBOL_ALPHAS["s"]

# Symbols whose size in pixels is rounded to an odd number by Points2D
_ODD_SIZE_SYMBOLS = {"+", "_", "|", "tickleft", "tickright", "tickup", "tickdown"}


def _symbolMaterial(symbol, color, colorMode) -> gfx.PointsMarkerMaterial | None:
    """Material drawing symbol as the OpenGL backend, None for no symbol.

    Its size is left to BackendPygfx._symbolSize at draw time.

    pygfx takes a signed distance d and makes the opacity sqrt(0.5 - d),
    so the distance returned here gives the OpenGL symbol alpha.
    Only fragments in the OpenGL point are drawn: those centered in
    [-size/2, size/2) along x and y upward, with a margin for float errors.
    """
    alpha = _SYMBOL_ALPHAS.get(symbol)
    if alpha is None:
        return None
    sdf = f"""
        let r = coord + 0.001;
        if (min(r.x, r.y) < -0.5 * size || max(r.x, r.y) >= 0.5 * size) {{
            return 1.0;
        }}
        let q = vec2<f32>(coord.x, -coord.y);
        var alpha = 0.0;
        {alpha}
        return 0.5 - alpha * alpha;
    """
    return _material(
        gfx.PointsMarkerMaterial,
        aa=True,
        marker="custom",
        custom_sdf=sdf,
        color=color,
        color_mode=colorMode,
        edge_width=0.0,
        edge_color=(0, 0, 0, 0),
    )


def _vertexColorsToFloatRGBA(color: numpy.ndarray, alpha: float) -> numpy.ndarray:
    """Convert per-vertex colors to a float32 RGBA array in [0, 1].

    :param color: (N, 3) or (N, 4) array of colors, either float in [0, 1]
        or unsigned bytes in [0, 255].
    :param alpha: Opacity in [0, 1] applied to the alpha channel.
    """
    color = numpy.asarray(color)
    if numpy.issubdtype(color.dtype, numpy.integer):
        rgba = color.astype(numpy.float32) / 255.0
    else:
        rgba = numpy.array(color, dtype=numpy.float32)

    if rgba.shape[1] == 3:
        return numpy.column_stack(
            [rgba, numpy.full(len(rgba), alpha, dtype=numpy.float32)]
        )
    rgba[:, 3] *= alpha
    return rgba


def _material(cls, *args, **kwargs):
    """Create a pygfx material that ignores the depth buffer.

    Items are drawn in silx order, so the paint order alone decides what is on
    top: with depth testing, the first object drawn would win at equal depth.
    """
    return cls(*args, depth_test=False, **kwargs)


def _pixelLines(vertices, width: float, color) -> gfx.Mesh:
    """Segments between pairs of screen vertices, filled as glLineWidth draws them.

    pygfx lines get no thinner than 1.415 pixels, so a 1 pixel line covers 1
    or 2 pixels depending on where it falls: fill rectangles of width instead.
    """
    start, end = numpy.asarray(vertices, numpy.float64).reshape(-1, 2, 2).swapaxes(0, 1)
    direction = end - start
    length = numpy.linalg.norm(direction, axis=1, keepdims=True)
    normal = (
        direction[:, ::-1] * (-0.5 * width, 0.5 * width) / numpy.maximum(length, 1e-12)
    )
    corners = numpy.stack(
        (start - normal, start + normal, end + normal, end - normal), 1
    )
    positions = numpy.zeros((corners.size // 2, 3), numpy.float32)
    positions[:, :2] = corners.reshape(-1, 2)
    quads = numpy.arange(len(positions), dtype=numpy.uint32).reshape(-1, 4)
    indices = numpy.concatenate((quads[:, (0, 1, 2)], quads[:, (0, 2, 3)]))
    return gfx.Mesh(
        gfx.Geometry(positions=positions, indices=indices),
        _material(gfx.MeshBasicMaterial, color=color, side="both"),
    )


_HATCH_STEP = 20
"""Pixels between the lines of the hatch of OpenGL FilledShape2D"""


def _hatchSegments(points, height: int, box) -> numpy.ndarray:
    """Return the vertices of the hatch segments inside a polygon of pixels.

    OpenGL draws the hatch at the pixels of column px and row py from the top
    with px + py + 1 = height mod 20, centered on the lines X + Y = c where
    c = height mod 20, inside the polygon by the even-odd rule of its stencil.
    As OpenGL fills only its viewport, the segments are clipped to the box
    (left, top, right, bottom) so that their count is bounded by its size.
    """
    left, top, right, bottom = box
    points = numpy.asarray(points, dtype=numpy.float64)
    s0 = points.sum(axis=1)
    s1 = numpy.roll(s0, -1)
    x0 = points[:, 0]
    x1 = numpy.roll(x0, -1)

    # Lines start + k * step crossing the box, for 0 <= k < kEnd, crossing each
    # edge, half-open to count vertices once
    start = left + top + (height - left - top) % _HATCH_STEP
    kEnd = (right + bottom - start) // _HATCH_STEP + 1
    first, last = (
        numpy.clip(numpy.ceil((bound - start) / _HATCH_STEP), 0, kEnd)
        for bound in (numpy.minimum(s0, s1), numpy.maximum(s0, s1))
    )
    count = (last - first).astype(numpy.int64)
    edge = numpy.repeat(numpy.arange(len(points)), count)
    k = numpy.arange(len(edge)) - numpy.repeat(
        numpy.cumsum(count) - count - first, count
    )
    c = start + _HATCH_STEP * k
    x = x0[edge] + (c - s0[edge]) / (s1[edge] - s0[edge]) * (x1[edge] - x0[edge])

    # Along each line, crossings pair up to bound the inside of the polygon
    order = numpy.lexsort((x, c))
    x = x[order].reshape(-1, 2)
    c = c[order][::2]
    xMin = numpy.maximum(x[:, 0], numpy.maximum(left, c - bottom))
    xMax = numpy.minimum(x[:, 1], numpy.minimum(right, c - top))
    inside = xMax > xMin
    x = numpy.stack((xMin[inside], xMax[inside]), axis=-1)
    segments = numpy.stack((x, c[inside, numpy.newaxis] - x), axis=-1)
    return segments.reshape(-1, 2)


def _setChildren(group: gfx.Group, objects: list) -> None:
    """Make objects the children of group, in this order"""
    if group.children != tuple(objects):
        group.clear()
        group.add(*objects)


# Item classes ################################################################


class _PygfxCurveItem:
    """Manages pygfx scene objects for a single curve."""

    def __init__(
        self,
        x,
        y,
        color,
        gapcolor,
        symbol,
        linewidth,
        linestyle,
        yaxis,
        xerror,
        yerror,
        fill,
        alpha,
        symbolsize,
        baseline,
    ):
        self.yaxis = yaxis
        self.lineMaterials = []
        """Line materials with their width in points, sized at draw time"""
        self.pointMaterials = []
        """Points materials with their symbol and size in points, sized at draw time"""
        self.group = gfx.Group()
        self._lineObj = None
        self._pointsObj = None

        x = numpy.asarray(x, dtype=numpy.float32)
        y = numpy.asarray(y, dtype=numpy.float32)

        # Per-vertex color handling
        if isinstance(color, numpy.ndarray) and color.ndim == 2:
            perVertexColor = True
            vertexColors = _vertexColorsToFloatRGBA(color, alpha)
            uniformColor = gfx.Color(1, 1, 1, 1)
        else:
            perVertexColor = False
            vertexColors = None
            rgba = colors.rgba(color)
            uniformColor = gfx.Color(rgba[0], rgba[1], rgba[2], rgba[3] * alpha)

        # Sub-objects are drawn in the order they are added, as in GLPlotCurve:
        # fill, error bars, lines (gap color under the dashes), then symbols

        # Fill between curve and baseline, none for per vertex colors as in OpenGL
        if fill and len(x) >= 2 and not perVertexColor:
            fillObj = self._buildFill(x, y, baseline, uniformColor)
            if fillObj is not None:
                self.group.add(fillObj)

        # Error bars, then their ticks, in OpenGL's black whatever the curve color
        if xerror is not None or yerror is not None:
            errorColor = gfx.Color(0, 0, 0, 1)
            xVertices, yVertices = self._errorBarVertices(x, y, xerror, yerror)
            segments = numpy.concatenate((xVertices, yVertices)).reshape(-1, 2, 3)
            segments = segments[numpy.isfinite(segments).all(axis=(1, 2))]
            if len(segments) > 0:
                errGeom = gfx.Geometry(positions=segments.reshape(-1, 3))
                errMat = _material(_GLLineSegmentMaterial, aa=True, color=errorColor)
                self.lineMaterials.append((errMat, 1.0))
                self.group.add(gfx.Line(errGeom, errMat))
            for tickSymbol, vertices in (("|", xVertices), ("_", yVertices)):
                vertices = vertices[numpy.isfinite(vertices).all(axis=1)]
                if len(vertices) > 0:
                    tickMat = _symbolMaterial(tickSymbol, errorColor, "uniform")
                    self.pointMaterials.append((tickMat, tickSymbol, 7))
                    self.group.add(
                        gfx.Points(gfx.Geometry(positions=vertices), tickMat)
                    )

        # Line
        dashOffset, dashPattern = _lineStyleToDashOffsetPattern(linestyle)
        if dashPattern is not None and len(x) > 1:
            positions = numpy.zeros((len(x), 3), dtype=numpy.float32)
            positions[:, 0] = x
            positions[:, 1] = y

            lineKwargs = {}
            if perVertexColor:
                lineKwargs["colors"] = vertexColors

            geom = gfx.Geometry(positions=positions, **lineKwargs)

            # Gap color line under the dashed line
            if gapcolor is not None and dashPattern:
                gapRgba = colors.rgba(gapcolor)
                gapMat = _material(_GLLineMaterial, aa=True, color=gfx.Color(*gapRgba))
                self.lineMaterials.append((gapMat, linewidth))
                self.group.add(gfx.Line(geom, gapMat))

            mat = _material(
                _GLLineMaterial,
                aa=True,
                color=uniformColor,
                color_mode="vertex" if perVertexColor else "uniform",
                dash_pattern=dashPattern,
                dash_offset=dashOffset,
            )
            self.lineMaterials.append((mat, linewidth))
            self._lineObj = gfx.Line(geom, mat)
            self.group.add(self._lineObj)

        # Symbol / Points
        mat = _symbolMaterial(
            symbol, uniformColor, "vertex" if perVertexColor else "uniform"
        )
        if mat is not None:
            positions = numpy.zeros((len(x), 3), dtype=numpy.float32)
            positions[:, 0] = x
            positions[:, 1] = y

            pointKwargs = {}
            if perVertexColor:
                pointKwargs["colors"] = vertexColors

            geom = gfx.Geometry(positions=positions, **pointKwargs)
            self.pointMaterials.append((mat, symbol, symbolsize))
            self._pointsObj = gfx.Points(geom, mat)
            self.group.add(self._pointsObj)

    @staticmethod
    def _errorBarVertices(x, y, xerror, yerror):
        """X and Y error bars vertices, laid out as in GLPlotCurve's _ErrorBars.

        Each data point has the 4 vertices: high, point, point, low, as 2
        segments and 4 ticks. A missing error gives NaN high and low vertices,
        leaving only the ticks on the data point.
        """

        def bounds(values, error):
            if error is None:
                error = numpy.nan
            error = numpy.asarray(error, dtype=numpy.float32)
            if error.ndim == 2 and error.shape[1] == 1:
                error = numpy.ravel(error)
            minus, plus = (error[0], error[1]) if error.ndim == 2 else (error, error)
            with numpy.errstate(invalid="ignore"):
                return numpy.stack(
                    numpy.broadcast_arrays(
                        values + plus, values, values, values - minus
                    ),
                    axis=1,
                ).ravel()

        xVertices = numpy.zeros((4 * len(x), 3), dtype=numpy.float32)
        xVertices[:, 0] = bounds(x, xerror)
        xVertices[:, 1] = numpy.repeat(y, 4)
        yVertices = numpy.zeros((4 * len(x), 3), dtype=numpy.float32)
        yVertices[:, 0] = numpy.repeat(x, 4)
        yVertices[:, 1] = bounds(y, yerror)
        return xVertices, yVertices

    @staticmethod
    def _buildFill(x, y, baseline, color):
        """Build a filled mesh between curve and baseline, in scaled coordinates."""
        n = len(x)
        # Create triangle strip: for each segment, two triangles
        vertices = []
        indices = []
        for i in range(n):
            vertices.append([x[i], y[i], 0])
            vertices.append([x[i], baseline[i], 0])

        for i in range(n - 1):
            idx = i * 2
            indices.append([idx, idx + 1, idx + 2])
            indices.append([idx + 1, idx + 3, idx + 2])

        if not indices:
            return None

        vertices = numpy.array(vertices, dtype=numpy.float32)
        indices = numpy.array(indices, dtype=numpy.int32)

        geom = gfx.Geometry(positions=vertices, indices=indices)
        mat = _material(gfx.MeshBasicMaterial, color=color, side="both")
        return gfx.Mesh(geom, mat)


# Image item ##################################################################


class _PygfxImageItem:
    """Manages pygfx scene objects for a single image."""

    def __init__(self, data, origin, scale, colormap, alpha):
        self.group = gfx.Group()
        self.yaxis = "left"
        self._imageObj = None
        self._scalarShape = None
        self._origin = origin
        self._scale = scale
        self._dataShape = numpy.asarray(data).shape[:2]

        self._build(data, origin, scale, colormap, alpha)

    def _build(self, data, origin, scale, colormap, alpha):
        data = numpy.asarray(data)
        self._origin = origin
        self._scale = scale
        self._dataShape = data.shape[:2]

        if data.ndim == 2:
            self._buildScalar(data, origin, scale, colormap, alpha)
        elif data.ndim == 3 and data.shape[2] in (3, 4):
            self._buildRGBA(data, origin, scale, alpha)
        else:
            _logger.warning("Unsupported image data shape: %s", data.shape)

    def _buildScalar(self, data, origin, scale, colormap, alpha):
        self._scalarShape = data.shape

        if colormap is not None:
            rgba = colormap.applyToData(data)  # (H, W, 4) uint8
        else:
            # No colormap: autoscale to grayscale
            finite = data[numpy.isfinite(data)]
            vmin = float(finite.min()) if finite.size else 0.0
            vmax = float(finite.max()) if finite.size else 1.0
            if vmin >= vmax:
                vmax = vmin + 1.0
            gray = (numpy.clip((data - vmin) / (vmax - vmin), 0.0, 1.0) * 255).astype(
                numpy.uint8
            )
            rgba = numpy.dstack([gray, gray, gray, numpy.full_like(gray, 255)])

        self._uploadRGBA(rgba, origin, scale, alpha)

    def _buildRGBA(self, data, origin, scale, alpha):
        self._scalarShape = None

        if data.dtype in (numpy.float32, numpy.float64):
            rgba = (numpy.clip(data, 0, 1) * 255).astype(numpy.uint8)
        else:
            rgba = numpy.asarray(data, dtype=numpy.uint8)
        if rgba.shape[2] == 3:
            alphaChannel = numpy.full(rgba.shape[:2] + (1,), 255, dtype=numpy.uint8)
            rgba = numpy.concatenate([rgba, alphaChannel], axis=-1)

        self._uploadRGBA(rgba, origin, scale, alpha)

    def _uploadRGBA(self, rgba, origin, scale, alpha):
        """Upload a (H, W, 4) uint8 RGBA array, creating or updating the image."""
        rgba = numpy.ascontiguousarray(rgba, dtype=numpy.uint8)
        if alpha < 1.0:
            rgba = rgba.copy()  # Do not modify caller's array
            rgba[:, :, 3] = numpy.rint(rgba[:, :, 3] * alpha).astype(numpy.uint8)

        if self._imageObj is None:
            geom = gfx.Geometry(grid=gfx.Texture(rgba, dim=2))
            # uint8 textures are sampled as unorm and scaled by 255 in the
            # shader, so contrast limits must match the uint8 range
            mat = _material(
                gfx.ImageBasicMaterial, interpolation="nearest", clim=(0, 255)
            )
            self._imageObj = gfx.Image(geom, mat)
            self.group.add(self._imageObj)
        else:
            self._imageObj.geometry.grid.set_data(rgba)

        ox, oy = origin
        sx, sy = scale
        # pygfx centers texels on integer coordinates (quad spans
        # [-0.5, size - 0.5]); silx pixel (i, j) covers [j, j + 1] x [i, i + 1]
        self._imageObj.local.position = (ox + 0.5 * sx, oy + 0.5 * sy, 0)
        self._imageObj.local.scale = (sx, sy, 1)


class _PygfxTrianglesItem:
    """Manages pygfx scene objects for triangles."""

    def __init__(self, x, y, triangles, color, alpha):
        self.group = gfx.Group()
        self.yaxis = "left"

        x = numpy.asarray(x, dtype=numpy.float32)
        y = numpy.asarray(y, dtype=numpy.float32)
        triangles = numpy.asarray(triangles, dtype=numpy.int32)

        self._x = x
        self._y = y
        self._triangles = triangles

        positions = numpy.zeros((len(x), 3), dtype=numpy.float32)
        positions[:, 0] = x
        positions[:, 1] = y

        if not isinstance(color, str) and numpy.ndim(color) == 2:
            color = _vertexColorsToFloatRGBA(color, alpha)
            geom = gfx.Geometry(positions=positions, indices=triangles, colors=color)
            mat = _material(_SRGBVertexMeshMaterial, color_mode="vertex", side="both")
        else:
            rgba = colors.rgba(color)
            geom = gfx.Geometry(positions=positions, indices=triangles)
            mat = _material(
                gfx.MeshBasicMaterial,
                color=gfx.Color(rgba[0], rgba[1], rgba[2], rgba[3] * alpha),
                side="both",
            )

        self.group.add(gfx.Mesh(geom, mat))


class _PygfxShapeItem(dict):
    """Shape parameters, with the points in data coordinates.

    Its scene objects are built in pixels at each draw, as OpenGL draws shapes.
    """

    def __init__(
        self,
        x,
        y,
        shape,
        color,
        fill,
        overlay,
        linewidth,
        linestyle,
        gapcolor,
    ):
        super().__init__()

        if shape not in ("polygon", "rectangle", "line", "vline", "hline", "polylines"):
            raise NotImplementedError(f"Unsupported shape {shape}")

        x = numpy.asarray(x)
        y = numpy.asarray(y)

        if shape == "rectangle":
            xMin, xMax = x
            x = numpy.array((xMin, xMin, xMax, xMax))
            yMin, yMax = y
            y = numpy.array((yMin, yMax, yMax, yMin))

        dashOffset, dashPattern = _lineStyleToDashOffsetPattern(linestyle)

        self.update(
            {
                "shape": shape,
                "color": colors.rgba(color),
                # Ignore fill for polylines to mimic matplotlib
                "fill": fill and shape != "polylines",
                "x": x,
                "y": y,
                "linewidth": linewidth,
                "dashoffset": dashOffset,
                "dashpattern": dashPattern,
                "gapcolor": None if gapcolor is None else colors.rgba(gapcolor),
                "overlay": overlay,
            }
        )


class _PygfxMarkerItem(dict):
    """Marker parameters, with the position in data coordinates.

    Its scene objects are built in pixels at each draw, as for OpenGL markers.
    """

    def __init__(
        self,
        x,
        y,
        text,
        color,
        symbol,
        symbolsize,
        linewidth,
        linestyle,
        constraint,
        yaxis,
        font,
        bgcolor,
    ):
        super().__init__()

        if symbol is None:
            symbol = "+"

        # Apply constraint
        isConstraint = constraint is not None and x is not None and y is not None
        if isConstraint:
            x, y = constraint(x, y)

        dashOffset, dashPattern = _lineStyleToDashOffsetPattern(linestyle)

        self.update(
            {
                "x": x,
                "y": y,
                "text": text,
                "color": colors.rgba(color),
                "constraint": constraint if isConstraint else None,
                "symbol": symbol,
                "symbolsize": symbolsize,
                "linewidth": linewidth,
                "linestyle": linestyle,
                "dashoffset": dashOffset,
                "dashpattern": dashPattern,
                "yaxis": yaxis,
                "font": font,
                "bgcolor": bgcolor,
            }
        )


# BackendPygfx ################################################################


class BackendPygfx(BackendBase.BackendBase, QRenderWidget):
    """pygfx/WGPU-based Plot backend.

    Uses pygfx for GPU-accelerated rendering via WGPU (Vulkan/Metal/DX12).
    """

    VSYNC = True
    """Enable VSync (default True). Set to False before creating the plot
    to unlock frame rates beyond the monitor refresh rate."""

    PRESENT_METHOD = "screen"
    """Present method for rendering. "screen" uses direct GPU rendering,
    "bitmap" uses CPU readback (works with remote desktops).
    Automatically forced to "bitmap" on native Wayland (see __init__).
    Set before creating the plot."""

    _TEXT_MARKER_PADDING = 4

    def __init__(self, plot, parent=None):
        present_method = self.PRESENT_METHOD
        # On native Wayland the "screen" present shares Qt's wl_display
        # connection and bypasses Qt's compositing, which triggers Wayland
        # protocol errors (wl_callback) and crashes (see wgpu-py#688).
        # Fall back to the robust "bitmap" present (CPU readback) there.
        # X11/XWayland keep the faster "screen" present.
        if (
            present_method == "screen"
            and qt.QGuiApplication.platformName() == "wayland"
        ):
            present_method = "bitmap"
        QRenderWidget.__init__(
            self,
            parent=parent,
            present_method=present_method,
            vsync=self.VSYNC,
        )
        # The canvas closes on its close event, which a child widget deleted
        # with its parent does not get: it would keep scheduling draws
        self.destroyed.connect(functools.partial(_closeCanvas, weakref.ref(self)))
        BackendBase.BackendBase.__init__(self, plot, parent)

        # Match OpenGLWidget: a layout is needed for Qt to respect sizeHint
        layout = qt.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        # Accept mouse events without requiring focus first (match OpenGL backend)
        self.setFocusPolicy(qt.Qt.NoFocus)

        # Raise max FPS for responsive interaction (zoom, pan, drag)
        self.set_update_mode("ondemand", max_fps=240)

        self._defaultFont = None

        self._backgroundColor = (1.0, 1.0, 1.0, 1.0)
        self._dataBackgroundColor = (1.0, 1.0, 1.0, 1.0)

        self._keepDataAspectRatio = False
        self._crosshairCursor = None
        self._mousePosInPixels = None

        # pygfx rendering objects
        _restrictWgpuToPrimaryBackends()  # Must be done before creating the renderer
        # No distance sorting: objects are drawn in scene order
        # Device resolution without post-processing, as the OpenGL backend:
        # supersampling and DDAA would blur text, images and thin lines
        self._renderer = gfx.WgpuRenderer(
            self, sort_objects=False, pixel_scale=1, ppaa="none"
        )
        self._renderer._blender = _SRGBBlender()  # pygfx has no public option

        # Widget pixels over the plot area, and the transforms to them from
        # each Y axis' scaled data coordinates: both updated by _syncCamera
        self._plotCamera = gfx.OrthographicCamera(maintain_aspect=False)
        self._dataToPixelMatrices = {
            "left": numpy.identity(4),
            "right": numpy.identity(4),
        }

        # Data area background: updated by setBackgroundColors
        self._bgObj = gfx.Background(
            None,
            _material(gfx.BackgroundMaterial, gfx.Color(*self._dataBackgroundColor)),
        )
        self._gridGroup = gfx.Group()
        self._plotBackgroundScene = gfx.Scene()
        self._plotBackgroundScene.add(self._bgObj, self._gridGroup)

        # Filled by _syncItemScenes in back to front order
        self._dataScene = gfx.Scene()
        self._overlayScene = gfx.Scene()

        # PlotFrame for coordinate transforms
        self._plotFrame = PlotFrame2DCore(
            foregroundColor=(0.0, 0.0, 0.0, 1.0),
            gridColor=(0.7, 0.7, 0.7, 1.0),
            marginRatios=(0.15, 0.1, 0.1, 0.15),
            font=self._getDefaultFont(),
        )
        self._plotFrame.size = (
            int(self.getDevicePixelRatio() * 640),
            int(self.getDevicePixelRatio() * 480),
        )

        # Screen-space scenes for the widget background and the frame/axes
        self._screenBg = gfx.Background(
            None, _material(gfx.BackgroundMaterial, gfx.Color(1, 1, 1, 1))
        )
        self._screenScene = gfx.Scene()
        self._screenScene.add(self._screenBg)
        self._screenFrameGroup = gfx.Group()
        self._frameScene = gfx.Scene()
        self._frameScene.add(self._screenFrameGroup)
        self._screenCamera = gfx.OrthographicCamera(maintain_aspect=False)
        self._cachedBgColor = (1.0, 1.0, 1.0, 1.0)

        self._reusableImageItem = None  # Pool for image item reuse

        self.request_draw(self._draw)
        self.setAutoFillBackground(False)
        self.setMouseTracking(True)

    def _getDefaultFont(self):
        """Returns the default font, the one of the OpenGL backend"""
        if self._defaultFont is None:
            fontId = qt.QFontDatabase.addApplicationFont(
                findfont(FontProperties(family=["sans-serif"]))
            )
            family = qt.QFontDatabase.applicationFontFamilies(fontId)[0]
            font = qt.QFont(family, 10, qt.QFont.Normal, False)
            font.setStyleStrategy(qt.QFont.PreferAntialias)
            self._defaultFont = font
        return self._defaultFont

    def _textObject(
        self,
        text: str,
        font: qt.QFont,
        color: RGBAColorType,
        x: float,
        y: float,
        align: str,
        valign: str,
        rotate: float = 0.0,
        bgcolor: RGBAColorType | None = None,
        padding: int = 0,
    ) -> gfx.Group | None:
        """Text in device pixels drawn as the OpenGL backend Text2D draws it,
        or None for blank text"""
        if not text.strip():
            return None
        coverage, baseline = _rasterText(text, font, self._plotFrame.dotsPerInch)
        padding = max(0, int(padding * self.getDevicePixelRatio()))
        coverage = numpy.pad(coverage, padding)
        height, width = coverage.shape
        if bgcolor is None:
            bgcolor = (*color[:3], 0.0)
        mix = coverage[..., numpy.newaxis] / 255
        rgba = numpy.round(
            255 * ((1 - mix) * numpy.asarray(bgcolor) + mix * numpy.asarray(color))
        ).astype(numpy.uint8)

        xOrig = {"left": 0, "right": -width, "center": -width // 2}[align]
        yOrig = {
            "baseline": -baseline,
            "top": 0,
            "bottom": -height,
            "center": -height // 2,
        }[valign]

        image = gfx.Image(
            gfx.Geometry(grid=gfx.Texture(rgba, dim=2)),
            _material(gfx.ImageBasicMaterial, interpolation="nearest", clim=(0, 255)),
        )
        # Texel centers are on integer coordinates
        image.local.position = (xOrig + 0.5, yOrig + 0.5, 0)
        group = gfx.Group()
        group.add(image)
        group.local.position = (int(x), int(y), 0)
        group.local.euler_z = math.radians(rotate)
        return group

    def getDevicePixelRatio(self):
        return self.devicePixelRatioF()

    def getDotsPerInch(self):
        screen = self.screen()
        if screen is None:
            return 96.0 * self.getDevicePixelRatio()

        # Qt sometimes reports a bogus screen DPI, clamp it to a sane range
        # (see silx.gui._glutils.OpenGLWidget.getDotsPerInch).
        physicalDPI = screen.physicalDotsPerInch()
        if physicalDPI < 55.0:
            defaultDPI = 72.0
            _logDpiOnce(
                f"Reported screen DPI too low: {int(physicalDPI)}, using {defaultDPI} instead"
            )
            physicalDPI = defaultDPI
        elif physicalDPI > 1000.0:
            defaultDPI = 96.0
            _logDpiOnce(
                f"Reported screen DPI too high: {int(physicalDPI)}, using {defaultDPI} instead"
            )
            physicalDPI = defaultDPI

        return physicalDPI * self.getDevicePixelRatio()

    # Drawing ###############################################################

    def _draw(self):
        plot = self._plotRef()
        if plot is None:
            return

        with plot._paintContext():
            self._syncPlotFrame()
            self._syncCamera()
            self._updateFrame()
            self._syncItemScenes()

            dpr = self.getDevicePixelRatio()
            left, top = self._plotFrame.plotOrigin
            pw, ph = self._plotFrame.plotSize
            # Viewport rect is in logical pixels
            plotRect = (left / dpr, top / dpr, pw / dpr, ph / dpr)

            # Same paint order as the OpenGL backend
            passes = (
                (self._screenScene, self._screenCamera, None),
                (self._plotBackgroundScene, self._plotCamera, plotRect),
                (self._dataScene, self._plotCamera, plotRect),
                (self._frameScene, self._screenCamera, None),
                (self._overlayScene, self._plotCamera, plotRect),
            )
            for index, (scene, camera, rect) in enumerate(passes):
                self._renderer.render(
                    scene,
                    camera,
                    rect=rect,
                    clear=index == 0,
                    flush=index == len(passes) - 1,
                )

    def _syncPlotFrame(self):
        """Sync plot frame size with widget size."""
        dpr = self.getDevicePixelRatio()
        w = int(self.width() * dpr)
        h = int(self.height() * dpr)
        if (w, h) != self._plotFrame.size:
            self._plotFrame.size = (w, h)
        self._plotFrame.devicePixelRatio = dpr
        self._plotFrame.dotsPerInch = self.getDotsPerInch()

    def _syncCamera(self):
        """Update the plot area camera and the data to pixel transforms."""
        # Plot area in widget pixels, y downward
        left, top = self._plotFrame.plotOrigin
        width, height = self._plotFrame.plotSize
        self._plotCamera.show_rect(
            left, left + width, top + height, top, depth=max(width, height, 1)
        )

        def displayedRange(range_, inverted):
            """(first, last) data values along the pixel axis, never empty"""
            first, last = range_
            if first == last:
                first, last = first - 0.5, last + 0.5
            return (last, first) if inverted else (first, last)

        trRanges = self._plotFrame.transformedDataRanges
        xMin, xMax = displayedRange(trRanges.x, self._plotFrame.isXAxisInverted)
        sx = width / (xMax - xMin)
        for axis, yRange in (("left", trRanges.y), ("right", trRanges.y2)):
            yMin, yMax = displayedRange(yRange, self._plotFrame.isYAxisInverted)
            sy = -height / (yMax - yMin)
            self._dataToPixelMatrices[axis] = numpy.array(
                (
                    (sx, 0, 0, left - xMin * sx),
                    (0, sy, 0, top - yMax * sy),
                    (0, 0, 1, 0),
                    (0, 0, 0, 1),
                ),
                dtype=numpy.float64,
            )

    def _updateFrame(self):
        """Update axes, ticks, grid, labels in screen space."""
        # Update background color only when changed
        bgColor = self._backgroundColor
        if self._cachedBgColor != bgColor:
            self._screenBg.material = _material(
                gfx.BackgroundMaterial, gfx.Color(*bgColor)
            )
            self._cachedBgColor = bgColor

        if not self._plotFrame.isDirty:
            return  # Frame unchanged, keep cached objects

        self._screenFrameGroup.clear()
        self._gridGroup.clear()

        if self._plotFrame.margins == self._plotFrame._NoDisplayMargins:
            return

        w, h = self._plotFrame.size
        if w <= 2 or h <= 2:
            return

        # Set screen camera to pixel coordinates (Y=0 at top, Y=h at bottom)
        # show_rect(left, right, top, bottom):
        # PlotFrameCore uses Y=0=top, Y=h=bottom (pixel convention)
        # In pygfx: height = bottom - top, so top=h, bottom=0 flips Y axis
        extent = max(w, h, 1.0)
        self._screenCamera.show_rect(0, w, h, 0, depth=extent)

        # Build vertices and labels from the core
        vertices, gridVertices, labelDicts = self._plotFrame._buildVerticesAndLabels()
        self._plotFrame._clearDirty()

        # Grid and frame lines are 1 device pixel wide, as in GLPlotFrame
        if len(gridVertices) >= 2:
            gridColor = gfx.Color(*self._plotFrame.gridColor)
            self._gridGroup.add(_pixelLines(gridVertices, 1.0, gridColor))

        if len(vertices) >= 2:
            fgColor = gfx.Color(*self._plotFrame.foregroundColor)
            self._screenFrameGroup.add(_pixelLines(vertices, 1.0, fgColor))

        # Render text labels (tick labels, axis titles, main title)
        for labelDict in labelDicts:
            textObj = self._textObject(
                labelDict["text"],
                labelDict["font"],
                labelDict["color"],
                labelDict["x"],
                labelDict["y"],
                labelDict["align"],
                labelDict["valign"],
                labelDict["rotate"],
            )
            if textObj is not None:
                self._screenFrameGroup.add(textObj)

    def _syncItemScenes(self):
        """Fill the data and overlay scenes with the visible items, back to front.

        As in the OpenGL backend, overlay items are drawn over the frame,
        followed by the marker labels and the crosshair.
        Both scenes are in plot area pixels: item groups hold scaled data,
        mapped through the transform of their Y axis, while markers and
        shapes are built in pixels.
        """
        dataObjects = []
        overlayObjects = []
        labels = []
        for plotItem in self.getItemsFromBackToFront(condition=lambda i: i.isVisible()):
            item = plotItem._backendRenderer
            if item is None:
                continue
            if isinstance(item, _PygfxMarkerItem):
                objects, label = self._markerObjects(item)
                overlayObjects.extend(objects)
                if label is not None:
                    labels.append(label)
                continue
            if isinstance(item, _PygfxShapeItem):
                objects = self._shapeObjects(item)
            else:
                item.group.local.matrix = self._dataToPixelMatrices[item.yaxis]
                if isinstance(item, _PygfxCurveItem):
                    for material, width in item.lineMaterials:
                        material.thickness = self._lineWidth(width)
                    for material, symbol, size in item.pointMaterials:
                        material.size = self._symbolSize(symbol, size)
                objects = [item.group]
            if plotItem.isOverlay():
                overlayObjects.extend(objects)
            else:
                dataObjects.extend(objects)

        _setChildren(self._dataScene, dataObjects)
        _setChildren(
            self._overlayScene, overlayObjects + labels + self._crosshairObjects()
        )

    def _symbolSize(self, symbol, size: float) -> float:
        """Size in logical pixels of a symbol of size points, as Points2D.render"""
        if symbol == ",":
            size = 1.0
        elif symbol == ".":
            size = math.ceil(0.5 * size) + 1.0
        size = size / 72.0 * self._plotFrame.dotsPerInch
        if symbol in _ODD_SIZE_SYMBOLS:
            size = size // 2 * 2 + 1.0
        return size / self._plotFrame.devicePixelRatio

    def _lineWidth(self, width: float) -> float:
        """Width in logical pixels of a line of width points, as GLLines2D.render"""
        return (
            width
            / 72.0
            * self._plotFrame.dotsPerInch
            / self._plotFrame.devicePixelRatio
        )

    def _markerObjects(self, item):
        """Return the scene objects of a marker and its text label or None"""
        xCoord = item["x"]
        yCoord = item["y"]
        yAxis = item.get("yaxis", "left")
        gfxColor = gfx.Color(*item["color"])
        text = item["text"]
        pixelOffset = 3

        if xCoord is not None and yCoord is not None:
            # Point marker, drawn only strictly inside the limits as in OpenGL
            xmin, xmax = self._plot.getXAxis().getLimits()
            ymin, ymax = self._plot.getYAxis(axis=yAxis).getLimits()
            if not xmin < xCoord < xmax or not ymin < yCoord < ymax:
                return [], None
            pixelPos = self._plotFrame.dataToPixel(xCoord, yCoord, axis=yAxis)
            if pixelPos is None:
                return [], None
            symbols = []
            mat = _symbolMaterial(item["symbol"], gfxColor, "uniform")
            if mat is not None:
                mat.size = self._symbolSize(item["symbol"], item["symbolsize"])
                geom = gfx.Geometry(
                    positions=numpy.array([[*pixelPos, 0]], dtype=numpy.float32)
                )
                symbols.append(gfx.Points(geom, mat))
            if text is None:
                return symbols, None
            if self._plotFrame.isYAxisInverted:
                valign, vPixelOffset = "bottom", -pixelOffset
            else:
                valign, vPixelOffset = "top", pixelOffset
            label = self._textObject(
                text,
                item["font"],
                item["color"],
                pixelPos[0] + pixelOffset,
                pixelPos[1] + vPixelOffset,
                "left",
                valign,
                bgcolor=item["bgcolor"],
                padding=self._TEXT_MARKER_PADDING,
            )
            return symbols, label

        label = None
        if xCoord is None:
            # Horizontal line at y
            pixelPos = self._plotFrame.dataToPixel(
                0.5 * sum(self._plotFrame.dataRanges[0]),
                yCoord,
                axis=yAxis,
            )
            if pixelPos is None:
                return [], None
            right = self._plotFrame.size[0] - self._plotFrame.margins.right
            # Edge to edge of the widget as in OpenGL, where the dashes start;
            # the plot area rect clips it
            positions = [[0, pixelPos[1], 0], [self._plotFrame.size[0], pixelPos[1], 0]]
            if text is not None:
                label = self._textObject(
                    text,
                    item["font"],
                    item["color"],
                    right - pixelOffset,
                    pixelPos[1] - pixelOffset,
                    "right",
                    "bottom",
                    bgcolor=item["bgcolor"],
                    padding=self._TEXT_MARKER_PADDING,
                )
        else:
            # Vertical line at x
            yRange = self._plotFrame.dataRanges[1 if yAxis == "left" else 2]
            pixelPos = self._plotFrame.dataToPixel(
                xCoord,
                0.5 * sum(yRange),
                axis=yAxis,
            )
            if pixelPos is None:
                return [], None
            top = self._plotFrame.margins.top
            # Edge to edge of the widget, as the horizontal line
            positions = [[pixelPos[0], 0, 0], [pixelPos[0], self._plotFrame.size[1], 0]]
            if text is not None:
                label = self._textObject(
                    text,
                    item["font"],
                    item["color"],
                    pixelPos[0] + pixelOffset,
                    top + pixelOffset,
                    "left",
                    "top",
                    bgcolor=item["bgcolor"],
                    padding=self._TEXT_MARKER_PADDING,
                )

        if item["dashpattern"] is None:
            return [], label
        geom = gfx.Geometry(positions=numpy.array(positions, dtype=numpy.float32))
        mat = _material(
            _GLLineMaterial,
            aa=True,
            thickness=self._lineWidth(item["linewidth"]),
            color=gfxColor,
            dash_pattern=item["dashpattern"],
            dash_offset=item["dashoffset"],
        )
        return [gfx.Line(geom, mat)], label

    def _shapeObjects(self, item):
        """Return the scene objects of a shape, in pixels as OpenGL draws it"""
        frame = self._plotFrame
        for values, axis in ((item["x"], frame.xAxis), (item["y"], frame.yAxis)):
            values = values[numpy.isfinite(values)]
            if len(values) == 0 or not axis_scale.isValid(axis.scale, values.min()):
                return []  # Out of the axes supported range

        shape = item["shape"]
        # hline and vline are widget long at their last point, where dashes start
        if shape == "hline":
            xMiddle = 0.5 * sum(frame.dataRanges[0])
            pixel = frame.dataToPixel(xMiddle, item["y"][-1], axis="left")
            if pixel is None:
                return []
            subShapes = [numpy.array(((0, pixel[1]), (frame.size[0], pixel[1])))]
        elif shape == "vline":
            yMiddle = 0.5 * sum(frame.dataRanges[1])
            pixel = frame.dataToPixel(item["x"][-1], yMiddle, axis="left")
            if pixel is None:
                return []
            subShapes = [numpy.array(((pixel[0], 0), (pixel[0], frame.size[1])))]
        else:
            # Sub-shapes between the not finite points
            finite = numpy.isfinite(item["x"]) & numpy.isfinite(item["y"])
            splits = numpy.concatenate(
                ([-1], numpy.flatnonzero(~finite), [len(finite)])
            )
            subShapes = [
                numpy.transpose(
                    frame.dataToPixel(
                        item["x"][begin:end], item["y"][begin:end], axis="left"
                    )
                )
                for begin, end in zip(splits[:-1] + 1, splits[1:])
                if end > begin
            ]

        left, top = frame.plotOrigin
        width, height = frame.plotSize
        plotBox = left, top, left + width, top + height
        color = gfx.Color(*item["color"])
        thickness = self._lineWidth(item["linewidth"])
        objects = []
        for points in subShapes:
            if item["fill"] and shape not in ("hline", "vline"):
                segments = _hatchSegments(points, frame.size[1], plotBox)
                if len(segments) > 0:
                    objects.append(_pixelLines(segments, 1.0, color))

            if item["dashpattern"] is None or len(points) < 2:
                continue
            if shape != "polylines":
                points = numpy.append(points, points[:1], axis=0)
            positions = numpy.zeros((len(points), 3), dtype=numpy.float32)
            positions[:, :2] = points
            geometry = gfx.Geometry(positions=positions)
            # The gap color line under the dashed line, as in OpenGL
            if item["gapcolor"] is not None and item["dashpattern"]:
                gapMaterial = _material(
                    _GLLineMaterial,
                    aa=True,
                    thickness=thickness,
                    color=gfx.Color(*item["gapcolor"]),
                )
                objects.append(gfx.Line(geometry, gapMaterial))
            material = _material(
                _GLLineMaterial,
                aa=True,
                thickness=thickness,
                color=color,
                dash_pattern=item["dashpattern"],
                dash_offset=item["dashoffset"],
            )
            objects.append(gfx.Line(geometry, material))
        return objects

    def _crosshairObjects(self):
        """Return the crosshair cursor lines"""
        if self._crosshairCursor is None or self._mousePosInPixels is None:
            return []

        color, linewidth = self._crosshairCursor
        gfxColor = gfx.Color(*color)
        # At the center of the pixel under the mouse, as BackendOpenGL
        mx, my = (v + 0.5 for v in self._mousePosInPixels)

        left, top = self._plotFrame.plotOrigin
        pw, ph = self._plotFrame.plotSize

        # linewidth is in device pixels, as BackendOpenGL glLineWidth
        segments = ((left, my), (left + pw, my), (mx, top), (mx, top + ph))
        return [_pixelLines(segments, linewidth, gfxColor)]

    # QWidget events ########################################################

    _MOUSE_BTNS = {
        qt.Qt.LeftButton: "left",
        qt.Qt.RightButton: "right",
        qt.Qt.MiddleButton: "middle",
    }

    def sizeHint(self):
        return qt.QSize(8 * 80, 6 * 80)

    def minimumSizeHint(self):
        return qt.QSize(0, 0)

    def enterEvent(self, event):
        # WA_NativeWindow (from screen present mode) requires OS-level focus.
        # Activate the top-level window when the mouse enters so that
        # mouse events and cursor changes work without an extra click.
        topLevel = self.window()
        if topLevel is not None:
            topLevel.activateWindow()
        super().enterEvent(event)

    def mousePressEvent(self, event):
        if event.button() not in self._MOUSE_BTNS:
            return super().mousePressEvent(event)
        x, y = qt.getMouseEventPosition(event)
        self._plot.onMousePress(x, y, self._MOUSE_BTNS[event.button()])
        event.accept()

    def mouseMoveEvent(self, event):
        qtPos = qt.getMouseEventPosition(event)

        previousMousePosInPixels = self._mousePosInPixels
        if qtPos == self._mouseInPlotArea(*qtPos):
            dpr = self.getDevicePixelRatio()
            devicePos = qtPos[0] * dpr, qtPos[1] * dpr
            self._mousePosInPixels = devicePos
        else:
            self._mousePosInPixels = None

        if (
            self._crosshairCursor is not None
            and previousMousePosInPixels != self._mousePosInPixels
        ):
            self._plot._setDirtyPlot(overlayOnly=True)

        self._plot.onMouseMove(*qtPos)
        event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() not in self._MOUSE_BTNS:
            return super().mouseReleaseEvent(event)
        x, y = qt.getMouseEventPosition(event)
        self._plot.onMouseRelease(x, y, self._MOUSE_BTNS[event.button()])
        event.accept()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        angleInDegrees = delta / 8.0
        x, y = qt.getMouseEventPosition(event)
        self._plot.onMouseWheel(x, y, angleInDegrees)
        event.accept()

    def leaveEvent(self, _):
        self._plot.onMouseLeaveWidget()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        w, h = self.width(), self.height()
        if w == 0 or h == 0:
            return
        dpr = self.getDevicePixelRatio()
        self._plotFrame.size = (int(w * dpr), int(h * dpr))

        # Store current ranges
        previousXRange = self.getGraphXLimits()
        previousYRange = self.getGraphYLimits(axis="left")
        previousYRightRange = self.getGraphYLimits(axis="right")

        # Re-apply current data ranges to the new size (same as OpenGL backend)
        (xMin, xMax), (yMin, yMax), (y2Min, y2Max) = self._plotFrame.dataRanges
        self.setLimits(xMin, xMax, yMin, yMax, y2Min, y2Max)

        # If plot range has changed, then emit signal
        if previousXRange != self.getGraphXLimits():
            self._plot.getXAxis()._emitLimitsChanged()
        if previousYRange != self.getGraphYLimits(axis="left"):
            self._plot.getYAxis(axis="left")._emitLimitsChanged()
        if previousYRightRange != self.getGraphYLimits(axis="right"):
            self._plot.getYAxis(axis="right")._emitLimitsChanged()

    # Backend API: Add methods ##############################################

    def addCurve(
        self,
        x,
        y,
        color,
        gapcolor,
        symbol,
        linewidth,
        linestyle,
        yaxis,
        xerror,
        yerror,
        fill,
        alpha,
        symbolsize,
        baseline,
    ):
        x = numpy.asarray(x, dtype=numpy.float64)
        y = numpy.asarray(y, dtype=numpy.float64)

        # Handle axes scale: convert data
        xAxis = self._plotFrame.xAxis
        if xAxis.scale != "linear":
            xScaled = xAxis.applyScale(x)

            if xerror is not None:
                # Transform xerror so that
                # xScaled +/- xerrorScaled = scaled(x +/- xerror)
                xerror = numpy.asarray(xerror, dtype=numpy.float32)
                if xerror.ndim == 2:
                    xErrMinus, xErrPlus = xerror[0], xerror[1]
                else:
                    xErrMinus, xErrPlus = xerror, xerror
                xerror = numpy.array(
                    (
                        xScaled - xAxis.applyScale(x - xErrMinus),
                        xAxis.applyScale(x + xErrPlus) - xScaled,
                    ),
                    dtype=numpy.float32,
                )

            x = xScaled

        yAxis = self._plotFrame.y2Axis if yaxis == "right" else self._plotFrame.yAxis

        if yAxis.scale != "linear":
            yScaled = yAxis.applyScale(y)

            if yerror is not None:
                # Transform yerror so that
                # yScaled +/- yerrorScaled = scaled(y +/- yerror)
                yerror = numpy.asarray(yerror, dtype=numpy.float32)
                if yerror.ndim == 2:
                    yErrMinus, yErrPlus = yerror[0], yerror[1]
                else:
                    yErrMinus, yErrPlus = yerror, yerror
                yerror = numpy.array(
                    (
                        yScaled - yAxis.applyScale(y - yErrMinus),
                        yAxis.applyScale(y + yErrPlus) - yScaled,
                    ),
                    dtype=numpy.float32,
                )

            y = yScaled

        if fill:
            # As GLPlotCurve2D: 0 by default, and the bottom of a log axis for <= 0
            dataBaseline = numpy.zeros(len(y)) + (0 if baseline is None else baseline)
            baseline = yAxis.applyScale(dataBaseline)
            if yAxis.scale == "log":
                baseline[dataBaseline <= 0] = -38

        item = _PygfxCurveItem(
            x,
            y,
            color,
            gapcolor,
            symbol,
            linewidth,
            linestyle,
            yaxis,
            xerror,
            yerror,
            fill,
            alpha,
            symbolsize,
            baseline,
        )
        if yaxis == "right":
            self._plotFrame.isY2Axis = True
        return item

    def addImage(self, data, origin, scale, colormap, alpha):
        data = numpy.asarray(data)
        ox, oy = origin
        sx, sy = scale
        h, w = data.shape[:2]

        xAxis = self._plotFrame.xAxis
        if xAxis.scale != "linear":
            xStart, xEnd = ox, ox + w * sx
            xMin = min(xStart, xEnd)
            if numpy.isfinite(xMin) and not axis_scale.isValid(xAxis.scale, xMin):
                raise RuntimeError(
                    "Cannot add image with coordinate outside X axis valid range"
                )
            scaledStart, scaledEnd = xAxis.applyScale((xStart, xEnd))
            ox = scaledStart
            sx = (scaledEnd - scaledStart) / w

        yAxis = self._plotFrame.yAxis
        if yAxis.scale != "linear":
            yStart, yEnd = oy, oy + h * sy
            yMin = min(yStart, yEnd)
            if numpy.isfinite(yMin) and not axis_scale.isValid(yAxis.scale, yMin):
                raise RuntimeError(
                    "Cannot add image with coordinate outside Y axis valid range"
                )
            scaledStart, scaledEnd = yAxis.applyScale((yStart, yEnd))
            oy = scaledStart
            sy = (scaledEnd - scaledStart) / h

        # Reuse pooled item if shape matches (avoids GPU object recreation)
        reuse = self._reusableImageItem
        if reuse is not None and data.ndim == 2 and reuse._scalarShape == data.shape:
            self._reusableImageItem = None
            reuse._build(data, (ox, oy), (sx, sy), colormap, alpha)
            return reuse

        self._reusableImageItem = None
        return _PygfxImageItem(data, (ox, oy), (sx, sy), colormap, alpha)

    def addTriangles(self, x, y, triangles, color, alpha):
        x = self._plotFrame.xAxis.applyScale(numpy.asarray(x, dtype=numpy.float64))
        y = self._plotFrame.yAxis.applyScale(numpy.asarray(y, dtype=numpy.float64))
        return _PygfxTrianglesItem(x, y, triangles, color, alpha)

    def addShape(
        self,
        x,
        y,
        shape,
        color,
        fill,
        overlay,
        linestyle,
        linewidth,
        gapcolor,
    ):
        x = numpy.asarray(x, dtype=numpy.float64)
        y = numpy.asarray(y, dtype=numpy.float64)

        xMin = x.min()
        if numpy.isfinite(xMin) and not axis_scale.isValid(
            self._plotFrame.xAxis.scale, xMin
        ):
            raise RuntimeError(
                "Cannot add item with coordinate outside X axis valid range"
            )
        yMin = y.min()
        if numpy.isfinite(yMin) and not axis_scale.isValid(
            self._plotFrame.yAxis.scale, yMin
        ):
            raise RuntimeError(
                "Cannot add item with coordinate outside Y axis valid range"
            )

        return _PygfxShapeItem(
            x,
            y,
            shape,
            color,
            fill,
            overlay,
            linewidth,
            linestyle,
            gapcolor,
        )

    def addMarker(
        self,
        x: float | None,
        y: float | None,
        text: str | None,
        color: str,
        symbol: str | None,
        symbolsize: float,
        linestyle: str | tuple[float, tuple[float, ...] | None],
        linewidth: float,
        constraint,
        yaxis: str,
        font: qt.QFont,
        bgcolor: RGBAColorType | None,
    ) -> object:
        if font is None:
            font = self._getDefaultFont()
        return _PygfxMarkerItem(
            x,
            y,
            text,
            color,
            symbol,
            symbolsize,
            linewidth,
            linestyle,
            constraint,
            yaxis,
            font,
            bgcolor,
        )

    # Backend API: Remove ####################################################

    def remove(self, item):
        if hasattr(item, "group"):
            # Check Y2 axis visibility
            if hasattr(item, "yaxis") and item.yaxis == "right":
                y2AxisItems = (
                    i
                    for i in self._plot.getItems()
                    if isinstance(i, items.YAxisMixIn) and i.getYAxis() == "right"
                )
                self._plotFrame.isY2Axis = next(y2AxisItems, None) is not None

            # Pool scalar image items for reuse (avoids GPU object recreation)
            if isinstance(item, _PygfxImageItem) and item._scalarShape is not None:
                self._reusableImageItem = item

    # Backend API: Interaction ###############################################

    _QT_CURSORS = {
        BackendBase.CURSOR_DEFAULT: qt.Qt.ArrowCursor,
        BackendBase.CURSOR_POINTING: qt.Qt.PointingHandCursor,
        BackendBase.CURSOR_SIZE_HOR: qt.Qt.SizeHorCursor,
        BackendBase.CURSOR_SIZE_VER: qt.Qt.SizeVerCursor,
        BackendBase.CURSOR_SIZE_ALL: qt.Qt.SizeAllCursor,
    }

    def setGraphCursorShape(self, cursor):
        if cursor is None:
            super().unsetCursor()
        else:
            cursor = self._QT_CURSORS[cursor]
            super().setCursor(qt.QCursor(cursor))

    def setGraphCursor(self, flag, color, linewidth, linestyle):
        if flag:
            color = colors.rgba(color)
            crosshairCursor = color, linewidth
        else:
            crosshairCursor = None

        if crosshairCursor != self._crosshairCursor:
            self._crosshairCursor = crosshairCursor

    _PICK_OFFSET = 3

    def _mouseInPlotArea(self, x, y):
        """Returns closest visible position in the plot."""
        left, top, width, height = self.getPlotBoundsInPixels()
        return (
            numpy.clip(x, left, left + width - 1),
            numpy.clip(y, top, top + height - 1),
        )

    def pickItem(self, x, y, item):
        dataPos = self._plot.pixelToData(x, y, axis="left", check=True)
        if dataPos is None:
            return None

        if item is None:
            _logger.error("No item provided for picking")
            return None

        # Pick markers
        if isinstance(item, _PygfxMarkerItem):
            yaxis = item["yaxis"]
            pixelPos = self._plot.dataToPixel(
                item["x"], item["y"], axis=yaxis, check=False
            )
            if pixelPos is None:
                return None

            if item["x"] is None:  # Horizontal line
                pt1 = self._plot.pixelToData(
                    x, y - self._PICK_OFFSET, axis=yaxis, check=False
                )
                pt2 = self._plot.pixelToData(
                    x, y + self._PICK_OFFSET, axis=yaxis, check=False
                )
                isPicked = min(pt1[1], pt2[1]) <= item["y"] <= max(pt1[1], pt2[1])

            elif item["y"] is None:  # Vertical line
                pt1 = self._plot.pixelToData(
                    x - self._PICK_OFFSET, y, axis=yaxis, check=False
                )
                pt2 = self._plot.pixelToData(
                    x + self._PICK_OFFSET, y, axis=yaxis, check=False
                )
                isPicked = min(pt1[0], pt2[0]) <= item["x"] <= max(pt1[0], pt2[0])

            else:
                isPicked = (
                    numpy.fabs(x - pixelPos[0]) <= self._PICK_OFFSET
                    and numpy.fabs(y - pixelPos[1]) <= self._PICK_OFFSET
                )

            return (0,) if isPicked else None

        # Pick curves
        if isinstance(item, _PygfxCurveItem):
            return self._pickCurve(item, x, y)

        # Pick images
        if isinstance(item, _PygfxImageItem):
            return self._pickImage(item, dataPos)

        # Pick triangles
        if isinstance(item, _PygfxTrianglesItem):
            return self._pickTriangles(item, dataPos)

        return None

    def _pickCurve(self, item, x, y):
        """Pick a curve item."""
        offset = self._PICK_OFFSET

        inAreaPos = self._mouseInPlotArea(x - offset, y - offset)
        dataPos = self._plot.pixelToData(
            inAreaPos[0], inAreaPos[1], axis=item.yaxis, check=True
        )
        if dataPos is None:
            return None
        xPick0, yPick0 = dataPos

        inAreaPos = self._mouseInPlotArea(x + offset, y + offset)
        dataPos = self._plot.pixelToData(
            inAreaPos[0], inAreaPos[1], axis=item.yaxis, check=True
        )
        if dataPos is None:
            return None
        xPick1, yPick1 = dataPos

        # The curve geometry holds scaled data
        yAxis = (
            self._plotFrame.y2Axis if item.yaxis == "right" else self._plotFrame.yAxis
        )
        xPick0, xPick1 = self._plotFrame.xAxis.applyScale((xPick0, xPick1))
        yPick0, yPick1 = yAxis.applyScale((yPick0, yPick1))
        xPickMin = min(xPick0, xPick1)
        xPickMax = max(xPick0, xPick1)
        yPickMin = min(yPick0, yPick1)
        yPickMax = max(yPick0, yPick1)

        # Get curve data from the line geometry
        if item._lineObj is not None:
            positions = item._lineObj.geometry.positions.data
            xData = positions[:, 0]
            yData = positions[:, 1]
        elif item._pointsObj is not None:
            positions = item._pointsObj.geometry.positions.data
            xData = positions[:, 0]
            yData = positions[:, 1]
        else:
            return None

        # Find points within the pick area
        indices = numpy.where(
            (xData >= xPickMin)
            & (xData <= xPickMax)
            & (yData >= yPickMin)
            & (yData <= yPickMax)
        )[0]

        if len(indices) > 0:
            return indices
        return None

    def _pickImage(self, item, dataPos):
        """Pick an image item."""
        ox, oy = item._origin
        sx, sy = item._scale
        h, w = item._dataShape

        xMin = ox if sx >= 0 else ox + sx * w
        xMax = ox + sx * w if sx >= 0 else ox
        yMin = oy if sy >= 0 else oy + sy * h
        yMax = oy + sy * h if sy >= 0 else oy

        x, y = dataPos
        if x < xMin or x > xMax or y < yMin or y > yMax:
            return None

        col = int((x - ox) / sx) if sx != 0 else 0
        row = int((y - oy) / sy) if sy != 0 else 0

        col = numpy.clip(col, 0, w - 1)
        row = numpy.clip(row, 0, h - 1)

        return (row,), (col,)

    def _pickTriangles(self, item, dataPos):
        """Pick a triangles item."""
        x, y = dataPos
        xPts = item._x
        yPts = item._y
        triangles = item._triangles

        if len(xPts) == 0 or len(triangles) == 0:
            return None

        # Bounding box check
        if x < xPts.min() or x > xPts.max() or y < yPts.min() or y > yPts.max():
            return None

        # Build triangle coordinates array (N, 3, 3) for intersection test
        triCoords = numpy.zeros((len(triangles), 3, 3), dtype=numpy.float32)
        triCoords[:, :, 0] = xPts[triangles]
        triCoords[:, :, 1] = yPts[triangles]

        # Create vertical segment through clicked point
        segment = numpy.array(((x, y, -1.0), (x, y, 1.0)), dtype=numpy.float32)
        indices = segmentTrianglesIntersection(segment, triCoords)[0]
        if len(indices) == 0:
            return None

        # Convert triangle indices to vertex indices
        indices = numpy.unique(numpy.ravel(triangles[indices]))

        # Sort from furthest to closest
        dists = (xPts[indices] - x) ** 2 + (yPts[indices] - y) ** 2
        indices = indices[numpy.flip(numpy.argsort(dists), axis=0)]

        return tuple(indices)

    # Backend API: Update curve ##############################################

    def setCurveColor(self, curve, color):
        pass  # TODO

    # Backend API: Widget ####################################################

    def getWidgetHandle(self):
        return self

    def _rc_close(self):
        if isValid(self):  # Else the widget is deleted, see _closeCanvas
            super()._rc_close()

    def paintEvent(self, event):
        # Flush dirty items inside the paint event, where GPU operations are
        # safe (same pattern as OpenGL's paintGL). This ensures _backendRenderer
        # is up-to-date before pick() is called. Qt's update() coalesces
        # multiple calls, naturally batching mutations.
        plot = self._plotRef()
        if plot is not None and plot._getDirtyPlot():
            with plot._paintContext():
                pass
        super().paintEvent(event)

    def postRedisplay(self):
        self.request_draw(self._draw)
        # Schedule a Qt paint event so processEvents() flushes dirty items.
        # rendercanvas's request_draw() uses an async scheduler that may not
        # fire during processEvents(). Qt's update() coalesces multiple calls,
        # naturally batching mutations before the paint event fires.
        qt.QWidget.update(self)

    def replot(self):
        self.request_draw(self._draw)
        qt.QWidget.update(self)

    def saveGraph(self, fileName, fileFormat, dpi):
        if dpi is not None:
            _logger.warning("saveGraph ignores dpi parameter")

        if fileFormat not in ["png", "ppm", "svg", "tif", "tiff"]:
            raise NotImplementedError("Unsupported format: %s" % fileFormat)

        # Force a synchronous render
        self._draw()
        snapshot = self._renderer.snapshot()  # (H, W, 4) RGBA uint8

        # Drop the alpha channel: saveImageToFile expects (H, W, 3) RGB
        data = numpy.ascontiguousarray(snapshot[:, :, :3])
        # fileName is either a file-like object or a str
        saveImageToFile(data, fileName, fileFormat)

    # Backend API: Labels ####################################################

    def setGraphTitle(self, title):
        self._plotFrame.title = title

    def setGraphXLabel(self, label):
        self._plotFrame.xAxis.title = label

    def setGraphYLabel(self, label, axis):
        if axis == "left":
            self._plotFrame.yAxis.title = label
        else:
            self._plotFrame.y2Axis.title = label

    # Backend API: Limits ####################################################

    def _setDataRanges(self, xlim=None, ylim=None, y2lim=None):
        self._plotFrame.setDataRanges(xlim, ylim, y2lim)

    def _ensureAspectRatio(self, keepDim=None):
        """Update plot bounds in order to keep aspect ratio.

        Warning: keepDim on right Y axis is not implemented !

        :param str keepDim: The dimension to maintain: 'x', 'y' or None.
            If None (the default), the dimension with the largest range.
        """
        plotWidth, plotHeight = self._plotFrame.plotSize
        xRange, yRange, y2Range = self._plotFrame.dataRanges
        if keepDim is None:
            ranges = self._plot.getDataRange()
            keepDim = findDimToKeep(plotWidth, plotHeight, ranges.x, ranges.y)
        newXRange, newYRange, newY2Range = ensureAspectRatio(
            plotWidth, plotHeight, xRange, yRange, y2Range, keepDim
        )

        # Update plot frame bounds
        self._setDataRanges(xlim=newXRange, ylim=newYRange, y2lim=newY2Range)

    def _setPlotBounds(self, xRange=None, yRange=None, y2Range=None, keepDim=None):
        self._setDataRanges(xlim=xRange, ylim=yRange, y2lim=y2Range)
        if self.isKeepDataAspectRatio():
            self._ensureAspectRatio(keepDim)

    def setLimits(self, xmin, xmax, ymin, ymax, y2min=None, y2max=None):
        if y2min is None or y2max is None:
            y2Range = None
        else:
            y2Range = y2min, y2max
        self._setPlotBounds((xmin, xmax), (ymin, ymax), y2Range)

    def getGraphXLimits(self):
        return self._plotFrame.dataRanges.x

    def setGraphXLimits(self, xmin, xmax):
        self._setPlotBounds(xRange=(xmin, xmax), keepDim="x")

    def getGraphYLimits(self, axis):
        assert axis in ("left", "right")
        if axis == "left":
            return self._plotFrame.dataRanges.y
        else:
            return self._plotFrame.dataRanges.y2

    def setGraphYLimits(self, ymin, ymax, axis):
        assert axis in ("left", "right")
        if axis == "left":
            self._setPlotBounds(yRange=(ymin, ymax), keepDim="y")
        else:
            self._setPlotBounds(y2Range=(ymin, ymax), keepDim="y")

    # Backend API: Axes ######################################################

    def getXAxisTimeZone(self):
        return self._plotFrame.xAxis.timeZone

    def setXAxisTimeZone(self, tz):
        self._plotFrame.xAxis.timeZone = tz

    def isXAxisTimeSeries(self):
        return self._plotFrame.xAxis.isTimeSeries

    def setXAxisTimeSeries(self, isTimeSeries):
        self._plotFrame.xAxis.isTimeSeries = isTimeSeries

    def setXAxisScale(self, scale: AxisScaleType):
        if scale != self._plotFrame.xAxis.scale:
            if scale != "linear" and self._keepDataAspectRatio:
                _logger.warning("KeepDataAspectRatio is ignored with non linear axes")

            self._plotFrame.xAxis.scale = scale

    def setYAxisScale(self, scale: AxisScaleType):
        if (
            scale != self._plotFrame.yAxis.scale
            or scale != self._plotFrame.y2Axis.scale
        ):
            if scale != "linear" and self._keepDataAspectRatio:
                _logger.warning("KeepDataAspectRatio is ignored with non linear axes")

            self._plotFrame.yAxis.scale = scale
            self._plotFrame.y2Axis.scale = scale

    def setYAxisInverted(self, flag: bool):
        self._plotFrame.isYAxisInverted = flag

    def isYAxisInverted(self) -> bool:
        return self._plotFrame.isYAxisInverted

    def setXAxisInverted(self, flag: bool):
        self._plotFrame.isXAxisInverted = flag

    def isXAxisInverted(self) -> bool:
        return self._plotFrame.isXAxisInverted

    def isYRightAxisVisible(self):
        return self._plotFrame.isY2Axis

    def _hasNonLinearScales(self) -> bool:
        nonLinearScales = [
            scale
            for scale in (
                self._plotFrame.xAxis.scale,
                self._plotFrame.yAxis.scale,
                self._plotFrame.y2Axis.scale,
            )
            if scale != "linear"
        ]
        return len(nonLinearScales) != 0

    def isKeepDataAspectRatio(self):
        return False if self._hasNonLinearScales() else self._keepDataAspectRatio

    def setKeepDataAspectRatio(self, flag):
        if flag and self._hasNonLinearScales():
            _logger.warning("KeepDataAspectRatio is ignored with non linear axes")
        self._keepDataAspectRatio = flag

    def setGraphGrid(self, which):
        assert which in (None, "major", "both")
        self._plotFrame.grid = which is not None

    # Backend API: Data <-> Pixel ############################################

    def dataToPixel(self, x, y, axis):
        result = self._plotFrame.dataToPixel(x, y, axis)
        if result is None:
            return None
        dpr = self.getDevicePixelRatio()
        return tuple(value / dpr for value in result)

    def pixelToData(self, x, y, axis):
        dpr = self.getDevicePixelRatio()
        return self._plotFrame.pixelToData(x * dpr, y * dpr, axis)

    def getPlotBoundsInPixels(self):
        dpr = self.getDevicePixelRatio()
        return tuple(
            int(value / dpr)
            for value in self._plotFrame.plotOrigin + self._plotFrame.plotSize
        )

    # Backend API: Margins & Colors ##########################################

    def setAxesMargins(self, left: float, top: float, right: float, bottom: float):
        self._plotFrame.marginRatios = left, top, right, bottom

    def setForegroundColors(self, foregroundColor, gridColor):
        self._plotFrame.foregroundColor = foregroundColor
        self._plotFrame.gridColor = gridColor

    def setBackgroundColors(self, backgroundColor, dataBackgroundColor):
        self._backgroundColor = backgroundColor
        self._dataBackgroundColor = dataBackgroundColor

        if dataBackgroundColor is None:
            self._bgObj.visible = False
        else:
            self._bgObj.material.set_colors(dataBackgroundColor)
            self._bgObj.visible = True
