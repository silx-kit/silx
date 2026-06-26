# /*##########################################################################
#
# Copyright (c) 2024 European Synchrotron Radiation Facility
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

__authors__ = ["S. Kim"]
__license__ = "MIT"

import logging
import math
import re
import threading

import numpy
import wgpu

from rendercanvas.qt import QRenderWidget
import pygfx as gfx

from .. import items
from .._utils import FLOAT32_MINPOS
from . import BackendBase
from ... import colors
from ... import qt
from ._PlotFrameCore import PlotFrame2DCore
from silx.gui.colors import RGBAColorType

_logger = logging.getLogger(__name__)

_MATHDEFAULT_RE = re.compile(r"\$\\mathdefault\{([^}]*)\}\$")


def _stripMathDefault(text):
    """Strip matplotlib's $\\mathdefault{...}$ LaTeX wrapping from tick labels."""
    if text is None:
        return text
    return _MATHDEFAULT_RE.sub(r"\1", text)


# Dash pattern mapping: silx linestyle -> pygfx dash_pattern
# pygfx dash_pattern is a tuple of (dash, gap, ...) relative to line thickness
_DASH_PATTERNS = {
    "": None,
    " ": None,
    "-": None,  # solid
    "--": (3.7, 1.6, 3.7, 1.6),
    "-.": (6.4, 1.6, 1, 1.6),
    ":": (1, 1.65, 1, 1.65),
}


def _lineStyleToDashPattern(linestyle):
    """Convert silx linestyle to pygfx dash_pattern tuple."""
    if linestyle is None or linestyle in ("", " "):
        return None
    if isinstance(linestyle, tuple) and len(linestyle) == 2:
        # Custom (offset, (on, off, on, off, ...))
        return linestyle[1]
    return _DASH_PATTERNS.get(linestyle)


# silx symbol -> pygfx marker shape mapping
_SYMBOL_MAP = {
    "o": "circle",
    ".": "circle",  # smaller via size
    ",": "square",  # pixel
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


def _rgbaToGfxColor(color):
    """Convert silx RGBA color (4-tuple of 0..1 floats) to pygfx Color."""
    if color is None:
        return gfx.Color(1, 1, 1, 1)
    if isinstance(color, str):
        color = colors.rgba(color)
    if len(color) == 3:
        return gfx.Color(*color, 1.0)
    return gfx.Color(*color)


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
        self.group = gfx.Group()
        self._lineObj = None
        self._gapLineObj = None
        self._pointsObj = None
        self._errorGroup = None
        self._fillObj = None

        x = numpy.asarray(x, dtype=numpy.float32)
        y = numpy.asarray(y, dtype=numpy.float32)

        # Per-vertex color handling
        if isinstance(color, numpy.ndarray) and color.ndim == 2:
            perVertexColor = True
            vertexColors = numpy.asarray(color, dtype=numpy.float32)
            if vertexColors.shape[1] == 3:
                vertexColors = numpy.column_stack(
                    [
                        vertexColors,
                        numpy.full(len(vertexColors), alpha, dtype=numpy.float32),
                    ]
                )
            uniformColor = gfx.Color(1, 1, 1, 1)
        else:
            perVertexColor = False
            vertexColors = None
            rgba = colors.rgba(color)
            uniformColor = gfx.Color(rgba[0], rgba[1], rgba[2], rgba[3] * alpha)

        # Line
        dashPattern = _lineStyleToDashPattern(linestyle)
        hasLine = linestyle not in (None, "", " ")
        if hasLine and len(x) > 1:
            positions = numpy.zeros((len(x), 3), dtype=numpy.float32)
            positions[:, 0] = x
            positions[:, 1] = y

            lineKwargs = {}
            if perVertexColor:
                lineKwargs["colors"] = vertexColors

            geom = gfx.Geometry(positions=positions, **lineKwargs)
            mat = gfx.LineMaterial(
                thickness=max(linewidth, 1.0),
                color=uniformColor,
                color_mode="vertex" if perVertexColor else "uniform",
                dash_pattern=dashPattern if dashPattern else (),
            )
            self._lineObj = gfx.Line(geom, mat)
            self.group.add(self._lineObj)

            # Gap color line (behind the dashed line via z-offset)
            if gapcolor is not None and dashPattern:
                gapPositions = positions.copy()
                gapPositions[:, 2] = -0.1  # slightly behind
                gapRgba = colors.rgba(gapcolor)
                gapMat = gfx.LineMaterial(
                    thickness=max(linewidth, 1.0),
                    color=gfx.Color(*gapRgba),
                )
                self._gapLineObj = gfx.Line(
                    gfx.Geometry(positions=gapPositions), gapMat
                )
                self.group.add(self._gapLineObj)

        # Symbol / Points
        hasSymbol = symbol not in (None, "", " ")
        if hasSymbol:
            positions = numpy.zeros((len(x), 3), dtype=numpy.float32)
            positions[:, 0] = x
            positions[:, 1] = y

            markerShape = _SYMBOL_MAP.get(symbol, "circle")
            pointSize = symbolsize if symbol != "," else 1.0
            if symbol == ".":
                pointSize = max(pointSize * 0.5, 1.0)

            pointKwargs = {}
            if perVertexColor:
                pointKwargs["colors"] = vertexColors

            geom = gfx.Geometry(positions=positions, **pointKwargs)
            mat = gfx.PointsMarkerMaterial(
                marker=markerShape,
                size=pointSize,
                color=uniformColor,
                color_mode="vertex" if perVertexColor else "uniform",
                edge_width=0.5,
                edge_color=uniformColor,
            )
            self._pointsObj = gfx.Points(geom, mat)
            self.group.add(self._pointsObj)

        # Error bars
        if xerror is not None or yerror is not None:
            self._errorGroup = gfx.Group()
            errSegments = self._buildErrorBarSegments(x, y, xerror, yerror)
            if len(errSegments) > 0:
                errGeom = gfx.Geometry(positions=errSegments.astype(numpy.float32))
                errMat = gfx.LineSegmentMaterial(
                    thickness=1.0,
                    color=uniformColor,
                )
                errLine = gfx.Line(errGeom, errMat)
                self._errorGroup.add(errLine)
            self.group.add(self._errorGroup)

        # Fill between curve and baseline
        if fill and len(x) >= 2:
            self._fillObj = self._buildFill(x, y, baseline, uniformColor, alpha)
            if self._fillObj is not None:
                self._fillObj.local.z = -0.2  # behind curve line
                self.group.add(self._fillObj)

    @staticmethod
    def _buildErrorBarSegments(x, y, xerror, yerror):
        """Build line segments for error bars."""
        parts = []

        if yerror is not None:
            yerror = numpy.asarray(yerror, dtype=numpy.float64)
            if yerror.ndim == 2 and yerror.shape[1] == 1:
                yerror = numpy.ravel(yerror)
            if yerror.ndim == 0:
                yErrMinus = numpy.full_like(y, yerror)
                yErrPlus = yErrMinus
            elif yerror.ndim == 1:
                yErrMinus = yerror
                yErrPlus = yerror
            else:
                yErrMinus = yerror[0]
                yErrPlus = yerror[1]
            n = len(x)
            seg = numpy.empty((n * 2, 3), dtype=numpy.float64)
            seg[0::2, 0] = x
            seg[0::2, 1] = y - yErrMinus
            seg[0::2, 2] = 0
            seg[1::2, 0] = x
            seg[1::2, 1] = y + yErrPlus
            seg[1::2, 2] = 0
            parts.append(seg)

        if xerror is not None:
            xerror = numpy.asarray(xerror, dtype=numpy.float64)
            if xerror.ndim == 2 and xerror.shape[1] == 1:
                xerror = numpy.ravel(xerror)
            if xerror.ndim == 0:
                xErrMinus = numpy.full_like(x, xerror)
                xErrPlus = xErrMinus
            elif xerror.ndim == 1:
                xErrMinus = xerror
                xErrPlus = xerror
            else:
                xErrMinus = xerror[0]
                xErrPlus = xerror[1]
            n = len(x)
            seg = numpy.empty((n * 2, 3), dtype=numpy.float64)
            seg[0::2, 0] = x - xErrMinus
            seg[0::2, 1] = y
            seg[0::2, 2] = 0
            seg[1::2, 0] = x + xErrPlus
            seg[1::2, 1] = y
            seg[1::2, 2] = 0
            parts.append(seg)

        if parts:
            return numpy.concatenate(parts)
        return numpy.empty((0, 3), dtype=numpy.float64)

    @staticmethod
    def _buildFill(x, y, baseline, color, alpha):
        """Build a filled mesh between curve and baseline."""
        if baseline is None:
            baseY = numpy.zeros_like(y)
        elif isinstance(baseline, numpy.ndarray):
            baseY = baseline
        else:
            baseY = numpy.full_like(y, float(baseline))

        n = len(x)
        # Create triangle strip: for each segment, two triangles
        vertices = []
        indices = []
        for i in range(n):
            vertices.append([x[i], y[i], 0])
            vertices.append([x[i], baseY[i], 0])

        for i in range(n - 1):
            idx = i * 2
            indices.append([idx, idx + 1, idx + 2])
            indices.append([idx + 1, idx + 3, idx + 2])

        if not indices:
            return None

        vertices = numpy.array(vertices, dtype=numpy.float32)
        indices = numpy.array(indices, dtype=numpy.int32)

        fillColor = gfx.Color(color.r, color.g, color.b, alpha * 0.5)
        geom = gfx.Geometry(positions=vertices, indices=indices)
        mat = gfx.MeshBasicMaterial(color=fillColor, side="both")
        return gfx.Mesh(geom, mat)


def _fastColormapRange(data, colormap):
    """Fast colormap range for common cases (avoids slow normalizer pipeline)."""
    vmin = colormap.getVMin()
    vmax = colormap.getVMax()
    if vmin is not None and vmax is not None:
        return float(vmin), float(vmax)

    # Fast path for linear + minmax (most common streaming case)
    norm = colormap.getNormalization()
    mode = colormap.getAutoscaleMode()
    if norm == "linear" and mode == "minmax":
        if vmin is None:
            vmin = float(numpy.nanmin(data))
        if vmax is None:
            vmax = float(numpy.nanmax(data))
        if vmin >= vmax:
            vmax = vmin + 1.0
        return vmin, vmax

    # Fallback to full pipeline (log, sqrt, percentile, etc.)
    return colormap.getColormapRange(data)


# GPU colormap helpers ########################################################


def _colormapToLUT(colormap):
    """Extract a 256x4 float32 LUT from a silx Colormap.

    :param colormap: silx Colormap object
    :returns: (lut, nanColor) where lut is (256, 4) float32 in [0, 1]
        and nanColor is (4,) float32 RGBA
    """
    lut_u8 = colormap.getNColors(nbColors=256)  # (256, 4) uint8
    lut = lut_u8.astype(numpy.float32) / 255.0

    qNanColor = colormap.getNaNColor()
    nanColor = numpy.array(
        [qNanColor.redF(), qNanColor.greenF(), qNanColor.blueF(), qNanColor.alphaF()],
        dtype=numpy.float32,
    )
    return lut, nanColor


def _prepareScalarForGPU(data, normalization, vmin, vmax, gamma):
    """Apply normalization pre-processing on CPU for GPU colormap path.

    For linear and gamma, no pre-processing is needed — pygfx handles them
    natively via clim and gamma material parameters.
    For log/sqrt/arcsinh, apply the transform to both data and clim bounds
    so that pygfx's linear clim mapping produces the correct result.

    :param data: 2D numpy array (scalar image data)
    :param normalization: one of "linear", "log", "sqrt", "gamma", "arcsinh"
    :param vmin: colormap lower bound
    :param vmax: colormap upper bound
    :param gamma: gamma parameter (used only for "gamma" normalization)
    :returns: (scalar_data, clim, use_gamma) ready for GPU
        scalar_data: float32 2D array
        clim: (float, float) for ImageBasicMaterial
        use_gamma: float for ImageBasicMaterial.gamma
    """
    scalar = numpy.asarray(data, dtype=numpy.float32)

    if normalization == "linear":
        return scalar, (float(vmin), float(vmax)), 1.0

    elif normalization == "gamma":
        # pygfx gamma: pow(normalized, 1/gamma), but silx gamma means
        # pow(normalized, gamma). So pass 1/gamma to pygfx.
        return scalar, (float(vmin), float(vmax)), 1.0 / gamma

    elif normalization == "log":
        minPos = max(vmin, FLOAT32_MINPOS) if vmin > 0 else FLOAT32_MINPOS
        scalar = numpy.log10(numpy.clip(scalar, minPos, None))
        clim = (
            float(numpy.log10(max(vmin, minPos))),
            float(numpy.log10(max(vmax, minPos))),
        )
        return scalar, clim, 1.0

    elif normalization == "sqrt":
        scalar = numpy.sqrt(numpy.clip(scalar, 0, None))
        clim = (float(numpy.sqrt(max(vmin, 0))), float(numpy.sqrt(max(vmax, 0))))
        return scalar, clim, 1.0

    elif normalization == "arcsinh":
        scalar = numpy.arcsinh(scalar)
        clim = (float(numpy.arcsinh(vmin)), float(numpy.arcsinh(vmax)))
        return scalar, clim, 1.0

    else:
        _logger.warning("Unknown normalization %r, using linear", normalization)
        return scalar, (float(vmin), float(vmax)), 1.0


def _handleNaN(scalar_data, clim, lut, nanColor):
    """Replace NaN pixels with a sentinel value and set LUT[0] to nanColor.

    The sentinel is placed below clim[0] so it maps to LUT index 0.
    With wrap='clamp', values below clim[0] also map to LUT[0].

    :param scalar_data: float32 2D array (may be modified in-place)
    :param clim: (vmin, vmax) tuple
    :param lut: (256, 4) float32 array (modified in-place)
    :param nanColor: (4,) float32 RGBA
    :returns: (scalar_data, clim) with sentinel applied
    """
    hasNan = numpy.any(numpy.isnan(scalar_data))
    if not hasNan:
        return scalar_data, clim

    scalar_data = scalar_data.copy()
    vmin, vmax = clim

    # Sentinel: well below vmin so it maps to LUT[0]
    rng = vmax - vmin if vmax != vmin else 1.0
    sentinel = vmin - rng * 0.01

    # Replace NaN with sentinel
    nanMask = numpy.isnan(scalar_data)
    scalar_data[nanMask] = sentinel

    # Adjust clim so sentinel maps to ~index 0 and vmin maps to ~index 1+
    # LUT[0] = nanColor, LUT[1..255] = original LUT[0..254]
    newLut = numpy.empty_like(lut)
    newLut[0] = nanColor
    # Remap: compress original 256 entries into indices 1..255
    indices = numpy.linspace(0, 255, 255).astype(numpy.int32)
    newLut[1:] = lut[indices]

    lut[:] = newLut

    # Expand clim so sentinel→0, vmin→~1/256, vmax→255/256
    newVmin = sentinel
    # vmin should map to index ~1 out of 256
    # index = (val - newVmin) / (newVmax - newVmin) * 255
    # For val=vmin, index=1: 1 = (vmin - sentinel) / (newVmax - sentinel) * 255
    # newVmax = sentinel + (vmin - sentinel) * 255
    newVmax = sentinel + (vmax - sentinel) * 256.0 / 255.0

    return scalar_data, (float(newVmin), float(newVmax))


# WGSL Compute shaders ########################################################

_MINMAX_SHADER = """
struct Params {
    num_elements: u32,
}

@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> s_min: array<f32, 256>;
var<workgroup> s_max: array<f32, 256>;
var<workgroup> s_min_pos: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wgid: vec3<u32>) {
    let tid = lid.x;
    let gid = wgid.x * 256u + tid;
    let stride = 256u * ((params.num_elements + 255u) / 256u);

    // Initialize with identity values
    var local_min: f32 = 3.402823e+38;
    var local_max: f32 = -3.402823e+38;
    var local_min_pos: f32 = 3.402823e+38;

    // Grid-stride loop: each thread processes multiple elements
    var idx = gid;
    while (idx < params.num_elements) {
        let val = input_data[idx];
        // Skip NaN and Inf
        if (!isNan(val) && !isInf(val)) {
            local_min = min(local_min, val);
            local_max = max(local_max, val);
            if (val > 0.0) {
                local_min_pos = min(local_min_pos, val);
            }
        }
        idx += stride;
    }

    s_min[tid] = local_min;
    s_max[tid] = local_max;
    s_min_pos[tid] = local_min_pos;
    workgroupBarrier();

    // Tree reduction
    var step = 128u;
    while (step > 0u) {
        if (tid < step) {
            s_min[tid] = min(s_min[tid], s_min[tid + step]);
            s_max[tid] = max(s_max[tid], s_max[tid + step]);
            s_min_pos[tid] = min(s_min_pos[tid], s_min_pos[tid + step]);
        }
        workgroupBarrier();
        step = step >> 1u;
    }

    // Write workgroup result
    if (tid == 0u) {
        let out_idx = wgid.x * 3u;
        output_data[out_idx] = s_min[0];
        output_data[out_idx + 1u] = s_max[0];
        output_data[out_idx + 2u] = s_min_pos[0];
    }
}

fn isNan(v: f32) -> bool {
    return !(v == v);
}

fn isInf(v: f32) -> bool {
    return (v == 3.402823e+38) || (v == -3.402823e+38) || abs(v) > 3.4e+38;
}
"""

_HISTOGRAM_SHADER = """
struct Params {
    num_elements: u32,
    data_min: f32,
    data_max: f32,
    num_bins: u32,
    norm_mode: u32,  // 0=linear, 1=log10, 2=sqrt, 3=arcsinh
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> histogram: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> params: Params;

fn apply_norm(val: f32, mode: u32) -> f32 {
    switch (mode) {
        case 1u: { // log10
            if (val <= 0.0) { return -1e30; }
            return log2(val) * 0.30102999566;  // log2(x) / log2(10)
        }
        case 2u: { return sqrt(max(val, 0.0)); }  // sqrt
        case 3u: { return asinh(val); }             // arcsinh
        default: { return val; }                    // linear/gamma
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let range = params.data_max - params.data_min;
    if (range <= 0.0) {
        return;
    }

    // Grid-stride loop for large data (>65535 workgroups)
    let total_threads = 256u * ((params.num_elements + 255u) / 256u);
    var idx = gid.x;
    while (idx < params.num_elements) {
        let val = input_data[idx];

        // Skip NaN
        if (val == val) {
            let transformed = apply_norm(val, params.norm_mode);
            var normalized = (transformed - params.data_min) / range;
            normalized = clamp(normalized, 0.0, 0.999999);
            let bin = u32(normalized * f32(params.num_bins));
            atomicAdd(&histogram[bin], 1u);
        }

        idx += total_threads;
    }
}
"""

# Normalization mode constants for histogram shader
_HIST_NORM_LINEAR = 0
_HIST_NORM_LOG = 1
_HIST_NORM_SQRT = 2
_HIST_NORM_ARCSINH = 3


class _WgpuComputeHelper:
    """GPU compute helper for min/max reduction and histogram computation."""

    _instance = None

    @classmethod
    def get(cls):
        """Get or create the singleton compute helper."""
        if cls._instance is None:
            try:
                cls._instance = cls()
            except Exception:
                _logger.debug("Failed to create GPU compute helper", exc_info=True)
                cls._instance = False  # Sentinel: tried and failed
        if cls._instance is False:
            return None
        return cls._instance

    def __init__(self):
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        self._device = adapter.request_device_sync()

        # Create minmax pipeline
        minmax_module = self._device.create_shader_module(code=_MINMAX_SHADER)
        self._minmax_bgl = self._device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "read-only-storage"},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "storage"},
                },
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "uniform"},
                },
            ]
        )
        self._minmax_pipeline = self._device.create_compute_pipeline(
            layout=self._device.create_pipeline_layout(
                bind_group_layouts=[self._minmax_bgl]
            ),
            compute={"module": minmax_module, "entry_point": "main"},
        )

        # Create histogram pipeline
        hist_module = self._device.create_shader_module(code=_HISTOGRAM_SHADER)
        self._hist_bgl = self._device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "read-only-storage"},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "storage"},
                },
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "uniform"},
                },
            ]
        )
        self._hist_pipeline = self._device.create_compute_pipeline(
            layout=self._device.create_pipeline_layout(
                bind_group_layouts=[self._hist_bgl]
            ),
            compute={"module": hist_module, "entry_point": "main"},
        )

    def compute_minmax(self, data):
        """Compute (min, minPositive, max) using GPU reduction.

        :param data: numpy array (will be flattened to float32)
        :returns: (min, minPositive, max) tuple of floats, or None on failure
        """
        flat = numpy.ascontiguousarray(data.ravel(), dtype=numpy.float32)
        num_elements = len(flat)
        if num_elements == 0:
            return None

        workgroup_size = 256
        num_workgroups = min(
            (num_elements + workgroup_size - 1) // workgroup_size, 65535
        )

        # Input buffer
        input_buf = self._device.create_buffer_with_data(
            data=flat.tobytes(),
            usage=wgpu.BufferUsage.STORAGE,
        )

        # Output buffer: 3 floats per workgroup (min, max, minPos)
        output_size = num_workgroups * 3 * 4  # float32
        output_buf = self._device.create_buffer(
            size=output_size,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )

        # Params uniform
        params = numpy.array([num_elements], dtype=numpy.uint32)
        params_buf = self._device.create_buffer_with_data(
            data=params.tobytes(),
            usage=wgpu.BufferUsage.UNIFORM,
        )

        # Bind group
        bind_group = self._device.create_bind_group(
            layout=self._minmax_bgl,
            entries=[
                {"binding": 0, "resource": {"buffer": input_buf}},
                {"binding": 1, "resource": {"buffer": output_buf}},
                {"binding": 2, "resource": {"buffer": params_buf}},
            ],
        )

        # Dispatch
        encoder = self._device.create_command_encoder()
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(self._minmax_pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(num_workgroups)
        compute_pass.end()

        # Readback
        readback_buf = self._device.create_buffer(
            size=output_size,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        )
        encoder.copy_buffer_to_buffer(output_buf, 0, readback_buf, 0, output_size)
        self._device.queue.submit([encoder.finish()])

        readback_buf.map_sync(wgpu.MapMode.READ)
        result_bytes = readback_buf.read_mapped()
        result = numpy.frombuffer(result_bytes, dtype=numpy.float32).copy()
        readback_buf.unmap()

        # CPU final reduction of per-workgroup results
        result = result.reshape(-1, 3)
        final_min = float(numpy.min(result[:, 0]))
        final_max = float(numpy.max(result[:, 1]))
        min_pos_vals = result[:, 2]
        valid_pos = min_pos_vals[min_pos_vals < 3.4e38]
        final_min_pos = (
            float(numpy.min(valid_pos)) if len(valid_pos) > 0 else float("inf")
        )

        # Clean up
        input_buf.destroy()
        output_buf.destroy()
        params_buf.destroy()
        readback_buf.destroy()

        return (final_min, final_min_pos, final_max)

    def compute_histogram(self, data, data_min, data_max, num_bins=256, norm_mode=0):
        """Compute histogram using GPU atomic operations.

        :param data: numpy array (will be flattened to float32)
        :param data_min: histogram lower bound (in normalized space)
        :param data_max: histogram upper bound (in normalized space)
        :param num_bins: number of bins (default 256)
        :param norm_mode: 0=linear, 1=log10, 2=sqrt, 3=arcsinh
        :returns: (counts, bin_edges) or None on failure
        """
        flat = numpy.ascontiguousarray(data.ravel(), dtype=numpy.float32)
        num_elements = len(flat)
        if num_elements == 0:
            return None

        workgroup_size = 256
        num_workgroups = min(
            (num_elements + workgroup_size - 1) // workgroup_size, 65535
        )

        # Input buffer
        input_buf = self._device.create_buffer_with_data(
            data=flat.tobytes(),
            usage=wgpu.BufferUsage.STORAGE,
        )

        # Histogram buffer (zero-initialized)
        hist_size = num_bins * 4  # uint32
        hist_buf = self._device.create_buffer(
            size=hist_size,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )

        # Params uniform (8 x u32/f32 = 32 bytes, matches Params struct)
        params = numpy.zeros(8, dtype=numpy.float32)
        params_view = params.view(numpy.uint32)
        params_view[0] = num_elements
        params[1] = numpy.float32(data_min)
        params[2] = numpy.float32(data_max)
        params_view[3] = num_bins
        params_view[4] = norm_mode
        # [5], [6], [7] = padding
        params_buf = self._device.create_buffer_with_data(
            data=params.tobytes(),
            usage=wgpu.BufferUsage.UNIFORM,
        )

        # Bind group
        bind_group = self._device.create_bind_group(
            layout=self._hist_bgl,
            entries=[
                {"binding": 0, "resource": {"buffer": input_buf}},
                {"binding": 1, "resource": {"buffer": hist_buf}},
                {"binding": 2, "resource": {"buffer": params_buf}},
            ],
        )

        # Dispatch
        encoder = self._device.create_command_encoder()
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(self._hist_pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(num_workgroups)
        compute_pass.end()

        # Readback
        readback_buf = self._device.create_buffer(
            size=hist_size,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        )
        encoder.copy_buffer_to_buffer(hist_buf, 0, readback_buf, 0, hist_size)
        self._device.queue.submit([encoder.finish()])

        readback_buf.map_sync(wgpu.MapMode.READ)
        result_bytes = readback_buf.read_mapped()
        counts = numpy.frombuffer(result_bytes, dtype=numpy.uint32).copy()
        readback_buf.unmap()

        bin_edges = numpy.linspace(data_min, data_max, num_bins + 1)

        # Clean up
        input_buf.destroy()
        hist_buf.destroy()
        params_buf.destroy()
        readback_buf.destroy()

        return (counts, bin_edges)


# Async compute for streaming ##################################################


class _AsyncCompute:
    """Non-blocking async computation for streaming image data.

    The render thread never blocks. It submits data and reads the latest
    completed result. A single worker thread processes requests, always
    skipping to the newest data (stale requests are dropped).

    Stats and histogram are computed on full data (no subsampling needed
    since computation runs off the render thread). GPU histogram at
    4096x4096 sustains ~50Hz; stats use optimized CPU (~20Hz for 4K).
    """

    def __init__(self, gpu_compute=None):
        self._gpu_compute = gpu_compute

        # Latest results (read by render thread)
        self._stats_result = None
        self._hist_result = None

        # Pending requests (written by render thread, read by worker)
        self._pending_stats_data = None
        self._pending_hist_request = None  # (data, data_min, data_max, num_bins)

        # Lock protects pending slots and results
        self._lock = threading.Lock()

        # Worker thread
        self._running = True
        self._event = threading.Event()  # Signals new work available
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def shutdown(self):
        """Stop the worker thread."""
        self._running = False
        self._event.set()

    def submit_stats(self, data):
        """Submit data for async stats computation. Non-blocking.

        :param data: numpy array (a reference is kept until processed)
        """
        with self._lock:
            self._pending_stats_data = data
        self._event.set()

    def submit_histogram(self, data, data_min, data_max, num_bins=256, norm_mode=0):
        """Submit data for async histogram computation. Non-blocking.

        :param data: numpy array
        :param data_min: histogram lower bound (in normalized space)
        :param data_max: histogram upper bound (in normalized space)
        :param num_bins: number of bins
        :param norm_mode: 0=linear, 1=log10, 2=sqrt, 3=arcsinh
        """
        with self._lock:
            self._pending_hist_request = (data, data_min, data_max, num_bins, norm_mode)
        self._event.set()

    def get_stats(self):
        """Return the latest computed stats, or None. Non-blocking."""
        return self._stats_result

    def get_histogram(self):
        """Return the latest computed histogram, or None. Non-blocking."""
        return self._hist_result

    def invalidate(self):
        """Clear cached results (e.g., when colormap changes)."""
        with self._lock:
            self._stats_result = None
            self._hist_result = None

    def _worker(self):
        """Worker thread: processes the latest pending request."""
        while self._running:
            self._event.wait()
            self._event.clear()

            if not self._running:
                break

            # Grab latest pending work (drop stale)
            with self._lock:
                stats_data = self._pending_stats_data
                self._pending_stats_data = None
                hist_req = self._pending_hist_request
                self._pending_hist_request = None

            # Process stats
            if stats_data is not None:
                result = self._compute_stats(stats_data)
                if result is not None:
                    self._stats_result = result

            # Process histogram
            if hist_req is not None:
                data, dmin, dmax, nbins, nmode = hist_req
                result = self._compute_histogram(data, dmin, dmax, nbins, nmode)
                if result is not None:
                    self._hist_result = result

    def _compute_stats(self, data):
        """Compute (min, minPositive, max) on full data."""
        try:
            data = numpy.asarray(data)
            if data.size == 0:
                return None
            min_ = float(numpy.nanmin(data))
            max_ = float(numpy.nanmax(data))
            if not (numpy.isfinite(min_) and numpy.isfinite(max_)):
                return None
            if min_ > 0:
                minPositive = min_
            else:
                pos = data[data > 0]
                minPositive = float(numpy.min(pos)) if len(pos) > 0 else float("inf")
            return (min_, minPositive, max_)
        except Exception:
            return None

    def _compute_histogram(self, data, data_min, data_max, num_bins, norm_mode=0):
        """Compute histogram, preferring GPU."""
        try:
            # GPU path
            if self._gpu_compute is not None:
                result = self._gpu_compute.compute_histogram(
                    data, data_min, data_max, num_bins, norm_mode=norm_mode
                )
                if result is not None:
                    return result
            # CPU fallback
            data = numpy.asarray(data)
            flat = data.ravel()
            finite = flat[numpy.isfinite(flat)]
            counts, edges = numpy.histogram(
                finite, bins=num_bins, range=(data_min, data_max)
            )
            return (counts, edges)
        except Exception:
            return None


# Image item ##################################################################


class _PygfxImageItem:
    """Manages pygfx scene objects for a single image."""

    def __init__(self, data, origin, scale, colormap, alpha):
        self.group = gfx.Group()
        self.yaxis = "left"
        self._imageObj = None
        self._scalarShape = None
        self._cmapName = None
        self._cmapTexture = None
        self._gpuColormapInfo = None  # Set when using GPU colormap path
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

        # Data: upload scalar float32 directly (no CPU colormap)
        if data.dtype == numpy.float32 and data.flags["C_CONTIGUOUS"]:
            scalarData = data
        else:
            scalarData = numpy.ascontiguousarray(data, dtype=numpy.float32)

        # Range: fast path for linear+minmax
        if colormap is not None:
            vmin, vmax = _fastColormapRange(data, colormap)
            cmapTex = self._getOrCreateCmapTexture(colormap, alpha)
        else:
            vmin = float(numpy.nanmin(data))
            vmax = float(numpy.nanmax(data))
            if vmin == vmax:
                vmax = vmin + 1.0
            cmapTex = None

        if self._imageObj is None:
            # First time: create GPU objects
            tex = gfx.Texture(scalarData, dim=2)
            geom = gfx.Geometry(grid=tex)
            mat = gfx.ImageBasicMaterial(
                clim=(vmin, vmax),
                map=cmapTex,
                interpolation="nearest",
            )
            self._imageObj = gfx.Image(geom, mat)
            self.group.add(self._imageObj)
        else:
            # Reuse: update texture data + clim (no GPU object creation)
            self._imageObj.geometry.grid.set_data(scalarData)
            self._imageObj.material.clim = (vmin, vmax)
            if cmapTex is not None:
                self._imageObj.material.map = cmapTex

        ox, oy = origin
        sx, sy = scale
        self._imageObj.local.position = (ox, oy, 0)
        self._imageObj.local.scale = (sx, sy, 1)

    def updateData(self, data, clim=None):
        """Fast path: update only the texture data (no item system overhead).

        Requires the image object to already exist and data shape to match.

        :param data: New image data (2D array)
        :param clim: (vmin, vmax) tuple for color limits, or None to compute from data
        """
        if self._imageObj is None:
            return
        if data.dtype == numpy.float32 and data.flags["C_CONTIGUOUS"]:
            scalarData = data
        else:
            scalarData = numpy.ascontiguousarray(data, dtype=numpy.float32)
        self._imageObj.geometry.grid.set_data(scalarData)
        if clim is None:
            dmin = float(numpy.nanmin(data))
            dmax = float(numpy.nanmax(data))
            if dmin >= dmax:
                dmax = dmin + 1.0
            clim = (dmin, dmax)

        self._imageObj.material.clim = clim

    def _buildRGBA(self, data, origin, scale, alpha):
        self._scalarShape = None

        if data.dtype == numpy.float64:
            data = data.astype(numpy.float32)
        if data.dtype in (numpy.float32, numpy.float64):
            rgbaData = (numpy.clip(data, 0, 1) * 255).astype(numpy.uint8)
        else:
            rgbaData = numpy.asarray(data, dtype=numpy.uint8)
        if rgbaData.shape[2] == 3:
            alphaChannel = numpy.full(rgbaData.shape[:2] + (1,), 255, dtype=numpy.uint8)
            rgbaData = numpy.concatenate([rgbaData, alphaChannel], axis=-1)

        rgbaFloat = rgbaData.astype(numpy.float32) / 255.0
        if alpha < 1.0:
            rgbaFloat = rgbaFloat.copy()
            rgbaFloat[:, :, 3] *= alpha
        rgbaFloat = numpy.ascontiguousarray(rgbaFloat)

        geom = gfx.Geometry(grid=gfx.Texture(rgbaFloat, dim=2))
        mat = gfx.ImageBasicMaterial(interpolation="nearest")
        self._imageObj = gfx.Image(geom, mat)
        self.group.add(self._imageObj)

        ox, oy = origin
        sx, sy = scale
        self._imageObj.local.position = (ox, oy, 0)
        self._imageObj.local.scale = (sx, sy, 1)

    def _getOrCreateCmapTexture(self, colormap, alpha):
        """Cache colormap LUT texture, recreate only when colormap changes."""
        name = colormap.getName()
        if name == self._cmapName and self._cmapTexture is not None:
            return self._cmapTexture

        lut = colormap.getNColors()  # (256, 4) uint8 RGBA
        lutFloat = lut.astype(numpy.float32) / 255.0
        if alpha < 1.0:
            lutFloat = lutFloat.copy()
            lutFloat[:, 3] *= alpha
        self._cmapTexture = gfx.Texture(lutFloat, dim=1)
        self._cmapName = name
        return self._cmapTexture

    def _initGPUColormap(self, data, origin, scale, colormap, alpha):
        """Initialize image using GPU-native colormap rendering.

        Uploads scalar data as a 1-channel texture and uses pygfx's
        ImageBasicMaterial.map for GPU-side colormap application.
        """
        normalization = colormap.getNormalization()
        cmapRange = colormap.getColormapRange(data)
        vmin, vmax = cmapRange
        gamma = colormap.getGammaNormalizationParameter()

        # 1. Normalization pre-processing
        scalar_data, clim, use_gamma = _prepareScalarForGPU(
            data, normalization, vmin, vmax, gamma
        )

        # 2. Build LUT and handle NaN
        lut, nanColor = _colormapToLUT(colormap)
        scalar_data, clim = _handleNaN(scalar_data, clim, lut, nanColor)

        # 3. Apply alpha to LUT
        if alpha < 1.0:
            lut = lut.copy()
            lut[:, 3] *= alpha

        # 4. Create GPU objects
        scalar_data = numpy.ascontiguousarray(scalar_data)
        lut_tex = gfx.Texture(lut, dim=1)
        cmap_map = gfx.TextureMap(lut_tex, filter="nearest", wrap="clamp")

        geom = gfx.Geometry(grid=gfx.Texture(scalar_data, dim=2))
        mat = gfx.ImageBasicMaterial(
            map=cmap_map,
            clim=clim,
            gamma=use_gamma,
            interpolation="nearest",
        )
        self._imageObj = gfx.Image(geom, mat)

        # Position and scale
        ox, oy = origin
        sx, sy = scale
        self._imageObj.local.position = (ox, oy, 0)
        self._imageObj.local.scale = (sx, sy, 1)

        self.group.add(self._imageObj)

        # Store info for dynamic updates (clim/LUT changes without re-upload)
        self._gpuColormapInfo = {
            "material": mat,
            "lut_texture": lut_tex,
            "normalization": normalization,
            "vmin": vmin,
            "vmax": vmax,
        }


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

        color = numpy.asarray(color, dtype=numpy.float32)
        if color.ndim == 2:
            if color.shape[1] == 3:
                color = numpy.column_stack(
                    [color, numpy.full(len(color), alpha, dtype=numpy.float32)]
                )
            geom = gfx.Geometry(positions=positions, indices=triangles, colors=color)
            mat = gfx.MeshBasicMaterial(color_mode="vertex", side="both")
        else:
            rgba = colors.rgba(color)
            geom = gfx.Geometry(positions=positions, indices=triangles)
            mat = gfx.MeshBasicMaterial(
                color=gfx.Color(rgba[0], rgba[1], rgba[2], rgba[3] * alpha),
                side="both",
            )

        self._meshObj = gfx.Mesh(geom, mat)
        self.group.add(self._meshObj)


class _PygfxShapeItem(dict):
    """Manages pygfx scene objects for shapes."""

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

        x = numpy.asarray(x, dtype=numpy.float32)
        y = numpy.asarray(y, dtype=numpy.float32)

        if shape == "rectangle":
            xMin, xMax = x
            x = numpy.array((xMin, xMin, xMax, xMax), dtype=numpy.float32)
            yMin, yMax = y
            y = numpy.array((yMin, yMax, yMax, yMin), dtype=numpy.float32)

        fill = fill if shape != "polylines" else False

        rgba = colors.rgba(color)
        dashPattern = _lineStyleToDashPattern(linestyle)

        self.update(
            {
                "shape": shape,
                "color": rgba,
                "fill": fill,
                "x": x,
                "y": y,
                "linewidth": linewidth,
                "overlay": overlay,
            }
        )

        self.group = gfx.Group()

        gfxColor = gfx.Color(*rgba)

        # Build outline
        if shape in ("polygon", "rectangle"):
            positions = numpy.zeros((len(x) + 1, 3), dtype=numpy.float32)
            positions[:-1, 0] = x
            positions[:-1, 1] = y
            positions[-1, 0] = x[0]
            positions[-1, 1] = y[0]
        elif shape == "polylines":
            positions = numpy.zeros((len(x), 3), dtype=numpy.float32)
            positions[:, 0] = x
            positions[:, 1] = y
        elif shape in ("line", "hline", "vline"):
            positions = numpy.zeros((len(x), 3), dtype=numpy.float32)
            positions[:, 0] = x
            positions[:, 1] = y
        else:
            positions = numpy.zeros((len(x), 3), dtype=numpy.float32)
            positions[:, 0] = x
            positions[:, 1] = y

        if len(positions) >= 2:
            # Gap color line: solid line behind the dashed foreground line.
            # Must be at a lower z to pass the strict '<' depth test.
            if gapcolor is not None and dashPattern:
                gapPositions = positions.copy()
                gapPositions[:, 2] = -0.1  # slightly behind
                gapRgba = colors.rgba(gapcolor)
                gapMat = gfx.LineMaterial(
                    thickness=max(linewidth, 1.0),
                    color=gfx.Color(*gapRgba),
                )
                gapLineObj = gfx.Line(gfx.Geometry(positions=gapPositions), gapMat)
                self.group.add(gapLineObj)

            # Foreground line (dashed or solid) at z=0 (in front of gap line)
            geom = gfx.Geometry(positions=positions)
            mat = gfx.LineMaterial(
                thickness=max(linewidth, 1.0),
                color=gfxColor,
                dash_pattern=dashPattern if dashPattern else (),
            )
            lineObj = gfx.Line(geom, mat)
            self.group.add(lineObj)

        # Build fill for closed shapes
        if fill and shape in ("polygon", "rectangle") and len(x) >= 3:
            fillObj = self._buildPolygonFill(x, y, rgba)
            if fillObj is not None:
                fillObj.local.z = -0.2  # behind lines
                self.group.add(fillObj)

    @staticmethod
    def _buildPolygonFill(x, y, rgba):
        """Create a semi-transparent polygon fill using a triangle fan mesh."""
        n = len(x)
        if n < 3:
            return None

        # Sort vertices by angle from centroid to avoid bowtie patterns
        cx, cy = numpy.nanmean(x), numpy.nanmean(y)
        angles = numpy.arctan2(y - cy, x - cx)
        order = numpy.argsort(angles)
        x = x[order]
        y = y[order]

        # Triangle fan from vertex 0
        positions = numpy.zeros((n, 3), dtype=numpy.float32)
        positions[:, 0] = x
        positions[:, 1] = y

        indices = numpy.zeros(((n - 2), 3), dtype=numpy.uint32)
        for i in range(n - 2):
            indices[i] = [0, i + 1, i + 2]

        fillColor = gfx.Color(rgba[0], rgba[1], rgba[2], 0.3)
        geom = gfx.Geometry(indices=indices, positions=positions)
        mat = gfx.MeshBasicMaterial(
            color=fillColor,
            side="both",
            depth_write=False,
        )
        return gfx.Mesh(geom, mat)


class _PygfxMarkerItem(dict):
    """Manages pygfx scene objects for markers."""

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

        dashPattern = _lineStyleToDashPattern(linestyle)

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
                "dashpattern": dashPattern,
                "yaxis": yaxis,
                "font": font,
                "bgcolor": bgcolor,
            }
        )

        self.group = gfx.Group()
        self._lineObj = None
        self._textObj = None
        rgba = colors.rgba(color)
        gfxColor = gfx.Color(*rgba)

        if x is not None and y is not None:
            # Point marker
            positions = numpy.array([[x, y, 0]], dtype=numpy.float32)
            markerShape = _SYMBOL_MAP.get(symbol, "plus")
            geom = gfx.Geometry(positions=positions)
            mat = gfx.PointsMarkerMaterial(
                marker=markerShape,
                size=symbolsize,
                color=gfxColor,
                edge_width=1.0,
                edge_color=gfxColor,
            )
            self._pointsObj = gfx.Points(geom, mat)
            self.group.add(self._pointsObj)


# BackendPygfx ################################################################


class BackendPygfx(BackendBase.BackendBase, QRenderWidget):
    """pygfx/WGPU-based Plot backend.

    Uses pygfx for GPU-accelerated rendering via WGPU (Vulkan/Metal/DX12).
    """

    _TEXT_MARKER_PADDING = 4
    VSYNC = True
    """Enable VSync (default True). Set to False before creating the plot
    to unlock frame rates beyond the monitor refresh rate."""

    PRESENT_METHOD = "screen"
    """Present method for rendering. "screen" uses direct GPU rendering
    (~3x faster), "image" uses CPU readback (works with remote desktops).
    Set before creating the plot."""

    def __init__(self, plot, parent=None):
        QRenderWidget.__init__(
            self,
            parent=parent,
            present_method=self.PRESENT_METHOD,
            vsync=self.VSYNC,
        )
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
        self._renderer = gfx.WgpuRenderer(self, pixel_ratio=4)
        self._scene = gfx.Scene()

        # Camera: orthographic for 2D plotting
        self._camera = gfx.OrthographicCamera(640, 480, maintain_aspect=False)

        # Scene hierarchy
        self._bgGroup = gfx.Group()
        self._dataGroup = gfx.Group()
        self._overlayGroup = gfx.Group()
        self._frameGroup = gfx.Group()

        # Shift overlays forward in z so they always render in front of data.
        # Camera z-range is wide (near=-100..far=100), so z=10 is safe.
        self._overlayGroup.local.z = 10

        self._scene.add(self._bgGroup)
        self._scene.add(self._dataGroup)
        self._scene.add(self._overlayGroup)
        self._scene.add(self._frameGroup)

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

        # Screen-space scene for frame/axes rendering (PR 9)
        self._screenScene = gfx.Scene()
        self._screenBg = gfx.Background(
            None, gfx.BackgroundMaterial(gfx.Color(1, 1, 1, 1))
        )
        self._screenScene.add(self._screenBg)
        self._screenFrameGroup = gfx.Group()
        self._screenScene.add(self._screenFrameGroup)
        self._screenCamera = gfx.OrthographicCamera(maintain_aspect=False)
        self._cachedBgColor = (1.0, 1.0, 1.0, 1.0)

        # Frame rendering objects (populated by _updateFrame)
        self._frameLines = None
        self._gridLines = None
        self._frameTexts = []
        self._titleText = None

        # Crosshair cursor lines
        self._crosshairHLine = None
        self._crosshairVLine = None

        self._reusableImageItem = None  # Pool for image item reuse

        # GPU compute helper (lazy singleton)
        self._gpuCompute = None

        # Async compute for streaming (non-blocking stats/histogram)
        self._asyncCompute = None  # Lazy init

        self.request_draw(self._draw)
        self.setAutoFillBackground(False)
        self.setMouseTracking(True)

    def _getGpuCompute(self):
        """Get or create the GPU compute helper (lazy initialization)."""
        if self._gpuCompute is None:
            self._gpuCompute = _WgpuComputeHelper.get()
        return self._gpuCompute

    def _getAsyncCompute(self):
        """Get or create the async compute helper."""
        if self._asyncCompute is None:
            self._asyncCompute = _AsyncCompute(gpu_compute=self._getGpuCompute())
        return self._asyncCompute

    def _computeGpuDataStats(self, data):
        """Submit data for async stats computation and return latest result.

        Called from ColormapMixIn._setColormappedData() to pre-fill the
        autoscale range cache. Non-blocking: submits work to background
        thread and returns the most recent completed result.

        :param data: numpy array
        :returns: (min, minPositive, max) or None
        """
        if data is None:
            return None
        ac = self._getAsyncCompute()
        ac.submit_stats(data)
        return ac.get_stats()

    def _computeGpuHistogram(self, data, data_min, data_max, num_bins=256, norm_mode=0):
        """Submit data for async histogram computation and return latest result.

        Non-blocking: submits work to background thread and returns
        the most recent completed histogram.

        :param data: numpy array
        :param data_min: histogram lower bound (in normalized space)
        :param data_max: histogram upper bound (in normalized space)
        :param num_bins: number of bins
        :param norm_mode: 0=linear, 1=log10, 2=sqrt, 3=arcsinh
        :returns: (counts, bin_edges) or None
        """
        ac = self._getAsyncCompute()
        ac.submit_histogram(data, data_min, data_max, num_bins, norm_mode)
        return ac.get_histogram()

    def _getDefaultFont(self):
        if self._defaultFont is None:
            app = qt.QApplication.instance()
            if app is not None:
                self._defaultFont = app.font()
            else:
                self._defaultFont = qt.QFont()
        return self._defaultFont

    def getDevicePixelRatio(self):
        return self.devicePixelRatioF()

    def getDotsPerInch(self):
        screen = self.screen()
        if screen is not None:
            return screen.logicalDotsPerInch() * self.getDevicePixelRatio()
        return 92

    # Drawing ###############################################################

    def _draw(self):
        plot = self._plotRef()
        if plot is None:
            return

        with plot._paintContext():
            self._syncPlotFrame()
            self._syncCamera()
            self._updateFrame()
            self._updateMarkers()
            self._updateCrosshair()

            # First pass: render frame (background + axes) in full widget
            self._renderer.render(self._screenScene, self._screenCamera, flush=False)

            # Second pass: render data scene in plot area viewport only
            dpr = self.getDevicePixelRatio()
            left, top = self._plotFrame.plotOrigin
            pw, ph = self._plotFrame.plotSize
            # Viewport rect is in logical pixels
            plotRect = (left / dpr, top / dpr, pw / dpr, ph / dpr)
            self._renderer.render(
                self._scene, self._camera, rect=plotRect, flush=True, clear=False
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
        """Update camera to match the current data ranges."""
        trRanges = self._plotFrame.transformedDataRanges
        xMin, xMax = trRanges.x
        yMin, yMax = trRanges.y

        if self._plotFrame.isXAxisInverted:
            xMin, xMax = xMax, xMin
        if self._plotFrame.isYAxisInverted:
            yMin, yMax = yMax, yMin

        # Ensure non-zero extent to avoid camera errors
        if xMin == xMax:
            xMin -= 0.5
            xMax += 0.5
        if yMin == yMax:
            yMin -= 0.5
            yMax += 0.5

        # show_rect(left, right, top, bottom)
        # height = bottom - top; positive height means Y increases upward
        # top=yMin, bottom=yMax → yMax at top of viewport, yMin at bottom
        extent = max(abs(xMax - xMin), abs(yMax - yMin), 1.0)
        self._camera.show_rect(xMin, xMax, yMin, yMax, depth=extent)

        # Populate projection matrix caches so isDirty returns False
        # (pygfx doesn't use OpenGL projection matrices, but isDirty checks them)
        _ = self._plotFrame.transformedDataProjMat
        _ = self._plotFrame.transformedDataY2ProjMat

    def _updateFrame(self):
        """Update axes, ticks, grid, labels in screen space."""
        # Update background color only when changed
        bgColor = self._backgroundColor
        if self._cachedBgColor != bgColor:
            self._screenBg.material = gfx.BackgroundMaterial(gfx.Color(*bgColor))
            self._cachedBgColor = bgColor

        if not self._plotFrame.isDirty:
            return  # Frame unchanged, keep cached objects

        # Clear previous frame objects (frame group only, not markers/crosshair)
        for child in list(self._screenFrameGroup.children):
            self._screenFrameGroup.remove(child)

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

        # Render grid lines
        if len(gridVertices) >= 2:
            gridColor = gfx.Color(*self._plotFrame.gridColor)
            geom = gfx.Geometry(
                positions=numpy.column_stack(
                    [gridVertices, numpy.zeros(len(gridVertices), dtype=numpy.float32)]
                )
            )
            mat = gfx.LineSegmentMaterial(thickness=1.0, color=gridColor)
            gridLine = gfx.Line(geom, mat)
            self._screenFrameGroup.add(gridLine)

        # Render frame lines (axes)
        if len(vertices) >= 2:
            fgColor = gfx.Color(*self._plotFrame.foregroundColor)
            geom = gfx.Geometry(
                positions=numpy.column_stack(
                    [vertices, numpy.zeros(len(vertices), dtype=numpy.float32)]
                )
            )
            mat = gfx.LineSegmentMaterial(thickness=1.0, color=fgColor)
            frameLine = gfx.Line(geom, mat)
            self._screenFrameGroup.add(frameLine)

        # Render text labels (tick labels, axis titles, main title)
        for labelDict in labelDicts:
            text = labelDict.get("text", "")
            if not text:
                continue
            # Strip matplotlib LaTeX formatting
            text = _stripMathDefault(text)

            lx = labelDict["x"]
            ly = labelDict["y"]
            labelColor = labelDict.get("color", (0, 0, 0, 1))
            rotate = labelDict.get("rotate", 0)

            # Map alignment strings to pygfx anchor
            align = labelDict.get("align", "center")
            valign = labelDict.get("valign", "center")
            anchor = self._mapAnchor(align, valign)

            fontSize = 12.0
            font = labelDict.get("font")
            if font is not None:
                ps = font.pointSizeF()
                if ps > 0:
                    fontSize = ps
                else:
                    px = font.pixelSize()
                    if px > 0:
                        fontSize = px * 72.0 / self.getDotsPerInch()

            textObj = gfx.Text(
                text=text,
                material=gfx.TextMaterial(color=gfx.Color(*labelColor)),
                font_size=fontSize,
                anchor=anchor,
                screen_space=True,
            )
            textObj.local.position = (lx, ly, 0)

            if rotate:
                import pylinalg as la

                # Negate angle because screen camera flips Y
                textObj.local.rotation = la.quat_from_axis_angle(
                    (0, 0, 1), math.radians(-rotate)
                )

            self._screenFrameGroup.add(textObj)

    def _updateMarkers(self):
        """Update marker lines and text labels in screen space."""
        plot = self._plotRef()
        if plot is None:
            return

        pixelOffset = 3

        for plotItem in self.getItemsFromBackToFront(condition=lambda i: i.isVisible()):
            if plotItem._backendRenderer is None:
                continue
            item = plotItem._backendRenderer
            if not isinstance(item, _PygfxMarkerItem):
                continue

            xCoord = item["x"]
            yCoord = item["y"]
            yAxis = item.get("yaxis", "left")
            color = item["color"]
            linewidth = item["linewidth"]
            dashPattern = item["dashpattern"]

            # Remove old line and text from the screen scene
            if item._lineObj is not None:
                if item._lineObj.parent is not None:
                    item._lineObj.parent.remove(item._lineObj)
                item._lineObj = None
            if item._textObj is not None:
                if item._textObj.parent is not None:
                    item._textObj.parent.remove(item._textObj)
                item._textObj = None

            gfxColor = gfx.Color(*color)

            if xCoord is None or yCoord is None:
                # hline or vline marker — render in screen space
                if xCoord is None:
                    # Horizontal line at y
                    pixelPos = self._plotFrame.dataToPixel(
                        0.5 * sum(self._plotFrame.dataRanges[0]),
                        yCoord,
                        axis=yAxis,
                    )
                    if pixelPos is None:
                        continue
                    left = self._plotFrame.margins.left
                    right = self._plotFrame.size[0] - self._plotFrame.margins.right
                    positions = numpy.array(
                        [
                            [left, pixelPos[1], 0],
                            [right, pixelPos[1], 0],
                        ],
                        dtype=numpy.float32,
                    )

                    if item["text"] is not None:
                        tx = right - pixelOffset
                        ty = pixelPos[1] - pixelOffset
                        textObj = gfx.Text(
                            material=gfx.TextMaterial(color=gfxColor),
                            text=item["text"],
                            font_size=self._getDefaultFont().pointSizeF() or 10,
                            anchor="bottom-right",
                            screen_space=True,
                        )
                        textObj.local.position = (tx, ty, 0)
                        item._textObj = textObj
                        self._screenScene.add(textObj)
                else:
                    # Vertical line at x
                    yRange = self._plotFrame.dataRanges[1 if yAxis == "left" else 2]
                    pixelPos = self._plotFrame.dataToPixel(
                        xCoord,
                        0.5 * sum(yRange),
                        axis=yAxis,
                    )
                    if pixelPos is None:
                        continue
                    top = self._plotFrame.margins.top
                    bottom = self._plotFrame.size[1] - self._plotFrame.margins.bottom
                    positions = numpy.array(
                        [
                            [pixelPos[0], top, 0],
                            [pixelPos[0], bottom, 0],
                        ],
                        dtype=numpy.float32,
                    )

                    if item["text"] is not None:
                        tx = pixelPos[0] + pixelOffset
                        ty = top + pixelOffset
                        textObj = gfx.Text(
                            material=gfx.TextMaterial(color=gfxColor),
                            text=item["text"],
                            font_size=self._getDefaultFont().pointSizeF() or 10,
                            anchor="top-left",
                            screen_space=True,
                        )
                        textObj.local.position = (tx, ty, 0)
                        item._textObj = textObj
                        self._screenScene.add(textObj)

                geom = gfx.Geometry(positions=positions)
                mat = gfx.LineMaterial(
                    thickness=max(linewidth, 1.0),
                    color=gfxColor,
                    dash_pattern=dashPattern if dashPattern else (),
                )
                item._lineObj = gfx.Line(geom, mat)
                self._screenScene.add(item._lineObj)

            else:
                # Point marker — text label in screen space
                if item["text"] is not None:
                    pixelPos = self._plotFrame.dataToPixel(
                        xCoord,
                        yCoord,
                        axis=yAxis,
                    )
                    if pixelPos is None:
                        continue
                    tx = pixelPos[0] + pixelOffset
                    ty = pixelPos[1] + pixelOffset
                    textObj = gfx.Text(
                        material=gfx.TextMaterial(color=gfxColor),
                        text=item["text"],
                        font_size=self._getDefaultFont().pointSizeF() or 10,
                        anchor="top-left",
                        screen_space=True,
                    )
                    textObj.local.position = (tx, ty, 0)
                    item._textObj = textObj
                    self._screenScene.add(textObj)

    def _updateCrosshair(self):
        """Update crosshair cursor lines."""
        # Remove old crosshair
        if self._crosshairHLine is not None:
            if self._crosshairHLine in self._screenScene.children:
                self._screenScene.remove(self._crosshairHLine)
            self._crosshairHLine = None
        if self._crosshairVLine is not None:
            if self._crosshairVLine in self._screenScene.children:
                self._screenScene.remove(self._crosshairVLine)
            self._crosshairVLine = None

        if self._crosshairCursor is None or self._mousePosInPixels is None:
            return

        color, linewidth = self._crosshairCursor
        gfxColor = gfx.Color(*color)
        mx, my = self._mousePosInPixels

        w, h = self._plotFrame.size
        left, top = self._plotFrame.plotOrigin
        pw, ph = self._plotFrame.plotSize

        # Horizontal line
        hPositions = numpy.array(
            [
                [left, my, 0],
                [left + pw, my, 0],
            ],
            dtype=numpy.float32,
        )
        hGeom = gfx.Geometry(positions=hPositions)
        hMat = gfx.LineMaterial(thickness=linewidth, color=gfxColor)
        self._crosshairHLine = gfx.Line(hGeom, hMat)
        self._screenScene.add(self._crosshairHLine)

        # Vertical line
        vPositions = numpy.array(
            [
                [mx, top, 0],
                [mx, top + ph, 0],
            ],
            dtype=numpy.float32,
        )
        vGeom = gfx.Geometry(positions=vPositions)
        vMat = gfx.LineMaterial(thickness=linewidth, color=gfxColor)
        self._crosshairVLine = gfx.Line(vGeom, vMat)
        self._screenScene.add(self._crosshairVLine)

    @staticmethod
    def _mapAnchor(align, valign):
        """Map silx align/valign strings to pygfx anchor string."""
        vmap = {"top": "top", "bottom": "bottom", "center": "middle"}
        hmap = {"left": "left", "right": "right", "center": "center"}
        v = vmap.get(str(valign), "middle")
        h = hmap.get(str(align), "center")
        return f"{v}-{h}"

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

    # Backend API: Log transform helpers #####################################

    def _logTransformX(self, x):
        """Apply log10 if X axis is log scale."""
        if not self._plotFrame.xAxis.isLog:
            return x
        x = numpy.array(x, copy=True, dtype=numpy.float64)
        mask = x < FLOAT32_MINPOS
        x[mask] = numpy.nan
        with numpy.errstate(divide="ignore"):
            return numpy.log10(x).astype(numpy.float32)

    def _logTransformY(self, y, yaxis="left"):
        """Apply log10 if Y axis is log scale."""
        isLog = (
            self._plotFrame.yAxis.isLog
            if yaxis == "left"
            else self._plotFrame.y2Axis.isLog
        )
        if not isLog:
            return y
        y = numpy.array(y, copy=True, dtype=numpy.float64)
        mask = y < FLOAT32_MINPOS
        y[mask] = numpy.nan
        with numpy.errstate(divide="ignore"):
            return numpy.log10(y).astype(numpy.float32)

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

        # Log transform errors before coordinates
        if self._plotFrame.xAxis.isLog and xerror is not None:
            xerror = numpy.asarray(xerror, dtype=numpy.float32)
            logX = numpy.log10(x)
            if xerror.ndim == 2:
                xErrMinus, xErrPlus = xerror[0], xerror[1]
            else:
                xErrMinus, xErrPlus = xerror, xerror
            with numpy.errstate(divide="ignore", invalid="ignore"):
                xErrMinus = logX - numpy.log10(x - xErrMinus)
            xErrPlus = numpy.log10(x + xErrPlus) - logX
            xerror = numpy.array((xErrMinus, xErrPlus), dtype=numpy.float32)

        isYLog = (yaxis == "left" and self._plotFrame.yAxis.isLog) or (
            yaxis == "right" and self._plotFrame.y2Axis.isLog
        )
        if isYLog and yerror is not None:
            yerror = numpy.asarray(yerror, dtype=numpy.float32)
            logY = numpy.log10(y)
            if yerror.ndim == 2:
                yErrMinus, yErrPlus = yerror[0], yerror[1]
            else:
                yErrMinus, yErrPlus = yerror, yerror
            with numpy.errstate(divide="ignore", invalid="ignore"):
                yErrMinus = logY - numpy.log10(y - yErrMinus)
            yErrPlus = numpy.log10(y + yErrPlus) - logY
            yerror = numpy.array((yErrMinus, yErrPlus), dtype=numpy.float32)

        x = self._logTransformX(x)
        y = self._logTransformY(y, yaxis)

        if baseline is not None and isYLog:
            if isinstance(baseline, numpy.ndarray):
                baseline = self._logTransformY(baseline, yaxis)
            else:
                bl = float(baseline)
                if bl > 0:
                    baseline = math.log10(bl)
                else:
                    baseline = numpy.nan

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
        self._dataGroup.add(item.group)
        return item

    def addImage(self, data, origin, scale, colormap, alpha):
        data = numpy.asarray(data)
        ox, oy = origin
        sx, sy = scale
        h, w = data.shape[:2]

        if self._plotFrame.xAxis.isLog:
            xMin = ox
            xMax = ox + w * sx
            if xMin > 0 and xMax > 0:
                logXMin = math.log10(xMin)
                logXMax = math.log10(xMax)
                ox = logXMin
                sx = (logXMax - logXMin) / w

        if self._plotFrame.yAxis.isLog:
            yMin = oy
            yMax = oy + h * sy
            if yMin > 0 and yMax > 0:
                logYMin = math.log10(yMin)
                logYMax = math.log10(yMax)
                oy = logYMin
                sy = (logYMax - logYMin) / h

        # Reuse pooled item if shape matches (avoids GPU object recreation)
        reuse = self._reusableImageItem
        if reuse is not None and data.ndim == 2 and reuse._scalarShape == data.shape:
            self._reusableImageItem = None
            reuse._build(data, (ox, oy), (sx, sy), colormap, alpha)
            self._dataGroup.add(reuse.group)
            return reuse

        self._reusableImageItem = None
        item = _PygfxImageItem(data, (ox, oy), (sx, sy), colormap, alpha)
        self._dataGroup.add(item.group)
        return item

    def addTriangles(self, x, y, triangles, color, alpha):
        x = self._logTransformX(numpy.asarray(x, dtype=numpy.float64))
        y = self._logTransformY(numpy.asarray(y, dtype=numpy.float64))
        item = _PygfxTrianglesItem(x, y, triangles, color, alpha)
        self._dataGroup.add(item.group)
        return item

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
        x = self._logTransformX(numpy.asarray(x, dtype=numpy.float64))
        y = self._logTransformY(numpy.asarray(y, dtype=numpy.float64))
        # Ensure overlay outlines (e.g. zoom selection) are clearly visible
        if overlay and linewidth < 2.0:
            linewidth = 2.0
        item = _PygfxShapeItem(
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
        if overlay:
            self._overlayGroup.add(item.group)
        else:
            self._dataGroup.add(item.group)
        return item

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
        # Log transform marker coordinates
        if x is not None and self._plotFrame.xAxis.isLog:
            x = math.log10(x) if x > 0 else numpy.nan
        if y is not None:
            isYLog = (
                self._plotFrame.yAxis.isLog
                if yaxis == "left"
                else self._plotFrame.y2Axis.isLog
            )
            if isYLog:
                y = math.log10(y) if y > 0 else numpy.nan

        item = _PygfxMarkerItem(
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

        self._overlayGroup.add(item.group)
        return item

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

            group = item.group
            if group.parent is not None:
                group.parent.remove(group)

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

        from silx.gui._glutils.utils import segmentTrianglesIntersection

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
        snapshot = self._renderer.snapshot()

        # snapshot is (H, W, 4) RGBA uint8
        from PIL import Image as PILImage

        img = PILImage.fromarray(snapshot)
        if fileFormat in ("tif", "tiff"):
            img.save(fileName, format="TIFF")
        elif fileFormat == "ppm":
            img.convert("RGB").save(fileName, format="PPM")
        elif fileFormat == "svg":
            raise NotImplementedError("SVG export not supported by pygfx backend")
        else:
            img.save(fileName, format=fileFormat.upper())

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
        plotWidth, plotHeight = self._plotFrame.plotSize
        if plotWidth <= 2 or plotHeight <= 2:
            return

        if keepDim is None:
            ranges = self._plot.getDataRange()
            if (
                ranges.y is not None
                and ranges.x is not None
                and (ranges.y[1] - ranges.y[0]) != 0.0
            ):
                dataRatio = (ranges.x[1] - ranges.x[0]) / float(
                    ranges.y[1] - ranges.y[0]
                )
                plotRatio = plotWidth / float(plotHeight)
                keepDim = "x" if dataRatio > plotRatio else "y"
            else:
                keepDim = "x"

        (xMin, xMax), (yMin, yMax), (y2Min, y2Max) = self._plotFrame.dataRanges
        if keepDim == "y":
            dataW = (yMax - yMin) * plotWidth / float(plotHeight)
            xCenter = 0.5 * (xMin + xMax)
            xMin = xCenter - 0.5 * dataW
            xMax = xCenter + 0.5 * dataW
        elif keepDim == "x":
            dataH = (xMax - xMin) * plotHeight / float(plotWidth)
            yCenter = 0.5 * (yMin + yMax)
            yMin = yCenter - 0.5 * dataH
            yMax = yCenter + 0.5 * dataH
            y2Center = 0.5 * (y2Min + y2Max)
            y2Min = y2Center - 0.5 * dataH
            y2Max = y2Center + 0.5 * dataH
        else:
            raise RuntimeError("Unsupported dimension to keep: %s" % keepDim)

        self._setDataRanges(xlim=(xMin, xMax), ylim=(yMin, yMax), y2lim=(y2Min, y2Max))

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

    def setXAxisLogarithmic(self, flag):
        if flag != self._plotFrame.xAxis.isLog:
            if flag and self._keepDataAspectRatio:
                _logger.warning("KeepDataAspectRatio is ignored with log axes")
            self._plotFrame.xAxis.isLog = flag

    def setYAxisLogarithmic(self, flag):
        if flag != self._plotFrame.yAxis.isLog or flag != self._plotFrame.y2Axis.isLog:
            if flag and self._keepDataAspectRatio:
                _logger.warning("KeepDataAspectRatio is ignored with log axes")
            self._plotFrame.yAxis.isLog = flag
            self._plotFrame.y2Axis.isLog = flag

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

    def isKeepDataAspectRatio(self):
        if self._plotFrame.xAxis.isLog or self._plotFrame.yAxis.isLog:
            return False
        return self._keepDataAspectRatio

    def setKeepDataAspectRatio(self, flag):
        if flag and (self._plotFrame.xAxis.isLog or self._plotFrame.yAxis.isLog):
            _logger.warning("KeepDataAspectRatio is ignored with log axes")
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

        # Remove old background
        if hasattr(self, "_bgObj") and self._bgObj is not None:
            if self._bgObj in self._scene.children:
                self._scene.remove(self._bgObj)

        # Update data scene background (plot area uses dataBackgroundColor)
        if dataBackgroundColor is not None:
            bgColor = gfx.Color(*dataBackgroundColor)
            self._bgObj = gfx.Background(None, gfx.BackgroundMaterial(bgColor))
            self._scene.add(self._bgObj)
        else:
            self._bgObj = None
