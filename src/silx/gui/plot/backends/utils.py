import logging
from typing import Literal


_logger = logging.getLogger(__name__)


Range = tuple[float, float]


def restrictWgpuToPrimaryBackends() -> None:
    """Exclude wgpu's GL/EGL backend from GPU adapter enumeration.

    When selecting a GPU adapter, wgpu enumerates every backend. Its GL/EGL
    path aborts the whole process with an uncatchable Rust panic
    (``EGL_BAD_ACCESS``) when a Qt OpenGL context is already current on the
    thread, e.g. after ``PlotWidget.setBackend("gl")`` then
    ``setBackend("pygfx")``. Restrict the wgpu instance to the "Primary"
    backends (Vulkan/Metal/DX12) that the pygfx backend targets so the fragile
    GL path is never probed.

    This must run before the wgpu instance is created (before the first
    ``request_adapter``/``enumerate_adapters`` call); it is a no-op afterwards.
    Call it from every code path that may be first to create the instance.
    """
    try:
        from wgpu.backends.wgpu_native.extras import set_instance_extras
    except ImportError:
        return  # Older wgpu without instance-backend selection
    try:
        set_instance_extras(backends=["Primary"])
    except Exception:
        # Instance already created, or unsupported flag: nothing we can do.
        _logger.debug("Could not restrict wgpu instance backends", exc_info=True)


def findDimToKeep(
    width: float, height: float, xRange: Range | None, yRange: Range | None
) -> Literal["x"] | Literal["y"]:
    if xRange is None or yRange is None or (yRange[1] - yRange[0]) == 0 or height == 0:
        return "x"
    dataRatio = (xRange[1] - xRange[0]) / float(yRange[1] - yRange[0])
    plotRatio = width / float(height)

    return "x" if dataRatio > plotRatio else "y"


def ensureAspectRatio(
    plotWidth: float,
    plotHeight: float,
    xRange: Range,
    yRange: Range,
    y2Range: Range,
    keepDim: Literal["x", "y"],
) -> tuple[Range, Range, Range]:
    """Update plot bounds in order to keep aspect ratio.

    Warning: keepDim on right Y axis is not implemented !

    :param keepDim: The dimension to maintain: 'x' or 'y'
    """
    if plotWidth <= 2 or plotHeight <= 2:
        return xRange, yRange, y2Range

    (xMin, xMax), (yMin, yMax), (y2Min, y2Max) = xRange, yRange, y2Range
    if keepDim == "y":
        dataW = (yMax - yMin) * plotWidth / float(plotHeight)
        xCenter = 0.5 * (xMin + xMax)
        xMin = xCenter - 0.5 * dataW
        xMax = xCenter + 0.5 * dataW
        return (xMin, xMax), yRange, y2Range

    if keepDim == "x":
        dataH = (xMax - xMin) * plotHeight / float(plotWidth)
        yCenter = 0.5 * (yMin + yMax)
        yMin = yCenter - 0.5 * dataH
        yMax = yCenter + 0.5 * dataH
        y2Center = 0.5 * (y2Min + y2Max)
        y2Min = y2Center - 0.5 * dataH
        y2Max = y2Center + 0.5 * dataH
        return xRange, (yMin, yMax), (y2Min, y2Max)
    raise RuntimeError("Unsupported dimension to keep: %s" % keepDim)
