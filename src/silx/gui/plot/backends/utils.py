from typing import Literal


Range = tuple[float, float]


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

    :param str keepDim: The dimension to maintain: 'x', 'y' or None.
        If None (the default), the dimension with the largest range.
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
