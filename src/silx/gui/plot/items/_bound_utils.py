from .types import ItemBounds, AxesInfo, AxisInfo


def bounds_outside_fixed_limits(bounds: ItemBounds, axesInfo: AxesInfo) -> bool:
    """
    Exlude when at least one axis has fixed limits and
    bound [min,max] falls outside these limits.
    """
    xmin, xmax, ymin, ymax = bounds
    return _outside_fixed_limits(xmin, xmax, axesInfo.x) or _outside_fixed_limits(
        ymin, ymax, axesInfo.y
    )


def _outside_fixed_limits(vmin: float, vmax: float, axis: AxisInfo) -> bool:
    """
    Exlude when the axis limits has fixed limits and
    bound [vmin,vmax] falls outside these limits.
    """
    if axis.auto:
        # Axis limits are not fixed
        return False
    lmin, lmax = axis.limits()
    return vmin > lmax or vmax < lmin
