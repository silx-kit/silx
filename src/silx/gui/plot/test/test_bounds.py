import numpy
import pytest

from silx.gui.plot import PlotWindow
from silx.gui.plot.items import BoundingRect, XAxisExtent, YAxisExtent
from silx.utils.enum import Enum


@pytest.mark.parametrize("free_axis", ["yleft", "yright", "x"])
def test_curve_bounds_vs_reset_zoom(qapp, qWidgetFactory, free_axis):
    """
    Test that resetting the zoom takes a fixed axis range into account.
    """
    plot = qWidgetFactory(PlotWindow)
    items = []

    # Zoomed field-of-view (FOV)
    fixed_min = 40
    fixed_max = 60

    # Curves with spike inside and outside the FOV
    x_in = x_out = numpy.linspace(0, 100, 5001)
    y_in = numpy.sin(x_in * 0.2) + 2.0 * numpy.exp(-0.5 * ((x_in - 50) / 0.3) ** 2)
    y_out = numpy.sin(x_out * 0.2) + 20.0 * numpy.exp(-0.5 * ((x_out - 85) / 0.2) ** 2)

    if free_axis == "yleft":
        fixed_axis = plot.getXAxis()
        fixed_args = dict(fixed_xmin=fixed_min, fixed_xmax=fixed_max)

        yaxisarg = "left"
    elif free_axis == "yright":
        fixed_axis = plot.getXAxis()
        fixed_args = dict(fixed_xmin=fixed_min, fixed_xmax=fixed_max)

        yaxisarg = "right"
    elif free_axis == "x":
        fixed_axis = plot.getYAxis()
        fixed_args = dict(fixed_ymin=fixed_min, fixed_ymax=fixed_max)

        yaxisarg = "left"
        x_in, y_in = y_in, x_in
        x_out, y_out = y_out, x_out
    else:
        raise ValueError(free_axis)

    # Curves
    item = plot.addCurve(x_in, y_in, legend="IN", yaxis=yaxisarg)
    items.append((x_in, y_in, item))

    item = plot.addCurve(x_out, y_out, legend="OUT", yaxis=yaxisarg)
    items.append((x_out, y_out, item))

    # Validate full bounds after resetting zoom
    _reset_zoom(qapp, plot)

    _assert_full_bounds(items)

    # Validate autoscale bounds after resetting zoom with fixed limits
    _fix_axis_limits(qapp, fixed_axis, fixed_min, fixed_max)
    _reset_zoom(qapp, plot)

    _assert_reset_bounds(plot, items, item_type=_TestItemType.POINTS, **fixed_args)
    _assert_limits(
        plot, items, item_type=_TestItemType.POINTS, yaxisarg=yaxisarg, **fixed_args
    )


@pytest.mark.parametrize("free_axis", ["y", "x"])
def test_scatter_bounds_vs_reset_zoom(qapp, qWidgetFactory, free_axis):
    """
    Test that resetting the zoom takes a fixed axis range into account.
    """
    plot = qWidgetFactory(PlotWindow)
    xaxis = plot.getXAxis()
    yaxis = plot.getYAxis()
    items = []

    # Zoomed field-of-view (FOV)
    fixed_min = 40
    fixed_max = 60

    # Scatters inside and outside the FOV
    rng = numpy.random.default_rng(seed=42)

    x_in = 50 + 2 * rng.random(400)
    y_in = 50 + 3 * rng.random(400)

    x_out = 200 + 5 * rng.random(400)
    y_out = 200 + 2 * rng.random(400)

    if free_axis == "y":
        fixed_axis = xaxis
        fixed_args = dict(fixed_xmin=fixed_min, fixed_xmax=fixed_max)
    elif free_axis == "x":
        fixed_axis = yaxis
        fixed_args = dict(fixed_ymin=fixed_min, fixed_ymax=fixed_max)

        x_in, y_in = y_in, x_in
        x_out, y_out = y_out, x_out
    else:
        raise ValueError(free_axis)

    # Scatters
    item = plot.addScatter(x_in, y_in, numpy.ones_like(x_in), legend="IN")
    items.append((x_in, y_in, item))

    item = plot.addScatter(x_out, y_out, numpy.ones_like(x_out), legend="OUT")
    items.append((x_out, y_out, item))

    # Validate full bounds after resetting zoom
    _reset_zoom(qapp, plot)

    _assert_full_bounds(items)

    # Validate autoscale bounds after resetting zoom with fixed X limits
    _fix_axis_limits(qapp, fixed_axis, fixed_min, fixed_max)
    _reset_zoom(qapp, plot)

    _assert_reset_bounds(plot, items, item_type=_TestItemType.POINTS, **fixed_args)
    _assert_limits(plot, items, item_type=_TestItemType.POINTS, **fixed_args)


@pytest.mark.parametrize("free_axis", ["y", "x"])
def test_histogram_bounds_vs_reset_zoom(qapp, qWidgetFactory, free_axis):
    """
    Test that resetting the zoom takes a fixed axis range into account.
    """
    plot = qWidgetFactory(PlotWindow)
    xaxis = plot.getXAxis()
    yaxis = plot.getYAxis()
    items = []

    # Zoomed field-of-view (FOV)
    if free_axis == "y":
        fixed_min = 40
        fixed_max = 60
    elif free_axis == "x":
        fixed_min = 5
        fixed_max = 20
    else:
        raise ValueError(free_axis)

    # Histogram inside FOV
    x_in = numpy.linspace(0, 100, 201)
    y_in = 10 * numpy.exp(-0.5 * ((x_in - 50) / 2.0) ** 2)

    # Histogram outside FOV
    x_out = numpy.linspace(0, 100, 201)
    y_out = numpy.exp(-0.5 * ((x_out - 85) / 1.0) ** 2)

    if free_axis == "y":
        fixed_axis = xaxis
        fixed_args = dict(fixed_xmin=fixed_min, fixed_xmax=fixed_max)
    elif free_axis == "x":
        fixed_axis = yaxis
        fixed_args = dict(fixed_ymin=fixed_min, fixed_ymax=fixed_max)
    else:
        raise ValueError(free_axis)

    item = plot.addHistogram(y_in, x_in, legend="IN")
    items.append((x_in, y_in, item))

    item = plot.addHistogram(y_out, x_out, legend="OUT")
    items.append((x_out, y_out, item))

    # Note: bounds and reset bounds and both extended with 0.25
    # with respect to the expected bounds.
    rtol = 1e-3
    atol = 0.5

    # Validate full bounds after resetting zoom
    _reset_zoom(qapp, plot)

    _assert_full_bounds(items, rtol=rtol, atol=atol)

    # Validate autoscale bounds after resetting zoom with fixed X limits
    _fix_axis_limits(qapp, fixed_axis, fixed_min, fixed_max)
    _reset_zoom(qapp, plot)

    _assert_reset_bounds(
        plot,
        items,
        item_type=_TestItemType.HISTOGRAM,
        **fixed_args,
        rtol=rtol,
        atol=atol,
    )
    _assert_limits(
        plot,
        items,
        item_type=_TestItemType.HISTOGRAM,
        **fixed_args,
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize("free_axis", ["y", "x"])
def test_image_bounds_vs_reset_zoom(qapp, qWidgetFactory, free_axis):
    """
    Test that resetting the zoom takes a fixed axis range into account.
    """
    plot = qWidgetFactory(PlotWindow)
    xaxis = plot.getXAxis()
    yaxis = plot.getYAxis()
    items = []

    # Zoomed field-of-view (FOV)
    height, width = 100, 100
    fixed_min = -10
    fixed_max = 150

    # Data
    image = numpy.zeros((height, width), dtype=float)

    x = numpy.array([0, width])
    y = numpy.array([0, height])
    item = plot.addImage(image, origin=(x[0], y[0]), scale=(1, 1), legend="IN")
    items.append((x, y, item))

    x = numpy.array([width, 2 * width])
    y = numpy.array([height, 2 * height])
    item = plot.addImage(image, origin=(x[0], y[0]), scale=(1, 1), legend="HALF")
    items.append((x, y, item))

    x = numpy.array([2 * width, 3 * width])
    y = numpy.array([2 * height, 3 * height])
    item = plot.addImage(image, origin=(x[0], y[0]), scale=(1, 1), legend="OUT")
    items.append((x, y, item))

    if free_axis == "y":
        fixed_axis = xaxis
        fixed_args = dict(fixed_xmin=fixed_min, fixed_xmax=fixed_max)
    elif free_axis == "x":
        fixed_axis = yaxis
        fixed_args = dict(fixed_ymin=fixed_min, fixed_ymax=fixed_max)
    else:
        raise ValueError(free_axis)

    # Validate full bounds after resetting zoom
    _reset_zoom(qapp, plot)

    _assert_full_bounds(items)

    # Validate autoscale bounds after resetting zoom with fixed X limits
    _fix_axis_limits(qapp, fixed_axis, fixed_min, fixed_max)
    _reset_zoom(qapp, plot)

    _assert_reset_bounds(plot, items, item_type=_TestItemType.IMAGE, **fixed_args)
    _assert_limits(plot, items, item_type=_TestItemType.IMAGE, **fixed_args)


@pytest.mark.parametrize("free_axis", ["yleft", "yright", "x"])
def test_bounding_rect_bounds_vs_reset_zoom(qapp, qWidgetFactory, free_axis):
    """
    Test that resetting the zoom takes a fixed axis range into account.
    """
    plot = qWidgetFactory(PlotWindow)
    items = []

    # Zoomed field-of-view (FOV)
    fixed_min = 40
    fixed_max = 60

    # Rectangles inside and outside the FOV
    if free_axis == "yleft":
        fixed_axis = plot.getXAxis()
        fixed_args = dict(fixed_xmin=fixed_min, fixed_xmax=fixed_max)
        yaxisarg = "left"

        item_params = {
            "IN": (45, 55, 10, 20),
            "HALF": (50, 70, 40, 80),
            "OUT": (80, 90, 100, 200),
        }

    elif free_axis == "yright":
        fixed_axis = plot.getXAxis()
        fixed_args = dict(fixed_xmin=fixed_min, fixed_xmax=fixed_max)
        yaxisarg = "right"

        item_params = {
            "IN": (45, 55, 10, 20),
            "HALF": (50, 70, 40, 80),
            "OUT": (80, 90, 100, 200),
        }

    elif free_axis == "x":
        fixed_axis = plot.getYAxis()
        fixed_args = dict(fixed_ymin=fixed_min, fixed_ymax=fixed_max)
        yaxisarg = "left"

        item_params = {
            "IN": (10, 20, 45, 55),
            "HALF": (40, 80, 50, 70),
            "OUT": (100, 200, 80, 90),
        }

    else:
        raise ValueError(free_axis)

    for name, bounds in item_params.items():
        item = BoundingRect()
        item.setName(name)
        if yaxisarg:
            item.setYAxis(yaxisarg)
        item.setBounds(bounds)
        plot.addItem(item)

        x = numpy.array([bounds[0], bounds[1]])
        y = numpy.array([bounds[2], bounds[3]])
        items.append((x, y, item))

    # Validate full bounds after resetting zoom
    _reset_zoom(qapp, plot)

    _assert_full_bounds(items)

    # Validate autoscale bounds after resetting zoom with fixed X limits
    _fix_axis_limits(qapp, fixed_axis, fixed_min, fixed_max)
    _reset_zoom(qapp, plot)

    _assert_reset_bounds(plot, items, item_type=_TestItemType.BOUNDRECT, **fixed_args)
    _assert_limits(
        plot, items, item_type=_TestItemType.BOUNDRECT, yaxisarg=yaxisarg, **fixed_args
    )


@pytest.mark.parametrize("axis", ["x", "y"])
def test_extent_bounds_vs_reset_zoom(qapp, qWidgetFactory, axis):
    plot = qWidgetFactory(PlotWindow)
    xaxis = plot.getXAxis()
    yaxis = plot.getYAxis()
    items = []

    # Zoomed field-of-view (FOV)
    fixed_min = 40
    fixed_max = 60

    # Extents inside and outside the FOV
    if axis == "x":
        item_type = _TestItemType.XEXTENT

        fixed_axis = xaxis
        fixed_args = dict(fixed_xmin=fixed_min, fixed_xmax=fixed_max)

        item = XAxisExtent()
        item.setName("IN")
        item.setRange(45, 55)
        items.append((numpy.array([45, 55]), numpy.array([numpy.nan, numpy.nan]), item))
        plot.addItem(item)

        item = XAxisExtent()
        item.setName("HALF")
        item.setRange(50, 70)
        items.append((numpy.array([50, 70]), numpy.array([numpy.nan, numpy.nan]), item))
        plot.addItem(item)

        item = XAxisExtent()
        item.setName("OUT")
        item.setRange(80, 90)
        items.append((numpy.array([80, 90]), numpy.array([numpy.nan, numpy.nan]), item))
        plot.addItem(item)

    else:
        item_type = _TestItemType.YEXTENT

        fixed_axis = yaxis
        fixed_args = dict(fixed_ymin=fixed_min, fixed_ymax=fixed_max)

        item = YAxisExtent()
        item.setName("IN")
        item.setRange(45, 55)
        items.append((numpy.array([numpy.nan, numpy.nan]), numpy.array([45, 55]), item))
        plot.addItem(item)

        item = YAxisExtent()
        item.setName("HALF")
        item.setRange(50, 70)
        items.append((numpy.array([numpy.nan, numpy.nan]), numpy.array([50, 70]), item))
        plot.addItem(item)

        item = YAxisExtent()
        item.setName("OUT")
        item.setRange(80, 90)
        items.append((numpy.array([numpy.nan, numpy.nan]), numpy.array([80, 90]), item))
        plot.addItem(item)

    # Validate full bounds after resetting zoom
    _reset_zoom(qapp, plot)

    _assert_full_bounds(items)

    # Validate autoscale bounds after resetting zoom with fixed X limits
    _fix_axis_limits(qapp, fixed_axis, fixed_min, fixed_max)
    _reset_zoom(qapp, plot)

    _assert_reset_bounds(plot, items, item_type=item_type, **fixed_args)
    _assert_limits(plot, items, item_type=item_type, **fixed_args)


def _fix_axis_limits(qapp, axis, fixed_min, fixed_max):
    """Fix the axis range to [fixed_min, fixed_max]."""
    axis.setLimits(fixed_min, fixed_max)
    axis.setAutoScale(False)
    qapp.processEvents()


def _reset_zoom(qapp, plot):
    """Reset the axes limits that are not fixed (autoscale=True)."""
    plot.resetZoom()
    qapp.processEvents()


def _assert_full_bounds(items, rtol=1e-5, atol=1e-8):
    """Validate Item.getBounds()."""
    for x, y, item in items:
        expected = _expected_full_bounds(x, y)
        actual = item.getBounds()
        assert actual is not None, item.getName()
        numpy.testing.assert_allclose(
            actual, expected, rtol=rtol, atol=atol, err_msg=item.getName()
        )


class _TestItemType(Enum):
    POINTS = "points"
    HISTOGRAM = "histogram"
    IMAGE = "image"
    BOUNDRECT = "boundrect"
    XEXTENT = "xextent"
    YEXTENT = "yextent"


def _assert_reset_bounds(
    plot,
    items,
    item_type: _TestItemType,
    fixed_xmin=None,
    fixed_xmax=None,
    fixed_ymin=None,
    fixed_ymax=None,
    rtol=1e-5,
    atol=1e-8,
):
    """Validate Item.getResetBounds()."""
    for x, y, item in items:
        expected = _expected_reset_bounds(
            x,
            y,
            item_type,
            fixed_xmin=fixed_xmin,
            fixed_xmax=fixed_xmax,
            fixed_ymin=fixed_ymin,
            fixed_ymax=fixed_ymax,
        )
        actual = plot._itemResetBounds(item)

        print()
        print(f"Item: {item.getName()}")
        print(f"Fixed X limits = [{fixed_xmin}, {fixed_xmax}]")
        print(f"Fixed Y limits = [{fixed_ymin}, {fixed_ymax}]")
        print("X data =", x)
        print("Y data =", y)
        print("Expected reset bounds =", expected)
        print("Actual reset bounds=", actual)

        if expected is None:
            assert actual is None, item.getName()
        else:
            assert actual is not None, item.getName()
            numpy.testing.assert_allclose(
                actual, expected, rtol=rtol, atol=atol, err_msg=item.getName()
            )


def _assert_limits(
    plot,
    items,
    item_type: _TestItemType,
    fixed_xmin=None,
    fixed_xmax=None,
    fixed_ymin=None,
    fixed_ymax=None,
    rtol=1e-5,
    atol=1e-8,
    yaxisarg="left",
):
    """Validate Axis.getLimits() after resetting zoom."""
    xaxis = plot.getXAxis()
    yaxis = plot.getYAxis(yaxisarg)

    bounds = []
    for x, y, _ in items:
        ibounds = _expected_reset_bounds(
            x,
            y,
            item_type,
            fixed_xmin=fixed_xmin,
            fixed_xmax=fixed_xmax,
            fixed_ymin=fixed_ymin,
            fixed_ymax=fixed_ymax,
        )
        if ibounds is None:
            continue
        bounds.append(ibounds)

    bxmin, bxmax, bymin, bymax = numpy.array(bounds).T
    actual = numpy.min(bxmin), numpy.max(bxmax), numpy.min(bymin), numpy.max(bymax)

    lxmin, lxmax = xaxis.getLimits()
    lymin, lymax = yaxis.getLimits()
    if item_type == _TestItemType.XEXTENT:
        expected = lxmin, lxmax, numpy.nan, numpy.nan
    elif item_type == _TestItemType.YEXTENT:
        expected = numpy.nan, numpy.nan, lymin, lymax
    else:
        expected = lxmin, lxmax, lymin, lymax

    numpy.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


def _expected_reset_bounds(
    x,
    y,
    item_type: _TestItemType,
    fixed_xmin=None,
    fixed_xmax=None,
    fixed_ymin=None,
    fixed_ymax=None,
):
    """
    Expected reset bounds from data, taking into account the fixed limits
    of axes with autoscale=False.
    """

    # Fixed axis limit (autoscale=False)
    xmin = -numpy.inf if fixed_xmin is None else fixed_xmin
    xmax = numpy.inf if fixed_xmax is None else fixed_xmax
    ymin = -numpy.inf if fixed_ymin is None else fixed_ymin
    ymax = numpy.inf if fixed_ymax is None else fixed_ymax

    # Coordinates within fixed limits
    xmask = (x >= xmin) & (x <= xmax)
    ymask = (y >= ymin) & (y <= ymax)

    # No coordinates within the fixed limits?
    if item_type == _TestItemType.XEXTENT:
        is_outside = not numpy.any(xmask)
    elif item_type == _TestItemType.YEXTENT:
        is_outside = not numpy.any(ymask)
    else:
        mask = xmask & ymask
        is_outside = not numpy.any(mask)
    if is_outside:
        return None

    # Keep only coordinates within fixed limits
    if item_type == _TestItemType.XEXTENT:
        xmasked = x[xmask]
        ymasked = y
    elif item_type == _TestItemType.YEXTENT:
        xmasked = x
        ymasked = y[ymask]
    elif item_type in (_TestItemType.IMAGE, _TestItemType.BOUNDRECT):
        # X: independent variable
        # Y: independent variable
        #
        # Fixed limits on X do not affect Y and vice versa
        xmasked = x[xmask]
        ymasked = y[ymask]
    else:
        # X: independent variable
        # Y: dependent variable
        #
        # Fixed limits on X affect Y and vice versa
        xmasked = x[mask]
        ymasked = y[mask]

    # Autoscale or keep fixed limits
    xmin = _nanmin(xmasked) if fixed_xmin is None else fixed_xmin
    xmax = _nanmax(xmasked) if fixed_xmax is None else fixed_xmax
    ymin = _nanmin(ymasked) if fixed_ymin is None else fixed_ymin
    ymax = _nanmax(ymasked) if fixed_ymax is None else fixed_ymax

    return xmin, xmax, ymin, ymax


def _expected_full_bounds(x, y):
    """Extract full bounds from data."""
    return _nanmin(x), _nanmax(x), _nanmin(y), _nanmax(y)


def _nanmin(arr):
    """Return nanmin, but nan if the array contains only nans."""
    if numpy.all(numpy.isnan(arr)):
        return numpy.nan
    return numpy.nanmin(arr)


def _nanmax(arr):
    """Return nanmax, but nan if the array contains only nans."""
    if numpy.all(numpy.isnan(arr)):
        return numpy.nan
    return numpy.nanmax(arr)
