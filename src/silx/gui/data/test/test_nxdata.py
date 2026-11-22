import numpy
import pytest

from silx.io.nxdata.parse import _get_default_slice_selection
from .._NxComplexImagePlot import NxComplexImagePlot
from .._NxCurvePlot import NxCurvePlot
from .._NxVolumePlot import NxVolumePlot


from .. import NXdataWidgets


def test_image_view_axis_permutation(qapp):
    """Test that the image view applies the selected axes permutation."""

    # Signal dimensions are ordered as (z, y, x):
    # dim0 -> z (3), dim1 -> y (4), dim2 -> x (5).
    data = numpy.arange(3 * 4 * 5).reshape(3, 4, 5)

    widget = NXdataWidgets.ArrayImagePlot()

    widget.setImageData(
        signals=[data],
        signals_names=["signal"],
        axes_names=["z", "y", "x"],
    )

    # Select index 1 along dim0 (z), keeping dim1 (y) and dim2 (x).
    # The resulting image before permutation is data[1, :, :], shape (4, 5).
    selection = (1, slice(None), slice(None))
    widget._axesSelector.setSelection(selection=selection, permutation=(1, 0))

    # Swap the displayed dimensions (y, x) -> (x, y).
    # Expected image: data[1, :, :].T, with shape (5, 4).
    expected_image = data[selection].T

    actual_image = widget._getImageToDisplay()
    numpy.testing.assert_array_equal(actual_image, expected_image)


@pytest.mark.parametrize(
    "default_slice, axes, shape, expected_n_dims, expected",
    [
        pytest.param(
            None,
            None,
            (10, 20, 30),
            2,
            None,
            id="no-default-slice",
        ),
        pytest.param(
            [".", ".", "0"],
            None,
            (10, 20, 30),
            2,
            (slice(None), slice(None), 0),
            id="index",
        ),
        pytest.param(
            [".", ".", "35"],
            None,
            (10, 20, 30),
            2,
            None,
            id="out-of-bound-index",
        ),
        pytest.param(
            [".", ".", "-1"],
            None,
            (10, 20, 30),
            2,
            None,
            id="negative-index",
        ),
        pytest.param(
            [".", "difference", "."],
            [
                numpy.arange(3),
                numpy.array(["threshold_1", "threshold_2", "difference"]),
                None,
            ],
            (3, 3, 100),
            2,
            (slice(None), 2, slice(None)),
            id="nxdata-channel-value",
        ),
        pytest.param(
            [".", "2", "."],
            [
                numpy.arange(3),
                numpy.array(["threshold_1", "threshold_2", "difference"]),
                None,
            ],
            (3, 3, 100),
            2,
            (slice(None), 2, slice(None)),
            id="nxdata-channel-index",
        ),
        pytest.param(
            [".", ".", "invalid"],
            None,
            (10, 20, 30),
            2,
            None,
            id="invalid-value",
        ),
        pytest.param(
            [".", "."],
            None,
            (10, 20, 30),
            2,
            None,
            id="invalid-length",
        ),
        pytest.param(
            ["0", "1", "2"],
            None,
            (10, 20, 30),
            2,
            None,
            id="not-two-slices",
        ),
        pytest.param(
            ["0", ".", "."],
            [numpy.ones((2, 2)), None, None],
            (10, 20, 30),
            2,
            None,
            id="axis-ndim-2",
        ),
    ],
)
def test_get_default_slice_selection(
    default_slice, axes, shape, expected_n_dims, expected
):
    result = _get_default_slice_selection(default_slice, axes, shape, expected_n_dims)
    assert result == expected


def test_image_view_default_slice(qapp):
    """Test that a default slice selection is applied to the image view."""

    data = numpy.arange(3 * 4 * 5).reshape(3, 4, 5)

    widget = NXdataWidgets.ArrayImagePlot()

    # dim0 -> z (3), dim1 -> y (4), dim2 -> x (5).
    # selects index 1 along dim2 (x).
    selection = slice(None), slice(None), 1

    widget.setImageData(
        signals=[data],
        signals_names=["signal"],
        axes_names=["z", "y", "x"],
        selection=selection,
    )

    actual_slice = widget._axesSelector.selection()
    assert actual_slice == selection

    actual_image = widget._getImageToDisplay()
    numpy.testing.assert_array_equal(actual_image, data[selection])


def test_complex_image_view_default_slice(qapp):
    data = (numpy.arange(3 * 4 * 5) * (1 + 1j)).reshape(3, 4, 5)
    selection = slice(None), 2, slice(None)

    widget = NxComplexImagePlot()
    widget.setImageData(
        signals=[data],
        signalsNames=["signal"],
        axesNames=["z", "y", "x"],
        selection=selection,
    )

    assert widget._axesSelector.selection() == selection
    numpy.testing.assert_array_equal(
        widget._imageItem.getComplexData(), data[selection]
    )


def test_curve_view_default_slice(qapp):
    data = numpy.arange(3 * 4).reshape(3, 4).astype(float)
    selection = 1, slice(None)

    widget = NxCurvePlot()
    widget.setCurvesData(signals=[data], signal_names=["signal"], selection=selection)

    assert widget._axesSelector.selection() == selection
    curve = widget._plot.getActiveCurve()
    numpy.testing.assert_array_equal(curve.getYData(), data[selection])


@pytest.mark.parametrize(
    "nxdata_scale, expected", [(None, "log"), ("linear", "linear")]
)
def test_curve_view_scale_on_data_change(qapp, nxdata_scale, expected):
    """Test that the user scale is kept when the new NXdata does not define a scale."""
    widget = NxCurvePlot()
    x = numpy.linspace(0.1, 1.0, 10)
    widget.setCurvesData(signals=[x], signal_names=["s1"], axes=[x])

    widget._plot.getYAxis().setScale("log")
    widget._plot.getXAxis().setScale("log")

    widget.setCurvesData(
        signals=[x],
        signal_names=["s2"],
        axes=[x],
        signal_scale=nxdata_scale,
        axes_scales=[nxdata_scale],
    )

    assert widget._plot.getYAxis().getScale() == expected
    assert widget._plot.getXAxis().getScale() == expected


@pytest.mark.usefixtures("use_opengl")
def test_volume_view_default_slice(qapp):
    data = numpy.arange(2 * 3 * 4 * 5, dtype=numpy.float32).reshape(2, 3, 4, 5)
    selection = 1, slice(None), slice(None), slice(None)

    widget = NxVolumePlot()
    widget.setVolumeData(signal=data, selection=selection)

    assert widget._axesSelector.selection() == selection

    (volume,) = widget._view.getSceneWidget().getItems()
    numpy.testing.assert_array_equal(volume.getData(copy=False), data[selection])
