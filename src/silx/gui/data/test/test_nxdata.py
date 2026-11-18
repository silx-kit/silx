import numpy
import pytest

from silx.io.nxdata.parse import _get_default_slice_selection

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
