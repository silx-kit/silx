import numpy

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
