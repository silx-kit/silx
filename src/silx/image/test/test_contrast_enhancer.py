import numpy
import pytest
from silx.image.ContrastEnhancer import _ContrastEnhancer


@pytest.mark.parametrize("saturation_factor", (0.1, 0.2, 0.8))
def test_ContrastEnhancer(qapp, saturation_factor):
    """test processing of _ContrastEnhancer"""
    contrast_enhancer = _ContrastEnhancer()
    contrast_enhancer.saturation = saturation_factor

    linear_image = numpy.linspace(
        0, 100, 100, endpoint=False, dtype=numpy.uint32
    ).reshape(10, 10)
    numpy.testing.assert_almost_equal(
        contrast_enhancer.get_min_max(linear_image),
        (100 * (saturation_factor / 2.0), 99 - (100 * (saturation_factor / 2.0))),
    )

    # x = numpy.outer(numpy.linspace(-10, 10, 200), numpy.linspace(-10, 5, 150))
    # sin_like_image = numpy.sin(x) / x

    # from silx.gui.plot import Plot2D
    # plot = Plot2D()
    # img = plot.addImage(sin_like_image)
    # img.getColormap().setVRange(contrast_enhancer.get_min_max(sin_like_image))
    # plot.show()
    # qapp.exec_()
    # assert contrast_enhancer.get_min_max(sin_like_image) == (100 * (saturation_factor / 2.0), 100 - (100 * (saturation_factor / 2.0)))
