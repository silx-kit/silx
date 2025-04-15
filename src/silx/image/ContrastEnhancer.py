import numpy
from silx.math.histogram import Histogramnd
from silx.math.combo import min_max


class _ContrastEnhancer:
    """ """

    DEFAULT_SATURATION = 0.35

    def __init__(self):
        self.saturation: float = _ContrastEnhancer.DEFAULT_SATURATION
        self.equalize: bool = True
        self.normalize: bool = True

    def get_min_max(
        self, image: numpy.ndarray, histogram_stats: dict | None = None
    ) -> tuple[float]:
        """
        Return min max value to be set to the colormap in order to enhance the contrast
        by saturating self.saturation ratio of pixels
        """

        if not isinstance(image, numpy.ndarray) and not image.ndim == 2:
            raise TypeError("image is expected to be a 2D array.")

        if histogram_stats is None:
            histogram_stats: dict = self.get_histogram(image=image)

        h_min = histogram_stats["hist_min"]
        h_max = histogram_stats["hist_max"]
        bin_size = histogram_stats["bin_size"]
        histogram = histogram_stats["histogram"]

        # compute histogram cumulative sum to determine new min / max location in the histogram
        # we know that we want to saturate 'nb_points_to_skip' on each side of the histogram
        hist_cum_sum = numpy.cumsum(histogram)
        hist_cum_sum_inv = numpy.cumsum(histogram[::-1])

        nb_points_to_skip = numpy.prod(image.shape) * self.DEFAULT_SATURATION / 2.0

        min_index_bin = numpy.where(hist_cum_sum > nb_points_to_skip)[0][0]
        max_index_bin = numpy.where(hist_cum_sum_inv > nb_points_to_skip)[0][0]

        min_val = h_min + bin_size * min_index_bin
        max_val = h_max - bin_size * max_index_bin

        return min_val, max_val

    @staticmethod
    def get_histogram(image: numpy.ndarray) -> dict:
        """
        Compute histogram image and metadata for downstream calculation.
        :return: dict with the following keys:

            - **histogram**: nupy.ndarray
            - **hist_min**: histogram range lower value
            - **hist_max**: histogram range higher value
            - **bin_size**: numpy.float
        """
        min, max = min_max(image)
        n_bins = 256
        # cast image to float32 to make it compatible with 'Histogramnd'
        image = image.ravel().astype(numpy.float32)
        histogram, _, _ = Histogramnd(image, histo_range=(min, max), n_bins=n_bins)
        # hist_min, hist_max = bin_edges
        return {
            "histogram": histogram,
            "hist_min": min,
            "hist_max": max,
            "bin_size": (max - min) / n_bins,
        }
