# coding: utf-8
# /*##########################################################################
# Copyright (C) 2018 European Synchrotron Radiation Facility
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

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "04/04/2018"


from ._mergeimpl import MarchingSquareMergeImpl


def _factory(engine, image, mask):
    """Factory to create the marching square implementation from the engine
    name"""
    if engine == "merge":
        return MarchingSquareMergeImpl(image, mask)
    elif engine == "skimage":
        from _skimage import MarchingSquareSciKitImage
        return MarchingSquareSciKitImage(image, mask)
    else:
        raise ValueError("Engine '%s' is not supported ('merge' or 'skimage' expected).")


def find_pixels(image, level, mask=None, engine="merge"):
    """
    Compute the pixels from the image over the requested iso contours
    at this `level`. Pixels are those over the bound of the segments.

    :param numpy.ndarray image: Image to process
    :param float level: Level of the requested iso contours.
    :param numpy.ndarray mask: An optional mask (a non-zero value invalidate
        the pixels of the image)
    :param str engine: Engine to use. Currently 2 implentation are available.

        - merge: An implentation using Cython and supporting OpenMP.
        - skimage: Provide an implementation based on the library skimage. If
          it is used with a mask, the computation will not be accurate nor
          efficient. Provided to compare implementation. The `skimage` library
          have to be installed.

    :returns: A list of array containg y-x coordinates of points
    :rtype: List[numpy.ndarray]
    """
    assert(image is not None)
    if mask is not None:
        assert(image.shape == mask.shape)
    impl = _factory(engine, image, mask)
    return impl.find_pixels(level)


def find_contours(image, level, mask=None, engine="merge"):
    """
    Compute the list of polygons of the iso contours at this `level`.

    :param numpy.ndarray image: Image to process
    :param float level: Level of the requested iso contours.
    :param numpy.ndarray mask: An optional mask (a non-zero value invalidate
        the pixels of the image)
    :param str engine: Engine to use. Currently 2 implentation are available.

        - merge: An implentation using Cython and supporting OpenMP.
        - skimage: Provide an implementation based on the library skimage. If
          it is used with a mask, the computation will not be accurate nor
          efficient. Provided to compare implementation. The `skimage` library
          have to be installed.

    :returns: A list of array containg y-x coordinates of points
    :rtype: List[numpy.ndarray]
    """
    assert(image is not None)
    if mask is not None:
        assert(image.shape == mask.shape)
    impl = _factory(engine, image, mask)
    return impl.find_contours(level)
