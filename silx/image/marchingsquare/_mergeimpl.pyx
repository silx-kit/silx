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
"""
Marching squares implementation based on a merge of segements and polygons.
"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "04/04/2018"

import numpy
cimport numpy as cnumpy

from libcpp.vector cimport vector
from libcpp.list cimport list as clist
from libcpp.map cimport map
from libcpp cimport bool
from libc.math cimport fabs

from cython.parallel import prange
from cython.operator cimport dereference
from cython.operator cimport preincrement
cimport cython

cdef double EPSILON = numpy.finfo(numpy.float64).eps

cdef extern from "include/patterns.h":
    cdef unsigned char EDGE_TO_POINT[][2]
    cdef unsigned char CELL_TO_EDGE[][5]

ctypedef cnumpy.uint32_t point_index_t

cdef struct point_t:
    cnumpy.float32_t x
    cnumpy.float32_t y

cdef cppclass polygon_description_t:
    point_index_t begin
    point_index_t end
    clist[point_t] points

    polygon_description_t() nogil:
        pass

cdef cppclass TileContext_t:
    int pos_x
    int pos_y
    int dim_x
    int dim_y

    clist[polygon_description_t*] final_polygons

    map[point_index_t, polygon_description_t*] polygons

    TileContext_t() nogil:
        pass


cdef class MarchingSquareMergeImpl(object):
    """
    Marching squares implementation based on a merge of segements and polygons.

    The main algorithm is based on the common marching squares algorims.
    Segments of the iso contours are identified using a pattern based on a group
    of 2*2 pixels. The image is  read sequencially and when a segment is
    identified it is inserted at a right place is a set of valid polygons.

    The algorithm can take care of a mask. If a pixel is invalidated by the
    mask, the computed pattern is simply cancelled and no segements are
    generated.

    It implements a multi process algorithms using OpenMP based on divide and
    conquer algorithm. The image is subdivised into many tiles, each one is
    processed independantly. The result is finally reduced by consecutives
    polygon merges.

    The OpenMP group size can also by used to skip part of the image using
    pre-computed informations (min and max of each tile groups). It was
    designed to improve the efficiency of the extraction of many contour levels
    from the same gradient image.

    :param numpy.ndarray image: Image to process.
        If the image is not a continuous array of native float 32bits, the data
        will be first normalized. This can reduce efficiency.
    :param numpy.ndarray mask: An optional mask (a non-zero value invalidate
        the pixels of the image)
        If the image is not a continuous array of signed integer 8bits, the
        data will be first normalized. This can reduce efficiency.
    :param int group_size: Specify the size of the tile to split the
        computation with OpenMP. It is also used as tile size to compute the
        min/max cache
    :param bool use_minmax_cache: If true the min/max cache is enabled.
    """

    cdef cnumpy.float32_t[:, :] _image
    cdef cnumpy.int8_t[:, :] _mask

    cdef cnumpy.float32_t *_image_ptr
    cdef cnumpy.int8_t *_mask_ptr
    cdef int _dim_x
    cdef int _dim_y
    cdef int _group_size
    cdef bool _use_minmax_cache

    cdef TileContext_t* _final_context

    cdef cnumpy.float32_t[:, :] _min_cache
    cdef cnumpy.float32_t[:, :] _max_cache

    def __init__(self,
                 image, mask=None,
                 group_size=256,
                 use_minmax_cache=False):
        self._image = numpy.ascontiguousarray(image, numpy.float32)
        self._image_ptr = &self._image[0][0]
        if mask is not None:
            assert(image.shape == mask.shape)
            self._mask = numpy.ascontiguousarray(mask, numpy.int8)
            self._mask_ptr = &self._mask[0][0]
        else:
            self._mask = None
            self._mask_ptr = NULL
        self._group_size = group_size
        self._use_minmax_cache = use_minmax_cache
        if self._use_minmax_cache:
            self._min_cache = None
            self._max_cache = None
        with nogil:
            self._dim_y = self._image.shape[0]
            self._dim_x = self._image.shape[1]

    def _get_minmax_block(self, array, block_size):
        """Python code to compute min/max cache per block of an image"""
        if block_size == 0:
            return None
        size = numpy.array(array.shape)
        size = size // block_size + (size % block_size > 0)
        min_per_block = numpy.empty(size, dtype=numpy.float32)
        max_per_block = numpy.empty(size, dtype=numpy.float32)
        for y in range(size[1]):
            yend = (y + 1) * block_size
            if y + 1 == size[1]:
                yy = slice(y * block_size, array.shape[1])
            else:
                yy = slice(y * block_size, yend)
            for x in range(size[0]):
                xend = (x + 1) * block_size
                if xend > size[0]:
                    xx = slice(x * block_size, array.shape[0])
                else:
                    xx = slice(x * block_size, xend)
                min_per_block[x, y] = numpy.min(array[xx, yy])
                max_per_block[x, y] = numpy.max(array[xx, yy])
        return (min_per_block, max_per_block, block_size)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _marching_squares(self, cnumpy.float64_t isovalue):
        cdef:
            int x, y, i
            vector[TileContext_t*] contexts
            TileContext_t *context
            int dim_x, dim_y
            int ix, iy

        iy = 0
        for y in range(0, self._dim_y - 1, self._group_size):
            ix = 0
            for x in range(0, self._dim_x - 1, self._group_size):
                if self._use_minmax_cache:
                    if isovalue < self._min_cache[iy, ix] or isovalue > self._max_cache[iy, ix]:
                        ix += 1
                        continue
                context = self._create_context(x, y, self._group_size, self._group_size)
                contexts.push_back(context)
                ix += 1
            iy += 1

        if contexts.size() == 0:
            # shortcut
            self._final_context = new TileContext_t()
            return

        # openmp
        for i in prange(contexts.size(), nogil=True):
            self._marching_squares_mp(contexts[i], isovalue)

        if contexts.size() == 1:
            # shortcut
            self._final_context = contexts[0]
            return

        # merge
        self._final_context = new TileContext_t()
        # self._final_context.polygons.reserve(self._dim_x * 2 + self._dim_y * 2)
        for i in range(contexts.size()):
            self._merge_context(self._final_context, contexts[i])
            del contexts[i]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef TileContext_t *_create_context(self, int x, int y, int dim_x, int dim_y) nogil:
        cdef:
            TileContext_t *context
        context = new TileContext_t()
        context.pos_x = x
        context.pos_y = y
        context.dim_x = dim_x
        context.dim_y = dim_y
        if x + context.dim_x > self._dim_x - 1:
            context.dim_x = self._dim_x - 1 - x
        if y + context.dim_y > self._dim_y - 1:
            context.dim_y = self._dim_y - 1 - y
        if context.dim_x <= 0 or context.dim_y <= 0:
            del context
            return NULL
        return context

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _marching_squares_mp(self, TileContext_t *context, cnumpy.float64_t isovalue) nogil:
        cdef:
            int x, y, index
            cnumpy.float64_t tmpf
            cnumpy.float32_t *image_ptr
            cnumpy.int8_t *mask_ptr

        image_ptr = self._image_ptr + (context.pos_y * self._dim_x + context.pos_x)
        if self._mask_ptr != NULL:
            mask_ptr = self._mask_ptr + (context.pos_y * self._dim_x + context.pos_x)
        else:
            mask_ptr = NULL

        for y in range(context.pos_y, context.pos_y + context.dim_y):
            for x in range(context.pos_x, context.pos_x + context.dim_x):
                # Calculate index.
                index = 0
                if image_ptr[0] > isovalue:
                    index += 1
                if image_ptr[1] > isovalue:
                    index += 2
                if image_ptr[self._dim_x] > isovalue:
                    index += 8
                if image_ptr[self._dim_x + 1] > isovalue:
                    index += 4

                # Resolve ambiguity
                if index == 5 or index == 10:
                    # Calculate value of cell center (i.e. average of corners)
                    tmpf = 0.25 * (image_ptr[0] +
                                   image_ptr[1] +
                                   image_ptr[self._dim_x] +
                                   image_ptr[self._dim_x + 1])
                    # If below isovalue, swap
                    if tmpf <= isovalue:
                        if index == 5:
                            index = 10
                        else:
                            index = 5

                # Cache mask information
                if mask_ptr != NULL:
                    # Note: Store the mask in the index. It could be usefull to
                    #     generate accurate segments in some cases, but yet it
                    #     is not used
                    if mask_ptr[0] > 0:
                        index += 16
                    if mask_ptr[1] > 0:
                        index += 32
                    if mask_ptr[self._dim_x] > 0:
                        index += 128
                    if mask_ptr[self._dim_x + 1] > 0:
                        index += 64
                    mask_ptr += 1

                if index < 16 and index != 0 and index != 15:
                    self._insert_pattern(context, x, y, index, isovalue)

                image_ptr += 1

            # There is a missing pixel at the end of each rows
            image_ptr += self._dim_x - context.dim_x
            if mask_ptr != NULL:
                mask_ptr += self._dim_x - context.dim_x

    cdef void _insert_pattern(self, TileContext_t *context, int x, int y, int pattern, cnumpy.float64_t isovalue) nogil:
        cdef:
            int segment
        for segment in range(CELL_TO_EDGE[pattern][0]):
            begin_edge = CELL_TO_EDGE[pattern][1 + segment * 2 + 0]
            end_edge = CELL_TO_EDGE[pattern][1 + segment * 2 + 1]
            self._insert_segment(context, x, y, begin_edge, end_edge, isovalue)

    cdef point_index_t _create_point_index(self, int yx, cnumpy.uint8_t edge) nogil:
        """Create an unique identifier for a point of a polygon.

        It can be shared by different pixel coordinates. For example, the tuple
        (x=0, y=0, edge=2) is equal to (x=1, y=0, edge=0).
        """
        if edge == 2:
            yx += self._dim_x
            edge = 0
        elif edge == 1:
            yx += 1
        elif edge == 3:
            edge = 1

        # Reserve the zero value
        yx += 1

        return edge + (yx << 1)

    cdef void _insert_segment(self, TileContext_t *context,
                              int x, int y,
                              cnumpy.uint8_t begin_edge,
                              cnumpy.uint8_t end_edge,
                              cnumpy.float64_t isovalue) nogil:
        cdef:
            int i, yx
            point_t point
            point_index_t begin, end
            polygon_description_t *description
            polygon_description_t *description_begin
            polygon_description_t *description_end
            map[point_index_t, polygon_description_t*].iterator it_begin
            map[point_index_t, polygon_description_t*].iterator it_end

        yx = self._dim_x * y + x
        begin = self._create_point_index(yx, begin_edge)
        end = self._create_point_index(yx, end_edge)

        it_begin = context.polygons.find(begin)
        it_end = context.polygons.find(end)
        if it_begin == context.polygons.end() and it_end == context.polygons.end():
            # insert a new polygon
            description = new polygon_description_t()
            description.begin = begin
            description.end = end
            self._compute_point(x, y, begin_edge, isovalue, &point)
            description.points.push_back(point)
            self._compute_point(x, y, end_edge, isovalue, &point)
            description.points.push_back(point)
            context.polygons[begin] = description
            context.polygons[end] = description
        elif it_begin == context.polygons.end():
            # insert the beggining point to an existing polygon
            self._compute_point(x, y, begin_edge, isovalue, &point)
            description = dereference(it_end).second
            # FIXME: We should erase using the iterator
            context.polygons.erase(end)
            if end == description.begin:
                # insert at start
                description.points.push_front(point)
                description.begin = begin
                context.polygons[begin] = description
            else:
                # insert on tail
                description.points.push_back(point)
                description.end = begin
                context.polygons[begin] = description
        elif it_end == context.polygons.end():
            # insert the endding point to an existing polygon
            self._compute_point(x, y, end_edge, isovalue, &point)
            description = dereference(it_begin).second
            # FIXME: We should erase using the iterator
            context.polygons.erase(begin)
            if begin == description.begin:
                # insert at start
                description.points.push_front(point)
                description.begin = end
                context.polygons[end] = description
            else:
                # insert on tail
                description.points.push_back(point)
                description.end = end
                context.polygons[end] = description
        else:
            # merge 2 polygons using this segment
            description_begin = dereference(it_begin).second
            description_end = dereference(it_end).second
            if description_begin == description_end:
                # The segment closes a polygon
                # FIXME: this intermediate assign is not needed
                point = description_begin.points.front()
                description_begin.points.push_back(point)
                context.polygons.erase(begin)
                context.polygons.erase(end)
                context.final_polygons.push_back(description_begin)
            else:
                if ((begin == description_begin.begin or end == description_begin.begin) and
                   (begin == description_end.end or end == description_end.end)):
                    # worst case, let's make it faster
                    description = description_end
                    description_end = description_begin
                    description_begin = description

                # FIXME: We can recycle a description instead of creating a new one
                description = new polygon_description_t()

                # Make sure the last element of the list is the one to connect
                if description_begin.begin == begin or description_begin.begin == end:
                    # O(n)
                    description_begin.points.reverse()
                    description.begin = description_begin.end
                else:
                    description.begin = description_begin.begin

                # O(1)
                description.points.splice(description.points.end(), description_begin.points)

                # Make sure the first element of the list is the one to connect
                if description_end.end == begin or description_end.end == end:
                    description_end.points.reverse()
                    description.end = description_end.begin
                else:
                    description.end = description_end.end

                description.points.splice(description.points.end(), description_end.points)

                # FIXME: We should erase using the iterator
                context.polygons.erase(begin)
                context.polygons.erase(end)
                context.polygons[description.begin] = description
                context.polygons[description.end] = description

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _merge_context(self, TileContext_t *context, TileContext_t *other) nogil:
        cdef:
            map[point_index_t, polygon_description_t*].iterator it_begin
            map[point_index_t, polygon_description_t*].iterator it_end
            map[point_index_t, polygon_description_t*].iterator it
            polygon_description_t *description_other
            polygon_description_t *description
            polygon_description_t *description2
            point_index_t point_index
            vector[polygon_description_t*] mergeable_polygons
            int i

        # merge final polygons
        context.final_polygons.splice(context.final_polygons.end(), other.final_polygons)

        # mergeable_polygons.reserve(other.polygons.size() / 2)
        it = other.polygons.begin()
        while it != other.polygons.end():
            point_index = dereference(it).first
            description_other = dereference(it).second
            if description_other.begin == point_index:
                mergeable_polygons.push_back(description_other)
            preincrement(it)

        for i in range(mergeable_polygons.size()):
            description_other = mergeable_polygons[i]
            it_begin = context.polygons.find(description_other.begin)
            it_end = context.polygons.find(description_other.end)

            if it_begin == context.polygons.end() and it_end == context.polygons.end():
                # It's a new polygon
                context.polygons[description_other.begin] = description_other
                context.polygons[description_other.end] = description_other
            elif it_end == context.polygons.end():
                # The head of the polygon have to be merged
                description = dereference(it_begin).second
                context.polygons.erase(description.begin)
                context.polygons.erase(description.end)
                if description.begin == description_other.begin:
                    description.begin = description.end
                    description.points.reverse()
                description.end = description_other.end
                # remove the dup element
                description_other.points.pop_front()
                description.points.splice(description.points.end(), description_other.points)
                context.polygons[description.begin] = description
                context.polygons[description.end] = description
                del description_other
            elif it_begin == context.polygons.end():
                # The tail of the polygon have to be merged
                description = dereference(it_end).second
                context.polygons.erase(description.begin)
                context.polygons.erase(description.end)
                if description.begin == description_other.end:
                    description.begin = description.end
                    description.points.reverse()
                description.end = description_other.begin
                description_other.points.reverse()
                # remove the dup element
                description_other.points.pop_front()
                description.points.splice(description.points.end(), description_other.points)
                context.polygons[description.begin] = description
                context.polygons[description.end] = description
                del description_other
            else:
                # Both sides have to be merged
                description = dereference(it_begin).second
                description2 = dereference(it_end).second
                if description == description2:
                    # It became a closed polygon
                    context.polygons.erase(description.begin)
                    context.polygons.erase(description.end)
                    if description.begin == description_other.begin:
                        description.begin = description.end
                        description.points.reverse()
                    description.end = description_other.end
                    # remove the dup element
                    description_other.points.pop_front()
                    description.points.splice(description.points.end(), description_other.points)
                    context.final_polygons.push_back(description)
                    del description_other
                else:
                    context.polygons.erase(description.begin)
                    context.polygons.erase(description.end)
                    context.polygons.erase(description2.begin)
                    context.polygons.erase(description2.end)
                    if description.begin == description_other.begin:
                        description.begin = description.end
                        description.points.reverse()
                    if description2.end == description_other.end:
                        description.end = description2.begin
                        description2.points.reverse()
                    else:
                        description.end = description2.end
                    description_other.points.pop_front()
                    description2.points.pop_front()
                    description.points.splice(description.points.end(), description_other.points)
                    description.points.splice(description.points.end(), description2.points)
                    context.polygons[description.begin] = description
                    context.polygons[description.end] = description
                    del description_other
                    del description2

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _compute_point(self,
                             cnumpy.uint_t x,
                             cnumpy.uint_t y,
                             cnumpy.uint8_t edge,
                             cnumpy.float64_t isovalue,
                             point_t *result_point) nogil:
        cdef:
            int dx1, dy1, index1
            int dx2, dy2, index2
            cnumpy.float64_t fx, fy, ff, weight1, weight2
        # Use these to look up the relative positions of the pixels to interpolate
        dx1, dy1 = EDGE_TO_POINT[edge][0], EDGE_TO_POINT[edge][1]
        dx2, dy2 = EDGE_TO_POINT[edge + 1][0], EDGE_TO_POINT[edge + 1][1]
        # Define "strength" of each corner of the cube that we need
        index1 = (y + dy1) * self._dim_x + x + dx1
        index2 = (y + dy2) * self._dim_x + x + dx2
        weight1 = 1.0 / (EPSILON + fabs(self._image_ptr[index1] - isovalue))
        weight2 = 1.0 / (EPSILON + fabs(self._image_ptr[index2] - isovalue))
        # Apply a kind of center-of-mass method
        fx, fy, ff = 0.0, 0.0, 0.0
        fx += dx1 * weight1
        fy += dy1 * weight1
        ff += weight1
        fx += dx2 * weight2
        fy += dy2 * weight2
        ff += weight2
        fx /= ff
        fy /= ff
        result_point.x = x + fx
        result_point.y = y + fy

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef _extract_polygons(self):
        cdef:
            int i, i_pixel
            cnumpy.uint8_t index
            map[point_index_t, polygon_description_t*].iterator it
            vector[polygon_description_t*] descriptions
            clist[point_t].iterator it_points
            polygon_description_t *description

        # move all the polygons in a final structure
        with nogil:
            it = self._final_context.polygons.begin()
            while it != self._final_context.polygons.end():
                description = dereference(it).second
                if dereference(it).first == description.begin:
                    # polygones are stored 2 times
                    # only use one
                    descriptions.push_back(description)
                preincrement(it)
            self._final_context.polygons.clear()

            descriptions.insert(descriptions.end(),
                                self._final_context.final_polygons.begin(),
                                self._final_context.final_polygons.end())
            self._final_context.final_polygons.clear()

        # create result and clean up allocated memory
        polygons = []
        for i in range(descriptions.size()):
            description = descriptions[i]
            polygon = numpy.empty(description.points.size() * 2, dtype=numpy.float32)
            it_points = description.points.begin()
            i_pixel = 0
            while it_points != description.points.end():
                polygon[i_pixel + 0] = dereference(it_points).y
                polygon[i_pixel + 1] = dereference(it_points).x
                i_pixel += 2
                preincrement(it_points)
            polygon.shape = -1, 2
            polygons.append(polygon)
            del description
        return polygons

    def find_pixels(self, level):
        """
        Compute the pixels from the image over the requested iso contours
        at this `level`.

        :param float level: Level of the requested iso contours.
        :returns: An array of y-x coordinates.
        :rtype: numpy.ndarray
        """
        # FIXME: Implement it
        pixels = numpy.zeros((0, 2), dtype=numpy.int32)
        return pixels

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def find_contours(self, level=None):
        """
        Compute the list of polygons of the iso contours at this `level`.

        :param float level: Level of the requested iso contours.
        :returns: A list of array containg y-x coordinates of points
        :rtype: List[numpy.ndarray]
        """
        if self._use_minmax_cache and self._min_cache is None:
            r = self._get_minmax_block(self._image, self._group_size)
            self._min_cache = r[0]
            self._max_cache = r[1]

        self._marching_squares(level)
        polygons = self._extract_polygons()
        return polygons
