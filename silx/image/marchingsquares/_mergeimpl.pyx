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

__authors__ = ["Almar Klein", "Jerome Kieffer", "Valentin Valls"]
__license__ = "MIT"
__date__ = "23/04/2018"

import numpy
cimport numpy as cnumpy

from libcpp.vector cimport vector
from libcpp.list cimport list as clist
from libcpp.set cimport set as cset
from libcpp.map cimport map
from libcpp cimport bool
from libc.math cimport fabs
from libc.math cimport floor

from cython.parallel import prange
from cython.operator cimport dereference
from cython.operator cimport preincrement
cimport libc.stdlib
cimport libc.string

cimport cython

include "../../utils/_have_openmp.pxi"
"""Store in the module if it was compiled with OpenMP"""

cdef double EPSILON = numpy.finfo(numpy.float64).eps

# Windows compatibility: Cross-platform INFINITY
from libc.float cimport DBL_MAX
cdef double INFINITY = DBL_MAX + DBL_MAX
# from libc.math cimport INFINITY

cdef extern from "include/patterns.h":
    cdef unsigned char EDGE_TO_POINT[][2]
    cdef unsigned char CELL_TO_EDGE[][5]
    cdef struct coord_t:
        short x
        short y

ctypedef cnumpy.uint32_t point_index_t
"""Type of the unique index identifying a connection for the polygons."""

"""Define a point of a polygon."""
cdef struct point_t:
    cnumpy.float32_t x
    cnumpy.float32_t y

"""Description of a non-final polygon."""
cdef cppclass PolygonDescription:
    point_index_t begin
    point_index_t end
    clist[point_t] points

    PolygonDescription() nogil:
        pass

"""Description of a tile context.

It contains structure to store intermediate and final data of a thread.
Pixels and contours structures are merged together as it looks to have
mostly no cost.
"""
cdef cppclass TileContext:
    int pos_x
    int pos_y
    int dim_x
    int dim_y

    # Only used to find contours
    clist[PolygonDescription*] final_polygons
    map[point_index_t, PolygonDescription*] polygons

    # Only used to find pixels
    clist[coord_t] final_pixels
    cset[coord_t] pixels

    TileContext() nogil:
        pass


cdef class _MarchingSquaresAlgorithm(object):
    """Abstract class managing a marching squares algorithm.

    It provides common methods to execute the process, with the support of
    OpenMP, plus some hooks. Mostly created to be able to reuse part of the
    logic between `_MarchingSquaresContours` and `_MarchingSquaresPixels`.
    """

    cdef cnumpy.float32_t *_image_ptr
    cdef cnumpy.int8_t *_mask_ptr
    cdef int _dim_x
    cdef int _dim_y
    cdef int _group_size
    cdef bool _use_minmax_cache
    cdef bool _force_sequencial_reduction

    cdef TileContext* _final_context

    cdef cnumpy.float32_t *_min_cache
    cdef cnumpy.float32_t *_max_cache

    def __cinit__(self):
        self._use_minmax_cache = False
        self._force_sequencial_reduction = False

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void marching_squares(self, cnumpy.float64_t level) nogil:
        """
        Main method to execute the marching squares.

        :param level: The level expected.
        """
        cdef:
            TileContext** contexts
            TileContext** valid_contexts
            int nb_contexts, nb_valid_contexts
            int i, j
            TileContext* context
            int dim_x, dim_y

        contexts = self.create_contexts(level, &dim_x, &dim_y, &nb_valid_contexts)
        nb_contexts = dim_x * dim_y

        if nb_valid_contexts == 0:
            # shortcut
            self._final_context = new TileContext()
            libc.stdlib.free(contexts)
            return

        j = 0
        valid_contexts = <TileContext **>libc.stdlib.malloc(nb_valid_contexts * sizeof(TileContext*))
        for i in xrange(nb_contexts):
            if contexts[i] != NULL:
                valid_contexts[j] = contexts[i]
                j += 1

        # openmp
        for i in prange(nb_valid_contexts, nogil=True):
            self.marching_squares_mp(valid_contexts[i], level)

        if nb_valid_contexts == 1:
            # shortcut
            self._final_context = valid_contexts[0]
            libc.stdlib.free(valid_contexts)
            libc.stdlib.free(contexts)
            return

        if self._force_sequencial_reduction:
            self.sequencial_reduction(nb_valid_contexts, valid_contexts)
        # FIXME can only be used if compiled with openmp
        # elif copenmp.omp_get_num_threads() <= 1:
        #     self._sequencial_reduction(nb_valid_contexts, valid_contexts)
        else:
            self.reduction_2d(dim_x, dim_y, contexts)

        libc.stdlib.free(valid_contexts)
        libc.stdlib.free(contexts)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void reduction_2d(self, int dim_x, int dim_y, TileContext **contexts) nogil:
        """
        Reduce the problem merging first neighbours together in a recursive
        process. Optimized with OpenMP.

        :param dim_x: Number of contexts in the x dimension
        :param dim_y: Number of contexts in the y dimension
        :param contexts: Array of contexts
        """
        cdef:
            int x1, y1, x2, y2, i1, i2
            int delta = 1

        while True:
            if delta >= dim_x and delta >= dim_y:
                break
            # NOTE: Cython 0.21.1 is buggy with prange + steps
            # It is needed to add a delta and the 'to'
            # Here is what we can use with Cython 0.28:
            #     for i in prange(0, dim_x, (delta + delta)):
            for i1 in prange(0, dim_x + (delta + delta - 1), delta + delta, nogil=True):
                x1 = i1
                if x1 + delta < dim_x:
                    y1 = 0
                    while y1 < dim_y:
                        self.merge_array_contexts(contexts, y1 * dim_x + x1, y1 * dim_x + x1 + delta)
                        y1 = y1 + delta

            # NOTE: Cython 0.21.1 is buggy with prange + steps
            # It is needed to add a delta and the 'to'
            # Here is what we can use with Cython 0.28:
            #     for i in prange(0, dim_y, (delta + delta)):
            for i2 in prange(0, dim_y + (delta + delta - 1), delta + delta, nogil=True):
                y2 = i2
                if y2 + delta < dim_y:
                    x2 = 0
                    while x2 < dim_x:
                        self.merge_array_contexts(contexts, y2 * dim_x + x2, (y2 + delta) * dim_x + x2)
                        x2 = x2 + delta + delta
            delta <<= 1

        self._final_context = contexts[0]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef inline void merge_array_contexts(self,
                                          TileContext **contexts,
                                          int index1,
                                          int index2) nogil:
        """
        Merge contexts from `index2` to `index1` and delete the one from index2.
        If the one from index1 was NULL, the one from index2 is moved to index1
        and is not deleted.

        This intermediate function was needed to avoid compilation problem of
        Cython + OpenMP.
        """
        cdef:
            TileContext *context1
            TileContext *context2

        context1 = contexts[index1]
        context2 = contexts[index2]
        if context1 != NULL and context2 != NULL:
            self.merge_context(context1, context2)
            del context2
        elif context2 != NULL:
            contexts[index1] = context2
        # for sanity
        # contexts[index2] = NULL

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void sequencial_reduction(self,
                                   int nb_contexts,
                                   TileContext **contexts) nogil:
        """
        Reduce the problem sequencially without taking care of the topology

        :param nb_contexts: Number of contexts
        :param contexts: Array of contexts
        """
        cdef:
            int i
        # merge
        self._final_context = new TileContext()
        for i in xrange(nb_contexts):
            if contexts[i] != NULL:
                self.merge_context(self._final_context, contexts[i])
                del contexts[i]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void marching_squares_mp(self,
                                  TileContext *context,
                                  cnumpy.float64_t level) nogil:
        """
        Main entry of the marching squares algorithm for each threads.

        :param context: Context used by the thread to store data
        :param level: The requested level
        """
        cdef:
            int x, y, pattern
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
                pattern = 0
                if image_ptr[0] > level:
                    pattern += 1
                if image_ptr[1] > level:
                    pattern += 2
                if image_ptr[self._dim_x] > level:
                    pattern += 8
                if image_ptr[self._dim_x + 1] > level:
                    pattern += 4

                # Resolve ambiguity
                if pattern == 5 or pattern == 10:
                    # Calculate value of cell center (i.e. average of corners)
                    tmpf = 0.25 * (image_ptr[0] +
                                   image_ptr[1] +
                                   image_ptr[self._dim_x] +
                                   image_ptr[self._dim_x + 1])
                    # If below level, swap
                    if tmpf <= level:
                        if pattern == 5:
                            pattern = 10
                        else:
                            pattern = 5

                # Cache mask information
                if mask_ptr != NULL:
                    # Note: Store the mask in the index. It could be usefull to
                    #     generate accurate segments in some cases, but yet it
                    #     is not used
                    if mask_ptr[0] > 0:
                        pattern += 16
                    if mask_ptr[1] > 0:
                        pattern += 32
                    if mask_ptr[self._dim_x] > 0:
                        pattern += 128
                    if mask_ptr[self._dim_x + 1] > 0:
                        pattern += 64
                    mask_ptr += 1

                if pattern < 16 and pattern != 0 and pattern != 15:
                    self.insert_pattern(context, x, y, pattern, level)

                image_ptr += 1

            # There is a missing pixel at the end of each rows
            image_ptr += self._dim_x - context.dim_x
            if mask_ptr != NULL:
                mask_ptr += self._dim_x - context.dim_x

        self.after_marching_squares(context)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void after_marching_squares(self, TileContext *context) nogil:
        """
        Called by each threads after execution of the marching squares
        algorithm. Called before merging together the contextes.

        :param context: Context used by the thread to store data
        """
        pass

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void insert_pattern(self,
                             TileContext *context,
                             int x,
                             int y,
                             int pattern,
                             cnumpy.float64_t level) nogil:
        """
        Called by the marching squares algorithm each time a pattern is found.

        :param context: Context used by the thread to store data
        :param x: X location of the pattern
        :param y: Y location of the pattern
        :param pattern: Binary-field identifying lower and higher pixel levels
        :param level: The requested level
        """
        pass

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void merge_context(self,
                            TileContext *context,
                            TileContext *other) nogil:
        """
        Merge into a context another context.

        :param context: Context which will contains the merge result
        :param other: Context to merge into the other one. The merging process
            is destructive. The context may returns empty.
        """
        pass

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef TileContext** create_contexts(self,
                                       cnumpy.float64_t level,
                                       int* dim_x,
                                       int* dim_y,
                                       int* nb_valid_contexts) nogil:
        """
        Create and initialize a 2d-array of contexts.

        If the minmax cache is used, only useful context will be created.
        Thous with the minmax range excluding the level will not be created and
        will have a `NULL` reference in the context array.

        :param level: The requested level
        :param dim_x: Resulting X dimension of context array
        :param dim_x: Resulting Y dimension of context array
        :param nb_valid_contexts: Resulting number of created contexts
        :return: The context array
        """
        cdef:
            int context_dim_x, context_dim_y
            int context_size, valid_contexts
            int x, y
            int icontext
            TileContext* context
            TileContext** contexts

        context_dim_x = self._dim_x // self._group_size + (self._dim_x % self._group_size > 0)
        context_dim_y = self._dim_y // self._group_size + (self._dim_y % self._group_size > 0)
        context_size = context_dim_x * context_dim_y
        contexts = <TileContext **>libc.stdlib.malloc(context_size * sizeof(TileContext*))
        libc.string.memset(contexts, 0, context_size * sizeof(TileContext*))

        valid_contexts = 0
        icontext = 0
        y = 0
        while y < self._dim_y - 1:
            x = 0
            while x < self._dim_x - 1:
                if self._use_minmax_cache:
                    if level < self._min_cache[icontext] or level > self._max_cache[icontext]:
                        icontext += 1
                        x += self._group_size
                        continue
                context = self.create_context(x, y, self._group_size, self._group_size)
                contexts[icontext] = context
                icontext += 1
                valid_contexts += 1
                x += self._group_size
            y += self._group_size

        # dereference is not working here... then we uses array index but
        # it is not the proper way
        dim_x[0] = context_dim_x
        dim_y[0] = context_dim_y
        nb_valid_contexts[0] = valid_contexts
        return contexts

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef TileContext *create_context(self,
                                     int x,
                                     int y,
                                     int dim_x,
                                     int dim_y) nogil:
        """
        Allocate and initialize a context.

        :param x: Left location of the context into the image
        :param y: Top location of the context into the image
        :param dim_x: Size of the context in the X dimension of the image
        :param dim_y: Size of the context in the Y dimension of the image
        :return: The context
        """
        cdef:
            TileContext *context
        context = new TileContext()
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
    cdef void compute_point(self,
                            cnumpy.uint32_t x,
                            cnumpy.uint32_t y,
                            cnumpy.uint8_t edge,
                            cnumpy.float64_t level,
                            point_t *result_point) nogil:
        """
        Compute the location of a point of the polygons according to the level
        and the neighbours.

        :param x: X location of the 4-pixels
        :param y: Y location of the 4-pixels
        :param edge: Enumeration identifying the 2-pixels to process
        :param level: The requested level
        :param result_point: Resulting value of the point
        """
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
        weight1 = 1.0 / (EPSILON + fabs(self._image_ptr[index1] - level))
        weight2 = 1.0 / (EPSILON + fabs(self._image_ptr[index2] - level))
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
    cdef void compute_ipoint(self,
                             cnumpy.uint32_t x,
                             cnumpy.uint32_t y,
                             cnumpy.uint8_t edge,
                             cnumpy.float64_t level,
                             coord_t *result_coord) nogil:
        """
        Compute the location of pixel which contains the point of the polygons
        according to the level and the neighbours.

        This implementation is supposed to be faster than `compute_point` when
        we only request the location of the pixel.

        :param x: X location of the 4-pixels
        :param y: Y location of the 4-pixels
        :param edge: Enumeration identifying the 2-pixels to process
        :param level: The requested level
        :param result_coord: Resulting location of the pixel
        """
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
        weight1 = EPSILON + fabs(self._image_ptr[index1] - level)
        weight2 = EPSILON + fabs(self._image_ptr[index2] - level)
        # Apply a kind of center-of-mass method
        if edge == 0:
            result_coord.x = x + (weight1 > weight2)
            result_coord.y = y
        elif edge == 1:
            result_coord.x = x + 1
            result_coord.y = y + (weight1 > weight2)
        elif edge == 2:
            result_coord.x = x + (weight1 < weight2)
            result_coord.y = y + 1
        elif edge == 3:
            result_coord.x = x
            result_coord.y = y + (weight1 < weight2)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef point_index_t create_point_index(self, int yx, cnumpy.uint8_t edge) nogil:
        """
        Create a unique identifier for a point of a polygon based on the
        pattern location and the edge.

        A index can be shared by different pixel coordinates. For example,
        the index of the tuple (x=0, y=0, edge=2) is equal to the one of
        (x=1, y=0, edge=0).

        :param yx: Index of the location of the pattern in the image
        :param edge: Enumeration identifying the edge of the pixel
        :return: An index
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


cdef class _MarchingSquaresContours(_MarchingSquaresAlgorithm):
    """Implementation of the marching squares algorithm to find iso contours.
    """

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void insert_pattern(self,
                             TileContext *context,
                             int x,
                             int y,
                             int pattern,
                             cnumpy.float64_t level) nogil:
        cdef:
            int segment
        for segment in range(CELL_TO_EDGE[pattern][0]):
            begin_edge = CELL_TO_EDGE[pattern][1 + segment * 2 + 0]
            end_edge = CELL_TO_EDGE[pattern][1 + segment * 2 + 1]
            self.insert_segment(context, x, y, begin_edge, end_edge, level)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void insert_segment(self, TileContext *context,
                             int x, int y,
                             cnumpy.uint8_t begin_edge,
                             cnumpy.uint8_t end_edge,
                             cnumpy.float64_t level) nogil:
        cdef:
            int i, yx
            point_t point
            point_index_t begin, end
            PolygonDescription *description
            PolygonDescription *description_begin
            PolygonDescription *description_end
            map[point_index_t, PolygonDescription*].iterator it_begin
            map[point_index_t, PolygonDescription*].iterator it_end

        yx = self._dim_x * y + x
        begin = self.create_point_index(yx, begin_edge)
        end = self.create_point_index(yx, end_edge)

        it_begin = context.polygons.find(begin)
        it_end = context.polygons.find(end)
        if it_begin == context.polygons.end() and it_end == context.polygons.end():
            # insert a new polygon
            description = new PolygonDescription()
            description.begin = begin
            description.end = end
            self.compute_point(x, y, begin_edge, level, &point)
            description.points.push_back(point)
            self.compute_point(x, y, end_edge, level, &point)
            description.points.push_back(point)
            context.polygons[begin] = description
            context.polygons[end] = description
        elif it_begin == context.polygons.end():
            # insert the beginning point to an existing polygon
            self.compute_point(x, y, begin_edge, level, &point)
            description = dereference(it_end).second
            context.polygons.erase(it_end)
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
            # insert the ending point to an existing polygon
            self.compute_point(x, y, end_edge, level, &point)
            description = dereference(it_begin).second
            context.polygons.erase(it_begin)
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
                description = new PolygonDescription()

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

                context.polygons.erase(it_begin)
                context.polygons.erase(it_end)
                context.polygons[description.begin] = description
                context.polygons[description.end] = description

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void merge_context(self, TileContext *context, TileContext *other) nogil:
        cdef:
            map[point_index_t, PolygonDescription*].iterator it_begin
            map[point_index_t, PolygonDescription*].iterator it_end
            map[point_index_t, PolygonDescription*].iterator it
            PolygonDescription *description_other
            PolygonDescription *description
            PolygonDescription *description2
            point_index_t point_index
            vector[PolygonDescription*] mergeable_polygons
            size_t i

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
    cdef extract_polygons(self):
        cdef:
            size_t i
            int i_pixel
            cnumpy.uint8_t index
            map[point_index_t, PolygonDescription*].iterator it
            vector[PolygonDescription*] descriptions
            clist[point_t].iterator it_points
            PolygonDescription *description
            cnumpy.float32_t[:, ::1] polygon

        if self._final_context == NULL:
            return []

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

        del self._final_context
        self._final_context = NULL

        # create result and clean up allocated memory
        polygons = []
        for i in range(descriptions.size()):
            description = descriptions[i]
            polygon = numpy.empty((description.points.size(), 2), dtype=numpy.float32)
            it_points = description.points.begin()
            i_pixel = 0
            while it_points != description.points.end():
                polygon[i_pixel, 0] = dereference(it_points).y
                polygon[i_pixel, 1] = dereference(it_points).x
                i_pixel += 1
                preincrement(it_points)
            polygons.append(numpy.asarray(polygon))
            del description

        return polygons


cdef class _MarchingSquaresPixels(_MarchingSquaresAlgorithm):
    """Implementation of the marching squares algorithm to find pixels of the
    image containing points of the polygons of the iso contours.
    """

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void insert_pattern(self,
                             TileContext *context,
                             int x,
                             int y,
                             int pattern,
                             cnumpy.float64_t level) nogil:
        cdef:
            int segment
        for segment in range(CELL_TO_EDGE[pattern][0]):
            begin_edge = CELL_TO_EDGE[pattern][1 + segment * 2 + 0]
            end_edge = CELL_TO_EDGE[pattern][1 + segment * 2 + 1]
            self.insert_segment(context, x, y, begin_edge, end_edge, level)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void insert_segment(self, TileContext *context,
                             int x, int y,
                             cnumpy.uint8_t begin_edge,
                             cnumpy.uint8_t end_edge,
                             cnumpy.float64_t level) nogil:
        cdef:
            coord_t coord
        self.compute_ipoint(x, y, begin_edge, level, &coord)
        context.pixels.insert(coord)
        self.compute_ipoint(x, y, end_edge, level, &coord)
        context.pixels.insert(coord)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void after_marching_squares(self, TileContext *context) nogil:
        cdef:
            coord_t coord
            cset[coord_t].iterator it_coord
            cset[coord_t].iterator it_coord_erase
        pass

        it_coord = context.pixels.begin()
        while it_coord != context.pixels.end():
            coord = dereference(it_coord)
            if (coord.x > context.pos_x and coord.x < context.pos_x + context.dim_x - 1 and
                    coord.y > context.pos_y and coord.y < context.pos_y + context.dim_y - 1):
                it_coord_erase = it_coord
                preincrement(it_coord)
                context.pixels.erase(it_coord_erase)
                context.final_pixels.push_back(coord)
            else:
                preincrement(it_coord)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void merge_context(self, TileContext *context, TileContext *other) nogil:
        cdef:
            cset[coord_t].iterator it_coord

        # merge final pixels
        context.final_pixels.splice(context.final_pixels.end(), other.final_pixels)

        # merge final pixels
        # NOTE: This is not declared in Cython
        #     context.final_pixels.insert(other.final_pixels.begin(), other.final_pixels.end())
        it_coord = other.pixels.begin()
        while it_coord != other.pixels.end():
            context.pixels.insert(dereference(it_coord))
            preincrement(it_coord)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef extract_pixels(self):
        cdef:
            int i, x, y
            point_index_t index
            cset[coord_t].iterator it
            clist[coord_t].iterator it_coord
            coord_t coord
            cnumpy.int32_t[:, ::1] pixels

        if self._final_context == NULL:
            return numpy.empty((0, 2), dtype=numpy.int32)

        # create result
        it = self._final_context.pixels.begin()
        while it != self._final_context.pixels.end():
            coord = dereference(it)
            self._final_context.final_pixels.push_back(coord)
            preincrement(it)

        pixels = numpy.empty((self._final_context.final_pixels.size(), 2), dtype=numpy.int32)
        i = 0

        it_coord = self._final_context.final_pixels.begin()
        while it_coord != self._final_context.final_pixels.end():
            coord = dereference(it_coord)
            pixels[i, 0] = coord.y
            pixels[i, 1] = coord.x
            i += 1
            preincrement(it_coord)

        del self._final_context
        self._final_context = NULL

        return numpy.asarray(pixels)


cdef class MarchingSquaresMergeImpl(object):
    """
    Marching squares implementation based on a merge of segements and polygons.

    The main logic is based on the common marching squares algorithms.
    Segments of the iso-valued contours are identified using a pattern based
    on blocks of 2*2 pixels. The image is read sequencially and when a segment
    is identified it is inserted at a right place is a set of valid polygons.
    This process can grow up polygons on bounds, or merge polygons together.

    The algorithm can take care of a mask. If a pixel is invalidated by a
    non-zero value of the mask at it's location, the computation of the pattern
    cancelled and no segements are generated.

    This implementation based on merge allow to use divide and conquer
    implementation in multi process using OpenMP.The image is subdivised into
    many tiles, each one is processed independantly. The result is finally
    reduced by consecutives polygon merges.

    The OpenMP group size can also by used to skip part of the image using
    pre-computed informations. `use_minmax_cache` can enable the computation of
    minimum and maximum pixel levels available on each tile groups. It was
    designed to improve the efficiency of the extraction of many contour levels
    from the same gradient image.

    Finally the implementation provides an implementation to reach polygons
    (:meth:`find_contours`) or pixels (:meth:`find_pixels`) from the iso-valued
    data.

    .. code-block:: python

        # Example using a mask
        shape = 100, 100
        image = numpy.random.random(shape)
        mask = numpy.random.random(shape) < 0.01
        ms = MarchingSquaresMergeImpl(image, mask)
        polygons = ms.find_contours(level=0.5)
        for polygon in polygons:
            print(polygon)

    .. code-block:: python

        # Example using multi requests
        shape = 1000, 1000
        image = numpy.random.random(shape)
        ms = MarchingSquaresMergeImpl(image)
        levels = numpy.arange(0, 1, 0.05)
        for level in levels:
            polygons = ms.find_contours(level=level)

    .. code-block:: python

        # Efficient cache using multi requests
        shape = 1000, 1000
        image = numpy.arange(shape[0] * shape[1]) / (shape[0] * shape[1])
        image.shape = shape
        ms = MarchingSquaresMergeImpl(image, use_minmax_cache=True)
        levels = numpy.arange(0, 1, 0.05)
        for level in levels:
            polygons = ms.find_contours(level=level)

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

    cdef cnumpy.float32_t[:, ::1] _image
    cdef cnumpy.int8_t[:, ::1] _mask

    cdef cnumpy.float32_t *_image_ptr
    cdef cnumpy.int8_t *_mask_ptr
    cdef int _dim_x
    cdef int _dim_y
    cdef int _group_size
    cdef bool _use_minmax_cache

    cdef cnumpy.float32_t *_min_cache
    cdef cnumpy.float32_t *_max_cache

    cdef _MarchingSquaresContours _contours_algo
    cdef _MarchingSquaresPixels _pixels_algo

    def __init__(self,
                 image, mask=None,
                 group_size=256,
                 use_minmax_cache=False):
        if not isinstance(image, numpy.ndarray) or len(image.shape) != 2:
            raise ValueError("Only 2D arrays are supported.")
        if image.shape[0] < 2 or image.shape[1] < 2:
            raise ValueError("Input array must be at least 2x2.")
        # Force contiguous native array
        self._image = numpy.ascontiguousarray(image, dtype='=f4')
        self._image_ptr = &self._image[0][0]
        if mask is not None:
            if not isinstance(mask, numpy.ndarray):
                raise ValueError("Only 2D arrays are supported.")
            if image.shape != mask.shape:
                raise ValueError("Mask size and image size must be the same.")
            # Force contiguous native array
            self._mask = numpy.ascontiguousarray(mask, dtype='=i1')
            self._mask_ptr = &self._mask[0][0]
        else:
            self._mask = None
            self._mask_ptr = NULL
        self._group_size = group_size
        self._use_minmax_cache = use_minmax_cache
        self._min_cache = NULL
        self._max_cache = NULL
        with nogil:
            self._dim_y = self._image.shape[0]
            self._dim_x = self._image.shape[1]
        self._contours_algo = None
        self._pixels_algo = None

    def __dealloc__(self):
        if self._min_cache != NULL:
            libc.stdlib.free(self._min_cache)
        if self._max_cache != NULL:
            libc.stdlib.free(self._max_cache)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _compute_minmax_on_block(self, int block_x, int block_y, int block_index) nogil:
        """
        Initialize the minmax cache.

        The cache is computed for each tiles of the image. It reuses the OpenMP
        group size for the size of the tile, which allow to skip a full OpenMP
        context in case the requested level do not match the cache.

        The minmax is compuded with an overlap of 1 pixel, in order to match
        the marching squares algorithm.

        The mask is taking into accound. As result if a tile is fully masked,
        the minmax cache result for this tile will have infinit values.

        :param block_x: X location of tile in block unit
        :param block_y: Y location of tile in block unit
        :param block_index: Index of the tile in the minmax cache structure
        """
        cdef:
            int x, y
            int pos_x, end_x, pos_y, end_y
            cnumpy.float32_t minimum, maximum, value
            cnumpy.float32_t *image_ptr
            cnumpy.int8_t *mask_ptr

        pos_x = block_x * self._group_size
        end_x = pos_x + self._group_size + 1
        if end_x > self._dim_x:
            end_x = self._dim_x
        pos_y = block_y * self._group_size
        end_y = pos_y + self._group_size + 1
        if end_y > self._dim_y:
            end_y = self._dim_y

        image_ptr = self._image_ptr + (pos_y * self._dim_x + pos_x)
        if self._mask_ptr != NULL:
            mask_ptr = self._mask_ptr + (pos_y * self._dim_x + pos_x)
        else:
            mask_ptr = NULL
        minimum = INFINITY
        maximum = -INFINITY

        for y in range(pos_y, end_y):
            for x in range(pos_x, end_x):
                if mask_ptr != NULL:
                    if mask_ptr[0] != 0:
                        image_ptr += 1
                        mask_ptr += 1
                        continue
                value = image_ptr[0]
                if value < minimum:
                    minimum = value
                if value > maximum:
                    maximum = value
                image_ptr += 1
                if mask_ptr != NULL:
                    mask_ptr += 1
            image_ptr += self._dim_x + pos_x - end_x
            if mask_ptr != NULL:
                mask_ptr += self._dim_x + pos_x - end_x

        self._min_cache[block_index] = minimum
        self._max_cache[block_index] = maximum

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _create_minmax_cache(self) nogil:
        """
        Create and initialize minmax cache.
        """
        cdef:
            int icontext, context_x, context_y
            int context_dim_x, context_dim_y, context_size

        context_dim_x = self._dim_x // self._group_size + (self._dim_x % self._group_size > 0)
        context_dim_y = self._dim_y // self._group_size + (self._dim_y % self._group_size > 0)
        context_size = context_dim_x * context_dim_y

        self._min_cache = <cnumpy.float32_t *>libc.stdlib.malloc(context_size * sizeof(cnumpy.float32_t))
        self._max_cache = <cnumpy.float32_t *>libc.stdlib.malloc(context_size * sizeof(cnumpy.float32_t))

        for icontext in prange(context_size, nogil=True):
            context_x = icontext % context_dim_x
            context_y = icontext // context_dim_x
            self._compute_minmax_on_block(context_x, context_y, icontext)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def find_pixels(self, level):
        """
        Compute the pixels from the image over the requested iso contours
        at this `level`. Pixels are those over the bound of the segments.

        :param float level: Level of the requested iso contours.
        :returns: An array of y-x coordinates.
        :rtype: numpy.ndarray
        """
        if self._use_minmax_cache and self._min_cache == NULL:
            self._create_minmax_cache()

        if self._pixels_algo is None:
            algo = _MarchingSquaresPixels()
            algo._image_ptr = self._image_ptr
            algo._mask_ptr = self._mask_ptr
            algo._dim_x = self._dim_x
            algo._dim_y = self._dim_y
            algo._group_size = self._group_size
            algo._use_minmax_cache = self._use_minmax_cache
            algo._force_sequencial_reduction = COMPILED_WITH_OPENMP == 0
            if self._use_minmax_cache:
                algo._min_cache = self._min_cache
                algo._max_cache = self._max_cache
            self._pixels_algo = algo
        else:
            algo = self._pixels_algo

        algo.marching_squares(level)
        pixels = algo.extract_pixels()
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
        if self._use_minmax_cache and self._min_cache == NULL:
            self._create_minmax_cache()

        if self._contours_algo is None:
            algo = _MarchingSquaresContours()
            algo._image_ptr = self._image_ptr
            algo._mask_ptr = self._mask_ptr
            algo._dim_x = self._dim_x
            algo._dim_y = self._dim_y
            algo._group_size = self._group_size
            algo._use_minmax_cache = self._use_minmax_cache
            algo._force_sequencial_reduction = COMPILED_WITH_OPENMP == 0
            if self._use_minmax_cache:
                algo._min_cache = self._min_cache
                algo._max_cache = self._max_cache
            self._contours_algo = algo
        else:
            algo = self._contours_algo

        algo.marching_squares(level)
        polygons = algo.extract_polygons()
        return polygons
