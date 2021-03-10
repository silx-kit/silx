# /*##########################################################################
#
# Copyright (c) 2018 European Synchrotron Radiation Facility
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
# ###########################################################################*/

/** This header provides static lookup table used by marching square
 * algorithms.
 */

#ifndef __MARCHINGSQUARE_PATTERNS_H__
#define __MARCHINGSQUARE_PATTERNS_H__

/**
 * Array containing pixel's coordinate linked together by an edge index.
 */
const unsigned char EDGE_TO_POINT[][2] = {
	{0, 0},
	{1, 0},
	{1, 1},
	{0, 1},
	{0, 0}
};

/**
 * Array of index containing 5 values: {nb, seg1b, seg2e, seg2b, seg2e}
 * nb: number of segments (up to 2)
 * seg1b: index of the start of the 1st edge
 * seg1e: index of the end of the 1st edge
 * seg2b: index of the start of the 2nd edge
 * seg2e: index of the end of the 2nd edge
 */
const unsigned char CELL_TO_EDGE[][5] = {
	{0, 0, 0, 0, 0},  // Case 0: 0000: nothing
	{1, 0, 3, 0, 0},  // Case 1: 0001
	{1, 0, 1, 0, 0},  // Case 2: 0010
	{1, 1, 3, 0, 0},  // Case 3: 0011

	{1, 1, 2, 0, 0},  // Case 4: 0100
	{2, 0, 1, 2, 3},  // Case 5: 0101 > ambiguous
	{1, 0, 2, 0, 0},  // Case 6: 0110
	{1, 2, 3, 0, 0},  // Case 7: 0111

	{1, 2, 3, 0, 0},  // Case 8: 1000
	{1, 0, 2, 0, 0},  // Case 9: 1001
	{2, 0, 3, 1, 2},  // Case 10: 1010 > ambiguous
	{1, 1, 2, 0, 0},  // Case 11: 1011

	{1, 1, 3, 0, 0},  // Case 12: 1100
	{1, 0, 1, 0, 0},  // Case 13: 1101
	{1, 0, 3, 0, 0},  // Case 14: 1110
	{0, 0, 0, 0, 0},  // Case 15: 1111
};


typedef struct coord_t {
	short x;
	short y;

	bool operator<(const coord_t& other) const {
	    if (y < other.y) {
	        return true;
	    } else if (y == other.y) {
	        return x < other.x;
	    } else {
	        return false;
	    }
	}
} coord_t;

#endif /*__MARCHINGSQUARE_PATTERNS_H__*/
