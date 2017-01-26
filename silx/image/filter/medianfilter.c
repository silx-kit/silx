/* This is a shameless copy with minor mofifications of the medianfilter
 * provided with scipy. Therefore is distributed under the terms of the
 * scipy license.
 *
 * The purpose of having it separately is not to introduce a dependency
 * on scipy that is big and potentially difficult to built on some
 * platforms.
 *
 * Using this code outside PyMca:
 *
 * The check_malloc function has to be provided for error handling.
 *
 *--------------------------------------------------------------------*/
/* Subset of SIGTOOLS module by Travis Oliphant

Copyright 2005 Travis Oliphant
Permission to use, copy, modify, and distribute this software without fee
is granted under the SciPy License.

Copyright (c) 2001, 2002 Enthought, Inc.
All rights reserved.

Copyright (c) 2003-2009 SciPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  a. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.
  b. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
  c. Neither the name of the Enthought nor the names of its contributors
     may be used to endorse or promote products derived from this software
     without specific prior written permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
*/
#include <stdlib.h>

extern char *check_malloc (int);

/* defined below */
void f_medfilt2(float*,float*,int*,int*, int);
void d_medfilt2(double*,double*,int*,int*, int);
void b_medfilt2(unsigned char*,unsigned char*,int*,int*,int);
void short_medfilt2(short*, short*,int*,int*,int);
void ushort_medfilt2(unsigned short*,unsigned short*,int*,int*,int);
void int_medfilt2(int*, int*,int*,int*,int);
void uint_medfilt2(unsigned int*,unsigned int*,int*,int*,int);
void long_medfilt2(long*, long*,int*,int*,int);
void ulong_medfilt2(unsigned long*,unsigned long*,int*,int*,int);

/* The QUICK_SELECT routine is based on Hoare's Quickselect algorithm,
 * with unrolled recursion.
 * Author: Thouis R. Jones, 2008
 */

#define ELEM_SWAP(t, a, x, y) {register t temp = (a)[x]; (a)[x] = (a)[y]; (a)[y] = temp;}
#define FIRST_LOWEST(x, y, z) (((x) < (y)) && ((x) < (z)))
#define FIRST_HIGHEST(x, y, z) (((x) > (y)) && ((x) > (z)))
#define LOWEST_IDX(a, x, y) (((a)[x] < (a)[y]) ? (x) : (y))
#define HIGHEST_IDX(a, x, y) (((a)[x] > (a)[y]) ? (x) : (y))

/* if (l is index of lowest) {return lower of mid,hi} else if (l is index of highest) {return higher of mid,hi} else return l */
#define MEDIAN_IDX(a, l, m, h) (FIRST_LOWEST((a)[l], (a)[m], (a)[h]) ? LOWEST_IDX(a, m, h) : (FIRST_HIGHEST((a)[l], (a)[m], (a)[h]) ? HIGHEST_IDX(a, m, h) : (l)))

#define QUICK_SELECT(NAME, TYPE)                                        \
TYPE NAME(TYPE arr[], int n)                                            \
{                                                                       \
    int lo, hi, mid, md;                                                \
    int median_idx;                                                     \
    int ll, hh;                                                         \
    TYPE piv;                                                           \
                                                                        \
    lo = 0; hi = n-1;                                                   \
    median_idx = (n - 1) / 2; /* lower of middle values for even-length arrays */ \
                                                                        \
    while (1) {                                                         \
        if ((hi - lo) < 2) {                                            \
            if (arr[hi] < arr[lo]) ELEM_SWAP(TYPE, arr, lo, hi);        \
            return arr[median_idx];                                     \
        }                                                               \
                                                                        \
        mid = (hi + lo) / 2;                                            \
        /* put the median of lo,mid,hi at position lo - this will be the pivot */ \
        md = MEDIAN_IDX(arr, lo, mid, hi);                              \
        ELEM_SWAP(TYPE, arr, lo, md);                                   \
                                                                        \
        /* Nibble from each end towards middle, swapping misordered items */ \
        piv = arr[lo];                                                  \
        for (ll = lo+1, hh = hi;; ll++, hh--) {                         \
        while (arr[ll] < piv) ll++;                                     \
        while (arr[hh] > piv) hh--;                                     \
        if (hh < ll) break;                                             \
        ELEM_SWAP(TYPE, arr, ll, hh);                                   \
        }                                                               \
        /* move pivot to top of lower partition */                      \
        ELEM_SWAP(TYPE, arr, hh, lo);                                   \
        /* set lo, hi for new range to search */                        \
        if (hh < median_idx) /* search upper partition */               \
            lo = hh+1;                                                  \
        else if (hh > median_idx) /* search lower partition */          \
            hi = hh-1;                                                  \
        else                                                            \
            return piv;                                                 \
    }                                                                   \
}

/* 2-D median filter with zero-padding on edges. */
#define MEDIAN_FILTER_2D(NAME, TYPE, SELECT)                            \
void NAME(TYPE* in, TYPE* out, int* Nwin, int* Ns, int flag)            \
{                                                                       \
    /* if flag is not 0, implements a conditional filter */             \
    int nx, ny, hN[2];                                                  \
    int pre_x, pre_y, pos_x, pos_y;                                     \
    int subx, suby, k, totN;                                            \
    TYPE *myvals, *fptr1, *fptr2, *ptr1, *ptr2, minval=0, maxval=0;     \
                                                                        \
    totN = Nwin[0] * Nwin[1];                                           \
    myvals = (TYPE *) check_malloc( totN * sizeof(TYPE));               \
                                                                        \
    hN[0] = Nwin[0] >> 1;                                               \
    hN[1] = Nwin[1] >> 1;                                               \
    ptr1 = in;                                                          \
    fptr1 = out;                                                        \
    for (ny = 0; ny < Ns[0]; ny++)                                      \
        for (nx = 0; nx < Ns[1]; nx++) {                                \
            pre_x = hN[1];                                              \
            pre_y = hN[0];                                              \
            pos_x = hN[1];                                              \
            pos_y = hN[0];                                              \
            if (nx < hN[1]) pre_x = nx;                                 \
            if (nx >= Ns[1] - hN[1]) pos_x = Ns[1] - nx - 1;            \
            if (ny < hN[0]) pre_y = ny;                                 \
            if (ny >= Ns[0] - hN[0]) pos_y = Ns[0] - ny - 1;            \
            fptr2 = myvals;                                             \
            ptr2 = ptr1 - pre_x - pre_y*Ns[1];                          \
            if (flag){                                                  \
                minval = maxval = *ptr1;                                \
                for (suby = -pre_y; suby <= pos_y; suby++) {            \
                    for (subx = -pre_x; subx <= pos_x; subx++){         \
                        minval = (*ptr2 < minval) ? *ptr2 : minval;     \
                        maxval = (*ptr2 > maxval) ? *ptr2 : maxval;     \
                        *fptr2++ = *ptr2++;                             \
                    }                                                   \
                    ptr2 += Ns[1] - (pre_x + pos_x + 1);                \
                }                                                       \
            }else{                                                      \
                for (suby = -pre_y; suby <= pos_y; suby++) {            \
                    for (subx = -pre_x; subx <= pos_x; subx++)          \
                        *fptr2++ = *ptr2++;                             \
                    ptr2 += Ns[1] - (pre_x + pos_x + 1);                \
                }                                                       \
            }                                                           \
            if ((flag == 0) || (*ptr1 == minval) || (*ptr1 == maxval)){ \
                ptr1++;                                                 \
                                                                        \
                k = (pre_x + pos_x + 1)*(pre_y + pos_y + 1);            \
                /* Prefer a shrinking window to zero padding */         \
                if (k > totN){                                          \
                    k = totN;                                           \
                }                                                       \
                *fptr1++ = SELECT(myvals, k);                           \
                /* Zero pad alternative*/                               \
                /*for ( ; k < totN; k++)                                \
                    *fptr2++ = 0;                                       \
                                                                        \
                *fptr1++ = SELECT(myvals,totN); */                      \
            }else{                                                      \
                *fptr1++ = *ptr1++;                                     \
            }                                                           \
        }                                                               \
    free(myvals);                                                       \
}

/* define quick_select for floats, doubles, and unsigned characters */
QUICK_SELECT(f_quick_select, float)
QUICK_SELECT(d_quick_select, double)
QUICK_SELECT(b_quick_select, unsigned char)

/*define quick_select for rest of common types */

QUICK_SELECT(short_quick_select, short);
QUICK_SELECT(ushort_quick_select, unsigned short);
QUICK_SELECT(int_quick_select, int);
QUICK_SELECT(uint_quick_select, unsigned int);
QUICK_SELECT(long_quick_select, long);
QUICK_SELECT(ulong_quick_select, unsigned long);

/* define medfilt for floats, doubles, and unsigned characters */
MEDIAN_FILTER_2D(f_medfilt2, float, f_quick_select)
MEDIAN_FILTER_2D(d_medfilt2, double, d_quick_select)
MEDIAN_FILTER_2D(b_medfilt2, unsigned char, b_quick_select)

/* define medfilt for rest of common types */
MEDIAN_FILTER_2D(short_medfilt2, short, short_quick_select)
MEDIAN_FILTER_2D(ushort_medfilt2, unsigned short, ushort_quick_select)
MEDIAN_FILTER_2D(int_medfilt2, int, int_quick_select)
MEDIAN_FILTER_2D(uint_medfilt2, unsigned int, uint_quick_select)
MEDIAN_FILTER_2D(long_medfilt2, long, long_quick_select)
MEDIAN_FILTER_2D(ulong_medfilt2, unsigned long, ulong_quick_select)

