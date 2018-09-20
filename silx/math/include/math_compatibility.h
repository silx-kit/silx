# /*##########################################################################
#
# Copyright (c) 2017-2018 European Synchrotron Radiation Facility
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
/* This header provides libc math functions and macros across platforms.

   Needed as VisualStudio 2008 (i.e., Python2.7) is missing some functions/macros.
*/

#ifndef __MATH_COMPATIBILITY_H__
#define __MATH_COMPATIBILITY_H__

#include <math.h>

#ifndef INFINITY
#define INFINITY (DBL_MAX+DBL_MAX)
#endif

#ifndef NAN
#define NAN (INFINITY-INFINITY)
#endif

#if (defined (_MSC_VER) && _MSC_VER < 1800)
#include <float.h>

/* Make sure asinh returns -inf rather than NaN for v=-inf */
#define asinh(v) (v == -INFINITY ? v : log((v) + sqrt((v)*(v) + 1)))

#define isnan(v) _isnan(v)
#define isfinite(v) _finite(v)
#define lrint(v) ((long int) (v))
#endif

#endif /*__MATH_COMPATIBILITY_H__*/
