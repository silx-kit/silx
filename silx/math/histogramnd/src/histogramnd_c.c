/*##########################################################################
# Copyright (C) 2016 European Synchrotron Radiation Facility
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

#include "histogramnd_c.h"

/*=====================
 * double sample, double cumul
 * =====================
*/
#ifdef HISTO_SAMPLE_T
#undef HISTO_SAMPLE_T
#endif
#define HISTO_SAMPLE_T double
#ifdef HISTO_WEIGHT_T
#undef HISTO_WEIGHT_T
#endif
#define HISTO_WEIGHT_T double
#ifdef HISTO_CUMUL_T
#undef HISTO_CUMUL_T
#endif
#define HISTO_CUMUL_T double
#include "histogramnd_template.c"

#ifdef HISTO_SAMPLE_T
#undef HISTO_SAMPLE_T
#endif
#define HISTO_SAMPLE_T double
#ifdef HISTO_WEIGHT_T
#undef HISTO_WEIGHT_T
#endif
#define HISTO_WEIGHT_T float
#ifdef HISTO_CUMUL_T
#undef HISTO_CUMUL_T
#endif
#define HISTO_CUMUL_T double
#include "histogramnd_template.c"

#ifdef HISTO_SAMPLE_T
#undef HISTO_SAMPLE_T
#endif
#define HISTO_SAMPLE_T double
#ifdef HISTO_WEIGHT_T
#undef HISTO_WEIGHT_T
#endif
#define HISTO_WEIGHT_T int32_t
#ifdef HISTO_CUMUL_T
#undef HISTO_CUMUL_T
#endif
#define HISTO_CUMUL_T double
#include "histogramnd_template.c"

/*=====================
 * float sample, double cumul
 * =====================
*/
#ifdef HISTO_SAMPLE_T
#undef HISTO_SAMPLE_T
#endif
#define HISTO_SAMPLE_T float
#ifdef HISTO_WEIGHT_T
#undef HISTO_WEIGHT_T
#endif
#define HISTO_WEIGHT_T double
#ifdef HISTO_CUMUL_T
#undef HISTO_CUMUL_T
#endif
#define HISTO_CUMUL_T double
#include "histogramnd_template.c"

#ifdef HISTO_SAMPLE_T
#undef HISTO_SAMPLE_T
#endif
#define HISTO_SAMPLE_T float
#ifdef HISTO_WEIGHT_T
#undef HISTO_WEIGHT_T
#endif
#define HISTO_WEIGHT_T float
#ifdef HISTO_CUMUL_T
#undef HISTO_CUMUL_T
#endif
#define HISTO_CUMUL_T double
#include "histogramnd_template.c"

#ifdef HISTO_SAMPLE_T
#undef HISTO_SAMPLE_T
#endif
#define HISTO_SAMPLE_T float
#ifdef HISTO_WEIGHT_T
#undef HISTO_WEIGHT_T
#endif
#define HISTO_WEIGHT_T int32_t
#ifdef HISTO_CUMUL_T
#undef HISTO_CUMUL_T
#endif
#define HISTO_CUMUL_T double
#include "histogramnd_template.c"

/*=====================
 * int32_t sample, double cumul
 * =====================
*/
#ifdef HISTO_SAMPLE_T
#undef HISTO_SAMPLE_T
#endif
#define HISTO_SAMPLE_T int32_t
#ifdef HISTO_WEIGHT_T
#undef HISTO_WEIGHT_T
#endif
#define HISTO_WEIGHT_T double
#ifdef HISTO_CUMUL_T
#undef HISTO_CUMUL_T
#endif
#define HISTO_CUMUL_T double
#include "histogramnd_template.c"

#ifdef HISTO_SAMPLE_T
#undef HISTO_SAMPLE_T
#endif
#define HISTO_SAMPLE_T int32_t
#ifdef HISTO_WEIGHT_T
#undef HISTO_WEIGHT_T
#endif
#define HISTO_WEIGHT_T float
#ifdef HISTO_CUMUL_T
#undef HISTO_CUMUL_T
#endif
#define HISTO_CUMUL_T double
#include "histogramnd_template.c"

#ifdef HISTO_SAMPLE_T
#undef HISTO_SAMPLE_T
#endif
#define HISTO_SAMPLE_T int32_t
#ifdef HISTO_WEIGHT_T
#undef HISTO_WEIGHT_T
#endif
#define HISTO_WEIGHT_T int32_t
#ifdef HISTO_CUMUL_T
#undef HISTO_CUMUL_T
#endif
#define HISTO_CUMUL_T double
#include "histogramnd_template.c"


/*=====================
 * double sample, float cumul
 * =====================
*/
#ifdef HISTO_SAMPLE_T
#undef HISTO_SAMPLE_T
#endif
#define HISTO_SAMPLE_T double
#ifdef HISTO_WEIGHT_T
#undef HISTO_WEIGHT_T
#endif
#define HISTO_WEIGHT_T double
#ifdef HISTO_CUMUL_T
#undef HISTO_CUMUL_T
#endif
#define HISTO_CUMUL_T float
#include "histogramnd_template.c"

#ifdef HISTO_SAMPLE_T
#undef HISTO_SAMPLE_T
#endif
#define HISTO_SAMPLE_T double
#ifdef HISTO_WEIGHT_T
#undef HISTO_WEIGHT_T
#endif
#define HISTO_WEIGHT_T float
#ifdef HISTO_CUMUL_T
#undef HISTO_CUMUL_T
#endif
#define HISTO_CUMUL_T float
#include "histogramnd_template.c"

#ifdef HISTO_SAMPLE_T
#undef HISTO_SAMPLE_T
#endif
#define HISTO_SAMPLE_T double
#ifdef HISTO_WEIGHT_T
#undef HISTO_WEIGHT_T
#endif
#define HISTO_WEIGHT_T int32_t
#ifdef HISTO_CUMUL_T
#undef HISTO_CUMUL_T
#endif
#define HISTO_CUMUL_T float
#include "histogramnd_template.c"

/*=====================
 * float sample, float cumul
 * =====================
*/
#ifdef HISTO_SAMPLE_T
#undef HISTO_SAMPLE_T
#endif
#define HISTO_SAMPLE_T float
#ifdef HISTO_WEIGHT_T
#undef HISTO_WEIGHT_T
#endif
#define HISTO_WEIGHT_T double
#ifdef HISTO_CUMUL_T
#undef HISTO_CUMUL_T
#endif
#define HISTO_CUMUL_T float
#include "histogramnd_template.c"

#ifdef HISTO_SAMPLE_T
#undef HISTO_SAMPLE_T
#endif
#define HISTO_SAMPLE_T float
#ifdef HISTO_WEIGHT_T
#undef HISTO_WEIGHT_T
#endif
#define HISTO_WEIGHT_T float
#ifdef HISTO_CUMUL_T
#undef HISTO_CUMUL_T
#endif
#define HISTO_CUMUL_T float
#include "histogramnd_template.c"

#ifdef HISTO_SAMPLE_T
#undef HISTO_SAMPLE_T
#endif
#define HISTO_SAMPLE_T float
#ifdef HISTO_WEIGHT_T
#undef HISTO_WEIGHT_T
#endif
#define HISTO_WEIGHT_T int32_t
#ifdef HISTO_CUMUL_T
#undef HISTO_CUMUL_T
#endif
#define HISTO_CUMUL_T float
#include "histogramnd_template.c"

/*=====================
 * int32_t sample, float cumul
 * =====================
*/
#ifdef HISTO_SAMPLE_T
#undef HISTO_SAMPLE_T
#endif
#define HISTO_SAMPLE_T int32_t
#ifdef HISTO_WEIGHT_T
#undef HISTO_WEIGHT_T
#endif
#define HISTO_WEIGHT_T double
#ifdef HISTO_CUMUL_T
#undef HISTO_CUMUL_T
#endif
#define HISTO_CUMUL_T float
#include "histogramnd_template.c"

#ifdef HISTO_SAMPLE_T
#undef HISTO_SAMPLE_T
#endif
#define HISTO_SAMPLE_T int32_t
#ifdef HISTO_WEIGHT_T
#undef HISTO_WEIGHT_T
#endif
#define HISTO_WEIGHT_T float
#ifdef HISTO_CUMUL_T
#undef HISTO_CUMUL_T
#endif
#define HISTO_CUMUL_T float
#include "histogramnd_template.c"

#ifdef HISTO_SAMPLE_T
#undef HISTO_SAMPLE_T
#endif
#define HISTO_SAMPLE_T int32_t
#ifdef HISTO_WEIGHT_T
#undef HISTO_WEIGHT_T
#endif
#define HISTO_WEIGHT_T int32_t
#ifdef HISTO_CUMUL_T
#undef HISTO_CUMUL_T
#endif
#define HISTO_CUMUL_T float
#include "histogramnd_template.c"
