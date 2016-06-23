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

#ifndef HISTOGRAMND_C_H
#define HISTOGRAMND_C_H

/* checking for MSVC version because VS 2008 doesnt fully support C99
   so inttypes.h and stdint.h are not provided with the compiler. */
#if defined(_MSC_VER) && _MSC_VER < 1600
    #include "msvc/stdint.h"
#else
    #include <inttypes.h>
#endif

#include "templates.h"

/** Allowed flag values for the i_opt_flags arguments. 
 */
typedef enum {
    HISTO_NONE              = 0,    /**< No options. */
    HISTO_WEIGHT_MIN        = 1,    /**< Filter weights with i_weight_min. */
    HISTO_WEIGHT_MAX        = 1<<1, /**< Filter weights with i_weight_max. */
    HISTO_LAST_BIN_CLOSED   = 1<<2  /**< Last bin is closed. */
} histo_opt_type;

/** Return codees for the histogramnd function. 
 */
typedef enum {
    HISTO_OK         = 0, /**< No error. */
    HISTO_ERR_ALLOC       /**< Failed to allocate memory. */
} histo_rc_t;

/*=====================
 * double sample, double cumul
 * ====================
*/

int histogramnd_double_double_double(double *i_sample,
                                     double *i_weigths,
                                     int i_n_dim,
                                     int i_n_elem,
                                     double *i_bin_ranges,
                                     int *i_n_bin,
                                     uint32_t *o_histo,
                                     double *o_cumul,
                                     double *o_bin_edges,
                                     int i_opt_flags,
                                     double i_weight_min,
                                     double i_weight_max);
                                
int histogramnd_double_float_double(double *i_sample,
                                    float *i_weigths,
                                    int i_n_dim,
                                    int i_n_elem,
                                    double *i_bin_ranges,
                                    int *i_n_bin,
                                    uint32_t *o_histo,
                                    double *o_cumul,
                                    double *o_bin_edges,
                                    int i_opt_flags,
                                    float i_weight_min,
                                    float i_weight_max);
                                
int histogramnd_double_int32_t_double(double *i_sample,
                                      int32_t *i_weigths,
                                      int i_n_dim,
                                      int i_n_elem,
                                      double *i_bin_ranges,
                                      int *i_n_bin,
                                      uint32_t *o_histo,
                                      double *o_cumul,
                                      double *o_bin_edges,
                                      int i_opt_flags,
                                      int32_t i_weight_min,
                                      int32_t i_weight_max);
                        
/*=====================
 * float sample, double cumul
 * ====================
*/
int histogramnd_float_double_double(float *i_sample,
                                    double *i_weigths,
                                    int i_n_dim,
                                    int i_n_elem,
                                    double *i_bin_ranges,
                                    int *i_n_bin,
                                    uint32_t *o_histo,
                                    double *o_cumul,
                                    double *o_bin_edges,
                                    int i_opt_flags,
                                    double i_weight_min,
                                    double i_weight_max);
                                
int histogramnd_float_float_double(float *i_sample,
                                   float *i_weigths,
                                   int i_n_dim,
                                   int i_n_elem,
                                   double *i_bin_ranges,
                                   int *i_n_bin,
                                   uint32_t *o_histo,
                                   double *o_cumul,
                                   double *o_bin_edges,
                                   int i_opt_flags,
                                   float i_weight_min,
                                   float i_weight_max);
                                
int histogramnd_float_int32_t_double(float *i_sample,
                                     int32_t *i_weigths,
                                     int i_n_dim,
                                     int i_n_elem,
                                     double *i_bin_ranges,
                                     int *i_n_bin,
                                     uint32_t *o_histo,
                                     double *o_cumul,
                                     double *o_bin_edges,
                                     int i_opt_flags,
                                     int32_t i_weight_min,
                                     int32_t i_weight_max);

/*=====================
 * int32_t sample, double cumul
 * ====================
*/
int histogramnd_int32_t_double_double(int32_t *i_sample,
                                      double *i_weigths,
                                      int i_n_dim,
                                      int i_n_elem,
                                      double *i_bin_ranges,
                                      int *i_n_bin,
                                      uint32_t *o_histo,
                                      double *o_cumul,
                                      double *o_bin_edges,
                                      int i_opt_flags,
                                      double i_weight_min,
                                      double i_weight_max);
                                
int histogramnd_int32_t_float_double(int32_t *i_sample,
                                     float *i_weigths,
                                     int i_n_dim,
                                     int i_n_elem,
                                     double *i_bin_ranges,
                                     int *i_n_bin,
                                     uint32_t *o_histo,
                                     double *o_cumul,
                                     double *o_bin_edges,
                                     int i_opt_flags,
                                     float i_weight_min,
                                     float i_weight_max);
                                
int histogramnd_int32_t_int32_t_double(int32_t *i_sample,
                                       int32_t *i_weigths,
                                       int i_n_dim,
                                       int i_n_elem,
                                       double *i_bin_ranges,
                                       int *i_n_bin,
                                       uint32_t *o_histo,
                                       double *o_cumul,
                                       double *o_bin_edges,
                                       int i_opt_flags,
                                       int32_t i_weight_min,
                                       int32_t i_weight_max);
                                       
/*=====================
 * double sample, float cumul
 * ====================
*/

int histogramnd_double_double_float(double *i_sample,
                                     double *i_weigths,
                                     int i_n_dim,
                                     int i_n_elem,
                                     double *i_bin_ranges,
                                     int *i_n_bin,
                                     uint32_t *o_histo,
                                     float *o_cumul,
                                     double *o_bin_edges,
                                     int i_opt_flags,
                                     double i_weight_min,
                                     double i_weight_max);
                                
int histogramnd_double_float_float(double *i_sample,
                                    float *i_weigths,
                                    int i_n_dim,
                                    int i_n_elem,
                                    double *i_bin_ranges,
                                    int *i_n_bin,
                                    uint32_t *o_histo,
                                    float *o_cumul,
                                    double *o_bin_edges,
                                    int i_opt_flags,
                                    float i_weight_min,
                                    float i_weight_max);
                                
int histogramnd_double_int32_t_float(double *i_sample,
                                      int32_t *i_weigths,
                                      int i_n_dim,
                                      int i_n_elem,
                                      double *i_bin_ranges,
                                      int *i_n_bin,
                                      uint32_t *o_histo,
                                      float *o_cumul,
                                      double *o_bin_edges,
                                      int i_opt_flags,
                                      int32_t i_weight_min,
                                      int32_t i_weight_max);
                        
/*=====================
 * float sample, float cumul
 * ====================
*/
int histogramnd_float_double_float(float *i_sample,
                                    double *i_weigths,
                                    int i_n_dim,
                                    int i_n_elem,
                                    double *i_bin_ranges,
                                    int *i_n_bin,
                                    uint32_t *o_histo,
                                    float *o_cumul,
                                    double *o_bin_edges,
                                    int i_opt_flags,
                                    double i_weight_min,
                                    double i_weight_max);
                                
int histogramnd_float_float_float(float *i_sample,
                                   float *i_weigths,
                                   int i_n_dim,
                                   int i_n_elem,
                                   double *i_bin_ranges,
                                   int *i_n_bin,
                                   uint32_t *o_histo,
                                   float *o_cumul,
                                   double *o_bin_edges,
                                   int i_opt_flags,
                                   float i_weight_min,
                                   float i_weight_max);
                                
int histogramnd_float_int32_t_float(float *i_sample,
                                     int32_t *i_weigths,
                                     int i_n_dim,
                                     int i_n_elem,
                                     double *i_bin_ranges,
                                     int *i_n_bin,
                                     uint32_t *o_histo,
                                     float *o_cumul,
                                     double *o_bin_edges,
                                     int i_opt_flags,
                                     int32_t i_weight_min,
                                     int32_t i_weight_max);

/*=====================
 * int32_t sample, double cumul
 * ====================
*/
int histogramnd_int32_t_double_float(int32_t *i_sample,
                                      double *i_weigths,
                                      int i_n_dim,
                                      int i_n_elem,
                                      double *i_bin_ranges,
                                      int *i_n_bin,
                                      uint32_t *o_histo,
                                      float *o_cumul,
                                      double *o_bin_edges,
                                      int i_opt_flags,
                                      double i_weight_min,
                                      double i_weight_max);
                                
int histogramnd_int32_t_float_float(int32_t *i_sample,
                                     float *i_weigths,
                                     int i_n_dim,
                                     int i_n_elem,
                                     double *i_bin_ranges,
                                     int *i_n_bin,
                                     uint32_t *o_histo,
                                     float *o_cumul,
                                     double *o_bin_edges,
                                     int i_opt_flags,
                                     float i_weight_min,
                                     float i_weight_max);
                                
int histogramnd_int32_t_int32_t_float(int32_t *i_sample,
                                       int32_t *i_weigths,
                                       int i_n_dim,
                                       int i_n_elem,
                                       double *i_bin_ranges,
                                       int *i_n_bin,
                                       uint32_t *o_histo,
                                       float *o_cumul,
                                       double *o_bin_edges,
                                       int i_opt_flags,
                                       int32_t i_weight_min,
                                       int32_t i_weight_max);
                        
#endif /* #define HISTOGRAMND_C_H */
