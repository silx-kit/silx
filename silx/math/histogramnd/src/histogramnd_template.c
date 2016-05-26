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

#include "templates.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdarg.h>

#ifdef HISTO_SAMPLE_T
#ifdef HISTO_WEIGHT_T
#ifdef HISTO_CUMUL_T

int TEMPLATE(histogramnd, HISTO_SAMPLE_T, HISTO_WEIGHT_T, HISTO_CUMUL_T)
                        (HISTO_SAMPLE_T *i_sample,
                         HISTO_WEIGHT_T *i_weights,
                         int i_n_dim,
                         int i_n_elem,
                         double *i_bin_ranges,
                         int *i_n_bins,
                         uint32_t *o_histo,
                         HISTO_CUMUL_T *o_cumul,
                         double *o_bin_edges,
                         int i_opt_flags,
                         HISTO_WEIGHT_T i_weight_min,
                         HISTO_WEIGHT_T i_weight_max)
{
    /* some counters */
    int i = 0, j = 0;
    long elem_idx = 0;
    
    HISTO_WEIGHT_T * weight_ptr = 0;
    HISTO_SAMPLE_T elem_coord = 0.;
    
    /* computed bin index (i_sample -> grid) */
    long bin_idx = 0;
    
    double * g_min = 0;
    double * g_max = 0;
    double * range = 0;
    
    /* ================================
     * Parsing options, if any.
     * ================================
     */
    
    int filt_min_weight = 0;
    int filt_max_weight = 0;
    int last_bin_closed = 0;
    
    /* Testing the option flags */
    if(i_opt_flags & HISTO_WEIGHT_MIN)
    {
        filt_min_weight = 1;
    }
        
    if(i_opt_flags & HISTO_WEIGHT_MAX)
    {
        filt_max_weight = 1;
    }
        
    if(i_opt_flags & HISTO_LAST_BIN_CLOSED)
    {
        last_bin_closed = 1;
    }
    
    /* storing the min & max bin coordinates in their own arrays because
     * i_bin_ranges = [[min0, max0], [min1, max1], ...]
     * (mostly for the sake of clarity)
     * (maybe faster access too?)
     */
    g_min = (double *) malloc(i_n_dim *sizeof(double));
    g_max = (double *) malloc(i_n_dim * sizeof(double));
    /* range used to convert from i_coords to bin indices in the grid */
    range = (double *) malloc(i_n_dim * sizeof(double));
            
    if(!g_min || !g_max || !range)
    {
        free(g_min);
        free(g_max);
        free(range);
        return HISTO_ERR_ALLOC;
    }
    
    j = 0;
    for(i=0; i<i_n_dim; i++)
    {
        g_min[i] = i_bin_ranges[i*2];
        g_max[i] = i_bin_ranges[i*2+1];
        range[i] = g_max[i]-g_min[i];
        
        for(bin_idx=0; bin_idx<i_n_bins[i]; j++, bin_idx++)
        {
            o_bin_edges[j] = g_min[i] +
                            bin_idx * (range[i] / i_n_bins[i]);
        }
        o_bin_edges[j++] = g_max[i];
    }
    
    weight_ptr = i_weights;
    
    if(!i_weights)
    {
        /* if weights are not provided there no point in trying to filter them
         * (!! careful if you change this, some code below relies on it !!)
         */
        filt_min_weight = 0;
        filt_max_weight = 0;
        
        /* If the weights array is not provided then there is no point
         * updating the weighted histogram, only the bin counts (o_histo)
         * will be filled.
         * (!! careful if you change this, some code below relies on it !!)
         */
        o_cumul = 0;
    }
    
    /* tried to use pointers instead of indices here, but it didn't
     * seem any faster (probably because the compiler 
     * optimizes stuff anyway),
     * so i'm keeping the "indices" version, for the sake of clarity
    */
    for(elem_idx=0;
        elem_idx<i_n_elem*i_n_dim;
        elem_idx+=i_n_dim, weight_ptr++)
    {
        /* no testing the validity of weight_ptr here, because if it is NULL
         * then filt_min_weight/filt_max_weight will be 0.
         * (see code above)
         */
        if(filt_min_weight && *weight_ptr<i_weight_min)
        {
            continue;
        }
        if(filt_max_weight && *weight_ptr>i_weight_max)
        {
            continue;
        }

        bin_idx = 0;
        
        for(i=0; i<i_n_dim; i++)
        {
            elem_coord = i_sample[elem_idx+i];
            
            /* =====================
             * Element is rejected if any of the following is NOT true :
             * 1. coordinate is >= than the minimum value
             * 2. coordinate is <= than the maximum value
             * 3. coordinate==maximum value and last_bin_closed is True
             * =====================
             */
            if(elem_coord<g_min[i])
            {
                bin_idx = -1;
                break;
            }
            
            /* Here we make the assumption that most of the time
             * there will be more coordinates inside the grid interval
             *  (one test)
             *  than coordinates higher or equal to the max
             *  (two tests)
             */
            if(elem_coord<g_max[i])
            {
                /* Warning : the following factorization seems to
                 *  increase the effect of precision error.
                 * bin_idx = (long)floor(
                 *                   (bin_idx +
                 *                   (elem_coord-g_min[i])/range[i]) *
                 *               i_n_bins[i]
                 *           );
                 */
                
                /* Not using floor to speed up things.
                 * We don't (?) need all the error checking provided by
                 * the built-in floor().
                 * Also the value is supposed to be always positive.
                 */
                bin_idx = bin_idx * i_n_bins[i] +
                        (long)(
                                ((elem_coord-g_min[i]) * i_n_bins[i]) /
                                range[i]
                              );
            }
            else /* ===> elem_coord>=g_max[i] */
            {
                /* if equal and the last bin is closed :
                 *  put it in the last bin
                 * else : discard
                 */
                if(last_bin_closed && elem_coord==g_max[i])
                {
                    bin_idx = (bin_idx + 1) * i_n_bins[i] - 1;
                }
                else
                {
                    bin_idx = -1;
                    break;
                }
            } /* if(elem_coord<g_max[i]) */
            
        } /* for(i=0; i<i_n_dim; i++) */
        
        /* element is out of the grid */
        if(bin_idx==-1)
        {
            continue;
        }
        
        if(o_histo)
        {
            o_histo[bin_idx] += 1;
        }
        if(o_cumul)
        {
            /* not testing the pointer since o_cumul is null if 
             * i_weights is null. 
             */
            o_cumul[bin_idx] += (HISTO_CUMUL_T) *weight_ptr;
        }
        
    } /* for(elem_idx=0; elem_idx<i_n_elem*i_n_dim; elem_idx+=i_n_dim) */
    
    free(g_min);
    free(g_max);
    free(range);
    
    /* For now just returning 0 (OK) since all the checks are done in
     * python. This might change later if people want to call this
     * function directly from C (might have to implement error codes).
     */
    return HISTO_OK;
}

#endif
#endif
#endif
