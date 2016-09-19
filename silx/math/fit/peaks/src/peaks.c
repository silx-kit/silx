#/*##########################################################################
# Copyright (c) 2004-2016 European Synchrotron Radiation Facility
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
#############################################################################*/
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "peaks.h"


#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))

/* Peak search function, adapted from PyMca SpecFitFuns

   This uses a convolution with the second-derivative of a gaussian curve, to
   smooth the data.

   Arguments:

      - begin_index: First index of the region of interest in the input data
         array
      - end_index: Last index of the region of interest in the input data
         array
      - nsamples: Number of samples in the input array
      - fwhm: Full width at half maximum for the gaussian used for smoothing.
      - sensitivity:
      - debug_info: If different from 0, print debugging messages
      - data: input array of 1D data
      - peaks: pointer to output array of peak indices
      - relevances: pointer to output array of peak relevances
*/
long seek(long begin_index,
           long end_index,
           long nsamples,
           double fwhm,
           double sensitivity,
           double debug_info,
           double *data,
           double **peaks,
           double **relevances)
{
    /* local variables */
    double *peaks0, *relevances0;
    double *realloc_peaks, *realloc_relevances;
    double  sigma, sigma2, sigma4;
    long    max_gfactor = 100;
    double  gfactor[100];
    long    nr_factor;
    double  lowthreshold;
    double  data2[2];
    double  nom;
    double  den2;
    long    channel1;
    long    lld;
    long    cch;
    long    cfac, cfac2, max_cfac;
    long    ihelp1, ihelp2;
    long    i;
    long    max_npeaks = 100;
    long    n_peaks = 0;
    double  peakstarted = 0;

    peaks0 = malloc(100 * sizeof(double));
    relevances0 = malloc(100 * sizeof(double));
    if (peaks0 == NULL || relevances0 == NULL) {
        printf("Error: failed to allocate memory for peaks array.");
        return(-123456);
    }
    /* Make sure the peaks matrix is filled with zeros */
    for (i=0;i<100;i++){
        peaks0[i]      = 0.0;
        relevances0[i] = 0.0;
    }
    /* Output pointers */
    *peaks = peaks0;
    *relevances = relevances0;

    /* prepare the calculation of the Gaussian scaling factors */

    sigma = fwhm / 2.35482;
    sigma2 = sigma * sigma;
    sigma4 = sigma2 * sigma2;
    lowthreshold = 0.01 / sigma2;

    /* calculate the factors until lower threshold reached */
    nr_factor = 0;
    max_cfac = MIN(max_gfactor, ((end_index - begin_index - 2) / 2) - 1);
    for (cfac=0; cfac < max_cfac; cfac++) {
        nr_factor++;
        cfac2 = (cfac+1) * (cfac+1);
        gfactor[cfac] = (sigma2 - cfac2) * exp(-cfac2/(sigma2*2.0)) / sigma4;

        if ((gfactor[cfac] < lowthreshold)
           && (gfactor[cfac] > (-lowthreshold))){
            break;
        }
    }

    /* What comes now is specific to MCA spectra ... */
    lld = 0;
    while (data[lld] == 0) {
        lld++;
    }
    lld = lld + (int) (0.5 * fwhm);

    channel1 = begin_index - nr_factor - 1;
    channel1 = MAX (channel1, lld);
    if(debug_info){
        printf("nrfactor  = %ld\n", nr_factor);
    }
    /* calculates smoothed value and variance at begincalc */
    cch = MAX(begin_index, 0);
    nom = data[cch] / sigma2;
    den2 = data[cch] / sigma4;
    for (cfac = 0; cfac < nr_factor; cfac++){
        ihelp1 = cch-cfac;
        if (ihelp1 < 0){
            ihelp1 = 0;
        }
        ihelp2 = cch+cfac;
        if (ihelp2 >= nsamples){
            ihelp2 = nsamples-1;
        }
        nom += gfactor[cfac] * (data[ihelp2] + data[ihelp1]);
        den2 += gfactor[cfac] * gfactor[cfac] *
                 (data[ihelp2] + data[ihelp1]);
    }

    /* now normalize the smoothed value to the standard deviation */
    if (den2 <= 0.0) {
        data2[1] = 0.0;
    }else{
        data2[1] = nom / sqrt(den2);
    }
    data[0] = data[1];

    while (cch <= MIN(end_index,nsamples-2)){
        /* calculate gaussian smoothed values */
        data2[0] = data2[1];
        cch++;
        nom = data[cch]/sigma2;
        den2 = data[cch] / sigma4;
        for (cfac = 1; cfac < nr_factor; cfac++){
            ihelp1 = cch-cfac;
            if (ihelp1 < 0){
                ihelp1 = 0;
            }
            ihelp2 = cch+cfac;
            if (ihelp2 >= nsamples){
                ihelp2 = nsamples-1;
            }
            nom += gfactor[cfac-1] * (data[ihelp2] + data[ihelp1]);
            den2 += gfactor[cfac-1] * gfactor[cfac-1] *
                     (data[ihelp2] + data[ihelp1]);
        }
        /* now normalize the smoothed value to the standard deviation */
        if (den2 <= 0) {
            data2[1] = 0;
        }else{
            data2[1] = nom / sqrt(den2);
        }
        /* look if the current point falls in a peak */
        if (data2[1] > sensitivity) {
            if(peakstarted == 0){
                if (data2[1] > data2[0]){
                    /* this second test is to prevent a peak from outside
                    the region from being detected at the beginning of the search */
                   peakstarted=1;
                }
            }
            /* there is a peak */
            if (debug_info){
                printf("At cch = %ld y[cch] = %g\n", cch, data[cch]);
                printf("data2[0] = %g\n", data2[0]);
                printf("data2[1] = %g\n", data2[1]);
                printf("sensitivity = %g\n", sensitivity);
            }
            if(peakstarted == 1){
                /* look for the top of the peak */
                if (data2[1] < data2[0]) {
                    /* we are close to the top of the peak */
                    if (debug_info){
                        printf("we are close to the top of the peak\n");
                    }
                    if (n_peaks == max_npeaks) {
                        max_npeaks = max_npeaks + 100;
                        realloc_peaks = realloc(peaks0, max_npeaks * sizeof(double));
                        realloc_relevances = realloc(relevances0, max_npeaks * sizeof(double));
                        if (realloc_peaks == NULL || realloc_relevances == NULL) {
                            printf("Error: failed to extend memory for peaks array.");
                            *peaks = peaks0;
                            *relevances = relevances0;
                            return(-n_peaks);
                        }
                        else {
                            peaks0 = realloc_peaks;
                            relevances0 = realloc_relevances;
                        }
                    }
                    peaks0[n_peaks] = cch-1;
                    relevances0[n_peaks] = data2[0];
                    n_peaks++;
                    peakstarted=2;
                }
            }
            /* Doublet case */
            if(peakstarted == 2){
                if ((cch-peaks0[n_peaks-1]) > 0.6 * fwhm) {
                    if (data2[1] > data2[0]){
                        if(debug_info){
                            printf("We may have a doublet\n");
                        }
                        peakstarted=1;
                    }
                }
            }
        }else{
            if (peakstarted==1){
            /* We were on a peak but we did not find the top */
                if(debug_info){
                    printf("We were on a peak but we did not find the top\n");
                }
            }
            peakstarted=0;
        }
    }
    if(debug_info){
      for (i=0;i< n_peaks;i++){
        printf("Peak %ld found at ",i+1);
        printf("index %g with y = %g\n", peaks0[i],data[(long ) peaks0[i]]);
      }
    }
    *peaks = peaks0;
    *relevances = relevances0;
    return (n_peaks);
}
