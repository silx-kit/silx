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
#include "functions.h"


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
      - max_npeaks: maximum nubmber of peaks. If the algorithm finds more
         peaks than this parameter allows for, it will fail due to insufficient
         memory being allocated to the output arrays.
      - data: input array of 1D data
      - peaks: output array of peak indices
      - relevances: output array of peak relevances
*/
long seek(long begin_index,
           long end_index,
           long nsamples,
           double fwhm,
           double sensitivity,
           double debug_info,
           long max_npeaks,
           double *data,
           double *peaks,
           double *relevances)
{
    /* local variables */
    double  sigma, sigma2, sigma4;
    long    max_gfactor = 100;
    double  gfactor[100];
    long    nr_factor;
    double  sum_factors;
    double  lowthreshold;
    double  data2[2];
    double  nom;
    double  den2;
    long    begincalc, endcalc;
    long    channel1;
    long    lld;
    long    cch;
    long    cfac, cfac2;
    long    ihelp1, ihelp2;
    long    i, j;
    long    n_peaks = 0;
    double  peakstarted = 0;

    /* Make sure the peaks matrix is filled with zeros */
    for (i=0;i<max_npeaks;i++){
        peaks[i]      = 0.0;
        relevances[i] = 0.0;
    }

    /* prepare the calculation of the Gaussian scaling factors */

    sigma = fwhm / 2.35482;
    sigma2 = sigma * sigma;
    sigma4 = sigma2 * sigma2;
    lowthreshold = 0.01 / sigma2;
    sum_factors = 0.0;

    /* calculate the factors until lower threshold reached */
    j = MIN(max_gfactor, ((end_index - begin_index -2)/2)-1);
    for (cfac=1; cfac < j+1; cfac++) {
        cfac2 = cfac * cfac;
        gfactor[cfac-1] = (sigma2 - cfac2) * exp(-cfac2/(sigma2*2.0)) / sigma4;
        sum_factors += gfactor[cfac-1];

        if ((gfactor[cfac-1] < lowthreshold)
           && (gfactor[cfac-1] > (-lowthreshold))){
            break;
        }
    }

    nr_factor = cfac;

    /* What comes now is specific to MCA spectra ... */
    lld = 0;
    while (data[lld] == 0) {
        lld++;
    }
    lld = lld + (int) (0.5 * fwhm);

    channel1 = begin_index - nr_factor - 1;
    channel1 = MAX (channel1, lld);
    begincalc = channel1+nr_factor+1;
    endcalc = MIN (end_index+nr_factor+1, nsamples-nr_factor-1);
    cch = begincalc;
    if(debug_info){
        printf("nrfactor  = %ld\n", nr_factor);
        printf("begincalc = %ld\n", begincalc);
        printf("endcalc   = %ld\n", endcalc);
    }
    /* calculates smoothed value and variance at begincalc */
    cch = MAX(begin_index,0);
    nom = data[cch] / sigma2;
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
                printf("At cch = %ld y[cch] = %g\n",cch,data[cch]);
                printf("data2[0] = %g\n",data2[0]);
                printf("data2[1] = %g\n",data2[1]);
                printf("sensitivity = %g\n",sensitivity);
            }
            if(peakstarted == 1){
                /* look for the top of the peak */
                if (data2[1] < data2[0]) {
                    /* we are close to the top of the peak */
                    if (debug_info){
                        printf("we are close to the top of the peak\n");
                    }
                    if (n_peaks < max_npeaks) {
                        peaks[n_peaks] = cch-1;
                        relevances[n_peaks] = data2[0];
                        n_peaks++;
                        peakstarted=2;
                    }else{
                        printf("Found too many peaks\n");
                        return (-2);
                    }
                }
            }
            /* Doublet case */
            if(peakstarted == 2){
                if ((cch-peaks[n_peaks-1]) > 0.6 * fwhm) {
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
        printf("index %g with y = %g\n",peaks[i],data[(long ) peaks[i]]);
      }
    }
    return (n_peaks);
}
