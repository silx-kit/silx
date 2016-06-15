#/*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2004-2014 European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
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
/*

*/
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "fitfunctions.h"

#ifndef M_PI
#define M_PI 3.1415926535
#endif

#define LOG2  0.69314718055994529

/*  sum_gauss
    Sum of gaussian functions, defined by (height, centroid, fwhm)

    *height* is the peak amplitude
    *centroid* is the peak x-coordinate
    *fwhm* is the full-width at half maximum

    Parameters:
    -----------

        - x: Independant variable where the gaussians are calculated.
        - len_x: Number of elements in the x array.
        - pvoigt: Array of gaussian parameters:
          (height1, centroid1, fwhm1, height2, centroid2, fwhm2,...)
        - len_pgauss: Number of elements in the pgauss array. Must be
          a multiple of 3.
        - y: Output array. Must have memory allocated for the same number
          of elements as x (len_x).

*/
void sum_gauss(double* x, int len_x, double* pgauss, int len_pgauss, double* y)
{
    int i, j;
    double dhelp, inv_two_sqrt_two_log2, sigma;
    double fwhm, centroid, height;

    if (len_pgauss % 3 || len_pgauss == 0) {
        printf("[sum_gauss]Error: Number of parameters must be a multiple of 3 (height1, centroid1, fwhm1, ...)\n");
        return;
    }

    /* Initialize output array */
    for (j=0; j<len_x;  j++) {
        y[j] = 0.;
    }

    inv_two_sqrt_two_log2 = 1.0 / (2.0 * sqrt(2.0 * LOG2));

    for (i=0; i<len_pgauss/3; i++) {
        height = pgauss[3*i];
        centroid = pgauss[3*i+1];
        fwhm = pgauss[3*i+2];

        sigma = fwhm * inv_two_sqrt_two_log2;

        for (j=0; j<len_x;  j++) {
            dhelp = (x[j] - centroid) / sigma;
            if (dhelp <= 20) {
                y[j] += height * exp (-0.5 * dhelp * dhelp);
            }
        }
    }
}

/*  sum_agauss
    Sum of gaussian functions defined by (area, centroid, fwhm)

    *area* is the area underneath the peak
    *centroid* is the peak x-coordinate
    *fwhm* is the full-width at half maximum

    Parameters:
    -----------

        - x: Independant variable where the gaussians are calculated.
        - len_x: Number of elements in the x array.
        - pgauss: Array of gaussian parameters:
          (area1, centroid1, fwhm1, area2, centroid2, fwhm2,...)
        - len_pgauss: Number of elements in the pgauss array. Must be
          a multiple of 3.
        - y: Output array. Must have memory allocated for the same number
          of elements as x (len_x).

*/
void sum_agauss(double* x, int len_x, double* pgauss, int len_pgauss, double* y)
{
    int i, j;
    double  dhelp, height, sqrt2PI, sigma, inv_two_sqrt_two_log2;
    double fwhm, centroid, area;

    if (len_pgauss % 3 || len_pgauss == 0) {
        printf("[sum_agauss]Error: Number of parameters must be a multiple of 3 (area1, centroid1, fwhm1, ...)\n");
        return;
    }

    /* Initialize output array */
    for (j=0; j<len_x;  j++) {
        y[j] = 0.;
    }

    inv_two_sqrt_two_log2 = 1.0 / (2.0 * sqrt(2.0 * LOG2));
    sqrt2PI = sqrt(2.0*M_PI);

    for (i=0; i<len_pgauss/3; i++) {
        area = pgauss[3*i];
        centroid = pgauss[3*i+1];
        fwhm = pgauss[3*i+2];

        sigma = fwhm * inv_two_sqrt_two_log2;
        height = area / (sigma * sqrt2PI);

        for (j=0; j<len_x;  j++) {
            dhelp = (x[j] - centroid)/sigma;
            if (dhelp <= 35) {
                y[j] += height * exp (-0.5 * dhelp * dhelp);
            }
        }
    }
}


/*  sum_fastagauss
    Sum of gaussian functions defined by (area, centroid, fwhm).
    This implementation uses a lookup table of precalculated exp values
    and a limited development (exp(-x) = 1 - x for small values of x)

    *area* is the area underneath the peak
    *centroid* is the peak x-coordinate
    *fwhm* is the full-width at half maximum

    Parameters:
    -----------

        - x: Independant variable where the gaussians are calculated.
        - len_x: Number of elements in the x array.
        - pgauss: Array of gaussian parameters:
          (area1, centroid1, fwhm1, area2, centroid2, fwhm2,...)
        - len_pgauss: Number of elements in the pgauss array. Must be
          a multiple of 3.
        - y: Output array. Must have memory allocated for the same number
          of elements as x (len_x).

*/

void sum_fastagauss(double* x, int len_x, double* pgauss, int len_pgauss, double* y)
{
    int i, j, expindex;
    double  dhelp, height, sqrt2PI, sigma, inv_two_sqrt_two_log2;
    double fwhm, centroid, area;
    static double EXP[5000];

    if (len_pgauss % 3 || len_pgauss == 0) {
        printf("[sum_fastgauss]Error: Number of parameters must be a multiple of 3 (area1, centroid1, fwhm1, ...)\n");
        return;
    }

    if (EXP[0] < 1){
        for (i=0; i<5000; i++){
            EXP[i] = exp(-0.01 * i);
        }
    }

    /* Initialize output array */
    for (j=0; j<len_x;  j++) {
        y[j] = 0.;
    }

    inv_two_sqrt_two_log2 = 1.0 / (2.0 * sqrt(2.0 * LOG2));
    sqrt2PI = sqrt(2.0*M_PI);

    for (i=0; i<len_pgauss/3; i++) {
        area = pgauss[3*i];
        centroid = pgauss[3*i+1];
        fwhm = pgauss[3*i+2];

        sigma = fwhm * inv_two_sqrt_two_log2;
        height = area / (sigma * sqrt2PI);

        for (j=0; j<len_x;  j++) {
            dhelp = (x[j] - centroid)/sigma;
            if (dhelp <= 15){
                dhelp = 0.5 * dhelp * dhelp;
                if (dhelp < 50){
                    expindex = (int) (dhelp * 100);
                    y[j] += height * EXP[expindex] * (1.0 - (dhelp - 0.01 * expindex));
                }
                else if (dhelp < 100) {
                    expindex = (int) (dhelp * 10);
                    y[j] += height * pow(EXP[expindex] * (1.0 - (dhelp - 0.1 * expindex)), 10);
                }
                else if (dhelp < 1000){
                    expindex = (int) (dhelp);
                    y[j] += height * pow(EXP[expindex] * (1.0 - (dhelp - expindex)), 20);
                }
            }
        }
    }
}

/*  sum_splitgauss
    Sum of split gaussian functions, defined by (height, centroid, fwhm1, fwhm2)

    *height* is the peak amplitude
    *centroid* is the peak x-coordinate
    *fwhm1* is the full-width at half maximum of the left half of the curve (x < centroid)
    *fwhm1* is the full-width at half maximum of the right half of the curve (x > centroid)

    Parameters:
    -----------

        - x: Independant variable where the gaussians are calculated.
        - len_x: Number of elements in the x array.
        - pgauss: Array of gaussian parameters:
          (height1, centroid1, fwhm11, fwhm21, height2, centroid2, fwhm12, fwhm22,...)
        - len_pgauss: Number of elements in the pgauss array. Must be
          a multiple of 4.
        - y: Output array. Must have memory allocated for the same number
          of elements as x (len_x).

*/
void sum_splitgauss(double* x, int len_x, double* pgauss, int len_pgauss, double* y)
{
    int i, j;
    double dhelp, inv_two_sqrt_two_log2, sigma1, sigma2;
    double fwhm1, fwhm2, centroid, height;

    if (len_pgauss % 4 || len_pgauss == 0) {
        printf("[sum_splitgauss]Error: Number of parameters must be a multiple of 4 (h1, c1, fwhm11, fwhm21...)\n");
        return;
    }

    /* Initialize output array */
    for (j=0; j<len_x;  j++) {
        y[j] = 0.;
    }

    inv_two_sqrt_two_log2 = 1.0 / (2.0 * sqrt(2.0 * LOG2));

    for (i=0; i<len_pgauss/4; i++) {
        height = pgauss[4*i];
        centroid = pgauss[4*i+1];
        fwhm1 = pgauss[4*i+2];
        fwhm2 = pgauss[4*i+3];

        sigma1 = fwhm1 * inv_two_sqrt_two_log2;
        sigma2 = fwhm2 * inv_two_sqrt_two_log2;

        for (j=0; j<len_x;  j++) {
            dhelp = (x[j] - centroid);
            if (dhelp > 0) {
                /* Use fwhm2 when x > centroid */
                dhelp = dhelp / sigma2;
            }
            else {
                /* Use fwhm1 when x < centroid */
                dhelp = dhelp / sigma1;
            }

            if (dhelp <= 20) {
                y[j] += height * exp (-0.5 * dhelp * dhelp);
            }
        }
    }
}

/*  sum_apvoigt
    Sum of pseudo-Voigt functions, defined by (area, centroid, fwhm, eta).

    The pseudo-Voigt profile PV(x) is an approximation of the Voigt profile
    using a linear combination of a Gaussian curve G(x) and a Lorentzian curve
    L(x) instead of their convolution.

    *area* is the area underneath both G(x) and L(x)
    *centroid* is the peak x-coordinate for both functions
    *fwhm* is the full-width at half maximum of both functions
    *eta* is the Lorentz factor: PV(x) = eta * L(x) + (1 - eta) * G(x)

    Parameters:
    -----------

        - x: Independant variable where the gaussians are calculated.
        - len_x: Number of elements in the x array.
        - pvoigt: Array of Voigt function parameters:
          (area1, centroid1, fwhm1, eta1, area2, centroid2, fwhm2, eta2,...)
        - len_voigt: Number of elements in the pvoigt array. Must be
          a multiple of 4.
        - y: Output array. Must have memory allocated for the same number
          of elements as x (len_x).

*/
void sum_apvoigt(double* x, int len_x, double* pvoigt, int len_pvoigt, double* y)
{
    int i, j;
    double dhelp, inv_two_sqrt_two_log2, sqrt2PI, sigma, height;
    double area, centroid, fwhm, eta;

    if (len_pvoigt % 4 || len_pvoigt == 0) {
        printf("[sum_apvoigt]Error: Number of parameters must be a multiple of 4 (a1, c1, fwhm1, eta1...)\n");
        return;
    }

    /* Initialize output array */
    for (j=0; j<len_x;  j++) {
        y[j] = 0.;
    }

    inv_two_sqrt_two_log2 = 1.0 / (2.0 * sqrt(2.0 * LOG2));
    sqrt2PI = sqrt(2.0*M_PI);


    for (i=0; i<len_pvoigt/4; i++) {
        area = pvoigt[4*i];
        centroid = pvoigt[4*i+1];
        fwhm = pvoigt[4*i+2];
        eta = pvoigt[4*i+3];

        sigma = fwhm * inv_two_sqrt_two_log2;
        height = area / (sigma * sqrt2PI);

        for (j=0; j<len_x;  j++) {
            /*  Lorentzian term */
            dhelp = (x[j] - centroid) / (0.5 * fwhm);
            dhelp = 1.0 + (dhelp * dhelp);
            y[j] += eta * (area / (0.5 * M_PI * fwhm * dhelp));

            /* Gaussian term */
            dhelp = (x[j] - centroid) / sigma;
            if (dhelp <= 35) {
                y[j] += (1.0 - eta) * height * exp (-0.5 * dhelp * dhelp);
            }
        }
    }
}

/*  sum_pvoigt
    Sum of pseudo-Voigt functions, defined by (height, centroid, fwhm, eta).

    The pseudo-Voigt profile PV(x) is an approximation of the Voigt profile
    using a linear combination of a Gaussian curve G(x) and a Lorentzian curve
    L(x) instead of their convolution.

    *height* is the peak amplitude of G(x) and L(x)
    *centroid* is the peak x-coordinate for both functions
    *fwhm* is the full-width at half maximum of both functions
    *eta* is the Lorentz factor: PV(x) = eta * L(x) + (1 - eta) * G(x)

    Parameters:
    -----------

        - x: Independant variable where the gaussians are calculated.
        - len_x: Number of elements in the x array.
        - pvoigt: Array of Voigt function parameters:
          (height1, centroid1, fwhm1, eta1, height2, centroid2, fwhm2, eta2,...)
        - len_voigt: Number of elements in the pvoigt array. Must be
          a multiple of 4.
        - y: Output array. Must have memory allocated for the same number
          of elements as x (len_x).

*/
void sum_pvoigt(double* x, int len_x, double* pvoigt, int len_pvoigt, double* y)
{
    int i, j;
    double dhelp, inv_two_sqrt_two_log2, sigma;
    double height, centroid, fwhm, eta;

    if (len_pvoigt % 4 || len_pvoigt == 0) {
        printf("[sum_pvoigt]Error: Number of parameters must be a multiple of 4 (h1, c1, fwhm1, eta1...)\n");
        return;
    }

    /* Initialize output array */
    for (j=0; j<len_x;  j++) {
        y[j] = 0.;
    }

    inv_two_sqrt_two_log2 = 1.0 / (2.0 * sqrt(2.0 * LOG2));

    for (i=0; i<len_pvoigt/4; i++) {
        height = pvoigt[4*i];
        centroid = pvoigt[4*i+1];
        fwhm = pvoigt[4*i+2];
        eta = pvoigt[4*i+3];

        sigma = fwhm * inv_two_sqrt_two_log2;

        for (j=0; j<len_x;  j++) {
            /*  Lorentzian term */
            dhelp = (x[j] - centroid) / (0.5 * fwhm);
            dhelp = 1.0 + (dhelp * dhelp);
            y[j] += eta * height / dhelp;

            /* Gaussian term */
            dhelp = (x[j] - centroid) / sigma;
            if (dhelp <= 35) {
                y[j] += (1.0 - eta) * height * exp (-0.5 * dhelp * dhelp);
            }
        }
    }
}

/*  sum_splitpvoigt
    Sum of split pseudo-Voigt functions, defined by
    (height, centroid, fwhm1, fwhm2, eta).

    The pseudo-Voigt profile PV(x) is an approximation of the Voigt profile
    using a linear combination of a Gaussian curve G(x) and a Lorentzian curve
    L(x) instead of their convolution.

    *height* is the peak amplitude of G(x) and L(x)
    *centroid* is the peak x-coordinate for both functions
    *fwhm1* is the full-width at half maximum of both functions for x < centroid
    *fwhm2* is the full-width at half maximum of both functions for x > centroid
    *eta* is the Lorentz factor: PV(x) = eta * L(x) + (1 - eta) * G(x)

    Parameters:
    -----------

        - x: Independant variable where the gaussians are calculated.
        - len_x: Number of elements in the x array.
        - pvoigt: Array of Voigt function parameters:
          (height1, centroid1, fwhm11, fwhm21, eta1, ...)
        - len_voigt: Number of elements in the pvoigt array. Must be
          a multiple of 5.
        - y: Output array. Must have memory allocated for the same number
          of elements as x (len_x).

*/
void sum_splitpvoigt(double* x, int len_x, double* pvoigt, int len_pvoigt, double* y)
{
    int i, j;
    double dhelp, inv_two_sqrt_two_log2, x_minus_centroid, sigma1, sigma2;
    double height, centroid, fwhm1, fwhm2, eta;

    if (len_pvoigt % 5 || len_pvoigt == 0) {
        printf("[sum_splitpvoigt]Error: Num of parameters must be a multiple of 5 (h1, c1, fwhm11, fwhm21, eta1...)\n");
        return;
    }

    /* Initialize output array */
    for (j=0; j<len_x;  j++) {
        y[j] = 0.;
    }

    inv_two_sqrt_two_log2 = 1.0 / (2.0 * sqrt(2.0 * LOG2));

    for (i=0; i<len_pvoigt/5; i++) {
        height = pvoigt[5*i];
        centroid = pvoigt[5*i+1];
        fwhm1 = pvoigt[5*i+2];
        fwhm2 = pvoigt[5*i+3];
        eta = pvoigt[5*i+4];

        sigma1 = fwhm1 * inv_two_sqrt_two_log2;
        sigma2 = fwhm2 * inv_two_sqrt_two_log2;

        for (j=0; j<len_x;  j++) {
            x_minus_centroid = (x[j] - centroid);

            /* Use fwhm2 when x > centroid */
            if (x_minus_centroid > 0) {
                /*  Lorentzian term */
                dhelp = x_minus_centroid / (0.5 * fwhm2);
                dhelp = 1.0 + (dhelp * dhelp);
                y[j] += eta * height / dhelp;

                /* Gaussian term */
                dhelp = x_minus_centroid / sigma2;
                if (dhelp <= 35) {
                    y[j] += (1.0 - eta) * height * exp (-0.5 * dhelp * dhelp);
                }
            }
            /* Use fwhm1 when x < centroid */
            else {
                /*  Lorentzian term */
                dhelp = x_minus_centroid / (0.5 * fwhm1);
                dhelp = 1.0 + (dhelp * dhelp);
                y[j] += eta * height / dhelp;

                /* Gaussian term */
                dhelp = x_minus_centroid / sigma1;
                if (dhelp <= 35) {
                    y[j] += (1.0 - eta) * height * exp (-0.5 * dhelp * dhelp);
                }
            }
        }
    }
}

/*  sum_lorentz
    Sum of Lorentz functions, defined by (height, centroid, fwhm).

    *height* is the peak amplitude
    *centroid* is the peak's x-coordinate
    *fwhm* is the full-width at half maximum

    Parameters:
    -----------

        - x: Independant variable where the Lorentzians are calculated.
        - len_x: Number of elements in the x array.
        - plorentz: Array of lorentz function parameters:
          (height1, centroid1, fwhm1, ...)
        - len_lorentz: Number of elements in the plorentz array. Must be
          a multiple of 3.
        - y: Output array. Must have memory allocated for the same number
          of elements as x (len_x).

*/
void sum_lorentz(double* x, int len_x, double* plorentz, int len_plorentz, double* y)
{
    int i, j;
    double dhelp;
    double height, centroid, fwhm;

    if (len_plorentz % 3 || len_plorentz == 0) {
        printf("[sum_lorentz]Error: Number of parameters must be a multiple of 4 (h1, c1, fwhm1, eta1...)\n");
        return;
    }

    /* Initialize output array */
    for (j=0; j<len_x;  j++) {
        y[j] = 0.;
    }

    for (i=0; i<len_plorentz/3; i++) {
        height = plorentz[3*i];
        centroid = plorentz[3*i+1];
        fwhm = plorentz[3*i+2];

        for (j=0; j<len_x;  j++) {
            dhelp = (x[j] - centroid) / (0.5 * fwhm);
            dhelp = 1.0 + (dhelp * dhelp);
            y[j] += height / dhelp;
        }
    }
}


/*  sum_alorentz
    Sum of Lorentz functions, defined by (area, centroid, fwhm).

    *area* is the area underneath the peak
    *centroid* is the peak's x-coordinate
    *fwhm* is the full-width at half maximum

    Parameters:
    -----------

        - x: Independant variable where the Lorentzians are calculated.
        - len_x: Number of elements in the x array.
        - plorentz: Array of lorentz function parameters:
          (area1, centroid1, fwhm1, ...)
        - len_lorentz: Number of elements in the plorentz array. Must be
          a multiple of 3.
        - y: Output array. Must have memory allocated for the same number
          of elements as x (len_x).

*/
void sum_alorentz(double* x, int len_x, double* plorentz, int len_plorentz, double* y)
{
    int i, j;
    double dhelp;
    double area, centroid, fwhm;

    if (len_plorentz % 3 || len_plorentz == 0) {
        printf("[sum_alorentz]Error: Number of parameters must be a multiple of 4 (h1, c1, fwhm1, eta1...)\n");
        return;
    }

    /* Initialize output array */
    for (j=0; j<len_x;  j++) {
        y[j] = 0.;
    }

    for (i=0; i<len_plorentz/3; i++) {
        area = plorentz[3*i];
        centroid = plorentz[3*i+1];
        fwhm = plorentz[3*i+2];

        for (j=0; j<len_x;  j++) {
            dhelp = (x[j] - centroid) / (0.5 * fwhm);
            dhelp = 1.0 + (dhelp * dhelp);
            y[j] += area / (0.5 * M_PI * fwhm * dhelp);
        }
    }
}






