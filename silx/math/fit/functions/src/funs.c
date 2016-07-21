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
/*
    This file provides fit functions.

    It is adapted from PyMca source file "SpecFitFuns.c". The main difference
    with the original code is that this code does not handle the python
    wrapping, which is done elsewhere using cython.

    Authors: V.A. Sole, P. Knobel
    License: MIT
    Last modified: 17/06/2016
*/
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "functions.h"

#ifndef M_PI
#define M_PI 3.1415926535
#endif

#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))

#if defined(_WIN32)
#define erf myerf
#define erfc myerfc
#endif

#define LOG2  0.69314718055994529


int test_params(int len_params,
                int len_params_one_function,
                char* fun_name,
                char* param_names)
{
    if (len_params % len_params_one_function) {
        printf("[%s]Error: Number of parameters must be a multiple of %d.",
               fun_name, len_params_one_function);
        printf("\nParameters expected for %s: %s\n",
               fun_name, param_names);
        return(1);
    }
    if (len_params == 0) {
        printf("[%s]Error: No parameters specified.", fun_name);
        printf("\nParameters expected for %s: %s\n",
               fun_name, param_names);
        return(1);
    }
    return(0);
}

/* Complementary error function for a single value*/
double myerfc(double x)
{
    double z;
    double t;
    double r;

    z=fabs(x);
    t=1.0/(1.0+0.5*z);
    r=t * exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.3740916 +
      t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 +
      t * (1.48851587 + t * (-0.82215223+t*0.17087277)))))))));
    if (x<0)
       r=2.0-r;
    return (r);
}

/* Gauss error function for a single value*/
double myerf(double x)
{
    return (1.0 - myerfc(x));
}

/* Gauss error function for an array
   y[i]=erf(x[i])
   returns status code 0
*/
int erf_array(double* x, int len_x, double* y)
{
    int j;
    for (j=0; j<len_x;  j++) {
        y[j] = erf(x[j]);
    }
    return(0);
}

/* Complementary error function for an array
   y[i]=erfc(x[i])
   returns status code 0*/
int erfc_array(double* x, int len_x, double* y)
{
    int j;
    for (j=0; j<len_x;  j++) {
        y[j] = erfc(x[j]);
    }
    return(0);
}

/* Use lookup table for fast exp computation */
double fastexp(double x)
{
    int expindex;
    static double EXP[5000] = {0.0};
    int i;

/*initialize */
    if (EXP[0] < 1){
        for (i=0;i<5000;i++){
            EXP[i] = exp(-0.01 * i);
        }
    }
/*calculate*/
    if (x < 0){
        x = -x;
        if (x < 50){
            expindex = (int) (x * 100);
            return EXP[expindex]*(1.0 - (x - 0.01 * expindex)) ;
        }else if (x < 100) {
            expindex = (int) (x * 10);
            return pow(EXP[expindex]*(1.0 - (x - 0.1 * expindex)),10) ;
        }else if (x < 1000){
            expindex = (int) x;
            return pow(EXP[expindex]*(1.0 - (x - expindex)),20) ;
        }else if (x < 10000){
            expindex = (int) (x * 0.1);
            return pow(EXP[expindex]*(1.0 - (x - 10.0 * expindex)),30) ;
        }else{
            return 0;
        }
    }else{
        if (x < 50){
            expindex = (int) (x * 100);
            return 1.0/EXP[expindex]*(1.0 - (x - 0.01 * expindex)) ;
        }else if (x < 100) {
            expindex = (int) (x * 10);
            return pow(EXP[expindex]*(1.0 - (x - 0.1 * expindex)),-10) ;
        }else{
            return exp(x);
        }
    }
}


/*  sum_gauss
    Sum of gaussian functions, defined by (height, centroid, fwhm)

    *height* is the peak amplitude
    *centroid* is the peak x-coordinate
    *fwhm* is the full-width at half maximum

    Parameters:
    -----------

        - x: Independant variable where the gaussians are calculated.
        - len_x: Number of elements in the x array.
        - pvoigt: Array of gaussian parameters:
          (height1, centroid1, fwhm1, height2, centroid2, fwhm2,...)
        - len_pgauss: Number of elements in the pgauss array. Must be
          a multiple of 3.
        - y: Output array. Must have memory allocated for the same number
          of elements as x (len_x).

    Adapted from PyMca module SpecFitFuns
*/
int sum_gauss(double* x, int len_x, double* pgauss, int len_pgauss, double* y)
{
    int i, j;
    double dhelp, inv_two_sqrt_two_log2, sigma;
    double fwhm, centroid, height;

    if (test_params(len_pgauss, 3, "sum_gauss", "height, centroid, fwhm")) {
        return(1);
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
    return(0);
}

/*  sum_agauss
    Sum of gaussian functions defined by (area, centroid, fwhm)

    *area* is the area underneath the peak
    *centroid* is the peak x-coordinate
    *fwhm* is the full-width at half maximum

    Parameters:
    -----------

        - x: Independant variable where the gaussians are calculated.
        - len_x: Number of elements in the x array.
        - pgauss: Array of gaussian parameters:
          (area1, centroid1, fwhm1, area2, centroid2, fwhm2,...)
        - len_pgauss: Number of elements in the pgauss array. Must be
          a multiple of 3.
        - y: Output array. Must have memory allocated for the same number
          of elements as x (len_x).

    Adapted from PyMca module SpecFitFuns
*/
int sum_agauss(double* x, int len_x, double* pgauss, int len_pgauss, double* y)
{
    int i, j;
    double  dhelp, height, sqrt2PI, sigma, inv_two_sqrt_two_log2;
    double fwhm, centroid, area;

    if (test_params(len_pgauss, 3, "sum_agauss", "area, centroid, fwhm")) {
        return(1);
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
    return(0);
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

        - x: Independant variable where the gaussians are calculated.
        - len_x: Number of elements in the x array.
        - pgauss: Array of gaussian parameters:
          (area1, centroid1, fwhm1, area2, centroid2, fwhm2,...)
        - len_pgauss: Number of elements in the pgauss array. Must be
          a multiple of 3.
        - y: Output array. Must have memory allocated for the same number
          of elements as x (len_x).

    Adapted from PyMca module SpecFitFuns
*/

int sum_fastagauss(double* x, int len_x, double* pgauss, int len_pgauss, double* y)
{
    int i, j, expindex;
    double  dhelp, height, sqrt2PI, sigma, inv_two_sqrt_two_log2;
    double fwhm, centroid, area;
    static double EXP[5000];

    if (test_params(len_pgauss, 3, "sum_fastagauss", "area, centroid, fwhm")) {
        return(1);
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
    return(0);
}

/*  sum_splitgauss
    Sum of split gaussian functions, defined by (height, centroid, fwhm1, fwhm2)

    *height* is the peak amplitude
    *centroid* is the peak x-coordinate
    *fwhm1* is the full-width at half maximum of the left half of the curve (x < centroid)
    *fwhm1* is the full-width at half maximum of the right half of the curve (x > centroid)

    Parameters:
    -----------

        - x: Independant variable where the gaussians are calculated.
        - len_x: Number of elements in the x array.
        - pgauss: Array of gaussian parameters:
          (height1, centroid1, fwhm11, fwhm21, height2, centroid2, fwhm12, fwhm22,...)
        - len_pgauss: Number of elements in the pgauss array. Must be
          a multiple of 4.
        - y: Output array. Must have memory allocated for the same number
          of elements as x (len_x).

    Adapted from PyMca module SpecFitFuns
*/
int sum_splitgauss(double* x, int len_x, double* pgauss, int len_pgauss, double* y)
{
    int i, j;
    double dhelp, inv_two_sqrt_two_log2, sigma1, sigma2;
    double fwhm1, fwhm2, centroid, height;

    if (test_params(len_pgauss, 4, "sum_splitgauss", "height, centroid, fwhm1, fwhm2")) {
        return(1);
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
    return(0);
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

        - x: Independant variable where the gaussians are calculated.
        - len_x: Number of elements in the x array.
        - pvoigt: Array of Voigt function parameters:
          (area1, centroid1, fwhm1, eta1, area2, centroid2, fwhm2, eta2,...)
        - len_voigt: Number of elements in the pvoigt array. Must be
          a multiple of 4.
        - y: Output array. Must have memory allocated for the same number
          of elements as x (len_x).

    Adapted from PyMca module SpecFitFuns
*/
int sum_apvoigt(double* x, int len_x, double* pvoigt, int len_pvoigt, double* y)
{
    int i, j;
    double dhelp, inv_two_sqrt_two_log2, sqrt2PI, sigma, height;
    double area, centroid, fwhm, eta;

    if (test_params(len_pvoigt, 4, "sum_apvoigt", "area, centroid, fwhm, eta")) {
        return(1);
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
    return(0);
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

        - x: Independant variable where the gaussians are calculated.
        - len_x: Number of elements in the x array.
        - pvoigt: Array of Voigt function parameters:
          (height1, centroid1, fwhm1, eta1, height2, centroid2, fwhm2, eta2,...)
        - len_voigt: Number of elements in the pvoigt array. Must be
          a multiple of 4.
        - y: Output array. Must have memory allocated for the same number
          of elements as x (len_x).

    Adapted from PyMca module SpecFitFuns
*/
int sum_pvoigt(double* x, int len_x, double* pvoigt, int len_pvoigt, double* y)
{
    int i, j;
    double dhelp, inv_two_sqrt_two_log2, sigma;
    double height, centroid, fwhm, eta;

    if (test_params(len_pvoigt, 4, "sum_pvoigt", "height, centroid, fwhm, eta")) {
        return(1);
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
    return(0);
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

        - x: Independant variable where the gaussians are calculated.
        - len_x: Number of elements in the x array.
        - pvoigt: Array of Voigt function parameters:
          (height1, centroid1, fwhm11, fwhm21, eta1, ...)
        - len_voigt: Number of elements in the pvoigt array. Must be
          a multiple of 5.
        - y: Output array. Must have memory allocated for the same number
          of elements as x (len_x).

    Adapted from PyMca module SpecFitFuns
*/
int sum_splitpvoigt(double* x, int len_x, double* pvoigt, int len_pvoigt, double* y)
{
    int i, j;
    double dhelp, inv_two_sqrt_two_log2, x_minus_centroid, sigma1, sigma2;
    double height, centroid, fwhm1, fwhm2, eta;

    if (test_params(len_pvoigt, 5, "sum_splitpvoigt", "height, centroid, fwhm1, fwhm2, eta")) {
        return(1);
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
    return(0);
}

/*  sum_lorentz
    Sum of Lorentz functions, defined by (height, centroid, fwhm).

    *height* is the peak amplitude
    *centroid* is the peak's x-coordinate
    *fwhm* is the full-width at half maximum

    Parameters:
    -----------

        - x: Independant variable where the Lorentzians are calculated.
        - len_x: Number of elements in the x array.
        - plorentz: Array of lorentz function parameters:
          (height1, centroid1, fwhm1, ...)
        - len_lorentz: Number of elements in the plorentz array. Must be
          a multiple of 3.
        - y: Output array. Must have memory allocated for the same number
          of elements as x (len_x).

    Adapted from PyMca module SpecFitFuns
*/
int sum_lorentz(double* x, int len_x, double* plorentz, int len_plorentz, double* y)
{
    int i, j;
    double dhelp;
    double height, centroid, fwhm;

    if (test_params(len_plorentz, 3, "sum_lorentz", "height, centroid, fwhm")) {
        return(1);
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
    return(0);
}


/*  sum_alorentz
    Sum of Lorentz functions, defined by (area, centroid, fwhm).

    *area* is the area underneath the peak
    *centroid* is the peak's x-coordinate
    *fwhm* is the full-width at half maximum

    Parameters:
    -----------

        - x: Independant variable where the Lorentzians are calculated.
        - len_x: Number of elements in the x array.
        - plorentz: Array of lorentz function parameters:
          (area1, centroid1, fwhm1, ...)
        - len_lorentz: Number of elements in the plorentz array. Must be
          a multiple of 3.
        - y: Output array. Must have memory allocated for the same number
          of elements as x (len_x).

    Adapted from PyMca module SpecFitFuns
*/
int sum_alorentz(double* x, int len_x, double* plorentz, int len_plorentz, double* y)
{
    int i, j;
    double dhelp;
    double area, centroid, fwhm;

    if (test_params(len_plorentz, 3, "sum_alorentz", "area, centroid, fwhm")) {
        return(1);
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
    return(0);
}


/*  sum_splitlorentz
    Sum of Lorentz functions, defined by (height, centroid, fwhm1, fwhm2).

    *height* is the peak amplitude
    *centroid* is the peak's x-coordinate
    *fwhm1* is the full-width at half maximum for x < centroid
    *fwhm2* is the full-width at half maximum for x > centroid

    Parameters:
    -----------

        - x: Independant variable where the Lorentzians are calculated.
        - len_x: Number of elements in the x array.
        - plorentz: Array of lorentz function parameters:
          (height1, centroid1, fwhm11, fwhm21 ...)
        - len_lorentz: Number of elements in the plorentz array. Must be
          a multiple of 4.
        - y: Output array. Must have memory allocated for the same number
          of elements as x (len_x).

    Adapted from PyMca module SpecFitFuns
*/
int sum_splitlorentz(double* x, int len_x, double* plorentz, int len_plorentz, double* y)
{
    int i, j;
    double dhelp;
    double height, centroid, fwhm1, fwhm2;

    if (test_params(len_plorentz, 4, "sum_splitlorentz", "height, centroid, fwhm1, fwhm2")) {
        return(1);
    }

    /* Initialize output array */
    for (j=0; j<len_x;  j++) {
        y[j] = 0.;
    }

    for (i=0; i<len_plorentz/4; i++) {
        height = plorentz[4*i];
        centroid = plorentz[4*i+1];
        fwhm1 = plorentz[4*i+2];
        fwhm2 = plorentz[4*i+3];

        for (j=0; j<len_x;  j++) {
            dhelp = (x[j] - centroid);
            if (dhelp>0) {
                dhelp = dhelp / (0.5 * fwhm2);
            }
            else {
                dhelp = dhelp / (0.5 * fwhm1);
            }
            dhelp = 1.0 + (dhelp * dhelp);
            y[j] += height / dhelp;
        }
    }
    return(0);
}

/*  sum_stepdown
    Sum of stepdown functions, defined by (height, centroid, fwhm).

    *height* is the step amplitude
    *centroid* is the step's x-coordinate
    *fwhm* is the full-width at half maximum of the derivative

    Parameters:
    -----------

        - x: Independant variable where the stepdown functions are calculated.
        - len_x: Number of elements in the x array.
        - pdstep: Array of downstpe function parameters:
          (height1, centroid1, fwhm1, ...)
        - len_pdstep: Number of elements in the pdstep array. Must be
          a multiple of 3.
        - y: Output array. Must have memory allocated for the same number
          of elements as x (len_x).

    Adapted from PyMca module SpecFitFuns
*/
int sum_stepdown(double* x, int len_x, double* pdstep, int len_pdstep, double* y)
{
    int i, j;
    double dhelp, sqrt2_inv_2_sqrt_two_log2 ;
    double height, centroid, fwhm;

    if (test_params(len_pdstep, 3, "sum_stepdown", "height, centroid, fwhm")) {
        return(1);
    }

    /* Initialize output array */
    for (j=0; j<len_x;  j++) {
        y[j] = 0.;
    }

    sqrt2_inv_2_sqrt_two_log2 = sqrt(2.0) / (2.0 * sqrt(2.0 * LOG2));

    for (i=0; i<len_pdstep/3; i++) {
        height = pdstep[3*i];
        centroid = pdstep[3*i+1];
        fwhm = pdstep[3*i+2];

        for (j=0; j<len_x;  j++) {
            dhelp = fwhm * sqrt2_inv_2_sqrt_two_log2;
            dhelp = (x[j] - centroid) / dhelp;
            y[j] += height * 0.5 * erfc(dhelp);
        }
    }
    return(0);
}

/*  sum_stepup
    Sum of stepup functions, defined by (height, centroid, fwhm).

    *height* is the step amplitude
    *centroid* is the step's x-coordinate
    *fwhm* is the full-width at half maximum of the derivative

    Parameters:
    -----------

        - x: Independant variable where the stepup functions are calculated.
        - len_x: Number of elements in the x array.
        - pustep: Array of stepdown function parameters:
          (height1, centroid1, fwhm1, ...)
        - len_pustep: Number of elements in the pustep array. Must be
          a multiple of 3.
        - y: Output array. Must have memory allocated for the same number
          of elements as x (len_x).

    Adapted from PyMca module SpecFitFuns
*/
int sum_stepup(double* x, int len_x, double* pustep, int len_pustep, double* y)
{
    int i, j;
    double dhelp, sqrt2_inv_2_sqrt_two_log2 ;
    double height, centroid, fwhm;

    if (test_params(len_pustep, 3, "sum_stepup", "height, centroid, fwhm")) {
        return(1);
    }

    /* Initialize output array */
    for (j=0; j<len_x;  j++) {
        y[j] = 0.;
    }

    sqrt2_inv_2_sqrt_two_log2 = sqrt(2.0) / (2.0 * sqrt(2.0 * LOG2));

    for (i=0; i<len_pustep/3; i++) {
        height = pustep[3*i];
        centroid = pustep[3*i+1];
        fwhm = pustep[3*i+2];

        for (j=0; j<len_x;  j++) {
            dhelp = fwhm * sqrt2_inv_2_sqrt_two_log2;
            dhelp = (x[j] - centroid) / dhelp;
            y[j] += height * 0.5 * (1.0 + erf(dhelp));
        }
    }
    return(0);
}


/*  sum_slit
    Sum of slit functions, defined by (height, position, fwhm, beamfwhm).

    *height* is the slit height
    *position* is the slit's center x-coordinate
    *fwhm* is the full-width at half maximum of the slit
    *beamfwhm* is the full-width at half maximum of derivative's peaks

    Parameters:
    -----------

        - x: Independant variable where the slit functions are calculated.
        - len_x: Number of elements in the x array.
        - pslit: Array of slit function parameters:
          (height1, centroid1, fwhm1, beamfwhm1 ...)
        - len_pslit: Number of elements in the pslit array. Must be
          a multiple of 3.
        - y: Output array. Must have memory allocated for the same number
          of elements as x (len_x).

    Adapted from PyMca module SpecFitFuns
*/
int sum_slit(double* x, int len_x, double* pslit, int len_pslit, double* y)
{
    int i, j;
    double dhelp, dhelp1, dhelp2, sqrt2_inv_2_sqrt_two_log2, centroid1, centroid2;
    double height, position, fwhm, beamfwhm;

    if (test_params(len_pslit, 4, "sum_slit", "height, centroid, fwhm, beamfwhm")) {
        return(1);
    }

    /* Initialize output array */
    for (j=0; j<len_x;  j++) {
        y[j] = 0.;
    }

    sqrt2_inv_2_sqrt_two_log2 = sqrt(2.0) / (2.0 * sqrt(2.0 * LOG2));

    for (i=0; i<len_pslit/4; i++) {
        height = pslit[4*i];
        position = pslit[4*i+1];
        fwhm = pslit[4*i+2];
        beamfwhm = pslit[4*i+3];

        centroid1 = position - 0.5 * fwhm;
        centroid2 = position + 0.5 * fwhm;

        for (j=0; j<len_x;  j++) {
            dhelp = beamfwhm * sqrt2_inv_2_sqrt_two_log2;
            dhelp1 = (x[j] - centroid1) / dhelp;
            dhelp2 = (x[j] - centroid2) / dhelp;
            y[j] += height * 0.25 * (1.0 + erf(dhelp1)) *  erfc(dhelp2);
        }
    }
    return(0);
}


/*  sum_ahypermet
    Sum of hypermet functions, defined by
    (area, position, fwhm, st_area_r, st_slope_r, lt_area_r, lt_slope_r, step_height_r).

    - *area* is the area underneath the gaussian peak
    - *position* is the center of the various peaks and the position of
      the step down
    - *fwhm* is the full-width at half maximum of the terms
    - *st_area_r* is factor between the gaussian area and the area of the
      short tail term
    - *st_slope_r* is a parameter related to the slope of the short tail
      in the low ``x`` values (the lower, the steeper)
    - *lt_area_r* is factor between the gaussian area and the area of the
      long tail term
    - *lt_slope_r* is a parameter related to the slope of the long tail
      in the low ``x`` values  (the lower, the steeper)
    - *step_height_r* is the factor between the height of the step down
      and the gaussian height

    Parameters:
    -----------

        - x: Independant variable where the functions are calculated.
        - len_x: Number of elements in the x array.
        - phypermet: Array of hypermet function parameters:
          *(area1, position1, fwhm1, st_area_r1, st_slope_r1, lt_area_r1,
          lt_slope_r1, step_height_r1, ...)*
        - len_phypermet: Number of elements in the phypermet array. Must be
          a multiple of 8.
        - y: Output array. Must have memory allocated for the same number
          of elements as x (len_x).
        - tail_flags: sum of binary flags to activate the various terms of the
          function:

              - 1 (b0001): Gaussian term
              - 2 (b0010): st term
              - 4 (b0100): lt term
              - 8 (b1000): step term

          E.g., to activate all termsof the hypermet, use ``tail_flags = 1 + 2 + 4 + 8 = 15``

    Adapted from PyMca module SpecFitFuns
*/
int sum_ahypermet(double* x, int len_x, double* phypermet, int len_phypermet, double* y, int tail_flags)
{
    int i, j;
    int g_term_flag, st_term_flag, lt_term_flag, step_term_flag;
    double c1, c2, sigma, height, sigma_sqrt2, sqrt2PI, inv_2_sqrt_2_log2, x_minus_position, epsilon;
    double area, position, fwhm, st_area_r, st_slope_r, lt_area_r, lt_slope_r, step_height_r;

    if (test_params(len_phypermet, 8, "sum_hypermet",
                    "height, centroid, fwhm, st_area_r, st_slope_r, lt_area_r, lt_slope_r, step_height_r")) {
        return(1);
    }

    g_term_flag    = tail_flags & 1;
    st_term_flag   = (tail_flags>>1) & 1;
    lt_term_flag   = (tail_flags>>2) & 1;
    step_term_flag = (tail_flags>>3) & 1;

    /* Initialize output array */
    for (j=0; j<len_x;  j++) {
        y[j] = 0.;
    }

    /* define epsilon to compare floating point values with 0. */
    epsilon = 0.00000000001;

    sqrt2PI= sqrt(2.0 * M_PI);
    inv_2_sqrt_2_log2 = 1.0 / (2.0 * sqrt(2.0 * LOG2));

    for (i=0; i<len_phypermet/8; i++) {
        area = phypermet[8*i];
        position = phypermet[8*i+1];
        fwhm = phypermet[8*i+2];
        st_area_r = phypermet[8*i+3];
        st_slope_r =  phypermet[8*i+4];
        lt_area_r = phypermet[8*i+5];
        lt_slope_r = phypermet[8*i+6];
        step_height_r = phypermet[8*i+7];

        sigma = fwhm * inv_2_sqrt_2_log2;
        height = area / (sigma * sqrt2PI);

        /* Prevent division by 0 */
        if (sigma == 0) {
            printf("fwhm must not be equal to 0");
            return(1);
        }
        sigma_sqrt2 = sigma * 1.4142135623730950488;

        for (j=0; j<len_x;  j++) {
            x_minus_position = x[j] - position;
            c2 = (0.5 * x_minus_position * x_minus_position) / (sigma * sigma);
            /* gaussian term */
            if (g_term_flag) {
                y[j] += exp(-c2) * height;
            }

            /* st term */
            if (st_term_flag) {
                if (fabs(st_slope_r) > epsilon) {
                    c1 = st_area_r * 0.5 * \
                         erfc((x_minus_position/sigma_sqrt2) + 0.5 * sigma_sqrt2 / st_slope_r);
                    y[j] += ((area * c1) / st_slope_r) * \
                            exp(0.5 * (sigma / st_slope_r) * (sigma / st_slope_r) + \
                                (x_minus_position / st_slope_r));
                }
            }

            /* lt term */
            if (lt_term_flag) {
                if (fabs(lt_slope_r) > epsilon) {
                    c1 = lt_area_r * \
                         0.5 * erfc((x_minus_position/sigma_sqrt2) + 0.5 * sigma_sqrt2 / lt_slope_r);
                    y[j] += ((area * c1) / lt_slope_r) * \
                            exp(0.5 * (sigma / lt_slope_r) * (sigma / lt_slope_r) + \
                                (x_minus_position / lt_slope_r));
                }
            }

            /* step term flag */
            if (step_term_flag) {
                y[j] += step_height_r * (area / (sigma * sqrt2PI)) * \
                        0.5 * erfc(x_minus_position / sigma_sqrt2);
            }
        }
    }
    return(0);
}

/*  sum_fastahypermet

    Sum of hypermet functions, defined by
    (area, position, fwhm, st_area_r, st_slope_r, lt_area_r, lt_slope_r, step_height_r).

    - *area* is the area underneath the gaussian peak
    - *position* is the center of the various peaks and the position of
      the step down
    - *fwhm* is the full-width at half maximum of the terms
    - *st_area_r* is factor between the gaussian area and the area of the
      short tail term
    - *st_slope_r* is a parameter related to the slope of the short tail
      in the low ``x`` values (the lower, the steeper)
    - *lt_area_r* is factor between the gaussian area and the area of the
      long tail term
    - *lt_slope_r* is a parameter related to the slope of the long tail
      in the low ``x`` values  (the lower, the steeper)
    - *step_height_r* is the factor between the height of the step down
      and the gaussian height

    Parameters:
    -----------

        - x: Independant variable where the functions are calculated.
        - len_x: Number of elements in the x array.
        - phypermet: Array of hypermet function parameters:
          *(area1, position1, fwhm1, st_area_r1, st_slope_r1, lt_area_r1,
          lt_slope_r1, step_height_r1, ...)*
        - len_phypermet: Number of elements in the phypermet array. Must be
          a multiple of 8.
        - y: Output array. Must have memory allocated for the same number
          of elements as x (len_x).
        - tail_flags: sum of binary flags to activate the various terms of the
          function:

              - 1 (b0001): Gaussian term
              - 2 (b0010): st term
              - 4 (b0100): lt term
              - 8 (b1000): step term

          E.g., to activate all termsof the hypermet, use ``tail_flags = 1 + 2 + 4 + 8 = 15``

    Adapted from PyMca module SpecFitFuns
*/
int sum_fastahypermet(double* x, int len_x, double* phypermet, int len_phypermet, double* y, int tail_flags)
{
    int i, j;
    int g_term_flag, st_term_flag, lt_term_flag, step_term_flag;
    double c1, c2, sigma, height, sigma_sqrt2, sqrt2PI, inv_2_sqrt_2_log2, x_minus_position, epsilon;
    double area, position, fwhm, st_area_r, st_slope_r, lt_area_r, lt_slope_r, step_height_r;

    if (test_params(len_phypermet, 8, "sum_hypermet",
                    "height, centroid, fwhm, st_area_r, st_slope_r, lt_area_r, lt_slope_r, step_height_r")) {
        return(1);
    }

    g_term_flag    = tail_flags & 1;
    st_term_flag   = (tail_flags>>1) & 1;
    lt_term_flag   = (tail_flags>>2) & 1;
    step_term_flag = (tail_flags>>3) & 1;

    /* Initialize output array */
    for (j=0; j<len_x;  j++) {
        y[j] = 0.;
    }

    /* define epsilon to compare floating point values with 0. */
    epsilon = 0.00000000001;

    sqrt2PI= sqrt(2.0 * M_PI);
    inv_2_sqrt_2_log2 = 1.0 / (2.0 * sqrt(2.0 * LOG2));

    for (i=0; i<len_phypermet/8; i++) {
        area = phypermet[8*i];
        position = phypermet[8*i+1];
        fwhm = phypermet[8*i+2];
        st_area_r = phypermet[8*i+3];
        st_slope_r =  phypermet[8*i+4];
        lt_area_r = phypermet[8*i+5];
        lt_slope_r = phypermet[8*i+6];
        step_height_r = phypermet[8*i+7];

        sigma = fwhm * inv_2_sqrt_2_log2;
        height = area / (sigma * sqrt2PI);

        /* Prevent division by 0 */
        if (sigma == 0) {
            printf("fwhm must not be equal to 0");
            return(1);
        }
        sigma_sqrt2 = sigma * 1.4142135623730950488;

        for (j=0; j<len_x;  j++) {
            x_minus_position = x[j] - position;
            c2 = (0.5 * x_minus_position * x_minus_position) / (sigma * sigma);
            /* gaussian term */
            if (g_term_flag && c2 < 100) {
                y[j] += fastexp(-c2) * height;
            }

            /* st term */
            if (st_term_flag && (fabs(st_slope_r) > epsilon) && (x_minus_position / st_slope_r) <= 612) {
                c1 = st_area_r * 0.5 * \
                     erfc((x_minus_position/sigma_sqrt2) + 0.5 * sigma_sqrt2 / st_slope_r);
                y[j] += ((area * c1) / st_slope_r) * \
                        fastexp(0.5 * (sigma / st_slope_r) * (sigma / st_slope_r) +\
                                (x_minus_position / st_slope_r));
            }

            /* lt term */
            if (lt_term_flag && (fabs(lt_slope_r) > epsilon) && (x_minus_position / lt_slope_r) <= 612) {
                c1 = lt_area_r * \
                     0.5 * erfc((x_minus_position/sigma_sqrt2) + 0.5 * sigma_sqrt2 / lt_slope_r);
                y[j] += ((area * c1) / lt_slope_r) * \
                        fastexp(0.5 * (sigma / lt_slope_r) * (sigma / lt_slope_r) +\
                                (x_minus_position / lt_slope_r));

            }

            /* step term flag */
            if (step_term_flag) {
                y[j] += step_height_r * (area / (sigma * sqrt2PI)) *\
                        0.5 * erfc(x_minus_position / sigma_sqrt2);
            }
        }
    }
    return(0);
}

void pileup(double* x, long len_x, double* ret, int input2, double zero, double gain)
{
    //int    input2=0;
    //double zero=0.0;
    //double gain=1.0;

    int i, j, k;
    double  *px, *pret, *pall;

    /* the pointer to the starting position of par data */
    px = x;
    pret = ret;

    *pret = 0;
    k = (int )(zero/gain);
    for (i=input2; i<len_x; i++){
        pall = x;
        if ((i+k) >= 0)
        {
            pret = (double *) ret+(i+k);
            for (j=0; j<len_x-i-k ;j++){
                *pret += *px * (*pall);
                pall++;
                pret++;
            }
        }
        px++;
    }
}
