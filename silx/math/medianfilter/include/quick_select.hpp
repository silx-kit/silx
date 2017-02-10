/*##########################################################################
#
# Copyright (c) 2017 European Synchrotron Radiation Facility
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
// __authors__ = ["P. Paleo"]
// __license__ = "MIT"
// __date__ = "10/02/2017"

#ifndef QUICK_SELECT
#define QUICK_SELECT

#include <vector>
#include <signal.h>
#include <assert.h>

template<typename T>
void swap(std::vector<const T*>& vec, int a, int b){
    const T* tmp = vec[a];
    vec[a] = vec[b];
    vec[b] = tmp;
}

template<typename T>
const T* quick_select(std::vector<const T*>& a, int n)
{
    int low, high;
    int median;
    int middle, ll, hh;

    low = 0 ; high = n-1 ; median = (low + high) / 2;
    for (;;) {
        if (high <= low) { /* One element only */
            return a[median] ;
        }

        if (high == low + 1) {  /* Two elements only */
            if (a[low] > a[high]) swap<T>(a, low, high);
            return a[median] ;
        }

    /* Find median of low, middle and high items; swap into position low */
        middle = (low + high) / 2;
        if (a[middle] > a[high])    swap<T>(a, middle, high) ;
        if (a[low] > a[high])       swap<T>(a, low, high) ;
        if (a[middle] > a[low])     swap<T>(a, middle, low) ;

    /* Swap low item (now in position middle) into position (low+1) */
        swap<T>(a, middle, low+1) ;

    /* Nibble from each end towards middle, swapping items when stuck */
        ll = low + 1;
        hh = high;
        for (;;) {
            do ll++; while (a[low] > a[ll]) ;
            do hh--; while (a[hh]  > a[low]) ;

            if (hh < ll) break;

            swap<T>(a, ll, hh) ;
        }

        /* Swap middle item (in position low) back into correct position */
        swap<T>(a, low, hh) ;

        /* Re-set active partition */
        if (hh <= median)
            low = ll;
        if (hh >= median)
            high = hh - 1;
    }
}

// Wirth
template<typename T>
const T* kth_smallest(std::vector<const T*>& a, int n, int k) {
    int i, j, l, m;
    T x;

    l=0, m=n-1;
    while (l<m) {
        x=*a[k];
        i=l ;
        j=m ;
        do {
            while (*a[i]<x) i++ ;
            while (x<*a[j]) j-- ;
            if (i<=j) {
                swap(a, i, j);
                i++, j--;
            }
        } while (i<=j);
        if (j<k) l=i;
        if (k<i) m=j;
    }
    return a[k];
}

template<typename T>
const T* medianwirth(std::vector<const T*>& a) {
    int n = a.size();
    return kth_smallest<T>(a, n, (((n)&1)?((n)/2):(((n)/2)-1)));
}

template<typename T>
void getMinMax(std::vector<const T*>& v, T& min, T&max){
    // init min and max values
    typename std::vector<const T*>::const_iterator it = v.begin();
    if (v.size() == 0){
        raise(SIGINT);
    }else{
        min = max = *(*it);
    }
    it++;

    // Browse all the vector
    while(it!=v.end()){
        if(*(*it) > max) max = *(*it);
        if(*(*it) < min) min = *(*it);
        it++;
    }
}

#endif // QUICK_SELECT