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

#ifndef KTH_SMALLEST
#define KTH_SMALLEST

#include <vector>
#include <signal.h>
#include <assert.h>

template<typename T>
void swap(std::vector<const T*>& vec, int a, int b){
    const T* tmp = vec[a];
    vec[a] = vec[b];
    vec[b] = tmp;
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

#endif // KTH_SMALLEST