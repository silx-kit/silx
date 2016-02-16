/* The original SpecFile.h has a `#define L 2`  directive
   that breaks cython lists and memory views.   */
#include "SpecFile.h"
#undef L
