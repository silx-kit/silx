


/** Flag the C module with a variable to know if it was compiled with OpenMP
 */
#ifdef _OPENMP
static const int COMPILED_WITH_OPENMP = 1;
#else
static const int COMPILED_WITH_OPENMP = 0;
#endif
