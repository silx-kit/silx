IF HAVE_OPENMP:
    cimport openmp
    print('In Cython with OpenMP. Max threads: %d' % openmp.omp_get_max_threads())
ELSE:
    print('In Cython without OpenMP')

