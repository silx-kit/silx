subdir('include')
subdir('test')


py.install_sources([
    '__init__.py',
    '_skimage.py',
],
pure: false,    # Will be installed next to binaries
subdir: 'silx/image/marchingsquares',  # Folder relative to site-packages to install to
)

# OpenMP can be used? several ways to disable it:
#  -> use -Duse_openmp=disabled
#  -> or the file SILX_WITH_OPENMP should contain False
#  -> or OpenMP can be unsuported on your achitecture/compiler

omp = dependency('openmp', required: get_option('use_openmp').disable_auto_if(openmp_var=='False'))


py.extension_module('_mergeimpl', '_mergeimpl.pyx',
        subdir: 'silx/image/marchingsquares',
        override_options : ['cython_language=cpp'],
        dependencies : [py_dep, omp], 
        install: true, 
        )
