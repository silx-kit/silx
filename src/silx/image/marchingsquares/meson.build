subdir('test')


py.install_sources([
    '__init__.py',
    '_skimage.py',
],
pure: false,    # Will be installed next to binaries
subdir: 'silx/image/marchingsquares',  # Folder relative to site-packages to install to
)


py.extension_module('_mergeimpl', '_mergeimpl.pyx',
        subdir: 'silx/image/marchingsquares',
        include_directories: ['include', '../../utils/include/'],
        override_options : ['cython_language=cpp'],
        dependencies : [py_dep, omp], 
        install: true, 
        )
