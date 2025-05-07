subdir('test')


py.install_sources([
    '__init__.py',
],
pure: false,    # Will be installed next to binaries
subdir: 'silx/math/medianfilter',  # Folder relative to site-packages to install to
)

py.extension_module('medianfilter', 'medianfilter.pyx',
        subdir: 'silx/math/medianfilter',
        include_directories: ['include'], 
        dependencies : [py_dep, omp],
        override_options : ['cython_language=cpp'],
        install: true, 
        )
