py.install_sources([
    '__init__.py',
],
subdir: 'silx/math/medianfilter',  # Folder relative to site-packages to install to
)

py.extension_module('medianfilter', 'medianfilter.pyx',
        subdir: 'silx/math/medianfilter',
        include_directories: ['include'], 
        dependencies : [py_dep, omp],
        override_options : ['cython_language=cpp'],
        install: true, 
        )

install_subdir('test', install_dir: silx_dir / 'math/medianfilter')
