py.install_sources([
    '__init__.py',
    '_skimage.py',
],
subdir: 'silx/image/marchingsquares',  # Folder relative to site-packages to install to
)


py.extension_module('_mergeimpl', '_mergeimpl.pyx',
        subdir: 'silx/image/marchingsquares',
        include_directories: ['include'],
        override_options : ['cython_language=cpp'],
        dependencies : [py_dep, omp], 
        install: true, 
        )

install_subdir('test', install_dir: silx_dir / 'image/marchingsquares')
