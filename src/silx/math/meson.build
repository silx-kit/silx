subdir('marchingcubes')
subdir('medianfilter')
subdir('fit')
subdir('histogramnd')


py.install_sources([
    '__init__.py',
    'calibration.py',
    'colormap.py',
    'combo.pyi',
    'histogram.py',
],
pure: false,    # Will be installed next to binaries
subdir: 'silx/math',  # Folder relative to site-packages to install to
)

py.extension_module('_colormap', '_colormap.pyx',
        subdir: 'silx/math',
        dependencies : [py_dep, omp],
        include_directories: ['include'], 
        install: true, 
        )

py.extension_module('combo', 'combo.pyx',
        subdir: 'silx/math',
        dependencies : [py_dep],
        include_directories: ['include'], 
        install: true, 
        )

py.extension_module('interpolate', 'interpolate.pyx',
        subdir: 'silx/math',
        dependencies : [py_dep, omp],
        install: true, 
        )

install_subdir('fft', install_dir: silx_dir / 'math')
install_subdir('test', install_dir: silx_dir / 'math')