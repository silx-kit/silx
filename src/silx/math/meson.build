subdir('marchingcubes')
subdir('medianfilter')
subdir('fit')
subdir('fft')
subdir('histogramnd')
subdir('test')


py.install_sources([
    '__init__.py',
    'calibration.py',
    'colormap.py',
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
