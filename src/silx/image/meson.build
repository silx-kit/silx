subdir('marchingsquares')
subdir('test')


py.install_sources([
    '__init__.py',
    '_boundingbox.py',
    'backprojection.py',
    'medianfilter.py',
    'phantomgenerator.py',
    'projection.py',
    'reconstruction.py',
    'sift.py',
    'tomography.py',
    'utils.py',
],
pure: false,    # Will be installed next to binaries
subdir: 'silx/image',  # Folder relative to site-packages to install to
)

py.extension_module('bilinear', 'bilinear.pyx',
        dependencies : py_dep, install: true, subdir: 'silx/image')

py.extension_module('shapes', 'shapes.pyx',
        dependencies : py_dep, install: true, subdir: 'silx/image')
