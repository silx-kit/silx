subdir('codec')
subdir('sift')
subdir('image')

py.install_sources(
   ['addition.cl',
    'array_utils.cl',
    'backproj.cl',
    'backproj_helper.cl',
    'bitonic.cl',
    'convolution.cl',
    'convolution_textures.cl',
    'doubleword.cl',
    'kahan.cl',
    'linalg.cl',
    'medfilt.cl',
    'preprocess.cl',
    'proj.cl',
    'sparse.cl',
    'statistics.cl'],
  pure: false,   # Will be installed next to binaries
  subdir: 'silx/resources/opencl'  # Folder relative to site-packages to install to
)