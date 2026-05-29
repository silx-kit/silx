subdir('codec')

py.install_sources([
    '__init__.py',
    'atomic.py',
	'common.py',
    'convolution.py',
    'medfilt.py',
    'projection.py',
    'sinofilter.py',
    'statistics.py',
    'backprojection.py',
    'conftest.py',
    'image.py',
    'linalg.py',
    'processing.py',
    'reconstruction.py',
    'sparse.py',
    'utils.py'
],
subdir: 'silx/opencl',  # Folder relative to site-packages to install to
)


pure_subdirs = [
  'sift',
  'test',
]

foreach subdir: pure_subdirs
  install_subdir(subdir, install_dir: silx_dir)
endforeach
