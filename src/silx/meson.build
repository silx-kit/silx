subdir('io')
subdir('math')
subdir('image')

py.install_sources([
    '__init__.py',
    '__main__.py',
    '_config.py',
    '_utils.py',
    '_version.py',
    'conftest.py',
    'py.typed',
],
subdir: 'silx',  # Folder relative to site-packages to install to
)

pure_subdirs = [
  'app',
  'gui',
  'opencl',
  'resources',
  'sx',
  'test',
  'third_party',
  'utils',
]

foreach subdir: pure_subdirs
  install_subdir(subdir, install_dir: silx_dir)
endforeach
