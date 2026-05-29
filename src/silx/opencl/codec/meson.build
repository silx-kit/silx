subdir('ext')

py.install_sources([
    '__init__.py',
    'bitshuffle_lz4.py',
    'byte_offset.py'
],
subdir: 'silx/opencl/codec',  # Folder relative to site-packages to install to
)


pure_subdirs = [
  'test',
]

foreach subdir: pure_subdirs
  install_subdir(subdir, install_dir: silx_dir)
endforeach