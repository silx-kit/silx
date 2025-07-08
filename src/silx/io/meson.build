subdir('specfile')


py.install_sources([
    '__init__.py',
    '_sliceh5.py',
    'commonh5.py',
    'configdict.py',
    'convert.py',
    'dictdump.py',
    'fabioh5.py',
    'fioh5.py',
    'h5link_utils.py',
    'h5py_utils.py',
    'octaveh5.py',
    'rawh5.py',
    'specfilewrapper.py',
    'spech5.py',
    'url.py',
    'utils.py',
    'zarrh5.py',
],
subdir: 'silx/io',  # Folder relative to site-packages to install to
)

install_subdir('nxdata', install_dir: silx_dir / 'io')
install_subdir('test', install_dir: silx_dir / 'io')
