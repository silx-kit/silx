subdir('filters')
subdir('functions')
subdir('peaks')


py.install_sources([
    '__init__.py',
    'bgtheories.py',
    'fitmanager.py',
    'fittheories.py',
    'fittheory.py',
    'leastsq.py',
],
subdir: 'silx/math/fit',  # Folder relative to site-packages to install to
)

install_subdir('test', install_dir: silx_dir / 'math/fit')
