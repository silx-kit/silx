# -*- mode: python -*-
import importlib.metadata
import os.path
from pathlib import Path
import shutil
import subprocess
import sys

import pkg_resources

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

datas = []

PROJECT_PATH = os.path.abspath(os.path.join(SPECPATH, "..", ".."))
datas.append((os.path.join(PROJECT_PATH, "README.rst"), "."))
datas.append((os.path.join(PROJECT_PATH, "LICENSE"), "."))
datas.append((os.path.join(PROJECT_PATH, "copyright"), "."))
datas += collect_data_files("silx.resources")
datas += collect_data_files("hdf5plugin")


hiddenimports = []
hiddenimports += collect_submodules('fabio')
hiddenimports += collect_submodules('hdf5plugin')


block_cipher = None


silx_a = Analysis(
    ['bootstrap.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False)

silx_view_a = Analysis(
    ['bootstrap-silx-view.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False)

MERGE(
    (silx_a, 'silx', os.path.join('silx', 'silx')),
    (silx_view_a, 'silx-view', os.path.join('silx-view', 'silx-view')),
)


silx_pyz = PYZ(
    silx_a.pure,
    silx_a.zipped_data,
    cipher=block_cipher)

silx_exe = EXE(
    silx_pyz,
    silx_a.scripts,
    silx_a.dependencies,
    [],
    exclude_binaries=True,
    name='silx',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    icon='silx.ico')

silx_coll = COLLECT(
    silx_exe,
    silx_a.binaries,
    silx_a.zipfiles,
    silx_a.datas,
    strip=False,
    upx=False,
    name='silx')


silx_view_pyz = PYZ(
    silx_view_a.pure,
    silx_view_a.zipped_data,
    cipher=block_cipher)

silx_view_exe = EXE(
    silx_view_pyz,
    silx_view_a.scripts,
    silx_view_a.dependencies,
    [],
    exclude_binaries=True,
    name='silx-view',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon='silx.ico')

silx_view_coll = COLLECT(
    silx_view_exe,
    silx_view_a.binaries,
    silx_view_a.zipfiles,
    silx_view_a.datas,
    strip=False,
    upx=False,
    name='silx-view')


# Fix MERGE by copying produced silx-view.exe file
def move_silx_view_exe():
    dist = Path(SPECPATH) / 'dist'
    shutil.copy2(
        src=dist / 'silx-view' / 'silx-view.exe',
        dst=dist / 'silx',
    )
    shutil.rmtree(dist / 'silx-view')

move_silx_view_exe()

# Generate license file from current Python env
def create_license_file(filename: str):
    import PyQt5.QtCore

    with open(filename, 'w') as f:
        f.write(f"""
This is free software.

This distribution of silx is provided under the
GNU General Public License v3 (https://www.gnu.org/licenses/gpl-3.0.en.html) since it includes PyQt5.

It includes mainy software packages with different licenses:

- Python ({sys.version}): PSF license, https://www.python.org/
- Qt ({PyQt5.QtCore.qVersion()}): GNU Lesser General Public License v3, https://www.qt.io/
""")

        for dist in sorted(pkg_resources.working_set, key=lambda d: d.key):
            license = importlib.metadata.metadata(dist.key).get('License')
            homepage = importlib.metadata.metadata(dist.key).get('Home-page')
            info = ", ".join(info for info in (license, homepage) if info)
            f.write(f"- {dist.project_name} ({importlib.metadata.version(dist.key)}): {info}\n")

create_license_file('LICENSE')

# Run innosetup
def innosetup():
    from silx import version

    config_name = "create-installer.iss"
    with open(config_name + '.template') as f:
        content = f.read().replace("#Version", version)
    with open(config_name, "w") as f:
        f.write(content)

    subprocess.call(["iscc", os.path.join(SPECPATH, config_name)])
    os.remove(config_name)

innosetup()
