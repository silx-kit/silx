# -*- mode: python -*-
import os.path
from PyInstaller.utils.hooks import collect_data_files

datas = []

PROJECT_PATH = os.path.abspath(os.path.join(SPECPATH, "..", ".."))
datas.append((os.path.join(PROJECT_PATH, "README.rst"), "."))
datas.append((os.path.join(PROJECT_PATH, "LICENSE"), "."))
datas.append((os.path.join(PROJECT_PATH, "copyright"), "."))


datas += collect_data_files("silx.resources")


block_cipher = None


a = Analysis(['bootstrap.py'],
             pathex=[],
             binaries=[],
             datas=datas,
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)

pyz = PYZ(a.pure,
          a.zipped_data,
          cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='silx',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=False,
          console=True,
          icon='silx.ico')

coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=False,
               name='silx')
