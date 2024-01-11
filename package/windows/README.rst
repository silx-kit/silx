Generate silx fat binary for Windows
====================================

Pre-requisites
--------------

- `PyInstaller <https://pyinstaller.readthedocs.io/>`_ must be installed.
  Run: ``pip install -r requirements-dev.txt``
- `InnoSetup <https://jrsoftware.org/isinfo.php>`_ must be installed and in the ``PATH``.
- silx and all its dependencies must be INSTALLED::

    pip install silx[full]

  or from the source directory::

    pip install .[full]


Procedure
---------

- Go to the ``package/windows`` folder in the source directory
- Run ``pyinstaller pyinstaller.spec``.
  This will generates the fat binary in ``package/windows/dist/``.
  It will then run InnoSetup to create the installer in ``package/windows/artifacts/``.


Troubleshooting
---------------

In case of issues with anti-virus during the build process, try to re-install PyInstaller
from source and rebuild the bootloader:

```
SET PYINSTALLER_COMPILE_BOOTLOADER=1
SET PYINSTALLER_BOOTLOADER_WAF_ARGS=--msvc_target=x64
pip install pyinstaller --no-binary pyinstaller
```