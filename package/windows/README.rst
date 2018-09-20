Generate silx fat binary for Windows
====================================

Pre-requisites
--------------

- PyInstaller must be installed.
  It is best to use the development version to use package specific hooks up-to-date.
  Run either::
  
    pip install -r requirements-dev.txt

  or::

    pip install https://github.com/pyinstaller/pyinstaller/archive/develop.zip

- silx and all its dependencies must be INSTALLED::

    pip install silx[full]
 
  or from the source directory::

    pip install .[full]


Procedure
---------

- Go to the `package/windows` folder in the source directory
- Run `pyinstaller pyinstaller.spec`.
  This generates a fat binary in `package/windows/dist/silx/` for the generic launcher `silx.exe`.
- Run `pyinstaller pyinstaller-silx-view.spec`.
  This generates a fat binary in `package/windows/dist/silx-view/` for the silx view command `silx-view.exe`.
- Copy `silx-view.exe` and `silx-view.exe.manifest` to `package/windows/dist/silx/`.
  This is a hack until PyInstaller supports multiple executables (see https://github.com/pyinstaller/pyinstaller/issues/1527).
- Zip `package\windows\dist\silx` to make the application available as a single zip file.

