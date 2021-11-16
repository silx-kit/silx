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
