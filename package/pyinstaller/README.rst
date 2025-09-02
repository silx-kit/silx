Generate silx installer
=======================

Pre-requisites
--------------

On all platforms, the following packages **must be installed**:

- silx and all its dependencies::

    pip install silx[full]

  or from the source directory::

    pip install .[full]

- `PyInstaller <https://pyinstaller.readthedocs.io/>`_
  
    pip install pyinstaller


Windows
-------

- Go to the ``package/pyinstaller`` folder in the source directory.
- Run ``pyinstaller pyinstaller.spec``. This will generate the executable in 
  ``package/pyinstaller/dist``. It will then run InnoSetup to create the
  installer in ``package/pyinstsaller/artifacts``.

Troubleshooting
~~~~~~~~~~~~~~~

In case of issues with anti-virus during the build process, try to re-install PyInstaller
from source and rebuild the bootloader:

```
SET PYINSTALLER_COMPILE_BOOTLOADER=1
SET PYINSTALLER_BOOTLOADER_WAF_ARGS=--msvc_target=x64
pip install pyinstaller --no-binary pyinstaller
```

macOS
-----

- Go to the ``package/pyinstaller`` folder in the source directory.
- Run ``pyinstaller pyinstaller.spec``. This will generate the app bundle in
  ``package/pyinstaller/dist`` and the dmg in ``package/pyinstaller/artifacts``.

On macOS, the scripts will also try to sign the application and notarize it if
the following environment variables are set:

- ``APPLE_ID``: The Apple ID used generate the certificate and application-specific password.
- ``APPLE_TEAM_ID``: The Apple Team ID associated with the Apple ID.
- ``CERTIFICATE_BASE64``: The Apple Developer ID Application Certificate exported as a .p12 file and encoded in base64.
- ``CERTIFICATE_PASSWORD``: The password to decode the .p12 file as set when exporting the certificate.
- ``KEYCHAIN_PASSWORD``: The password used in the temporary keychain created to import the certificate. It can be any value, and is not related to the Apple ID or certificate passwords. It is only used to protect the temporary keychain.
- ``APPLICATION_SPECIFIC_PASSWORD`: An application-specific password generated for the Apple ID and used to connect to the notarization service.

A number of steps need to be performed to create the certificate and application-specific password. Step-by-step instructions can be found here: `Mac Signing and Notarization Demo <https://github.com/omkarcloud/macos-code-signing-example>`_. Even though the instructions are for a different application-type, the steps to create the required credentials are the same.