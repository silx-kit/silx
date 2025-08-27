# -*- mode: python -*-
import importlib.metadata
import os.path
import shutil
import subprocess
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

from silx import strictversion

if sys.platform == "darwin":
    icon = "silx.icns"
elif sys.platform == "win32":
    icon = "silx.ico"
else:
    raise RuntimeError("Unsupported platform")

icon = os.path.join(os.getcwd(), icon)

datas = []

PROJECT_PATH = os.path.abspath(os.path.join(SPECPATH, "..", ".."))
datas.append((os.path.join(PROJECT_PATH, "README.rst"), "."))
datas.append((os.path.join(PROJECT_PATH, "LICENSE"), "."))
datas.append((os.path.join(PROJECT_PATH, "copyright"), "."))
datas += collect_data_files("silx.resources")

hiddenimports = ["hdf5plugin"]
hiddenimports += collect_submodules("fabio")

block_cipher = None

silx_a = Analysis(
    ["bootstrap.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    cipher=block_cipher,
    noarchive=False,
)

silx_pyz = PYZ(silx_a.pure, silx_a.zipped_data, cipher=block_cipher)

silx_exe = EXE(
    silx_pyz,
    silx_a.scripts,
    silx_a.dependencies,
    [],
    exclude_binaries=True,
    name="silx",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    icon=icon,
)

silx_view_a = Analysis(
    ["bootstrap-silx-view.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    cipher=block_cipher,
    noarchive=False,
)

silx_view_pyz = PYZ(silx_view_a.pure, silx_view_a.zipped_data, cipher=block_cipher)

silx_view_exe = EXE(
    silx_view_pyz,
    silx_view_a.scripts,
    silx_view_a.dependencies,
    [],
    exclude_binaries=True,
    name="silx-view",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=icon,
)
if sys.platform == "win32":
    silx_coll = COLLECT(
        silx_view_exe,
        silx_view_a.binaries,
        silx_view_a.zipfiles,
        silx_view_a.datas,
        silx_exe,
        silx_a.binaries,
        silx_a.zipfiles,
        silx_a.datas,
        strip=False,
        upx=False,
        name="silx",
    )
elif sys.platform == "darwin":
    # macOS application-bundle only for silx-view.
    silx_view_coll = COLLECT(
        silx_view_exe,
        silx_view_a.binaries,
        silx_view_a.zipfiles,
        silx_view_a.datas,
        strip=False,
        upx=False,
        name="silx-view",
    )

    app = BUNDLE(
        silx_view_coll,
        name="silx-view.app",
        icon=icon,
        bundle_identifier="org.silx.silxview",
        info_plist={
            "CFBundleIdentifier": "org.silx",
            "CFBundleShortVersionString": strictversion,
            "CFBundleVersion": "silx-view " + strictversion,
            "LSTypeIsPackage": True,
            "LSMinimumSystemVersion": "10.13.0",
            "NSHumanReadableCopyright": "MIT",
            "NSHighResolutionCapable": True,
            "NSPrincipalClass": "NSApplication",
            "NSAppleScriptEnabled": False,
        },
    )


def create_license_file(filename: str):
    """Generate a LICENSE file with the licenses of the main dependencies."""
    import PySide6.QtCore

    with open(filename, "w") as f:
        f.write(
            f"""
This is free software.

It includes mainy software packages with different licenses:

- Python ({sys.version}): PSF license, https://www.python.org/
- Qt ({PySide6.QtCore.qVersion()}): GNU Lesser General Public License v3, https://www.qt.io/
"""
        )

        for dist in sorted(
            importlib.metadata.distributions(), key=lambda d: d.name.lower()
        ):
            license = dist.metadata.get("License")
            homepage = dist.metadata.get("Home-page")
            info = ", ".join(info for info in (license, homepage) if info)
            f.write(f"- {dist.name} ({dist.version}): {info}\n")


def run_script(script_name: str, description: str):
    """Run a bash script and exit on failure."""
    print(f"{description}...")
    if subprocess.call(["bash", script_name]) != 0:
        print(f"Error: {description} failed.")
        sys.exit(1)


def innosetup():
    """Create an installer using Inno Setup."""
    from silx import strictversion

    config_name = "create-installer.iss"
    with open(config_name + ".template") as f:
        content = f.read().replace("#Version", strictversion)
    with open(config_name, "w") as f:
        f.write(content)
    subprocess.call(["iscc", os.path.join(SPECPATH, config_name)])
    os.remove(config_name)


def make_zip():
    """Create a zip archive of the installer."""
    from silx import strictversion

    base_name = os.path.join(
        SPECPATH, "artifacts", f"silx-{strictversion}-windows-application"
    )
    shutil.make_archive(
        base_name,
        format="zip",
        root_dir=os.path.join(SPECPATH, "dist"),
        base_dir="silx",
    )


create_license_file("LICENSE")


if sys.platform == "darwin":
    run_script("codesign.sh", "Codesigning the application")
    run_script("create-dmg.sh", "Packing the application in a .dmg image")
    run_script("notarize.sh", "Submitting the image for notarization")

    # Rename the created .dmg image.
    os.rename(
        os.path.join("artifacts", "silx-view.dmg"),
        os.path.join("artifacts", f"silx-view-{strictversion}.dmg"),
    )
elif sys.platform == "win32":
    innosetup()
    make_zip()

os.remove("LICENSE")
