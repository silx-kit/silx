#!/usr/bin/env python3
# /*##########################################################################
#
# Copyright (c) 2015-2023 European Synchrotron Radiation Facility
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ###########################################################################*/

__authors__ = ["Jérôme Kieffer", "Thomas Vincent"]
__date__ = "07/11/2022"
__license__ = "MIT"

import sys
import os
import platform
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("silx.setup")

from setuptools import Command, Extension, find_packages
from setuptools.command.sdist import sdist
from setuptools.command.build_ext import build_ext

try:
    import numpy
except ImportError:
    raise ImportError(
        "To install this package, you must install numpy first\n"
        "(See https://pypi.org/project/numpy)")


PROJECT = "silx"
if sys.version_info.major < 3:
    logger.error(PROJECT + " no longer supports Python2")

if "LANG" not in os.environ and sys.platform == "darwin":
    print("""WARNING: the LANG environment variable is not defined,
an utf-8 LANG is mandatory to use setup.py, you may face unexpected UnicodeError.
export LANG=en_US.utf-8
export LC_ALL=en_US.utf-8
""")


def get_version(debian=False):
    """Returns current version number from _version.py file"""
    dirname = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "src", PROJECT)
    sys.path.insert(0, dirname)
    import _version
    sys.path = sys.path[1:]
    return _version.debianversion if debian else _version.strictversion


def get_readme():
    """Returns content of README.rst file"""
    dirname = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(dirname, "README.rst")
    with open(filename, "r", encoding="utf-8") as fp:
        long_description = fp.read()
    return long_description


classifiers = ["Development Status :: 5 - Production/Stable",
               "Environment :: Console",
               "Environment :: MacOS X",
               "Environment :: Win32 (MS Windows)",
               "Environment :: X11 Applications :: Qt",
               "Intended Audience :: Education",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: MIT License",
               "Natural Language :: English",
               "Operating System :: MacOS",
               "Operating System :: Microsoft :: Windows",
               "Operating System :: POSIX",
               "Programming Language :: Cython",
               "Programming Language :: Python :: 3",
               "Programming Language :: Python :: Implementation :: CPython",
               "Topic :: Scientific/Engineering :: Physics",
               "Topic :: Software Development :: Libraries :: Python Modules",
               ]


class BuildMan(Command):
    """Command to build man pages"""

    description = "Build man pages of the provided entry points"

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def entry_points_iterator(self):
        """Iterate other entry points available on the project."""
        entry_points = self.distribution.entry_points
        console_scripts = entry_points.get('console_scripts', [])
        gui_scripts = entry_points.get('gui_scripts', [])
        scripts = []
        scripts.extend(console_scripts)
        scripts.extend(gui_scripts)
        for script in scripts:
            # Remove ending extra dependencies
            script = script.split("[")[0]
            elements = script.split("=")
            target_name = elements[0].strip()
            elements = elements[1].split(":")
            module_name = elements[0].strip()
            function_name = elements[1].strip()
            yield target_name, module_name, function_name

    def run_targeted_script(self, target_name, script_name, env, log_output=False):
        """Execute targeted script using --help and --version to help checking
        errors. help2man is not very helpful to do it for us.

        :return: True is both return code are equal to 0
        :rtype: bool
        """
        import subprocess

        if log_output:
            extra_args = {}
        else:
            try:
                # Python 3
                from subprocess import DEVNULL
            except ImportError:
                # Python 2
                import os
                DEVNULL = open(os.devnull, 'wb')
            extra_args = {'stdout': DEVNULL, 'stderr': DEVNULL}

        succeeded = True
        command_line = [sys.executable, script_name, "--help"]
        if log_output:
            logger.info("See the following execution of: %s", " ".join(command_line))
        p = subprocess.Popen(command_line, env=env, **extra_args)
        status = p.wait()
        if log_output:
            logger.info("Return code: %s", status)
        succeeded = succeeded and status == 0
        command_line = [sys.executable, script_name, "--version"]
        if log_output:
            logger.info("See the following execution of: %s", " ".join(command_line))
        p = subprocess.Popen(command_line, env=env, **extra_args)
        status = p.wait()
        if log_output:
            logger.info("Return code: %s", status)
        succeeded = succeeded and status == 0
        return succeeded

    @staticmethod
    def _write_script(target_name, lst_lines=None):
        """Write a script to a temporary file and return its name
        :paran target_name: base of the script name
        :param lst_lines: list of lines to be written in the script
        :return: the actual filename of the script (for execution or removal)
        """
        import tempfile
        import stat
        script_fid, script_name = tempfile.mkstemp(prefix="%s_" % target_name, text=True)
        with os.fdopen(script_fid, 'wt') as script:
            for line in lst_lines:
                if not line.endswith("\n"):
                    line += "\n"
                script.write(line)
        # make it executable
        mode = os.stat(script_name).st_mode
        os.chmod(script_name, mode + stat.S_IEXEC)
        return script_name

    def get_synopsis(self, module_name, env, log_output=False):
        """Execute a script to retrieve the synopsis for help2man
        :return: synopsis
        :rtype: single line string
        """
        import subprocess
        script_name = None
        synopsis = None
        script = ["#!%s\n" % sys.executable,
                  "import logging",
                  "logging.basicConfig(level=logging.ERROR)",
                  "import %s as app" % module_name,
                  "print(app.__doc__)"]
        try:
            script_name = self._write_script(module_name, script)
            command_line = [sys.executable, script_name]
            p = subprocess.Popen(command_line, env=env, stdout=subprocess.PIPE)
            status = p.wait()
            if status != 0:
                logger.warning("Error while getting synopsis for module '%s'.", module_name)
            synopsis = p.stdout.read().decode("utf-8").strip()
            if synopsis == 'None':
                synopsis = None
        finally:
            # clean up the script
            if script_name is not None:
                os.remove(script_name)
        return synopsis

    def run(self):
        build = self.get_finalized_command('build')
        path = sys.path
        path.insert(0, os.path.abspath(build.build_lib))

        env = dict((str(k), str(v)) for k, v in os.environ.items())
        env["PYTHONPATH"] = os.pathsep.join(path)
        if not os.path.isdir("build/man"):
            os.makedirs("build/man")
        import subprocess
        import tempfile
        import stat
        script_name = None
        workdir = tempfile.mkdtemp()

        entry_points = self.entry_points_iterator()
        for target_name, module_name, function_name in entry_points:
            logger.info("Build man for entry-point target '%s'" % target_name)
            # help2man expect a single executable file to extract the help
            # we create it, execute it, and delete it at the end

            try:
                # create a launcher using the right python interpreter
                script_name = os.path.join(workdir, target_name)
                with open(script_name, "wt") as script:
                    script.write("#!%s\n" % sys.executable)
                    script.write("import %s as app\n" % module_name)
                    script.write("app.%s()\n" % function_name)
                # make it executable
                mode = os.stat(script_name).st_mode
                os.chmod(script_name, mode + stat.S_IEXEC)

                # execute help2man
                man_file = "build/man/%s.1" % target_name
                command_line = ["help2man", "-N", script_name, "-o", man_file]

                synopsis = self.get_synopsis(module_name, env)
                if synopsis:
                    command_line += ["-n", synopsis]

                p = subprocess.Popen(command_line, env=env)
                status = p.wait()
                if status != 0:
                    logger.info("Error while generating man file for target '%s'.", target_name)
                    self.run_targeted_script(target_name, script_name, env, True)
                    raise RuntimeError("Fail to generate '%s' man documentation" % target_name)
            finally:
                # clean up the script
                if script_name is not None:
                    os.remove(script_name)
        os.rmdir(workdir)

# ############## #
# Compiler flags #
# ############## #


def parse_env_as_bool(key: str, default: Optional[bool]=None) -> Optional[bool]:
    """Parse `key` env. var. and convert its value to a boolean or None.

    If it cannot parse it or if None, `default` is returned.
    """
    content = os.environ.get(key, "")
    value = content.lower()
    if value in ["1", "true", "yes", "y"]:
        return True
    if value in ["0", "false", "no", "n"]:
        return False
    if value in ["none", ""]:
        return default
    msg = "Env variable '%s' contains '%s'. But a boolean or an empty \
        string was expected. Variable ignored."
    logger.warning(msg, key, content)
    return default


def get_use_openmp_from_env_var() -> bool:
    """Returns whether or not to build with OpenMP"""
    use_openmp = parse_env_as_bool("SILX_WITH_OPENMP", default=True)
    if use_openmp and platform.system() == "Darwin":
        logger.warning("OpenMP support ignored. Your platform does not support it.")
        return False
    return use_openmp


USE_OPENMP = get_use_openmp_from_env_var()
FORCE_CYTHON = parse_env_as_bool("SILX_FORCE_CYTHON", default=False)


class BuildExt(build_ext):
    """Handle extension compilation.

    Command-line argument and environment can custom:

    - The use of cython to cythonize files, else a default version is used
    - Build extension with support of OpenMP (by default it is enabled)
    - If building with MSVC, compiler flags are converted from gcc flags.
    """

    COMPILE_ARGS_CONVERTER = {'-fopenmp': '/openmp'}

    LINK_ARGS_CONVERTER = {'-fopenmp': ''}

    description = 'Build extensions'

    def patch_extension(self, ext):
        """
        Patch an extension according to requested Cython and OpenMP usage.

        :param Extension ext: An extension
        """
        # Cytonize
        from Cython.Build import cythonize
        patched_exts = cythonize(
                                 [ext],
                                 compiler_directives={'embedsignature': True,
                                 'language_level': 3},
                                 force=FORCE_CYTHON,
        )
        ext.sources = patched_exts[0].sources

        # Remove OpenMP flags if OpenMP is disabled
        if not USE_OPENMP:
            ext.extra_compile_args = [
                f for f in ext.extra_compile_args if f != '-fopenmp']
            ext.extra_link_args = [
                f for f in ext.extra_link_args if f != '-fopenmp']

        # Convert flags from gcc to MSVC if required
        if self.compiler.compiler_type == 'msvc':
            extra_compile_args = [self.COMPILE_ARGS_CONVERTER.get(f, f)
                                  for f in ext.extra_compile_args]
            # Avoid empty arg
            ext.extra_compile_args = [arg for arg in extra_compile_args if arg]

            extra_link_args = [self.LINK_ARGS_CONVERTER.get(f, f)
                               for f in ext.extra_link_args]
            # Avoid empty arg
            ext.extra_link_args = [arg for arg in extra_link_args if arg]

    def build_extensions(self):
        for ext in self.extensions:
            self.patch_extension(ext)
        build_ext.build_extensions(self)


################################################################################
# Debian source tree
################################################################################


class sdist_debian(sdist):
    """
    Tailor made sdist for debian
    * remove auto-generated doc
    * remove cython generated .c files
    * remove cython generated .cpp files
    * remove .bat files
    * include .l man files
    """

    description = "Create a source distribution for Debian (tarball, zip file, etc.)"

    @staticmethod
    def get_debian_name():
        name = "%s_%s" % (PROJECT, get_version(debian=True))
        return name

    def prune_file_list(self):
        sdist.prune_file_list(self)
        to_remove = ["doc/build", "doc/pdf", "doc/html", "pylint", "epydoc"]
        print("Removing files for debian")
        for rm in to_remove:
            self.filelist.exclude_pattern(pattern="*", anchor=False, prefix=rm)

        # this is for Cython files specifically: remove C & html files
        search_root = os.path.dirname(os.path.abspath(__file__))
        for root, _, files in os.walk(search_root):
            for afile in files:
                if os.path.splitext(afile)[1].lower() == ".pyx":
                    base_file = os.path.join(root, afile)[len(search_root) + 1:-4]
                    self.filelist.exclude_pattern(pattern=base_file + ".c")
                    self.filelist.exclude_pattern(pattern=base_file + ".cpp")
                    self.filelist.exclude_pattern(pattern=base_file + ".html")

        # do not include third_party/_local files
        self.filelist.exclude_pattern(pattern="*", prefix="silx/third_party/_local")

    def make_distribution(self):
        self.prune_file_list()
        sdist.make_distribution(self)
        dest = self.archive_files[0]
        dirname, basename = os.path.split(dest)
        base, ext = os.path.splitext(basename)
        while ext in [".zip", ".tar", ".bz2", ".gz", ".Z", ".lz", ".orig"]:
            base, ext = os.path.splitext(base)
        # if ext:
        #     dest = "".join((base, ext))
        # else:
        #     dest = base
        # sp = dest.split("-")
        # base = sp[:-1]
        # nr = sp[-1]
        debian_arch = os.path.join(dirname, self.get_debian_name() + ".orig.tar.gz")
        os.rename(self.archive_files[0], debian_arch)
        self.archive_files = [debian_arch]
        print("Building debian .orig.tar.gz in %s" % self.archive_files[0])

# ##### #
# setup #
# ##### #


def get_project_configuration():
    """Returns project arguments for setup"""
    # Use installed numpy version as minimal required version
    # This is useful for wheels to advertise the numpy version they were built with
    numpy_requested_version = ">=%s" % numpy.version.version
    logger.info("Install requires: numpy %s", numpy_requested_version)

    install_requires = [
        # for most of the computation
        "numpy%s" % numpy_requested_version,
        # for the script launcher and pkg_resources
        "setuptools",
        # for io support
        "h5py",
        "fabio>=0.9",
        ]

    # extras requirements: target 'full' to install all dependencies at once
    full_requires = [
        # opencl
        'pyopencl',
        'Mako',
        # gui
        'qtconsole',
        'matplotlib>=1.2.0',
        'PyOpenGL',
        'python-dateutil',
        'PyQt5',
        # extra
        'hdf5plugin',
        'scipy',
        'Pillow',
        'bitshuffle']

    test_requires = [
        "pytest",
        "pytest-xvfb",
        "pytest-mock",
        'bitshuffle'
    ]

    extras_require = {
        'full': full_requires,
        'test': test_requires,
    }

    # Here for packaging purpose only
    # Setting the SILX_FULL_INSTALL_REQUIRES environment variable
    # put all dependencies as install_requires
    if os.environ.get('SILX_FULL_INSTALL_REQUIRES') is not None:
        install_requires += full_requires

    # Set the SILX_INSTALL_REQUIRES_STRIP env. var. to a comma-separated
    # list of package names to remove them from install_requires
    install_requires_strip = os.environ.get('SILX_INSTALL_REQUIRES_STRIP')
    if install_requires_strip is not None:
        for package_name in install_requires_strip.split(','):
            install_requires.remove(package_name)


    package_data = {
        # Resources files for silx
        'silx.resources': [
            'gui/logo/*.png',
            'gui/logo/*.svg',
            'gui/icons/*.png',
            'gui/icons/*.svg',
            'gui/icons/*.mng',
            'gui/icons/*.gif',
            'gui/icons/*/*.png',
            'opencl/*.cl',
            'opencl/image/*.cl',
            'opencl/sift/*.cl',
            'opencl/codec/*.cl',
            'gui/colormaps/*.npy'],
        'silx.examples': ['*.png'],
    }

    entry_points = {
        'console_scripts': ['silx = silx.__main__:main'],
        # 'gui_scripts': [],
    }

    cmdclass = dict(
        build_ext=BuildExt,
        build_man=BuildMan,
        debian_src=sdist_debian)

    def silx_io_specfile_define_macros():
        # Locale and platform management
        if sys.platform == "win32":
            return [('WIN32', None), ('SPECFILE_POSIX', None)]
        elif os.name.lower().startswith('posix'):
            # the best choice is to have _GNU_SOURCE defined
            # as a compilation flag because that allows the
            # use of strtod_l
            use_gnu_source = os.environ.get("SPECFILE_USE_GNU_SOURCE", "False")
            if use_gnu_source in ("True", "1"):  # 1 was the initially supported value
                return [('_GNU_SOURCE', 1)]
            return [('SPECFILE_POSIX', None)]
        else:
            return []

    ext_modules = [

        # silx.image

        Extension(
            name='silx.image.bilinear',
            sources=["src/silx/image/bilinear.pyx"],
            language='c',
        ),
        Extension(
            name='silx.image.marchingsquares._mergeimpl',
            sources=['src/silx/image/marchingsquares/_mergeimpl.pyx'],
            include_dirs=[
                numpy.get_include(),
                os.path.join(os.path.dirname(__file__), "src", "silx", "utils", "include")
            ],
            language='c++',
            extra_link_args=['-fopenmp'],
            extra_compile_args=['-fopenmp'],
        ),
        Extension(
            name='silx.image.shapes',
            sources=["src/silx/image/shapes.pyx"],
            language='c',
        ),

        # silx.io

        Extension(
            name='silx.io.specfile',
            sources=[
                'src/silx/io/specfile/src/sfheader.c',
                'src/silx/io/specfile/src/sfinit.c',
                'src/silx/io/specfile/src/sflists.c',
                'src/silx/io/specfile/src/sfdata.c',
                'src/silx/io/specfile/src/sfindex.c',
                'src/silx/io/specfile/src/sflabel.c',
                'src/silx/io/specfile/src/sfmca.c',
                'src/silx/io/specfile/src/sftools.c',
                'src/silx/io/specfile/src/locale_management.c',
                'src/silx/io/specfile.pyx',
            ],
            define_macros=silx_io_specfile_define_macros(),
            include_dirs=['src/silx/io/specfile/include'],
            language='c',
        ),

        # silx.math

        Extension(
            name='silx.math._colormap',
            sources=["src/silx/math/_colormap.pyx"],
            language='c',
            include_dirs=[
                'src/silx/math/include',
                numpy.get_include(),
            ],
            extra_link_args=['-fopenmp'],
            extra_compile_args=['-fopenmp'],
        ),
        Extension(
            name='silx.math.chistogramnd',
            sources=[
                'src/silx/math/histogramnd/src/histogramnd_c.c',
                'src/silx/math/chistogramnd.pyx',
            ],
            include_dirs=[
                'src/silx/math/histogramnd/include',
                numpy.get_include(),
            ],
            language='c',
        ),
        Extension(
            name='silx.math.chistogramnd_lut',
            sources=['src/silx/math/chistogramnd_lut.pyx'],
            include_dirs=[
                'src/silx/math/histogramnd/include',
                numpy.get_include(),
            ],
            language='c',
        ),
        Extension(
            name='silx.math.combo',
            sources=['src/silx/math/combo.pyx'],
            include_dirs=['src/silx/math/include'],
            language='c',
        ),
        Extension(
            name='silx.math.interpolate',
            sources=["src/silx/math/interpolate.pyx"],
            language='c',
            include_dirs=[
                'src/silx/math/include',
                numpy.get_include(),
            ],
            extra_link_args=['-fopenmp'],
            extra_compile_args=['-fopenmp'],
        ),
        Extension(
            name='silx.math.marchingcubes',
            sources=[
                'src/silx/math/marchingcubes/mc_lut.cpp',
                'src/silx/math/marchingcubes.pyx',
            ],
            include_dirs=[
                'src/silx/math/marchingcubes',
                numpy.get_include(),
            ],
            language='c++',
        ),
        Extension(
            name='silx.math.medianfilter.medianfilter',
            sources=['src/silx/math/medianfilter/medianfilter.pyx'],
            include_dirs=[
                'src/silx/math/medianfilter/include',
                numpy.get_include(),
            ],
            language='c++',
            extra_link_args=['-fopenmp'],
            extra_compile_args=['-fopenmp'],
        ),

        # silx.math.fit

        Extension(
            name='silx.math.fit.filters',
            sources=[
                'src/silx/math/fit/filters/src/smoothnd.c',
                'src/silx/math/fit/filters/src/snip1d.c',
                'src/silx/math/fit/filters/src/snip2d.c',
                'src/silx/math/fit/filters/src/snip3d.c',
                'src/silx/math/fit/filters/src/strip.c',
                'src/silx/math/fit/filters.pyx',
            ],
            include_dirs=['src/silx/math/fit/filters/include'],
            language='c',
        ),
        Extension(
            name='silx.math.fit.functions',
            sources=[
                'src/silx/math/fit/functions/src/funs.c',
                'src/silx/math/fit/functions.pyx',
            ],
            include_dirs=['src/silx/math/fit/functions/include'],
            language='c',
        ),
        Extension(
            name='silx.math.fit.peaks',
            sources=[
                'src/silx/math/fit/peaks/src/peaks.c',
                'src/silx/math/fit/peaks.pyx',
            ],
            include_dirs=['src/silx/math/fit/peaks/include'],
            language='c',
        ),
    ]

    return dict(
        name=PROJECT,
        version=get_version(),
        license="MIT",
        url="http://www.silx.org/",
        author="data analysis unit",
        author_email="silx@esrf.fr",
        classifiers=classifiers,
        description="Software library for X-ray data analysis",
        long_description=get_readme(),
        install_requires=install_requires,
        extras_require=extras_require,
        python_requires='>=3.7',
        cmdclass=cmdclass,
        zip_safe=False,
        entry_points=entry_points,
        packages=find_packages(where='src', include=['silx*']) + ['silx.examples'],
        package_dir={
            "": "src",
            "silx.examples": "examples",
        },
        ext_modules=ext_modules,
        package_data=package_data,
    )


if __name__ == "__main__":
    from setuptools import setup

    setup(**get_project_configuration())
