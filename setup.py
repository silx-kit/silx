#!/usr/bin/python
# coding: utf8
# /*##########################################################################
#
# Copyright (c) 2015-2017 European Synchrotron Radiation Facility
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
__date__ = "26/04/2017"
__license__ = "MIT"


# This import is here only to fix a bug on Debian 7 with python2.7
# Without this, the system io module is not loaded from numpy.distutils
# the silx.io module seems to be loaded instead
import io

import sys
import os
import platform
import shutil
import logging
import glob

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("silx.setup")


from distutils.command.clean import clean as Clean
try:
    from setuptools import Command
    from setuptools.command.build_py import build_py as _build_py
    from setuptools.command.build_ext import build_ext
    from setuptools.command.sdist import sdist
    logger.info("Use setuptools")
except ImportError:
    try:
        from numpy.distutils.core import Command
    except ImportError:
        from distutils.core import Command
    from distutils.command.build_py import build_py as _build_py
    from distutils.command.build_ext import build_ext
    from distutils.command.sdist import sdist
    logger.info("Use distutils")

try:
    import sphinx
    import sphinx.util.console
    sphinx.util.console.color_terminal = lambda: False
    from sphinx.setup_command import BuildDoc
except ImportError:
    sphinx = None


USE_OPENMP = None
"""Refere if the compilation will use OpenMP or not.
It have to be initialized before the setup."""


PROJECT = "silx"

if "LANG" not in os.environ and sys.platform == "darwin" and sys.version_info[0] > 2:
    print("""WARNING: the LANG environment variable is not defined,
an utf-8 LANG is mandatory to use setup.py, you may face unexpected UnicodeError.
export LANG=en_US.utf-8
export LC_ALL=en_US.utf-8
""")


def get_version():
    """Returns current version number from version.py file"""
    import version
    return version.strictversion


def get_readme():
    """Returns content of README.rst file"""
    dirname = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(dirname, "README.rst")
    with io.open(filename, "r", encoding="utf-8") as fp:
        long_description = fp.read()
    return long_description


classifiers = ["Development Status :: 4 - Beta",
               "Environment :: Console",
               "Environment :: MacOS X",
               "Environment :: Win32 (MS Windows)",
               "Environment :: X11 Applications :: Qt",
               "Intended Audience :: Education",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: MIT License",
               "License :: OSI Approved :: GNU Lesser General Public License v2 or later (LGPLv2+)",
               "Natural Language :: English",
               "Operating System :: MacOS",
               "Operating System :: Microsoft :: Windows",
               "Operating System :: POSIX",
               "Programming Language :: Cython",
               "Programming Language :: Python :: 2.7",
               "Programming Language :: Python :: 3.4",
               "Programming Language :: Python :: 3.5",
               "Programming Language :: Python :: Implementation :: CPython",
               "Topic :: Scientific/Engineering :: Physics",
               "Topic :: Software Development :: Libraries :: Python Modules",
               ]


# ########## #
# version.py #
# ########## #

class build_py(_build_py):
    """
    Enhanced build_py which copies version.py to <PROJECT>._version.py
    """
    def find_package_modules(self, package, package_dir):
        modules = _build_py.find_package_modules(self, package, package_dir)
        if package == PROJECT:
            modules.append((PROJECT, '_version', 'version.py'))
        return modules


########
# Test #
########

class PyTest(Command):
    """Command to start tests running the script: run_tests.py -i"""
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import subprocess
        errno = subprocess.call([sys.executable, 'run_tests.py', '-i'])
        if errno != 0:
            raise SystemExit(errno)


# ################### #
# build_doc command   #
# ################### #

if sphinx is None:
    class SphinxExpectedCommand(Command):
        """Command to inform that sphinx is missing"""
        user_options = []

        def initialize_options(self):
            pass

        def finalize_options(self):
            pass

        def run(self):
            raise RuntimeError(
                'Sphinx is required to build or test the documentation.\n'
                'Please install Sphinx (http://www.sphinx-doc.org).')


class BuildMan(Command):
    """Command to build man pages"""
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        build = self.get_finalized_command('build')
        path = sys.path
        path.insert(0, os.path.abspath(build.build_lib))

        env = dict((str(k), str(v)) for k, v in os.environ.items())
        env["PYTHONPATH"] = os.pathsep.join(path)

        import subprocess

        status = subprocess.call(["mkdir", "-p", "build/man"])
        if status != 0:
            raise RuntimeError("Fail to create build/man directory")

        try:
            import tempfile
            import stat
            script_name = None

            # help2man expect a single executable file to extract the help
            # we create it, execute it, and delete it at the end

            # create a launcher using the right python interpreter
            script_fid, script_name = tempfile.mkstemp(prefix="%s_" % PROJECT, text=True)
            script = os.fdopen(script_fid, 'wt')
            script.write("#!%s\n" % sys.executable)
            script.write("import runpy\n")
            script.write("runpy.run_module('%s', run_name='__main__')\n" % PROJECT)
            script.close()

            # make it executable
            mode = os.stat(script_name).st_mode
            os.chmod(script_name, mode + stat.S_IEXEC)

            # execute help2man
            p = subprocess.Popen(["help2man", script_name, "-o", "build/man/silx.1"], env=env)
            status = p.wait()
            if status != 0:
                raise RuntimeError("Fail to generate man documentation")
        finally:
            # clean up the script
            if script_name is not None:
                os.remove(script_name)


if sphinx is not None:
    class BuildDocCommand(BuildDoc):
        """Command to build documentation using sphinx.

        Project should have already be built.
        """

        def run(self):
            # make sure the python path is pointing to the newly built
            # code so that the documentation is built on this and not a
            # previously installed version

            build = self.get_finalized_command('build')
            sys.path.insert(0, os.path.abspath(build.build_lib))

            # # Copy .ui files to the path:
            # dst = os.path.join(
            #     os.path.abspath(build.build_lib), "silx", "gui")
            # if not os.path.isdir(dst):
            #     os.makedirs(dst)
            # for i in os.listdir("gui"):
            #     if i.endswith(".ui"):
            #         src = os.path.join("gui", i)
            #         idst = os.path.join(dst, i)
            #         if not os.path.exists(idst):
            #             shutil.copy(src, idst)

            # Build the Users Guide in HTML and TeX format
            for builder in ['html', 'latex']:
                self.builder = builder
                self.builder_target_dir = os.path.join(self.build_dir, builder)
                self.mkpath(self.builder_target_dir)
                BuildDoc.run(self)
            sys.path.pop(0)
else:
    BuildDocCommand = SphinxExpectedCommand


# ################### #
# test_doc command    #
# ################### #

if sphinx is not None:
    class TestDocCommand(BuildDoc):
        """Command to test the documentation using sphynx doctest.

        http://www.sphinx-doc.org/en/1.4.8/ext/doctest.html
        """
        def run(self):
            # make sure the python path is pointing to the newly built
            # code so that the documentation is built on this and not a
            # previously installed version

            build = self.get_finalized_command('build')
            sys.path.insert(0, os.path.abspath(build.build_lib))

            # Build the Users Guide in HTML and TeX format
            for builder in ['doctest']:
                self.builder = builder
                self.builder_target_dir = os.path.join(self.build_dir, builder)
                self.mkpath(self.builder_target_dir)
                BuildDoc.run(self)
            sys.path.pop(0)

else:
    TestDocCommand = SphinxExpectedCommand


# ############## #
# OpenMP support #
# ############## #

def check_openmp():
    """Do we compile with OpenMP?

    Store the result in WITH_OPENMP environment variable

    TODO: It would be much better to take care of command line arguments in the
          initialize_options and finalize_options.

    :return: True if available and not disabled.
    """
    if "WITH_OPENMP" in os.environ:
        return os.environ["WITH_OPENMP"] == "False"

    elif "--no-openmp" in sys.argv:
        sys.argv.remove("--no-openmp")
        os.environ["WITH_OPENMP"] = "False"
        print("No OpenMP requested by command line")
        return False

    elif ("--openmp" in sys.argv):
        sys.argv.remove("--openmp")
        os.environ["WITH_OPENMP"] = "True"
        print("OpenMP requested by command line")
        return True

    if platform.system() == "Darwin":
        # By default Xcode5 & XCode6 do not support OpenMP, Xcode4 is OK.
        osx = tuple([int(i) for i in platform.mac_ver()[0].split(".")])
        if osx >= (10, 8):
            os.environ["WITH_OPENMP"] = "False"
            return False

    os.environ["WITH_OPENMP"] = "True"
    return True


# ############## #
# Cython support #
# ############## #

def check_cython(min_version=None):
    """
    Check if cython must be activated fron te command line or the environment.

    Store the result in WITH_CYTHON environment variable.

    TODO: It would be much better to take care of command line arguments in the
          initialize_options and finalize_options.

    :param string min_version: Minimum version of Cython requested
    :return: True if available and not disabled.
    """

    if "WITH_CYTHON" in os.environ:
        if os.environ["WITH_CYTHON"] in ["False", "0", 0]:
            os.environ["WITH_CYTHON"] = "False"
            return False

    if "--no-cython" in sys.argv:
        sys.argv.remove("--no-cython")
        print("No Cython requested by command line")
        os.environ["WITH_CYTHON"] = "False"
        return False

    try:
        import Cython.Compiler.Version
    except ImportError:
        os.environ["WITH_CYTHON"] = "False"
        return False
    else:
        if min_version and Cython.Compiler.Version.version < min_version:
            os.environ["WITH_CYTHON"] = "False"
            return False

    os.environ["WITH_CYTHON"] = "True"

    if "--force-cython" in sys.argv:
        sys.argv.remove("--force-cython")
        print("Force Cython re-generation requested by command line")
        os.environ["FORCE_CYTHON"] = "True"
    return True


# ############################# #
# numpy.distutils Configuration #
# ############################# #

def configuration(parent_package='', top_path=None):
    """Recursive construction of package info to be used in setup().

    See http://docs.scipy.org/doc/numpy/reference/distutils.html#numpy.distutils.misc_util.Configuration
    """
    try:
        from numpy.distutils.misc_util import Configuration
    except ImportError:
        raise ImportError(
            "To install this package, you must install numpy first\n"
            "(See https://pypi.python.org/pypi/numpy)")
    config = Configuration(None, parent_package, top_path)
    config.set_options(
        ignore_setup_xxx_py=True,
        assume_default_configuration=True,
        delegate_options_to_subpackages=True,
        quiet=True)
    config.add_subpackage(PROJECT)
    return config

# ############## #
# Compiler flags #
# ############## #


class BuildExtFlags(build_ext):
    """Handle compiler and linker flags.

    If OpenMP is disabled, it removes OpenMP compile flags.
    If building with MSVC, compiler flags are converted from gcc flags.
    """

    COMPILE_ARGS_CONVERTER = {'-fopenmp': '/openmp'}

    LINK_ARGS_CONVERTER = {'-fopenmp': ' '}

    def build_extensions(self):
        # Remove OpenMP flags if OpenMP is disabled
        if not USE_OPENMP:
            for ext in self.extensions:
                ext.extra_compile_args = [
                    f for f in ext.extra_compile_args if f != '-fopenmp']
                ext.extra_link_args = [
                    f for f in ext.extra_link_args if f != '-fopenmp']

        # Convert flags from gcc to MSVC if required
        if self.compiler.compiler_type == 'msvc':
            for ext in self.extensions:
                ext.extra_compile_args = [self.COMPILE_ARGS_CONVERTER.get(f, f)
                                          for f in ext.extra_compile_args]
                ext.extra_link_args = [self.LINK_ARGS_CONVERTER.get(f, f)
                                       for f in ext.extra_link_args]

        build_ext.build_extensions(self)


def fake_cythonize(extensions):
    """Replace cython files by .c or .cpp files in extension's sources.

    It replaces the *.pyx and *.py source files of the extensions
    to either *.cpp or *.c source files.
    No compilation is performed.

    :param iterable extensions: List of extensions to patch.
    """
    for ext_module in extensions:
        new_sources = []
        for source in ext_module.sources:
            base, ext = os.path.splitext(source)
            if ext in ('.pyx', '.py'):
                if ext_module.language == 'c++':
                    source = base + '.cpp'
                else:
                    source = base + '.c'
                if not os.path.isfile(source):
                    raise RuntimeError("Source file not found: %s" % source)
            new_sources.append(source)
        ext_module.sources = new_sources

################################################################################
# Clean command
################################################################################


class CleanCommand(Clean):
    description = "Remove build artifacts from the source tree"

    def expand(self, path_list):
        """Expand a list of path using glob magic.

        :param list[str] path_list: A list of path which may contains magic
        :rtype: list[str]
        :returns: A list of path without magic
        """
        path_list2 = []
        for path in path_list:
            if glob.has_magic(path):
                iterator = glob.iglob(path)
                path_list2.extend(iterator)
            else:
                path_list2.append(path)
        return path_list2

    def run(self):
        Clean.run(self)
        # really remove the directories
        # and not only if they are empty
        to_remove = [self.build_base]
        to_remove = self.expand(to_remove)

        if not self.dry_run:
            for path in to_remove:
                try:
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                    else:
                        os.remove(path)
                    logger.info("removing '%s'", path)
                except OSError:
                    pass

################################################################################
# Debian source tree
################################################################################


class sdist_debian(sdist):
    """
    Tailor made sdist for debian
    * remove auto-generated doc
    * remove cython generated .c files
    * remove cython generated .c files
    * remove .bat files
    * include .l man files
    """
    @staticmethod
    def get_debian_name():
        import version
        name = "%s_%s" % (PROJECT, version.debianversion)
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
        if ext:
            dest = "".join((base, ext))
        else:
            dest = base
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

def get_project_configuration(dry_run):
    """Returns project arguments for setup"""
    install_requires = [
        # for most of the computation
        "numpy",
        # for the script launcher
        "setuptools"]

    setup_requires = ["setuptools", "numpy"]

    package_data = {
        'silx.resources': [
            # Add here all resources files
            'gui/icons/*.png',
            'gui/icons/*.svg',
            'gui/icons/*.mng',
            'gui/icons/*.gif',
            'gui/icons/animated/*.png',
            'opencl/sift/*.cl']
    }

    entry_points = {
        'console_scripts': ['silx = silx.__main__:main'],
        # 'gui_scripts': [],
    }

    cmdclass = dict(
        build_py=build_py,
        test=PyTest,
        build_doc=BuildDocCommand,
        test_doc=TestDocCommand,
        build_ext=BuildExtFlags,
        build_man=BuildMan,
        clean=CleanCommand,
        debian_src=sdist_debian)

    if dry_run:
        # DRY_RUN implies actions which do not require NumPy
        #
        # And they are required to succeed without Numpy for example when
        # pip is used to install silx when Numpy is not yet present in
        # the system.
        setup_kwargs = {}
    else:
        use_cython = check_cython(min_version='0.21.1')

        use_openmp = check_openmp()
        USE_OPENMP = use_openmp
        print('  ===> user open MP is %s' % USE_OPENMP)
        print('  ===> use_openmp is %s' % use_openmp)

        config = configuration()
        for ext_module in config.ext_modules:
            print('  ===> config.ext_modules is %s' % ext_module.name)
            print('  ===> %s' % ext_module.extra_compile_args)
            print('  ===> %s' % ext_module.extra_link_args)
        # exit(0)

        if use_cython:
            # Cythonize extensions
            from Cython.Build import cythonize

            config.ext_modules = cythonize(
                config.ext_modules,
                compiler_directives={'embedsignature': True},
                force=(os.environ.get("FORCE_CYTHON") is "True"),
                compile_time_env={"HAVE_OPENMP": use_openmp}
            )
        else:
            # Do not use Cython but convert source names from .pyx to .c or .cpp
            fake_cythonize(config.ext_modules)

        setup_kwargs = config.todict()

    setup_kwargs.update(name=PROJECT,
                        version=get_version(),
                        url="https://github.com/silx-kit/silx",
                        author="data analysis unit",
                        author_email="silx@esrf.fr",
                        classifiers=classifiers,
                        description="Software library for X-Ray data analysis",
                        long_description=get_readme(),
                        install_requires=install_requires,
                        setup_requires=setup_requires,
                        cmdclass=cmdclass,
                        package_data=package_data,
                        zip_safe=False,
                        entry_points=entry_points,
                        )
    return setup_kwargs


def setup_package():
    """Run setup(**kwargs)

    Depending on the command, it either runs the complete setup which depends on numpy,
    or a *dry run* setup with no dependency on numpy.
    """

    # Check if action requires build/install
    dry_run = len(sys.argv) == 1 or (len(sys.argv) >= 2 and (
        '--help' in sys.argv[1:] or
        sys.argv[1] in ('--help-commands', 'egg_info', '--version',
                        'clean', '--name')))

    if dry_run:
        # DRY_RUN implies actions which do not require dependancies, like NumPy
        try:
            from setuptools import setup
            logger.info("Use setuptools.setup")
        except ImportError:
            from distutils.core import setup
            logger.info("Use distutils.core.setup")
    else:
        try:
            from setuptools import setup
        except ImportError:
            from numpy.distutils.core import setup
            logger.info("Use numpydistutils.setup")

    setup_kwargs = get_project_configuration(dry_run)
    setup(**setup_kwargs)

if __name__ == "__main__":
    setup_package()
