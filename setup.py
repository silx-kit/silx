#!/usr/bin/python
# coding: utf8

__author__ = "Jérôme Kieffer"
__date__ = "27/11/2015"
__license__ = "MIT"


import sys
import os
import shutil

from numpy.distutils.misc_util import Configuration

try:
    from setuptools import setup
    from setuptools.command.build_py import build_py as _build_py
except ImportError:
    from numpy.distutils.core import setup
    from distutils.command.build_py import build_py as _build_py

PROJECT = "silx"
cmdclass = {}


def get_version():
    import version
    return version.strictversion


def get_readme():
    dirname = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(dirname, "README.md"), "r") as fp:
        long_description = fp.read()
    return long_description


classifiers = ["Development Status :: 1 - Planning",
               "Environment :: Console",
               "Environment :: MacOS X",
               "Environment :: Win32 (MS Windows)",
               "Environment :: X11 Applications :: Qt",
               "Intended Audience :: Education",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: MIT License",
               "Natural Language :: English",
               "Operating System :: Microsoft :: Windows",
               "Operating System :: POSIX",
               "Programming Language :: Cython",
               "Programming Language :: Python :: 2",
               "Programming Language :: Python :: 3",
               "Programming Language :: Python :: Implementation :: CPython",
               "Topic :: Documentation :: Sphinx",
               "Topic :: Scientific/Engineering :: Physics",
               "Topic :: Software Development :: Libraries :: Python Modules",
               ]


class build_py(_build_py):
    """
    Enhanced build_py which copies version to the built
    """
    def build_package_data(self):
        """Copy data files into build directory
        Patched in such a way version.py -> silx/_version.py"""
        print(self.data_files)
        _build_py.build_package_data(self)
        for package, src_dir, build_dir, filenames in self.data_files:
            if package == PROJECT:
                filename = "version.py"
                target = os.path.join(build_dir, "_" + filename)
                self.mkpath(os.path.dirname(target))
                self.copy_file(os.path.join(filename), target,
                               preserve_mode=False)
                break

cmdclass['build_py'] = build_py

# ################### #
# build_doc commandes #
# ################### #
try:
    import sphinx
    import sphinx.util.console
    sphinx.util.console.color_terminal = lambda: False
    from sphinx.setup_command import BuildDoc
except ImportError:
    sphinx = None

if sphinx:
    class build_doc(BuildDoc):

        def run(self):
            # make sure the python path is pointing to the newly built
            # code so that the documentation is built on this and not a
            # previously installed version

            build = self.get_finalized_command('build')
            sys.path.insert(0, os.path.abspath(build.build_lib))

#             # Copy gui files to the path:
#             dst = os.path.join(os.path.abspath(build.build_lib), "pyFAI", "gui")
#             if not os.path.isdir(dst):
#                 os.makedirs(dst)
#             for i in os.listdir("gui"):
#                 if i.endswith(".ui"):
#                     src = os.path.join("gui", i)
#                     idst = os.path.join(dst, i)
#                     if not os.path.exists(idst):
#                         shutil.copy(src, idst)

            # Build the Users Guide in HTML and TeX format
            for builder in ('html', 'latex'):
                self.builder = builder
                self.builder_target_dir = os.path.join(self.build_dir, builder)
                self.mkpath(self.builder_target_dir)
                BuildDoc.run(self)
            sys.path.pop(0)
    cmdclass['build_doc'] = build_doc


# numpy.distutils Configuration

def configuration(parent_package='', top_path=None):
    config = Configuration(None, parent_package, top_path)
    config.set_options(
        ignore_setup_xxx_py=True,
        assume_default_configuration=True,
        delegate_options_to_subpackages=True,
        quiet=True)
    config.add_subpackage('silx')
    return config

setup_kwargs = configuration().todict()


install_requires = ["numpy", "h5py"]
setup_requires = ["numpy", "cython"]

setup_kwargs.update(dict(
    name=PROJECT,
    version=get_version(),
    url="https://github.com/silex-kit/silx",
    author="data analysis unit",
    author_email="silx@esrf.fr",
    classifiers=classifiers,
    description="Software library for X-Ray data analysis",
    long_description=get_readme(),
    install_requires=install_requires,
    setup_requires=setup_requires,
    cmdclass=cmdclass,
    ))

setup(**setup_kwargs)
