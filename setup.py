#!/usr/bin/python
# coding: utf8

__author__ = "Jérôme Kieffer"
__date__ = "09/10/2015"
__license__ = "MIT"

import sys
import os
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


def get_version():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "silx"))
    import _version
    sys.path.pop(0)
    return _version.strictversion


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


install_requires = ["numpy", "h5py"]
setup_requires = ["numpy", "cython"]


setup(name='silx',
      version=get_version(),
      url="https://github.com/silex-kit/silx",
      author="data analysis unit",
      author_email="silx@esrf.fr",
      classifiers = classifiers,
      description="Software library for X-Ray data analysis",
      long_description=get_readme(),
      packages=["silx", "silx.io", "silx.third_party", "silx.visu"],
      install_requires=install_requires,
      setup_requires=setup_requires,
      )


