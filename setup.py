#!/usr/bin/python
# coding: utf8

__author__ = "Jérôme Kieffer"
__date__ = "09/10/2015"
__license__ = "MIT"

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

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

with open("README.md", "r")as fp:
    long_description = fp.read()

setup(name='silx',
      version='0.0.1',
      url="https://github.com/silex-kit/silx",
      author="data analysis unit",
      author_email="silx@esrf.fr",
      classifiers = classifiers,
      description="Software library for X-Ray data analysis",
      long_description=long_description,
      packages=["silx", "silx.io", "silx.third_party", "silx.visu"],

      )


