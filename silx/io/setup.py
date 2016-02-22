import os
import sys

import numpy
from numpy.distutils.misc_util import Configuration

# Locale and platform management
SPECFILE_USE_GNU_SOURCE = os.getenv("SPECFILE_USE_GNU_SOURCE")
if SPECFILE_USE_GNU_SOURCE is None:
    SPECFILE_USE_GNU_SOURCE = 0
    if sys.platform.lower().startswith("linux"):
        print("WARNING:")
        print("A cleaner locale independent implementation")
        print("may be achieved setting SPECFILE_USE_GNU_SOURCE to 1")
        print("For instance running this script as:")
        print("SPECFILE_USE_GNU_SOURCE=1 python setup.py build")
else:
    SPECFILE_USE_GNU_SOURCE = int(SPECFILE_USE_GNU_SOURCE)

if sys.platform == "win32":
    define_macros = [('WIN32',None)]
elif os.name.lower().startswith('posix'):
    define_macros = [('SPECFILE_POSIX', None)]
    #this one is more efficient but keeps the locale
    #changed for longer time
    #define_macros = [('PYMCA_POSIX', None)]
    #the best choice is to have _GNU_SOURCE defined
    #as a compilation flag because that allows the
    #use of strtod_l
    if SPECFILE_USE_GNU_SOURCE:
        define_macros = [('_GNU_SOURCE', 1)]
else:
    define_macros = []


def configuration(parent_package='', top_path=None):
    config = Configuration('io', parent_package, top_path)
    config.add_subpackage('test')

    srcfiles = ['sfheader','sfinit','sflists','sfdata','sfindex',
                'sflabel' ,'sfmca', 'sftools','locale_management']
    sources = ['specfile/src/'+ffile+'.c' for ffile in srcfiles]
    sources.append('specfile/specfile.pyx')


    config.add_extension('specfile',
                         sources=sources,
                         define_macros = define_macros,
                         include_dirs = ['specfile/include', numpy.get_include()],
                         language='c')

    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup

    setup(configuration=configuration)
