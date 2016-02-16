import numpy
from numpy.distutils.misc_util import Configuration


def configuration(parent_package='', top_path=None):
    config = Configuration('io', parent_package, top_path)
    config.add_subpackage('test')

    srcfiles = ['sfheader','sfinit','sflists','sfdata','sfindex',
                'sflabel' ,'sfmca', 'sftools','locale_management']
    sources = ['specfile/src/'+ffile+'.c' for ffile in srcfiles]
    sources.append('specfile/specfile.pyx')


    config.add_extension('specfile',
                         sources=sources,
                         include_dirs = ['specfile/include', numpy.get_include()],
                         language='c')

    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup

    setup(configuration=configuration)
