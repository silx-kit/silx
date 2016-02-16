from numpy.distutils.misc_util import Configuration


def configuration(parent_package='', top_path=None):
    config = Configuration('silx', parent_package, top_path)
    config.add_subpackage('gui')
    config.add_subpackage('io')
    config.add_subpackage('test')
    config.add_subpackage('third_party')

    config.add_extension('dummy',
                         sources=['dummy.pyx'],
                         extra_compile_args=['-fopenmp'],
                         extra_link_args=['-fopenmp'],
                         language='c',
                         )

    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup

    setup(configuration=configuration)
