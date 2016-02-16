from numpy.distutils.misc_util import Configuration


def configuration(parent_package='', top_path=None):
    config = Configuration('gui', parent_package, top_path)
    config.add_subpackage('plot')
    config.add_subpackage('test')

    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup

    setup(configuration=configuration)
