py.extension_module('chistogramnd', 
        ['chistogramnd.pyx', 'src/histogramnd_c.c'],
        subdir: 'silx/math',
        dependencies : [py_dep],
        include_directories: ['include'], 
        install: true, 
        )

py.extension_module('chistogramnd_lut', 
        ['chistogramnd_lut.pyx'],
        subdir: 'silx/math',
        dependencies : [py_dep],
        include_directories: ['include'], 
        install: true, 
        )
