# Definitions of macros specific to specfile
if target_machine.system() == 'windows'
    macros = [ '-DSPECFILE_POSIX', '-DWIN32']
elif target_machine.system() in ['linux', 'darwin', 'dragonfly', 'freebsd', 'gnu']
    use_gnu_source = get_option('specfile_use_gnu_source')
    if use_gnu_source
        macros = ['-D_GNU_SOURCE=1']
    else
        macros = ['-DSPECFILE_POSIX']
    endif
else
    macros = []
endif


py.extension_module('specfile', 
            [
                'src/sfheader.c',
                'src/sfinit.c',
                'src/sflists.c',
                'src/sfdata.c',
                'src/sfindex.c',
                'src/sflabel.c',
                'src/sfmca.c',
                'src/sftools.c',
                'src/locale_management.c',
                'specfile.pyx',
            ],
        subdir: 'silx/io',
        include_directories: ['include'],
        dependencies : [py_dep], 
        install: true, 
        c_args: macros
        )
