# Definitions of macros specific to specfile
macros = []
if target_machine.system() == 'windows'
    macros = [ '-DSPECFILE_POSIX', '-DWIN32']
elif target_machine.system() in ['linux', 'darwin', 'dragonfly', 'freebsd', 'gnu']
    if os.is_file('SPECFILE_USE_GNU_SOURCE')
        specfile_use_gnu = os.read('SPECFILE_USE_GNU_SOURCE').strip()
    else
        res = run_command(py, '-c', 'import os; print(os.environ["SPECFILE_USE_GNU_SOURCE"])',
                          check:false)
        if res.returncode() == 0
            specfile_use_gnu = res.stdout().strip()
        else
            specfile_use_gnu = ''
        endif
    endif
    if specfile_use_gnu in ['1', 'true', 'True', 'TRUE']
        macros += ['-D_GNU_SOURCE=1']
    else
        macros += ['-DSPECFILE_POSIX']
    endif
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
