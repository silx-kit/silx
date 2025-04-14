#
#    def silx_io_specfile_define_macros():
#        # Locale and platform management
#        if sys.platform == "win32":
#            return [("WIN32", None), ("SPECFILE_POSIX", None)]
#        elif os.name.lower().startswith("posix"):
#            # the best choice is to have _GNU_SOURCE defined
#            # as a compilation flag because that allows the
#            # use of strtod_l
#            use_gnu_source = os.environ.get("SPECFILE_USE_GNU_SOURCE", "False")
#            if use_gnu_source in ("True", "1"):  # 1 was the initially supported value
#                return [("_GNU_SOURCE", 1)]
#            return [("SPECFILE_POSIX", None)]
#        else:
#            return []
#if build_machine.system() == "windows":
#TODO


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
        #define_macros=
        )
