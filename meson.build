project('silx',
  'c', 'cpp', 'cython',
  license: 'MIT',
  meson_version: '>= 1.1',
  version: run_command(['src/silx/_version.py', '--wheel'],
                       check:true).stdout().strip(),
  default_options: ['buildtype=plain', ],
)

py_mod = import('python')
py = py_mod.find_installation()
os = import('fs')

# Seek the backend
if meson.backend() != 'ninja'
  error('Ninja backend required')
endif

# OpenMP can be used? several ways to disable it:
#  -> use -Duse_openmp=disabled
#  -> or the file SILX_WITH_OPENMP should contain False
#  -> or OpenMP can be unsuported on your achitecture/compiler


# How to disable OpenMP:
# The 'SILX_WITH_OPENMP' file should contain 'False'
if os.is_file('SILX_WITH_OPENMP')
  openmp_var = os.read('SILX_WITH_OPENMP').strip()
else
  res = run_command(py, '-c', 'import os; print(os.environ["SILX_WITH_OPENMP"])',
                    check:false)
  if res.returncode() == 0
    openmp_var = res.stdout().strip()
  else
    openmp_var = ''
  endif
endif
omp = dependency('openmp', required: get_option('use_openmp').disable_auto_if(openmp_var=='False'))


cc = meson.get_compiler('c')
cpp = meson.get_compiler('cpp')
cy = meson.get_compiler('cython')
# We need -lm for all C code (assuming it uses math functions, which is safe).
# For C++ it isn't needed, because libstdc++/libc++ is guaranteed to depend on it.
m_dep = cc.find_library('m', required : false)
if m_dep.found()
  add_project_link_arguments('-lm', language : 'c')
endif

# https://mesonbuild.com/Python-module.html
py_dep = py.dependency()


subdir('src/silx')
