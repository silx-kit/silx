project('silx',
  'c', 'cpp', 'cython',
  license: 'MIT',
  meson_version: '>= 1.1',
  version: run_command(['src/silx/_version.py', '--wheel'],
                       check:true).stdout().strip(),
  default_options: ['buildtype=plain', ],
)

py_mod = import('python')
py = py_mod.find_installation(pure: false)
os = import('fs')

# Seek the backend
if meson.backend() != 'ninja'
  error('Ninja backend required')
endif


omp = dependency('openmp', required: get_option('use_openmp'))


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


silx_dir = py.get_install_dir() / 'silx'

subdir('src/silx')
install_subdir('examples', install_dir: silx_dir)
