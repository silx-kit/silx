#!/bin/sh
#
#    Project: Silx
#             https://github.com/silx-kit/silx
#
#    Copyright (C) 2015-2016 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE

# Script that builds a debian package from this library

project=silx
version=$(python -c"import version; print(version.version)")
strictversion=$(python -c"import version; print(version.strictversion)")
debianversion=$(python -c"import version; print(version.debianversion)")
tarname=${project}_${debianversion}.orig.tar.gz
deb_name=$(echo "$project" | tr '[:upper:]' '[:lower:]')

# target system
debian_version=$(grep -o '[0-9]*' /etc/issue)
target_system=debian${debian_version}

project_directory="`dirname \"$0\"`"
project_directory="`( cd \"$project_directory\" && pwd )`" # absolutized
dist_directory=${project_directory}/dist/${target_system}
build_directory=${project_directory}/build/${target_system}

if [ -d /usr/lib/ccache ];
then
   export PATH=/usr/lib/ccache:$PATH
fi


usage="usage: $(basename "$0") [options]

Build the Debian ${debian_version} package of the ${project} library.

If the build succeed the directory dist/debian7 will
contains the pachages.

optional arguments:
    --help     show this help text
    --install  install the packages generated at the end of
               the process using 'sudo dpkg'
"

install=0

while :
do
    case "$1" in
      -h | --help)
          echo "$usage"
          exit 0
          ;;
      --install)
          install=1
          shift
          ;;
      -*)
          echo "Error: Unknown option: $1" >&2
          echo "$usage"
          exit 1
          ;;
      *)  # No more options
          break
          ;;
    esac
done



# clean up previous build
rm -rf ${build_directory}
# create the build context
mkdir -p ${build_directory}
python setup.py debian_src
cp -f dist/${tarname} ${build_directory}
if [ -f dist/${project}-testimages.tar.gz ]
then
  cp -f dist/${project}-testimages.tar.gz ${build_directory}
fi

cd ${build_directory}
tar -xzf ${tarname}
newname=${deb_name}_${debianversion}.orig.tar.gz
directory=${project}-${strictversion}
echo tarname $tarname newname $newname
if [ $tarname != $newname ]
then
  if [ -h $newname ]
  then
    rm ${newname}
  fi
    ln -s ${tarname} ${newname}
fi

if [ -f ${project}-testimages.tar.gz ]
then
  if [ ! -h  ${deb_name}_${debianversion}.orig-testimages.tar.gz ]
  then
    ln -s ${project}-testimages.tar.gz ${deb_name}_${debianversion}.orig-testimages.tar.gz
  fi
fi

cd ${directory}
cp -r ${project_directory}/package/${target_system} debian
cp ${project_directory}/copyright debian

#handle test images
if [ -f ../${deb_name}_${debianversion}.orig-testimages.tar.gz ]
then
  if [ ! -d testimages ]
  then
    mkdir testimages
  fi
  cd testimages
  tar -xzf  ../${deb_name}_${debianversion}.orig-testimages.tar.gz
  cd ..
else
  # Disable to skip tests during build
  echo No test data
  #export PYBUILD_DISABLE_python2=test
  #export PYBUILD_DISABLE_python3=test
  #export DEB_BUILD_OPTIONS=nocheck
fi

dch -v ${debianversion}-1 "upstream development build of ${project} ${version}"
dch --bpo "${project} snapshot ${version} built for ${target_system}"
dpkg-buildpackage -r
rc=$?

if [ $rc -eq 0 ]
then

  # move packages to dist directory
  rm -rf ${dist_directory}
  mkdir -p ${dist_directory}
  mv ${build_directory}/*.deb ${dist_directory}
  mv ${build_directory}/*.x* ${dist_directory}
  mv ${build_directory}/*.dsc ${dist_directory}
  mv ${build_directory}/*.changes ${dist_directory}
  cd ../../..

else
  echo Build failed, please investigate ...
  exit "$rc"
fi

if [ $install = 1 ]; then
  sudo -v su -c  "dpkg -i ${dist_directory}/*.deb"
fi

exit 0
