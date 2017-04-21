#!/bin/sh
#
#    Project: Silx
#             https://github.com/silx-kit/silx
#
#    Copyright (C) 2015-2.16 European Synchrotron Radiation Facility, Grenoble, France
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

PROJECT=silx
VERSION=$(python -c"import version; print(version.version)")
STRICT_VERSION=$(python -c"import version; print(version.strictversion)")
DEBIAN_VERSION=$(python -c"import version; print(version.debianversion)")
TARNAME=${PROJECT}-${STRICT_VERSION}.tar.gz
DEB_NAME=$(echo "$PROJECT" | tr '[:upper:]' '[:lower:]')

if [ -d /usr/lib/ccache ];
then 
   CCPATH=/usr/lib/ccache:$PATH
else  
   CCPATH=$PATH
fi
export PYBUILD_DISABLE_python2=test
export PYBUILD_DISABLE_python3=test
export DEB_BUILD_OPTIONS=nocheck

usage="usage: $(basename "$0") [options]

Build the Debian 7 package of the ${PROJECT} library.

If the build succeed the directory dist/debian7 will
contains the packages.

optional arguments:
    --help     show this help text
    --install  install the packages generated at the end of
               the process using 'sudo dpkg'
    --python3  use python3 to build the packages
               (it looks to be broken)
"


use_python3=0
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
      --python3)
          use_python3=1
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


BUILD_DIRECTORY=build/debian7
DIST_DIRECTORY=dist/debian7

# clean up previous build
rm -rf ${BUILD_DIRECTORY}

# create the build context
mkdir -p ${BUILD_DIRECTORY}
python setup.py sdist
cp -f dist/${TARNAME} ${BUILD_DIRECTORY}
cd ${BUILD_DIRECTORY}
tar -xzf ${TARNAME}
cd ${PROJECT}-${STRICT_VERSION}

if [ $use_python3 = 1 ]
then
  echo Using Python 2+3
  PATH=$CCPATH  python3 setup.py --command-packages=stdeb.command sdist_dsc --with-python2=True --with-python3=True --no-python3-scripts=True bdist_deb --no-cython
  rc=$?
else
  echo Using Python 2
  # bdist_deb feed /usr/bin using setup.py entry-points
  PATH=$CCPATH python setup.py --command-packages=stdeb.command bdist_deb --no-cython
  rc=$?
fi

# move packages to dist directory
rm -rf ../../../${DIST_DIRECTORY}
mkdir -p ../../../${DIST_DIRECTORY}
mv -f deb_dist/*.deb ../../../${DIST_DIRECTORY}

# back to the root
cd ../../..

if [ $install = 1 ]; then
  sudo -v su -c  "dpkg -i dist/debian7/python*-${PROJECT}*.deb"
fi

exit "$rc"
